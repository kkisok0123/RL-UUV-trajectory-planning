from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamics_wrapper import backend_name as dynamics_backend_name
from dynamics_wrapper import step as dynamics_step
from dynamics_wrapper.params import validate_params
from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, STATE_DIM, VX, VY, VZ, WX, WY, WZ
from rl.configs.fish_env import build_fish_env_config
from rl.envs.geometry import (
    nearest_obstacle_info,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
    vector_angle,
)
from rl.envs.reward import compute_reward
from simulation.local_planning.fin_controller import fin_controller, velocity_to_attitude_refs


def _quat_from_forward_vector(forward_world: np.ndarray) -> np.ndarray:
    forward_world = np.asarray(forward_world, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(forward_world))
    if speed < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw = float(np.arctan2(forward_world[1], forward_world[0]))
    pitch = float(np.arctan2(forward_world[2], np.linalg.norm(forward_world[:2])))
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    quat = np.array([cp * cy, sp * sy, -sp * cy, cp * sy], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-8)


class FishAvoidEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.cfg = deepcopy(config) if config is not None else build_fish_env_config()
        self.dt = float(self.cfg["dt"])
        self.c_A = float(self.cfg["c_A"])
        self.fin_f = float(self.cfg["fin_f"])
        self.max_steps = int(self.cfg["max_steps"])
        self.fish_radius = float(self.cfg["fish_radius"])
        self.obs_clip = float(self.cfg["obs_clip"])

        self.body_params = np.asarray(self.cfg["body_params"], dtype=np.float64)
        self.fin_params = np.asarray(self.cfg["fin_params"], dtype=np.float64)
        validate_params(self.body_params, self.fin_params)
        self.initial_state_template = np.asarray(
            self.cfg.get(
                "initial_state_template",
                np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ),
            dtype=np.float64,
        )
        if self.initial_state_template.shape != (STATE_DIM,):
            raise ValueError(
                f"initial_state_template must have shape ({STATE_DIM},), got {self.initial_state_template.shape}"
            )

        self.trim_refs = np.asarray(self.cfg["trim_refs"], dtype=np.float64)
        self.ref_min = np.asarray(self.cfg["ref_min"], dtype=np.float64)
        self.ref_max = np.asarray(self.cfg["ref_max"], dtype=np.float64)
        self.action_velocity_limits = np.asarray(self.cfg["action_velocity_limits"], dtype=np.float64)
        self.controller_params = deepcopy(self.cfg["controller_params"])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-self.obs_clip, high=self.obs_clip, shape=(23,), dtype=np.float32
        )

        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.hist = self.trim_refs.copy()
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.goal_xyz = np.zeros(3, dtype=np.float64)
        self.obstacles: list[dict] = []
        self.scene_bounds_low = np.zeros(3, dtype=np.float64)
        self.scene_bounds_high = np.zeros(3, dtype=np.float64)
        self.prev_goal_dist = 0.0
        self.t_k = 0.0
        self.step_count = 0
        self.backend = dynamics_backend_name()
        self.ctrl_state = {
            "e_theta_prev": 0.0,
            "e_theta_int": 0.0,
            "e_psi_prev": 0.0,
            "e_psi_int": 0.0,
        }

    def _corridor_basis(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        corridor = np.asarray(goal_xyz, dtype=np.float64).reshape(3) - np.asarray(start_xyz, dtype=np.float64).reshape(3)
        corridor_len = float(np.linalg.norm(corridor))
        corridor_dir = corridor / max(corridor_len, 1e-8)
        seed_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(corridor_dir, seed_axis))) > 0.95:
            seed_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        side = np.cross(corridor_dir, seed_axis)
        side = side / max(np.linalg.norm(side), 1e-8)
        lift = np.cross(corridor_dir, side)
        return corridor_dir, side, lift, corridor_len

    def _sample_goal(self, start_xyz: np.ndarray) -> np.ndarray:
        spawn = self.cfg["spawn"]
        if "goal_distance_low" not in spawn:
            return self.np_random.uniform(spawn["goal_xyz_low"], spawn["goal_xyz_high"]).astype(np.float64)
        distance = float(self.np_random.uniform(spawn["goal_distance_low"], spawn["goal_distance_high"]))
        azimuth = float(self.np_random.uniform(-spawn["goal_azimuth_range"], spawn["goal_azimuth_range"]))
        elevation = float(self.np_random.uniform(-spawn["goal_elevation_range"], spawn["goal_elevation_range"]))
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )
        return np.asarray(start_xyz, dtype=np.float64).reshape(3) + distance * direction

    def _sample_dynamic_velocity(
        self,
        corridor_dir: np.ndarray,
        side: np.ndarray,
        lift: np.ndarray,
    ) -> np.ndarray:
        speed = float(
            self.np_random.uniform(
                self.cfg["spawn"]["dynamic_speed_low"],
                self.cfg["spawn"]["dynamic_speed_high"],
            )
        )
        lateral_sign = -1.0 if bool(self.np_random.integers(0, 2)) else 1.0
        direction = (
            self.np_random.uniform(-0.30, 0.20) * np.asarray(corridor_dir, dtype=np.float64).reshape(3)
            + lateral_sign * self.np_random.uniform(0.85, 1.20) * np.asarray(side, dtype=np.float64).reshape(3)
            + self.np_random.uniform(-0.25, 0.25) * np.asarray(lift, dtype=np.float64).reshape(3)
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            direction = np.asarray(side, dtype=np.float64).reshape(3)
        else:
            direction = direction / direction_norm
        return speed * direction

    def _sample_initial_forward(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> np.ndarray:
        spawn = self.cfg["spawn"]
        corridor_dir, side, lift, _ = self._corridor_basis(start_xyz, goal_xyz)
        lateral = np.tan(float(self.np_random.uniform(-spawn["initial_heading_range"], spawn["initial_heading_range"])))
        vertical = np.tan(float(self.np_random.uniform(-spawn["initial_pitch_range"], spawn["initial_pitch_range"])))
        forward = corridor_dir + lateral * side + vertical * lift
        return forward / max(np.linalg.norm(forward), 1e-8)

    def _reset_scene_bounds(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> None:
        margin = float(self.cfg["spawn"].get("scene_margin", 3.0))
        self.scene_bounds_low = np.minimum(start_xyz, goal_xyz) - margin
        self.scene_bounds_high = np.maximum(start_xyz, goal_xyz) + margin

    def _sample_obstacles(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> list[dict]:
        spawn = self.cfg["spawn"]
        corridor_dir, side, lift, corridor_len = self._corridor_basis(start_xyz, goal_xyz)
        num_obstacles = int(
            self.np_random.integers(spawn["num_obstacles_low"], spawn["num_obstacles_high"] + 1)
        )
        obstacles = []
        attempts = 0
        while len(obstacles) < num_obstacles and attempts < 200:
            attempts += 1
            radius = float(
                self.np_random.uniform(spawn["obstacle_radius_low"], spawn["obstacle_radius_high"])
            )
            along = float(self.np_random.uniform(spawn.get("corridor_along_low", 0.2), spawn.get("corridor_along_high", 0.9))) * corridor_len
            lateral = float(self.np_random.uniform(-spawn.get("corridor_lateral_span", 2.5), spawn.get("corridor_lateral_span", 2.5)))
            vertical = float(self.np_random.uniform(-spawn.get("corridor_vertical_span", 1.8), spawn.get("corridor_vertical_span", 1.8)))
            center = (
                np.asarray(start_xyz, dtype=np.float64).reshape(3)
                + corridor_dir * along
                + side * lateral
                + lift * vertical
            )
            if np.linalg.norm(center - start_xyz) <= spawn["start_clearance"] + radius:
                continue
            if np.linalg.norm(center - goal_xyz) <= spawn["goal_clearance"] + radius:
                continue
            valid = True
            for obstacle in obstacles:
                min_gap = obstacle["r"] + radius + spawn["obstacle_clearance"]
                if np.linalg.norm(center - obstacle["c"]) <= min_gap:
                    valid = False
                    break
            if valid:
                obstacles.append({"c": center.astype(np.float64), "r": radius, "v": np.zeros(3, dtype=np.float64)})

        if obstacles:
            dynamic_low = int(spawn.get("dynamic_count_low", spawn.get("dynamic_count_min", 0)))
            dynamic_high = int(spawn.get("dynamic_count_high", len(obstacles)))
            dynamic_low = max(0, min(dynamic_low, len(obstacles)))
            dynamic_high = max(dynamic_low, min(dynamic_high, len(obstacles)))
            dynamic_count = int(self.np_random.integers(dynamic_low, dynamic_high + 1))
            if dynamic_count > 0:
                dynamic_indices = self.np_random.choice(len(obstacles), size=dynamic_count, replace=False)
                for idx in np.asarray(dynamic_indices, dtype=np.int64):
                    obstacles[int(idx)]["v"] = self._sample_dynamic_velocity(corridor_dir, side, lift)
        return obstacles

    def _rotation_body_to_world(self, state: np.ndarray | None = None) -> np.ndarray:
        state = self.state if state is None else np.asarray(state, dtype=np.float64)
        return quaternion_to_rotation_matrix(state[Q0], state[Q1], state[Q2], state[Q3])

    def _world_velocity(self, state: np.ndarray | None = None) -> np.ndarray:
        state = self.state if state is None else np.asarray(state, dtype=np.float64)
        rotation = self._rotation_body_to_world(state)
        return rotate_body_to_world(state[[VX, VY, VZ]], rotation)

    def _map_action_to_references(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cmd_vel_body = action * self.action_velocity_limits
        cmd_vel_world = rotate_body_to_world(cmd_vel_body, self._rotation_body_to_world())
        psi_ref, theta_ref, cmd_speed = velocity_to_attitude_refs(cmd_vel_world, self.state)
        a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, self.ctrl_state = fin_controller(
            psi_ref,
            theta_ref,
            cmd_speed,
            self.state,
            self.ctrl_state,
            self.dt,
            self.controller_params,
        )
        action_ref = np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64)
        return np.clip(action_ref, self.ref_min, self.ref_max), cmd_vel_body, cmd_vel_world

    def _update_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle["c"] = obstacle["c"] + obstacle["v"] * self.dt
            legal_low = self.scene_bounds_low + obstacle["r"]
            legal_high = self.scene_bounds_high - obstacle["r"]
            for axis in range(3):
                if obstacle["c"][axis] < legal_low[axis]:
                    obstacle["c"][axis] = legal_low[axis]
                    obstacle["v"][axis] = abs(obstacle["v"][axis])
                elif obstacle["c"][axis] > legal_high[axis]:
                    obstacle["c"][axis] = legal_high[axis]
                    obstacle["v"][axis] = -abs(obstacle["v"][axis])

    def _build_obs(self) -> tuple[np.ndarray, dict]:
        position_world = self.state[[PX, PY, PZ]]
        rotation = self._rotation_body_to_world()
        velocity_world = self._world_velocity()
        goal_rel_world = self.goal_xyz - position_world
        goal_rel_body = rotate_world_to_body(goal_rel_world, rotation)
        goal_dist = float(np.linalg.norm(goal_rel_world))

        nearest = nearest_obstacle_info(position_world, velocity_world, self.obstacles, self.fish_radius)
        obs_rel_body = rotate_world_to_body(nearest["rel_world"], rotation)
        obs_rel_vel_body = rotate_world_to_body(nearest["rel_vel_world"], rotation)

        obs = np.array(
            [
                self.state[VX],
                self.state[VY],
                self.state[VZ],
                self.state[WX],
                self.state[WY],
                self.state[WZ],
                goal_rel_body[0],
                goal_rel_body[1],
                goal_rel_body[2],
                goal_dist,
                obs_rel_body[0],
                obs_rel_body[1],
                obs_rel_body[2],
                nearest["clearance"],
                nearest["r"],
                obs_rel_vel_body[0],
                obs_rel_vel_body[1],
                obs_rel_vel_body[2],
                self.hist[0],
                self.hist[1],
                self.hist[2],
                self.hist[3],
                self.hist[4],
            ],
            dtype=np.float64,
        )
        obs = np.clip(obs, -self.obs_clip, self.obs_clip).astype(np.float32)
        info = {
            "goal_dist": goal_dist,
            "goal": self.goal_xyz.copy(),
            "goal_rel_world": goal_rel_world,
            "goal_rel_body": goal_rel_body,
            "nearest_obstacle": nearest,
            "velocity_world": velocity_world,
            "backend": self.backend,
        }
        return obs, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        spawn = self.cfg["spawn"]
        self.step_count = 0
        self.t_k = 0.0
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.hist = self.trim_refs.copy()
        self.ctrl_state = {
            "e_theta_prev": 0.0,
            "e_theta_int": 0.0,
            "e_psi_prev": 0.0,
            "e_psi_int": 0.0,
        }

        start_xyz = self.np_random.uniform(spawn["start_xyz_low"], spawn["start_xyz_high"]).astype(np.float64)
        self.state = self.initial_state_template.copy()
        self.state[[PX, PY, PZ]] = start_xyz

        self.goal_xyz = self._sample_goal(start_xyz)
        initial_forward = self._sample_initial_forward(start_xyz, self.goal_xyz)
        initial_speed = float(self.np_random.uniform(spawn.get("initial_speed_low", 0.01), spawn.get("initial_speed_high", 0.05)))
        self.state[Q0 : Q3 + 1] = _quat_from_forward_vector(initial_forward)
        self.state[VX] = initial_speed
        self.state[VY] = 0.0
        self.state[VZ] = 0.0
        self.state[WX] = 0.0
        self.state[WY] = 0.0
        self.state[WZ] = 0.0
        self.obstacles = self._sample_obstacles(start_xyz, self.goal_xyz)
        self._reset_scene_bounds(start_xyz, self.goal_xyz)
        obs, info = self._build_obs()
        self.prev_goal_dist = float(info["goal_dist"])
        return obs, info

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action_ref, cmd_vel_body, cmd_vel_world = self._map_action_to_references(action)

        next_state, next_hist = dynamics_step(
            self.state,
            self.body_params,
            self.fin_params,
            self.dt,
            self.t_k,
            action_ref,
            self.hist,
            self.c_A,
            self.fin_f,
        )
        self.t_k += self.dt
        self.state = next_state
        self.hist = next_hist
        self._update_obstacles()

        obs, info = self._build_obs()
        goal_dist = float(info["goal_dist"])
        nearest = info["nearest_obstacle"]
        clearance = float(nearest["clearance"])
        direction_error = vector_angle(info["velocity_world"], info["goal_rel_world"])
        angular_rate_norm = float(np.linalg.norm(self.state[[WX, WY, WZ]]))

        reward, terms = compute_reward(
            self.prev_goal_dist,
            goal_dist,
            clearance,
            direction_error,
            angular_rate_norm,
            action,
            self.prev_action,
            action_ref,
            self.cfg,
        )
        self.prev_goal_dist = goal_dist
        self.prev_action = action

        reached_goal = goal_dist < self.cfg["goal_radius"]
        collided = clearance <= 0.0
        terminated = reached_goal or collided
        truncated = self.step_count >= self.max_steps

        info.update(
            {
                "action_ref": action_ref,
                "cmd_vel_body": cmd_vel_body,
                "cmd_vel_world": cmd_vel_world,
                "reward_terms": terms,
                "reached_goal": reached_goal,
                "collided": collided,
                "t_k": self.t_k,
            }
        )
        return obs, reward, terminated, truncated, info
