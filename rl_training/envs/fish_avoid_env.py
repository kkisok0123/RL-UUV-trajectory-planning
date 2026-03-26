from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamics import backend_name as dynamics_backend_name
from dynamics import step as dynamics_step
from dynamics.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, STATE_DIM, VX, VY, VZ, WX, WY, WZ
from dynamics.params import validate_params
from rl_training.config.fish_env_config import build_fish_env_config
from rl_training.envs.geometry import (
    nearest_obstacle_info,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
    vector_angle,
)
from rl_training.envs.reward import compute_reward
from simulation.fin_controller.controller import fin_controller, velocity_to_attitude_refs
from simulation.local_planner.common import get_visible_obstacles


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
        self.sensor_cfg = deepcopy(self.cfg.get("sensor", {}))
        self.observation_noise_cfg = deepcopy(self.cfg.get("observation_noise", {}))
        self.disturbance_cfg = deepcopy(self.cfg.get("disturbance", {}))
        self.termination_cfg = deepcopy(self.cfg.get("termination", {}))

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
        self.current_world = np.zeros(3, dtype=np.float64)
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
        local_direction = self.np_random.normal(0.0, 1.0, size=3).astype(np.float64)
        local_direction *= np.array([1.0, 1.15, 0.9], dtype=np.float64)
        direction = (
            local_direction[0] * np.asarray(corridor_dir, dtype=np.float64).reshape(3)
            + local_direction[1] * np.asarray(side, dtype=np.float64).reshape(3)
            + local_direction[2] * np.asarray(lift, dtype=np.float64).reshape(3)
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            fallback = self.np_random.normal(0.0, 1.0, size=3).astype(np.float64)
            fallback_norm = float(np.linalg.norm(fallback))
            direction = fallback / max(fallback_norm, 1e-8)
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

    def _sample_current_world(self) -> np.ndarray:
        current_low = float(self.disturbance_cfg.get("current_speed_low", 0.0))
        current_high = float(self.disturbance_cfg.get("current_speed_high", 0.0))
        if current_high <= 1e-9:
            return np.zeros(3, dtype=np.float64)

        speed = float(self.np_random.uniform(current_low, current_high))
        azimuth = float(self.np_random.uniform(-np.pi, np.pi))
        elevation = float(
            self.np_random.uniform(
                -float(self.disturbance_cfg.get("current_elevation_range", 0.0)),
                float(self.disturbance_cfg.get("current_elevation_range", 0.0)),
            )
        )
        return speed * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )

    def _update_current_world(self) -> None:
        walk_std = float(self.disturbance_cfg.get("current_walk_std", 0.0))
        decay = float(self.disturbance_cfg.get("current_decay", 1.0))
        if walk_std > 0.0:
            self.current_world = decay * self.current_world + self.np_random.normal(
                0.0, walk_std, size=3
            ).astype(np.float64)

        current_speed_max = float(self.disturbance_cfg.get("current_speed_max", 0.0))
        current_norm = float(np.linalg.norm(self.current_world))
        if current_speed_max > 0.0 and current_norm > current_speed_max:
            self.current_world *= current_speed_max / max(current_norm, 1e-8)

    def _collect_visible_obstacles(
        self,
        position_world: np.ndarray,
        velocity_world: np.ndarray,
    ) -> tuple[list[dict], dict]:
        safe_zone = get_visible_obstacles(position_world, velocity_world, self.obstacles, self.sensor_cfg)
        blocked_idx = np.flatnonzero(safe_zone["is_blocked"])
        if blocked_idx.size == 0:
            return [], safe_zone

        hit_points = (
            safe_zone["origin"].reshape(1, 3)
            + safe_zone["rays"][:, blocked_idx].T * safe_zone["dists"][blocked_idx].reshape(-1, 1)
        )
        hit_tolerance = float(self.sensor_cfg.get("hit_tolerance", 0.1))
        miss_rate = float(self.sensor_cfg.get("miss_rate", 0.0))
        visible_obstacles: list[dict] = []

        for obstacle in self.obstacles:
            center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
            boundary_error = np.abs(np.linalg.norm(hit_points - center.reshape(1, 3), axis=1) - float(obstacle["r"]))
            if not np.any(boundary_error < hit_tolerance):
                continue
            if miss_rate > 0.0 and float(self.np_random.uniform()) < miss_rate:
                continue
            visible_obstacles.append(
                {
                    "c": center.copy(),
                    "r": float(obstacle["r"]),
                    "v": np.asarray(obstacle.get("v", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3).copy(),
                }
            )
        return visible_obstacles, safe_zone

    def _apply_noise(self, value: np.ndarray | float, std: float, nonnegative: bool = False):
        if std <= 0.0:
            return value

        arr = np.asarray(value, dtype=np.float64)
        if not np.isfinite(arr).all():
            return value

        noisy = arr + self.np_random.normal(0.0, std, size=arr.shape)
        if nonnegative:
            noisy = np.maximum(noisy, 0.0)
        if np.isscalar(value):
            return float(np.asarray(noisy).reshape(-1)[0])
        return noisy

    def _fallback_visible_obstacle(self) -> dict:
        sensor_range = float(self.sensor_cfg.get("range", self.obs_clip))
        return {
            "c": np.full(3, np.nan, dtype=np.float64),
            "r": 0.0,
            "v": np.zeros(3, dtype=np.float64),
            "distance": sensor_range,
            "clearance": sensor_range,
            "rel_world": np.array([sensor_range, 0.0, 0.0], dtype=np.float64),
            "rel_vel_world": np.zeros(3, dtype=np.float64),
        }

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
        visible_obstacles, safe_zone = self._collect_visible_obstacles(position_world, velocity_world)

        nearest_visible = nearest_obstacle_info(position_world, velocity_world, visible_obstacles, self.fish_radius)
        if not np.isfinite(nearest_visible["distance"]):
            nearest_visible = self._fallback_visible_obstacle()
        nearest_true = nearest_obstacle_info(position_world, velocity_world, self.obstacles, self.fish_radius)
        obs_rel_body = rotate_world_to_body(nearest_visible["rel_world"], rotation)
        obs_rel_vel_body = rotate_world_to_body(nearest_visible["rel_vel_world"], rotation)
        velocity_std = float(self.observation_noise_cfg.get("velocity_std", 0.0))
        angular_rate_std = float(self.observation_noise_cfg.get("angular_rate_std", 0.0))
        goal_rel_std = float(self.observation_noise_cfg.get("goal_rel_std", 0.0))
        goal_dist_std = float(self.observation_noise_cfg.get("goal_dist_std", 0.0))
        obstacle_rel_std = float(self.observation_noise_cfg.get("obstacle_rel_std", 0.0))
        obstacle_vel_std = float(self.observation_noise_cfg.get("obstacle_vel_std", 0.0))
        clearance_std = float(self.observation_noise_cfg.get("clearance_std", 0.0))
        radius_std = float(self.observation_noise_cfg.get("radius_std", 0.0))
        hist_std = float(self.observation_noise_cfg.get("hist_std", 0.0))

        lin_vel_obs = self._apply_noise(self.state[[VX, VY, VZ]], velocity_std)
        ang_vel_obs = self._apply_noise(self.state[[WX, WY, WZ]], angular_rate_std)
        goal_rel_body_obs = self._apply_noise(goal_rel_body, goal_rel_std)
        goal_dist_obs = self._apply_noise(goal_dist, goal_dist_std, nonnegative=True)
        obs_rel_body_obs = self._apply_noise(obs_rel_body, obstacle_rel_std)
        obs_rel_vel_body_obs = self._apply_noise(obs_rel_vel_body, obstacle_vel_std)
        clearance_obs = self._apply_noise(nearest_visible["clearance"], clearance_std)
        radius_obs = self._apply_noise(nearest_visible["r"], radius_std, nonnegative=True)
        hist_obs = self._apply_noise(self.hist, hist_std)

        obs = np.array(
            [
                lin_vel_obs[0],
                lin_vel_obs[1],
                lin_vel_obs[2],
                ang_vel_obs[0],
                ang_vel_obs[1],
                ang_vel_obs[2],
                goal_rel_body_obs[0],
                goal_rel_body_obs[1],
                goal_rel_body_obs[2],
                goal_dist_obs,
                obs_rel_body_obs[0],
                obs_rel_body_obs[1],
                obs_rel_body_obs[2],
                clearance_obs,
                radius_obs,
                obs_rel_vel_body_obs[0],
                obs_rel_vel_body_obs[1],
                obs_rel_vel_body_obs[2],
                hist_obs[0],
                hist_obs[1],
                hist_obs[2],
                hist_obs[3],
                hist_obs[4],
            ],
            dtype=np.float64,
        )
        obs = np.clip(obs, -self.obs_clip, self.obs_clip).astype(np.float32)
        info = {
            "goal_dist": goal_dist,
            "goal": self.goal_xyz.copy(),
            "goal_rel_world": goal_rel_world,
            "goal_rel_body": goal_rel_body,
            "nearest_obstacle": nearest_true,
            "nearest_obstacle_visible": nearest_visible,
            "visible_obstacles": visible_obstacles,
            "visible_obstacle_count": len(visible_obstacles),
            "velocity_world": velocity_world,
            "backend": self.backend,
            "current_world": self.current_world.copy(),
            "sensor_hits": int(np.count_nonzero(safe_zone["is_blocked"])),
        }
        return obs, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        spawn = self.cfg["spawn"]
        self.step_count = 0
        self.t_k = 0.0
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.hist = self.trim_refs.copy()
        self.current_world = self._sample_current_world()
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
        if not np.isfinite(next_state).all() or not np.isfinite(next_hist).all():
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "goal_dist": float("inf"),
                "goal": self.goal_xyz.copy(),
                "goal_rel_world": np.full(3, np.nan, dtype=np.float64),
                "goal_rel_body": np.full(3, np.nan, dtype=np.float64),
                "nearest_obstacle": self._fallback_visible_obstacle(),
                "nearest_obstacle_visible": self._fallback_visible_obstacle(),
                "visible_obstacles": [],
                "visible_obstacle_count": 0,
                "velocity_world": np.zeros(3, dtype=np.float64),
                "backend": self.backend,
                "current_world": self.current_world.copy(),
                "sensor_hits": 0,
            }
            info.update(
                {
                    "action_ref": action_ref,
                    "cmd_vel_body": cmd_vel_body,
                    "cmd_vel_world": cmd_vel_world,
                    "reward_terms": {"numerical_failure": -self.cfg["reward"]["collision"]},
                    "reached_goal": False,
                    "collided": False,
                    "unsafe_terminated": False,
                    "numerical_issue": True,
                    "t_k": self.t_k,
                }
            )
            return obs, -self.cfg["reward"]["collision"], True, False, info

        self.state = next_state
        self.hist = next_hist
        self._update_current_world()
        self.state[[PX, PY, PZ]] = self.state[[PX, PY, PZ]] + self.current_world * self.dt
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
        collision_clearance = float(self.termination_cfg.get("collision_clearance", 0.0))
        unsafe_clearance = float(self.termination_cfg.get("unsafe_clearance", collision_clearance))
        collided = clearance <= collision_clearance
        unsafe_terminated = (clearance <= unsafe_clearance) and not collided
        if unsafe_terminated:
            terms["unsafe_termination"] = -float(self.cfg["reward"].get("unsafe_termination", 0.0))
            reward += terms["unsafe_termination"]
        terminated = reached_goal or collided or unsafe_terminated
        truncated = self.step_count >= self.max_steps

        info.update(
            {
                "action_ref": action_ref,
                "cmd_vel_body": cmd_vel_body,
                "cmd_vel_world": cmd_vel_world,
                "reward_terms": terms,
                "reached_goal": reached_goal,
                "collided": collided,
                "unsafe_terminated": unsafe_terminated,
                "t_k": self.t_k,
            }
        )
        return obs, reward, terminated, truncated, info
