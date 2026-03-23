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

    def _sample_goal(self) -> np.ndarray:
        spawn = self.cfg["spawn"]
        return self.np_random.uniform(spawn["goal_xyz_low"], spawn["goal_xyz_high"]).astype(np.float64)

    def _sample_dynamic_velocity(self) -> np.ndarray:
        speed = float(
            self.np_random.uniform(
                self.cfg["spawn"]["dynamic_speed_low"],
                self.cfg["spawn"]["dynamic_speed_high"],
            )
        )
        direction = self.np_random.normal(size=3)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / direction_norm
        return speed * direction

    def _sample_obstacles(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> list[dict]:
        spawn = self.cfg["spawn"]
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
            center = self.np_random.uniform(
                spawn["obstacle_xyz_low"], spawn["obstacle_xyz_high"]
            ).astype(np.float64)
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
                obstacles.append({"c": center, "r": radius, "v": np.zeros(3, dtype=np.float64)})

        if obstacles:
            min_dynamic = min(int(spawn["dynamic_count_min"]), len(obstacles))
            dynamic_count = int(self.np_random.integers(min_dynamic, len(obstacles) + 1))
            dynamic_indices = self.np_random.choice(len(obstacles), size=dynamic_count, replace=False)
            for idx in np.asarray(dynamic_indices, dtype=np.int64):
                obstacles[int(idx)]["v"] = self._sample_dynamic_velocity()
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
        spawn = self.cfg["spawn"]
        bounds_low = np.asarray(spawn["obstacle_bounds_low"], dtype=np.float64)
        bounds_high = np.asarray(spawn["obstacle_bounds_high"], dtype=np.float64)
        for obstacle in self.obstacles:
            obstacle["c"] = obstacle["c"] + obstacle["v"] * self.dt
            legal_low = bounds_low + obstacle["r"]
            legal_high = bounds_high - obstacle["r"]
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

        self.goal_xyz = self._sample_goal()
        self.obstacles = self._sample_obstacles(start_xyz, self.goal_xyz)
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
