from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamics_wrapper import backend_name as dynamics_backend_name
from dynamics_wrapper import step as dynamics_step
from dynamics_wrapper.params import validate_params
from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, STATE_DIM, VX, VY, VZ, WX, WY, WZ
from rl.action_mapping import map_structured_action_to_fins
from rl.configs.fish_env import build_fish_env_config
from rl.envs.geometry import (
    nearest_obstacle_info,
    quaternion_from_yaw,
    rotate_world_to_body_2d,
    wrap_angle,
    yaw_from_quaternion,
)
from rl.envs.reward import compute_reward


class FishAvoidEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.cfg = deepcopy(config) if config is not None else build_fish_env_config()
        self.dt = float(self.cfg["dt"])
        self.c_A = float(self.cfg["c_A"])
        self.fin_f = float(self.cfg["fin_f"])
        self.max_steps = int(self.cfg["max_steps"])
        self.plane_z = float(self.cfg["plane_z"])
        self.fish_radius = float(self.cfg["fish_radius"])
        self.obs_clip = float(self.cfg["obs_clip"])

        self.body_params = np.asarray(self.cfg["body_params"], dtype=np.float64)
        self.fin_params = np.asarray(self.cfg["fin_params"], dtype=np.float64)
        validate_params(self.body_params, self.fin_params)

        self.trim_refs = np.asarray(self.cfg["trim_refs"], dtype=np.float64)
        self.ref_min = np.asarray(self.cfg["ref_min"], dtype=np.float64)
        self.ref_max = np.asarray(self.cfg["ref_max"], dtype=np.float64)
        self.action_map = self.cfg["action_map"]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-self.obs_clip, high=self.obs_clip, shape=(17,), dtype=np.float32
        )

        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.hist = self.trim_refs.copy()
        self.prev_action = np.zeros(3, dtype=np.float64)
        self.goal_xy = np.zeros(2, dtype=np.float64)
        self.obstacles: list[dict] = []
        self.prev_goal_dist = 0.0
        self.t_k = 0.0
        self.step_count = 0
        self.backend = dynamics_backend_name()

    def _sample_goal(self) -> np.ndarray:
        spawn = self.cfg["spawn"]
        return self.np_random.uniform(spawn["goal_xy_low"], spawn["goal_xy_high"]).astype(np.float64)

    def _sample_obstacles(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> list[dict]:
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
                spawn["obstacle_xy_low"], spawn["obstacle_xy_high"]
            ).astype(np.float64)
            if np.linalg.norm(center - start_xy) <= spawn["start_clearance"] + radius:
                continue
            if np.linalg.norm(center - goal_xy) <= spawn["goal_clearance"] + radius:
                continue
            valid = True
            for obstacle in obstacles:
                min_gap = obstacle["radius"] + radius + spawn["obstacle_clearance"]
                if np.linalg.norm(center - obstacle["center"]) <= min_gap:
                    valid = False
                    break
            if valid:
                obstacles.append({"center": center, "radius": radius})
        return obstacles

    def _map_action_to_references(self, action: np.ndarray) -> np.ndarray:
        return map_structured_action_to_fins(
            np.asarray(action, dtype=np.float64),
            self.trim_refs,
            self.action_map,
            self.ref_min,
            self.ref_max,
        )

    def _project_to_2d(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).copy()
        yaw = yaw_from_quaternion(state[Q0], state[Q1], state[Q2], state[Q3])
        quat = quaternion_from_yaw(yaw)
        state[VZ] = 0.0
        state[WX] = 0.0
        state[WY] = 0.0
        state[PZ] = self.plane_z
        state[Q0 : Q3 + 1] = quat
        return state

    def _build_obs(self) -> tuple[np.ndarray, dict]:
        pos_xy = self.state[[PX, PY]]
        yaw = yaw_from_quaternion(self.state[Q0], self.state[Q1], self.state[Q2], self.state[Q3])
        goal_rel_world = self.goal_xy - pos_xy
        goal_rel_body = rotate_world_to_body_2d(goal_rel_world, yaw)
        goal_dist = float(np.linalg.norm(goal_rel_world))

        nearest = nearest_obstacle_info(pos_xy, self.obstacles, self.fish_radius)
        obs_rel_body = rotate_world_to_body_2d(nearest["rel_world"], yaw)

        obs = np.array(
            [
                self.state[VX],
                self.state[VY],
                self.state[WZ],
                np.sin(yaw),
                np.cos(yaw),
                goal_rel_body[0],
                goal_rel_body[1],
                goal_dist,
                obs_rel_body[0],
                obs_rel_body[1],
                nearest["clearance"],
                nearest["radius"],
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
            "yaw": yaw,
            "goal_dist": goal_dist,
            "goal_rel_body": goal_rel_body,
            "nearest_obstacle": nearest,
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

        start_xy = self.np_random.uniform(spawn["start_xy_low"], spawn["start_xy_high"]).astype(np.float64)
        yaw0 = float(self.np_random.uniform(-np.pi, np.pi))
        quat = quaternion_from_yaw(yaw0)

        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.state[Q0 : Q3 + 1] = quat
        self.state[PX] = start_xy[0]
        self.state[PY] = start_xy[1]
        self.state[PZ] = self.plane_z

        self.goal_xy = self._sample_goal()
        self.obstacles = self._sample_obstacles(start_xy, self.goal_xy)
        obs, info = self._build_obs()
        self.prev_goal_dist = float(info["goal_dist"])
        return obs, info

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action_ref = self._map_action_to_references(action)

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
        self.state = self._project_to_2d(next_state)
        self.hist = next_hist

        obs, info = self._build_obs()
        goal_dist = float(info["goal_dist"])
        nearest = info["nearest_obstacle"]
        clearance = float(nearest["clearance"])

        goal_heading = np.arctan2(self.goal_xy[1] - self.state[PY], self.goal_xy[0] - self.state[PX])
        yaw_err = wrap_angle(goal_heading - float(info["yaw"]))

        reward, terms = compute_reward(
            self.prev_goal_dist,
            goal_dist,
            clearance,
            yaw_err,
            float(self.state[WZ]),
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
                "reward_terms": terms,
                "reached_goal": reached_goal,
                "collided": collided,
                "t_k": self.t_k,
            }
        )
        return obs, reward, terminated, truncated, info
