from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, VX, VY, VZ, WX, WY, WZ
from rl.configs.fish_env import build_fish_env_config
from rl.envs.geometry import (
    nearest_obstacle_info,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
)
from simulation.local_planning.fin_controller import (
    build_attitude_reference,
    fin_controller,
    run_layer1_controller,
    run_layer2_controller,
)


class RLLocalPlanner:
    def __init__(self, model_path: str | Path | None = None, config: dict | None = None):
        self.config = build_fish_env_config() if config is None else config
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[1] / "artifacts" / "models" / "best_model.zip"
            )
        self.model_path = Path(model_path)
        self.model = PPO.load(str(self.model_path))
        expected_obs_shape = (23,)
        if tuple(self.model.observation_space.shape) != expected_obs_shape:
            raise ValueError(
                f"Model at {self.model_path} expects observation shape "
                f"{tuple(self.model.observation_space.shape)}, but FishAvoidEnv now uses "
                f"{expected_obs_shape}. Retrain the policy with the upgraded 3D environment."
            )
        self.ref_min = np.asarray(self.config["ref_min"], dtype=np.float64)
        self.ref_max = np.asarray(self.config["ref_max"], dtype=np.float64)
        self.action_velocity_limits = np.asarray(self.config["action_velocity_limits"], dtype=np.float64)
        self.fish_radius = float(self.config["fish_radius"])
        self.obs_clip = float(self.config["obs_clip"])
        self.prev_action = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.prev_action[:] = 0.0

    def _rotation_body_to_world(self, fish_state: np.ndarray) -> np.ndarray:
        fish_state = np.asarray(fish_state, dtype=np.float64)
        return quaternion_to_rotation_matrix(
            fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
        )

    def _world_velocity(self, fish_state: np.ndarray) -> np.ndarray:
        fish_state = np.asarray(fish_state, dtype=np.float64)
        rotation = self._rotation_body_to_world(fish_state)
        return rotate_body_to_world(fish_state[[VX, VY, VZ]], rotation)

    def build_observation(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
    ) -> np.ndarray:
        fish_state = np.asarray(fish_state, dtype=np.float64)
        hist = np.asarray(hist, dtype=np.float64)
        rotation = self._rotation_body_to_world(fish_state)
        position_world = fish_state[[PX, PY, PZ]]
        velocity_world = self._world_velocity(fish_state)

        goal_rel_world = np.asarray(local_target, dtype=np.float64).reshape(3) - position_world
        goal_rel_body = rotate_world_to_body(goal_rel_world, rotation)
        goal_dist = float(np.linalg.norm(goal_rel_world))

        nearest = nearest_obstacle_info(position_world, velocity_world, visible_obstacles, self.fish_radius)
        obs_rel_body = rotate_world_to_body(nearest["rel_world"], rotation)
        obs_rel_vel_body = rotate_world_to_body(nearest["rel_vel_world"], rotation)

        obs = np.array(
            [
                fish_state[VX],
                fish_state[VY],
                fish_state[VZ],
                fish_state[WX],
                fish_state[WY],
                fish_state[WZ],
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
                hist[0],
                hist[1],
                hist[2],
                hist[3],
                hist[4],
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -self.obs_clip, self.obs_clip)

    def plan(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
        controller_state: dict,
        dt: float,
        controller_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs = self.build_observation(fish_state, hist, local_target, visible_obstacles)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        cmd_vel_body = action * self.action_velocity_limits
        cmd_vel_global = rotate_body_to_world(cmd_vel_body, self._rotation_body_to_world(fish_state))
        ref_cmd = build_attitude_reference(cmd_vel_body, fish_state, controller_params)
        a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, controller_state = fin_controller(
            ref_cmd,
            fish_state,
            hist,
            controller_state,
            dt,
            controller_params,
        )
        action_ref = np.clip(
            np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64),
            self.ref_min,
            self.ref_max,
        )
        self.prev_action = action
        return action_ref, action, obs, cmd_vel_global, controller_state

    def plan_layer1_wrench(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
        controller_state: dict,
        dt: float,
        controller_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs = self.build_observation(fish_state, hist, local_target, visible_obstacles)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        cmd_vel_body = action * self.action_velocity_limits
        cmd_vel_global = rotate_body_to_world(cmd_vel_body, self._rotation_body_to_world(fish_state))
        ref_cmd = build_attitude_reference(cmd_vel_body, fish_state, controller_params)
        tau_d, controller_state = run_layer1_controller(
            ref_cmd,
            fish_state,
            controller_state,
            dt,
            controller_params,
        )
        self.prev_action = action
        return tau_d, ref_cmd, action, obs, cmd_vel_global, controller_state

    def plan_layer2_actuator(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
        controller_state: dict,
        dt: float,
        controller_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs = self.build_observation(fish_state, hist, local_target, visible_obstacles)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        cmd_vel_body = action * self.action_velocity_limits
        cmd_vel_global = rotate_body_to_world(cmd_vel_body, self._rotation_body_to_world(fish_state))
        ref_cmd = build_attitude_reference(cmd_vel_body, fish_state, controller_params)
        actuator_cmd, controller_state = run_layer2_controller(
            ref_cmd,
            fish_state,
            hist,
            controller_state,
            dt,
            controller_params,
        )
        self.prev_action = action
        return actuator_cmd, ref_cmd, action, obs, cmd_vel_global, controller_state
