from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from dynamics.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, VX, VY, VZ, WX, WY, WZ
from rl_training.config.fish_env_config import build_fish_env_config
from rl_training.envs.geometry import (
    nearest_obstacle_info,
    quaternion_to_rotation_matrix,
    rotate_body_to_world,
    rotate_world_to_body,
)
from simulation.fin_controller.controller import fin_controller, velocity_to_attitude_refs


class RLLocalPlanner:
    """Run a trained PPO policy and convert its output into fin references."""

    def __init__(self, model_path: str | Path | None = None, config: dict | None = None):
        self.config = build_fish_env_config() if config is None else config
        if model_path is None:
            model_dir = Path(__file__).resolve().parents[3] / "artifacts" / "models"
            latest_model = model_dir / "latest_model.zip"
            best_model = model_dir / "best_model.zip"
            model_path = latest_model if latest_model.exists() else best_model
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RL model not found: {self.model_path}")
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
        """Reset internal policy state between avoidance episodes."""
        self.prev_action[:] = 0.0

    def _fallback_nearest_obstacle(self) -> dict:
        """Return a synthetic 'no obstacle' observation when the sensor sees nothing."""
        sensor_range = float(self.obs_clip)
        return {
            "c": np.full(3, np.nan, dtype=np.float64),
            "r": 0.0,
            "v": np.zeros(3, dtype=np.float64),
            "distance": sensor_range,
            "clearance": sensor_range,
            "rel_world": np.array([sensor_range, 0.0, 0.0], dtype=np.float64),
            "rel_vel_world": np.zeros(3, dtype=np.float64),
        }

    def _rotation_body_to_world(self, fish_state: np.ndarray) -> np.ndarray:
        """Build the body-to-world rotation matrix from the fish quaternion."""
        fish_state = np.asarray(fish_state, dtype=np.float64)
        return quaternion_to_rotation_matrix(
            fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
        )

    def _world_velocity(self, fish_state: np.ndarray) -> np.ndarray:
        """Convert current body-frame velocity into the world frame."""
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
        """Assemble the policy observation vector from state, goal, and obstacle data."""
        fish_state = np.asarray(fish_state, dtype=np.float64)
        hist = np.asarray(hist, dtype=np.float64)
        rotation = self._rotation_body_to_world(fish_state)
        position_world = fish_state[[PX, PY, PZ]]
        velocity_world = self._world_velocity(fish_state)

        goal_rel_world = np.asarray(local_target, dtype=np.float64).reshape(3) - position_world
        goal_rel_body = rotate_world_to_body(goal_rel_world, rotation)
        goal_dist = float(np.linalg.norm(goal_rel_world))

        nearest = nearest_obstacle_info(position_world, velocity_world, visible_obstacles, self.fish_radius)
        if not np.isfinite(nearest["distance"]):
            nearest = self._fallback_nearest_obstacle()
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
        """Run the policy and map its action to fin references and a desired velocity."""
        obs = self.build_observation(fish_state, hist, local_target, visible_obstacles)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        cmd_vel_body = action * self.action_velocity_limits
        cmd_vel_global = rotate_body_to_world(cmd_vel_body, self._rotation_body_to_world(fish_state))
        psi_ref, theta_ref, cmd_speed = velocity_to_attitude_refs(cmd_vel_global, fish_state)
        a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, controller_state = fin_controller(
            psi_ref,
            theta_ref,
            cmd_speed,
            fish_state,
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
