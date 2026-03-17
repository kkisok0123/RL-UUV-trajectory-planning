from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from dynamics_wrapper.indices import PX, PY, Q0, Q1, Q2, Q3, VX, VY, WZ
from rl.action_mapping import map_structured_action_to_fins
from rl.configs.fish_env import build_fish_env_config
from rl.envs.geometry import rotate_world_to_body_2d, wrap_angle, yaw_from_quaternion


class RLLocalPlanner:
    def __init__(self, model_path: str | Path | None = None, config: dict | None = None):
        self.config = build_fish_env_config() if config is None else config
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[1] / "artifacts" / "models" / "best_model.zip"
            )
        self.model_path = Path(model_path)
        self.model = PPO.load(str(self.model_path))
        self.trim_refs = np.asarray(self.config["trim_refs"], dtype=np.float64)
        self.ref_min = np.asarray(self.config["ref_min"], dtype=np.float64)
        self.ref_max = np.asarray(self.config["ref_max"], dtype=np.float64)
        self.action_map = self.config["action_map"]
        self.fish_radius = float(self.config["fish_radius"])
        self.obs_clip = float(self.config["obs_clip"])
        self.prev_action = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.prev_action[:] = 0.0

    def _nearest_visible_xy(self, position_xy: np.ndarray, visible_obstacles: list[dict]) -> dict:
        best = None
        best_distance = np.inf
        for obstacle in visible_obstacles:
            center_xy = np.asarray(obstacle["c"], dtype=np.float64)[:2]
            rel_xy = center_xy - position_xy
            distance = float(np.linalg.norm(rel_xy))
            clearance = distance - float(obstacle["r"]) - self.fish_radius
            if distance < best_distance:
                best_distance = distance
                best = {
                    "center_xy": center_xy,
                    "rel_xy": rel_xy,
                    "radius": float(obstacle["r"]),
                    "clearance": clearance,
                }
        if best is None:
            best = {
                "center_xy": np.array([np.inf, np.inf], dtype=np.float64),
                "rel_xy": np.array([self.obs_clip, 0.0], dtype=np.float64),
                "radius": 0.0,
                "clearance": self.obs_clip,
            }
        return best

    def build_observation(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
    ) -> np.ndarray:
        fish_state = np.asarray(fish_state, dtype=np.float64)
        hist = np.asarray(hist, dtype=np.float64)
        position_xy = fish_state[[PX, PY]]
        yaw = yaw_from_quaternion(
            fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
        )

        goal_rel_world = np.asarray(local_target, dtype=np.float64)[:2] - position_xy
        goal_rel_body = rotate_world_to_body_2d(goal_rel_world, yaw)
        goal_dist = float(np.linalg.norm(goal_rel_world))

        nearest = self._nearest_visible_xy(position_xy, visible_obstacles)
        obs_rel_body = rotate_world_to_body_2d(nearest["rel_xy"], yaw)

        obs = np.array(
            [
                fish_state[VX],
                fish_state[VY],
                fish_state[WZ],
                np.sin(yaw),
                np.cos(yaw),
                goal_rel_body[0],
                goal_rel_body[1],
                goal_dist,
                obs_rel_body[0],
                obs_rel_body[1],
                nearest["clearance"],
                nearest["radius"],
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
        alpha5_fallback: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = self.build_observation(fish_state, hist, local_target, visible_obstacles)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(3)
        action_ref = map_structured_action_to_fins(
            action, self.trim_refs, self.action_map, self.ref_min, self.ref_max
        )
        action_ref[4] = alpha5_fallback
        self.prev_action = action
        return action_ref, action, obs
