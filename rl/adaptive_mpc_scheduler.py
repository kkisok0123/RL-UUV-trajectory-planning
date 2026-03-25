from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl.adaptive_mpc_common import (
    DEFAULT_SCHEDULER_MODEL_PATH,
    SCHEDULER_OBS_DIM,
    build_scheduler_observation,
    build_scheduler_settings,
    resolve_scheduler_action,
)
from rl.configs.fish_env import build_fish_env_config


class AdaptiveMPCScheduler:
    def __init__(self, model_path: str | Path | None = None, config: dict | None = None):
        self.config = build_fish_env_config() if config is None else config
        self.scheduler_cfg = build_scheduler_settings(self.config)
        self.fish_radius = float(self.config["fish_radius"])
        self.prev_mode: str | None = None
        self.model_path = Path(DEFAULT_SCHEDULER_MODEL_PATH if model_path is None else model_path)
        self.model: PPO | None = None
        self.using_heuristic = not self.model_path.exists()
        if not self.using_heuristic:
            self.model = PPO.load(str(self.model_path))
            expected_obs_shape = (SCHEDULER_OBS_DIM,)
            if tuple(self.model.observation_space.shape) != expected_obs_shape:
                raise ValueError(
                    f"Scheduler model at {self.model_path} expects observation shape "
                    f"{tuple(self.model.observation_space.shape)}, but scheduler env uses "
                    f"{expected_obs_shape}. Retrain the scheduler policy."
                )

    def reset(self) -> None:
        self.prev_mode = None

    def plan(
        self,
        fish_state: np.ndarray,
        avoidance_goal: np.ndarray,
        visible_obstacles: list[dict],
    ) -> tuple[dict, dict]:
        obs, diagnostics = build_scheduler_observation(
            fish_state,
            avoidance_goal,
            visible_obstacles,
            self.fish_radius,
            self.prev_mode,
            self.scheduler_cfg,
        )
        if self.model is None:
            selected_mode = str(diagnostics["target_mode"])
            strategy_params = resolve_scheduler_action(
                np.array([self.scheduler_cfg["strategies"][selected_mode]["mode_index"]], dtype=np.int64),
                self.scheduler_cfg,
            )[1]
        else:
            action, _ = self.model.predict(obs, deterministic=True)
            selected_mode, strategy_params = resolve_scheduler_action(action, self.scheduler_cfg)
        self.prev_mode = selected_mode
        return strategy_params, {
            "observation": np.asarray(obs, dtype=np.float32),
            "selected_mode": selected_mode,
            "target_mode": diagnostics["target_mode"],
            "collision_risk": float(diagnostics["collision_risk"]),
            "ttc": float(diagnostics["dangerous_obstacle"]["ttc"]),
            "closing_speed": float(diagnostics["dangerous_obstacle"]["closing_speed"]),
            "is_static": bool(diagnostics["dangerous_obstacle"]["is_static"]),
            "moving_away": bool(diagnostics["dangerous_obstacle"]["moving_away"]),
            "num_visible_obs": int(diagnostics["num_visible_obs"]),
            "strategy_params": strategy_params.copy(),
            "policy_source": "heuristic" if self.model is None else "ppo",
        }
