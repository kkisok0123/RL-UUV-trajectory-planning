from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class CollisionAwareEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str | Path | None = None,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.best_model_save_path = None if best_model_save_path is None else Path(best_model_save_path)
        self.deterministic = deterministic
        self.best_score: tuple[float, float, float, float] | None = None

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        metrics = self._evaluate_policy()
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/mean_length", metrics["mean_length"])
        self.logger.record("eval/success_rate", metrics["success_rate"])
        self.logger.record("eval/collision_rate", metrics["collision_rate"])
        self.logger.record("eval/unsafe_termination_rate", metrics["unsafe_termination_rate"])
        self.logger.record("eval/safety_failure_rate", metrics["safety_failure_rate"])

        score = (
            -metrics["safety_failure_rate"],
            -metrics["collision_rate"],
            metrics["success_rate"],
            metrics["mean_reward"],
        )
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            if self.best_model_save_path is not None:
                self.model.save(str(self.best_model_save_path / "best_model"))
            if self.verbose > 0:
                print(
                    "New best model: "
                    f"safety_failure_rate={metrics['safety_failure_rate']:.3f}, "
                    f"collision_rate={metrics['collision_rate']:.3f}, "
                    f"success_rate={metrics['success_rate']:.3f}, "
                    f"mean_reward={metrics['mean_reward']:.3f}"
                )
        elif self.verbose > 0:
            print(
                "Eval: "
                f"safety_failure_rate={metrics['safety_failure_rate']:.3f}, "
                f"collision_rate={metrics['collision_rate']:.3f}, "
                f"success_rate={metrics['success_rate']:.3f}, "
                f"mean_reward={metrics['mean_reward']:.3f}"
            )
        return True

    def _evaluate_policy(self) -> dict[str, float]:
        rewards: list[float] = []
        lengths: list[int] = []
        successes = 0
        collisions = 0
        unsafe_terminations = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = np.array([False])
            last_info: dict = {}

            while not bool(done[0]):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, infos = self.eval_env.step(action)
                episode_reward += float(reward[0])
                episode_length += 1
                last_info = infos[0]

            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes += int(last_info.get("reached_goal", False))
            collisions += int(last_info.get("collided", False))
            unsafe_terminations += int(last_info.get("unsafe_terminated", False))

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        collision_rate = collisions / self.n_eval_episodes
        unsafe_rate = unsafe_terminations / self.n_eval_episodes
        success_rate = successes / self.n_eval_episodes
        safety_failure_rate = (collisions + unsafe_terminations) / self.n_eval_episodes
        return {
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "unsafe_termination_rate": unsafe_rate,
            "safety_failure_rate": safety_failure_rate,
        }
