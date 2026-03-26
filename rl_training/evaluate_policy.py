from __future__ import annotations

import sys
from pathlib import Path

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_training.envs.fish_avoid_env import FishAvoidEnv


def main(episodes: int = 5) -> None:
    model_path = ROOT / "artifacts" / "models" / "latest_model.zip"
    env = FishAvoidEnv()
    model = PPO.load(str(model_path))

    rewards = []
    successes = 0
    collisions = 0
    unsafe_terminations = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
        successes += int(info["reached_goal"])
        collisions += int(info["collided"])
        unsafe_terminations += int(info.get("unsafe_terminated", False))

    print(f"episodes={episodes}")
    print(f"mean_reward={sum(rewards) / len(rewards):.3f}")
    print(f"success_rate={successes / episodes:.3f}")
    print(f"collision_rate={collisions / episodes:.3f}")
    print(f"unsafe_termination_rate={unsafe_terminations / episodes:.3f}")


if __name__ == "__main__":
    main()
