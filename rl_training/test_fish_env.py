from __future__ import annotations

import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_training.envs.fish_avoid_env import FishAvoidEnv


def main() -> None:
    env = FishAvoidEnv()
    check_env(env)
    obs, info = env.reset(seed=7)
    print("backend:", info["backend"])
    print("obs shape:", obs.shape)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print("reward:", reward, "terminated:", terminated, "truncated:", truncated)
        if terminated or truncated:
            env.reset()


if __name__ == "__main__":
    main()
