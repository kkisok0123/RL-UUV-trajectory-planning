from __future__ import annotations

import sys
from pathlib import Path

from gymnasium.envs.registration import register

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def register_fish_avoid_env() -> None:
    try:
        register(
            id="FishAvoid-v0",
            entry_point="rl_training.envs.fish_avoid_env:FishAvoidEnv",
        )
    except Exception as exc:  # Gymnasium raises on duplicate registration.
        if "already registered" not in str(exc):
            raise


if __name__ == "__main__":
    register_fish_avoid_env()
