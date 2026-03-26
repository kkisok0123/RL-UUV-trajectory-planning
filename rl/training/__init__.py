from .evaluate_policy import main as evaluate_policy_main
from .register_env import register_fish_avoid_env
from .test_fish_env import main as test_fish_env_main
from .train_local_avoid import main as train_local_avoid_main

__all__ = [
    "evaluate_policy_main",
    "register_fish_avoid_env",
    "test_fish_env_main",
    "train_local_avoid_main",
]
