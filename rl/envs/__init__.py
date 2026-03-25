__all__ = ["FishAvoidEnv", "MPCSchedulerEnv"]


def __getattr__(name: str):
    if name == "FishAvoidEnv":
        from .fish_avoid_env import FishAvoidEnv

        return FishAvoidEnv
    if name == "MPCSchedulerEnv":
        from .mpc_scheduler_env import MPCSchedulerEnv

        return MPCSchedulerEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
