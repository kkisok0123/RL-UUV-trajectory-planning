__all__ = ["FishAvoidEnv"]


def __getattr__(name: str):
    if name == "FishAvoidEnv":
        from .fish_avoid_env import FishAvoidEnv

        return FishAvoidEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
