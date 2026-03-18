__all__ = ["HybridSimulationConfig", "run_hybrid_los_rl_simulation"]


def __getattr__(name: str):
    if name in __all__:
        from .main import HybridSimulationConfig, run_hybrid_los_rl_simulation

        exports = {
            "HybridSimulationConfig": HybridSimulationConfig,
            "run_hybrid_los_rl_simulation": run_hybrid_los_rl_simulation,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
