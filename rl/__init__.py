__all__ = ["RLLocalPlanner", "map_structured_action_to_fins"]


def __getattr__(name: str):
    if name == "RLLocalPlanner":
        from .local_planner import RLLocalPlanner

        return RLLocalPlanner
    if name == "map_structured_action_to_fins":
        from .action_mapping import map_structured_action_to_fins

        return map_structured_action_to_fins
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
