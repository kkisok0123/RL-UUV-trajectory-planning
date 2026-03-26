from __future__ import annotations

import sys
from pathlib import Path

# Allow running this demo file directly from the repo root or an IDE run config.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from simulation.main.hybrid_simulation import (
    run_hybrid_los_comparison_simulation,
    run_hybrid_los_mpc_simulation,
    run_hybrid_los_rl_simulation,
)
from simulation.main.hybrid_simulation import HybridSimulationConfig

__all__ = [
    "HybridSimulationConfig",
    "run_hybrid_los_rl_simulation",
    "run_hybrid_los_mpc_simulation",
    "run_hybrid_los_comparison_simulation",
]


def main() -> None:
    """Run one demo entrypoint; switch by uncommenting the desired block below."""
    result = run_hybrid_los_rl_simulation(
        visualize=True,
        animation_path="data_saving/hybrid_los_rl.gif",
        animation_fps=30,
    )
    # result = run_hybrid_los_mpc_simulation(
    #     visualize=True,
    #     animation_path="data_saving/hybrid_los_mpc.gif",
    #     animation_fps=30,
    # )
    # result = run_hybrid_los_comparison_simulation(
    #     visualize=True,
    #     animation_path="data_saving/hybrid_los_compare.gif",
    #     animation_fps=30,
    # )

    if "local_planner_kind" not in result:
        for planner_name, planner_result in result.items():
            print("local_planner:", planner_name)
            print("reached_goal:", planner_result["reached_goal"])
            print("collided:", planner_result["collided"])
            if planner_result["collision_info"] is not None:
                print("collision_info:", planner_result["collision_info"])
            print("path_len:", len(planner_result["path_hist"]))
            timing = planner_result["local_planner_timing"]
            print("local_decision_count:", timing["count"])
            if timing["count"] > 0:
                print(
                    "local_decision_time_ms:"
                    f" mean={timing['mean_ms']:.3f}, min={timing['min_ms']:.3f}, max={timing['max_ms']:.3f}"
                )
            print("tracking_metrics:", planner_result["tracking_metrics"])
        return

    print("local_planner:", result["local_planner_kind"])
    print("reached_goal:", result["reached_goal"])
    print("collided:", result["collided"])
    if result["collision_info"] is not None:
        print("collision_info:", result["collision_info"])
    print("path_len:", len(result["path_hist"]))
    timing = result["local_planner_timing"]
    print("local_decision_count:", timing["count"])
    if timing["count"] > 0:
        print(
            "local_decision_time_ms:"
            f" mean={timing['mean_ms']:.3f}, min={timing['min_ms']:.3f}, max={timing['max_ms']:.3f}"
        )
    print("tracking_metrics:", result["tracking_metrics"])


if __name__ == "__main__":
    main()
