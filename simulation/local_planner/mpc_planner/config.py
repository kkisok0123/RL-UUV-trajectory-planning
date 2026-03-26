from __future__ import annotations

import numpy as np


def build_mpc_planner_config() -> dict:
    """Build the default MPC tuning block used by local planners."""
    return {
        "N": 10,
        "max_obstacles": 4,
        "obstacle_margin": 4.0,
        "collision_margin": 2.0,
        "heading_tau": 0.2,
        "speed_tau": 0.3,
        "max_theta_ref": np.deg2rad(60.0),
        "speed_min": 0.04,
        "speed_cap_floor": 0.35,
        "W_goal": 10.0,
        "W_terminal": 2.0,
        "W_obs": 3500.0,
        "W_collision": 50000.0,
        "W_speed": 35.0,
        "risk_speed_ref": 0.25,
        "risk_acc_ref": 0.15,
        "risk_size_ref": 1.0,
        "track_stale_steps": 5,
        "adaptive_risk": {
            "clearance_weight": 0.40,
            "speed_weight": 0.30,
            "acc_weight": 0.20,
            "size_weight": 0.10,
            "obstacle_margin_gain": 1.2,
            "obstacle_margin_scale_max": 2.2,
            "collision_margin_gain": 1.5,
            "collision_margin_scale_max": 2.5,
            "w_obs_gain": 2.0,
            "w_obs_scale_max": 3.0,
            "w_collision_gain": 3.0,
            "w_collision_scale_max": 4.0,
            "speed_risk_slowdown": 0.65,
        },
    }


MPC_PLANNER_CONFIG = build_mpc_planner_config()
