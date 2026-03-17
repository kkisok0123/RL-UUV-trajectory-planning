from __future__ import annotations

import numpy as np

from dynamics_wrapper.params import load_body_params, load_fin_params


def build_fish_env_config() -> dict:
    trim_refs = np.deg2rad(np.array([30.0, 15.0, 30.0, 15.0, 0.0], dtype=np.float64))
    ref_min = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0, -45.0], dtype=np.float64))
    ref_max = np.deg2rad(np.array([45.0, 30.0, 45.0, 30.0, 45.0], dtype=np.float64))
    return {
        "dt": 0.20,
        "c_A": 5.0,
        "fin_f": 4.0,
        "max_steps": 300,
        "plane_z": 0.0,
        "fish_radius": 0.18,
        "goal_radius": 0.40,
        "safe_margin": 0.80,
        "obs_clip": 50.0,
        "body_params": load_body_params(),
        "fin_params": load_fin_params(),
        "trim_refs": trim_refs,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "action_map": {
            "forward_flap": np.deg2rad(15.0),
            "turn_rot": np.deg2rad(15.0),
            "tail": np.deg2rad(45.0),
        },
        "spawn": {
            "start_xy_low": np.array([-1.0, -1.0], dtype=np.float64),
            "start_xy_high": np.array([1.0, 1.0], dtype=np.float64),
            "goal_xy_low": np.array([4.0, -2.5], dtype=np.float64),
            "goal_xy_high": np.array([7.0, 2.5], dtype=np.float64),
            "num_obstacles_low": 1,
            "num_obstacles_high": 3,
            "obstacle_xy_low": np.array([0.5, -3.0], dtype=np.float64),
            "obstacle_xy_high": np.array([6.5, 3.0], dtype=np.float64),
            "obstacle_radius_low": 0.30,
            "obstacle_radius_high": 0.80,
            "start_clearance": 1.50,
            "goal_clearance": 1.25,
            "obstacle_clearance": 0.40,
        },
        "reward": {
            "progress": 8.0,
            "goal": 250.0,
            "collision": 250.0,
            "near_obstacle": 6.0,
            "heading": 0.4,
            "yaw_rate": 0.08,
            "smooth": 0.15,
            "energy": 0.03,
        },
    }


FISH_ENV_CONFIG = build_fish_env_config()
