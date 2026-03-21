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
        "fish_radius": 0.18,
        "goal_radius": 0.40,
        "safe_margin": 0.80,
        "obs_clip": 50.0,
        "body_params": load_body_params(),
        "fin_params": load_fin_params(),
        "trim_refs": trim_refs,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "action_velocity_limits": np.array([0.60, 0.35, 0.30], dtype=np.float64),
        "controller_params": {
            "Kp_z": 1.2,
            "Ki_z": 0,
            "Kd_z": 6.0,
            "Kp_psi": 1.5,
            "Ki_psi": 2,
            "Kd_psi": 2,
            "A_base": np.deg2rad(45.0),
            "A_rot_base": np.deg2rad(15.0),
            "alpha5_min": np.deg2rad(-45.0),
            "alpha5_max": np.deg2rad(45.0),
            "delta_rot_max": np.deg2rad(15.0),
            "e_theta_int_limit": 0.75,
            "e_psi_int_limit": 0.75,
        },
        "spawn": {
            "start_xyz_low": np.array([-1.0, -1.0, -1.0], dtype=np.float64),
            "start_xyz_high": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "goal_xyz_low": np.array([4.0, -2.5, -1.5], dtype=np.float64),
            "goal_xyz_high": np.array([7.0, 2.5, 1.5], dtype=np.float64),
            "num_obstacles_low": 1,
            "num_obstacles_high": 3,
            "dynamic_count_min": 1,
            "obstacle_xyz_low": np.array([0.5, -3.0, -2.0], dtype=np.float64),
            "obstacle_xyz_high": np.array([6.5, 3.0, 2.0], dtype=np.float64),
            "obstacle_bounds_low": np.array([-1.5, -4.0, -3.0], dtype=np.float64),
            "obstacle_bounds_high": np.array([7.5, 4.0, 3.0], dtype=np.float64),
            "obstacle_radius_low": 0.30,
            "obstacle_radius_high": 0.80,
            "dynamic_speed_low": 0.15,
            "dynamic_speed_high": 0.45,
            "start_clearance": 1.50,
            "goal_clearance": 1.25,
            "obstacle_clearance": 0.40,
        },
        "reward": {
            "progress": 8.0,
            "goal": 250.0,
            "collision": 250.0,
            "near_obstacle": 6.0,
            "direction_alignment": 0.4,
            "angular_rate": 0.08,
            "smooth": 0.15,
            "energy": 0.03,
        },
    }


FISH_ENV_CONFIG = build_fish_env_config()
