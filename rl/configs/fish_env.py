from __future__ import annotations

import numpy as np

from dynamics_wrapper.params import load_body_params, load_fin_params


def build_fish_env_config() -> dict:
    fin_f = 4.0
    trim_refs = np.deg2rad(np.array([30.0, 15.0, 30.0, 15.0, 0.0], dtype=np.float64))
    ref_min = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0, -45.0], dtype=np.float64))
    ref_max = np.deg2rad(np.array([45.0, 30.0, 45.0, 30.0, 45.0], dtype=np.float64))
    fin_params = load_fin_params()
    return {
        "dt": 0.20,
        "c_A": 5.0,
        "fin_f": fin_f,
        "max_steps": 300,
        "fish_radius": 0.18,
        "goal_radius": 0.40,
        "safe_margin": 0.80,
        "obs_clip": 50.0,
        "body_params": load_body_params(),
        "fin_params": fin_params,
        "trim_refs": trim_refs,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "action_velocity_limits": np.array([0.60, 0.35, 0.30], dtype=np.float64),
        "controller_params": {
            "phi_ref": 0.0,
            "u_ref_max": 0.75,
            "theta_ref_max": np.deg2rad(35.0),
            "zero_speed_epsilon": 0.05,
            "ref_filter_alpha": 1.0,
            "c_A": 5.0,
            "fin_f": fin_f,
            "fin_params": fin_params.copy(),
            "trim_refs": trim_refs.copy(),
            "actuator_min": ref_min.copy(),
            "actuator_max": ref_max.copy(),
            "actuator_rate_limit": np.array([0.70, 0.70, 0.70, 0.70, 0.90], dtype=np.float64),
            "backstepping_force_gains": np.array(
                [
                    [6.0, 1.5],
                    [4.0, 1.5],
                    [4.0, 1.5],
                ],
                dtype=np.float64,
            ),
            "backstepping_virtual_rate_gains": np.array([1.5, 1.8, 1.5], dtype=np.float64),
            "backstepping_attitude_gains": np.array([8.0, 10.0, 8.0], dtype=np.float64),
            "backstepping_rate_gains": np.array([2.0, 2.5, 2.0], dtype=np.float64),
            "eso_beta1": np.array([8.0, 8.0, 8.0, 10.0, 10.0, 8.0], dtype=np.float64),
            "eso_beta2": np.array([20.0, 20.0, 20.0, 25.0, 25.0, 20.0], dtype=np.float64),
            "eso_b0": np.ones(6, dtype=np.float64),
            "tau_limits": np.array([6.0, 2.0, 2.0, 1.5, 2.0, 1.5], dtype=np.float64),
            "fin_share_matrix": np.array(
                [
                    [0.45, 0.50, 0.10, 0.50, 0.10, 0.50],
                    [0.45, 0.50, 0.10, 0.50, 0.10, 0.50],
                    [0.10, 0.00, 0.80, 0.00, 0.80, 0.00],
                ],
                dtype=np.float64,
            ),
            "fin_wrench_max": np.array(
                [
                    [4.0, 1.25, 0.60, 1.25, 0.60, 1.00],
                    [4.0, 1.25, 0.60, 1.25, 0.60, 1.00],
                    [1.5, 0.30, 2.00, 0.30, 2.00, 0.30],
                ],
                dtype=np.float64,
            ),
            "alloc_effort_weight": np.ones(6, dtype=np.float64),
            "alloc_smooth_weight": np.full(6, 0.15, dtype=np.float64),
            "alloc_asym_weight": np.array([0.25, 0.10, 0.50, 0.05, 0.50, 0.05], dtype=np.float64),
            "alloc_iterations": 12,
            "alloc_step_size": 0.35,
            "jacobian_epsilon": np.deg2rad(1.0),
            "inverse_damping": 0.08,
            "inverse_phase_average_samples": 9,
            "inverse_phase_average_window": 1.0 / fin_f,
            "adapt_gain_rate": 0.03,
            "adapt_bias_rate": 0.02,
            "adapt_gain_min": 0.6,
            "adapt_gain_max": 1.6,
            "adapt_bias_min": np.deg2rad(-10.0),
            "adapt_bias_max": np.deg2rad(10.0),
            "inverse_min_scale": np.deg2rad(2.0),
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
