from __future__ import annotations

import numpy as np

from dynamics.params import load_body_params, load_fin_params
from simulation.fin_controller.config import build_fin_controller_config
from simulation.local_planner.mpc_planner.config import build_mpc_planner_config


def _build_sensor_config() -> dict:
    """Build sensor parameters used by the RL environment."""
    return {
        "range": 8.0,
        "fov_angle": 60.0,
        "num_rays": 100,
        "hit_tolerance": 0.12,
        "miss_rate": 0.02,
    }


def _build_observation_noise_config() -> dict:
    """Build observation-noise settings for domain randomization."""
    return {
        "velocity_std": 0.01,
        "angular_rate_std": 0.01,
        "goal_rel_std": 0.03,
        "goal_dist_std": 0.02,
        "obstacle_rel_std": 0.04,
        "obstacle_vel_std": 0.03,
        "clearance_std": 0.03,
        "radius_std": 0.02,
        "hist_std": np.deg2rad(0.25),
    }


def _build_disturbance_config() -> dict:
    """Build ambient-current disturbance settings."""
    return {
        "current_speed_low": 0.0,
        "current_speed_high": 0.05,
        "current_elevation_range": np.deg2rad(8.0),
        "current_walk_std": 0.003,
        "current_decay": 0.98,
        "current_speed_max": 0.08,
    }


def _build_spawn_config() -> dict:
    """Build scene-sampling ranges for RL training episodes."""
    return {
        "start_xyz_low": np.array([-1.0, -1.0, -1.0], dtype=np.float64),
        "start_xyz_high": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "goal_xyz_low": np.array([4.0, -2.5, -1.5], dtype=np.float64),
        "goal_xyz_high": np.array([7.0, 2.5, 1.5], dtype=np.float64),
        "goal_distance_low": 6.0,
        "goal_distance_high": 11.0,
        "goal_azimuth_range": np.deg2rad(28.0),
        "goal_elevation_range": np.deg2rad(14.0),
        "num_obstacles_low": 1,
        "num_obstacles_high": 5,
        "dynamic_count_min": 1,
        "dynamic_count_low": 1,
        "dynamic_count_high": 5,
        "obstacle_xyz_low": np.array([0.5, -3.0, -2.0], dtype=np.float64),
        "obstacle_xyz_high": np.array([6.5, 3.0, 2.0], dtype=np.float64),
        "obstacle_bounds_low": np.array([-1.5, -4.0, -3.0], dtype=np.float64),
        "obstacle_bounds_high": np.array([7.5, 4.0, 3.0], dtype=np.float64),
        "corridor_along_low": 0.18,
        "corridor_along_high": 0.88,
        "corridor_lateral_span": 2.2,
        "corridor_vertical_span": 1.4,
        "obstacle_radius_low": 0.30,
        "obstacle_radius_high": 0.80,
        "dynamic_speed_low": 0.15,
        "dynamic_speed_high": 0.40,
        "start_clearance": 1.80,
        "goal_clearance": 1.50,
        "obstacle_clearance": 0.70,
        "scene_margin": 3.0,
        "initial_speed_low": 0.02,
        "initial_speed_high": 0.08,
        "initial_heading_range": np.deg2rad(18.0),
        "initial_pitch_range": np.deg2rad(10.0),
    }


def _build_termination_config() -> dict:
    """Build episode termination thresholds."""
    return {
        "collision_clearance": 0.0,
        "unsafe_clearance": 0.20,
    }


def _build_reward_config() -> dict:
    """Build the RL reward weights."""
    return {
        "progress": 5.0,
        "goal": 200.0,
        "collision": 6000.0,
        "near_obstacle": 60.0,
        "danger_obstacle": 180.0,
        "unsafe_obstacle": 420.0,
        "unsafe_termination": 1200.0,
        "direction_alignment": 0.4,
        "angular_rate": 0.08,
        "smooth": 0.05,
        "energy": 0.03,
        "time": 0.03,
    }


def build_fish_env_config() -> dict:
    """Build the shared environment config used by training and simulation."""
    trim_refs = np.deg2rad(np.array([30.0, 15.0, 30.0, 15.0, 0.0], dtype=np.float64))
    ref_min = np.deg2rad(np.array([0.0, 0.0, 0.0, 0.0, -45.0], dtype=np.float64))
    ref_max = np.deg2rad(np.array([45.0, 30.0, 45.0, 30.0, 45.0], dtype=np.float64))
    initial_state_template = np.array(
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    return {
        "dt": 0.20,
        "c_A": 5.0,
        "fin_f": 4.0,
        "max_steps": 300,
        "fish_radius": 0.18,
        "goal_radius": 0.40,
        "safe_margin": 1.10,
        "danger_margin": 0.55,
        "unsafe_margin": 0.30,
        "obs_clip": 50.0,
        "body_params": load_body_params(),
        "fin_params": load_fin_params(),
        "initial_state_template": initial_state_template,
        "trim_refs": trim_refs,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "action_velocity_limits": np.array([0.60, 0.35, 0.30], dtype=np.float64),
        "sensor": _build_sensor_config(),
        "observation_noise": _build_observation_noise_config(),
        "disturbance": _build_disturbance_config(),
        "mpc": build_mpc_planner_config(),
        "controller_params": build_fin_controller_config(),
        "spawn": _build_spawn_config(),
        "termination": _build_termination_config(),
        "reward": _build_reward_config(),
    }


FISH_ENV_CONFIG = build_fish_env_config()
