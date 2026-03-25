from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, VX, VY, VZ, WX, WY, WZ
from rl.configs.fish_env import build_fish_env_config
from rl.envs.geometry import quaternion_to_rotation_matrix, rotate_body_to_world, rotate_world_to_body
from simulation.local_planning.sensor import get_visible_obstacles


SCHEDULER_MODE_NAMES = ("aggressive", "normal", "conservative")
SCHEDULER_MODE_TO_INDEX = {name: idx for idx, name in enumerate(SCHEDULER_MODE_NAMES)}
SCHEDULER_OBS_DIM = 31
DEFAULT_SCHEDULER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "models" / "mpc_scheduler" / "best_model.zip"
)

DEFAULT_STRATEGY_TABLE = {
    "aggressive": {
        "mode_name": "aggressive",
        "mode_index": 0,
        "speed_scale": 1.15,
        "lookahead": 8.0,
        "W_goal": 18.0,
        "W_terminal": 4.0,
        "W_obs": 1200.0,
        "W_collision": 18000.0,
        "obstacle_margin": 3.0,
        "collision_margin": 1.5,
    },
    "normal": {
        "mode_name": "normal",
        "mode_index": 1,
        "speed_scale": 1.00,
        "lookahead": 6.0,
        "W_goal": 12.0,
        "W_terminal": 3.0,
        "W_obs": 2200.0,
        "W_collision": 25000.0,
        "obstacle_margin": 4.0,
        "collision_margin": 2.0,
    },
    "conservative": {
        "mode_name": "conservative",
        "mode_index": 2,
        "speed_scale": 0.70,
        "lookahead": 4.0,
        "W_goal": 8.0,
        "W_terminal": 2.0,
        "W_obs": 3500.0,
        "W_collision": 40000.0,
        "obstacle_margin": 5.0,
        "collision_margin": 2.5,
    },
}

DEFAULT_SCHEDULER_SETTINGS = {
    "propagation_model": "kinematic",
    "kinematics": {
        "heading_tau": 0.20,
        "speed_tau": 0.30,
    },
    "sensor": {
        "range": 8.0,
        "fov_angle": 60.0,
        "num_rays": 100,
    },
    "static_speed_threshold": 0.05,
    "risk_clearance_ref": 4.0,
    "risk_ttc_ref": 6.0,
    "risk_closing_speed_ref": 0.60,
    "risk_clearance_weight": 0.35,
    "risk_ttc_weight": 0.30,
    "risk_closing_weight": 0.20,
    "risk_heading_weight": 0.10,
    "risk_dynamic_weight": 0.10,
    "risk_away_bonus": 0.40,
    "target_mode_low": 0.25,
    "target_mode_high": 0.60,
    "goal_distance_low": 5.0,
    "goal_distance_high": 12.0,
    "base_command_speed": 0.18,
    "max_steps": 220,
    "max_consecutive_solver_fails": 5,
    "scene_margin": 3.0,
    "reward": {
        "goal_bonus": 40.0,
        "progress_weight": 1.2,
        "mode_match_bonus": 1.0,
        "over_aggressive_penalty": 1.5,
        "over_conservative_penalty": 0.8,
        "clearance_penalty_weight": 0.8,
        "collision_penalty": 2.0,
        "solver_fail_penalty": 1.0,
        "switch_penalty": 0.4,
        "detour_penalty_weight": 0.6,
        "aggressive_detour_scale": 1.5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_scheduler_settings(config: dict | None = None) -> dict:
    merged = deepcopy(DEFAULT_SCHEDULER_SETTINGS)
    if config is not None:
        scheduler_override = config.get("scheduler", {})
        merged = _deep_merge(merged, scheduler_override)
    strategies_override = {} if config is None else config.get("scheduler", {}).get("strategies", {})
    merged["strategies"] = _deep_merge(DEFAULT_STRATEGY_TABLE, strategies_override)
    return merged


def default_scheduler_config() -> dict:
    return build_scheduler_settings(build_fish_env_config())


def rotation_body_to_world(fish_state: np.ndarray) -> np.ndarray:
    fish_state = np.asarray(fish_state, dtype=np.float64).reshape(-1)
    return quaternion_to_rotation_matrix(fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3])


def world_velocity(fish_state: np.ndarray) -> np.ndarray:
    fish_state = np.asarray(fish_state, dtype=np.float64).reshape(-1)
    return rotate_body_to_world(fish_state[[VX, VY, VZ]], rotation_body_to_world(fish_state))


def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return vec / norm


def _safe_goal_direction(goal_rel_world: np.ndarray, forward_world: np.ndarray) -> np.ndarray:
    goal_rel_world = np.asarray(goal_rel_world, dtype=np.float64).reshape(3)
    if np.linalg.norm(goal_rel_world) < 1e-8:
        return _unit(forward_world, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    return goal_rel_world / max(np.linalg.norm(goal_rel_world), 1e-8)


def resolve_scheduler_action(action: int | np.ndarray, settings: dict | None = None) -> tuple[str, dict]:
    settings = default_scheduler_config() if settings is None else settings
    action_value = int(np.asarray(action, dtype=np.int64).reshape(-1)[0])
    action_value = int(np.clip(action_value, 0, len(SCHEDULER_MODE_NAMES) - 1))
    mode_name = SCHEDULER_MODE_NAMES[action_value]
    strategy = deepcopy(settings["strategies"][mode_name])
    strategy["mode_name"] = mode_name
    strategy["mode_index"] = action_value
    return mode_name, strategy


def strategy_for_mode(mode_name: str, settings: dict | None = None) -> dict:
    settings = default_scheduler_config() if settings is None else settings
    if mode_name not in settings["strategies"]:
        raise KeyError(f"Unknown scheduler mode: {mode_name}")
    strategy = deepcopy(settings["strategies"][mode_name])
    strategy["mode_name"] = mode_name
    strategy["mode_index"] = int(SCHEDULER_MODE_TO_INDEX[mode_name])
    return strategy


def project_goal_with_lookahead(current_pos: np.ndarray, goal_pos: np.ndarray, lookahead: float) -> np.ndarray:
    current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    goal_rel = goal_pos - current_pos
    distance = float(np.linalg.norm(goal_rel))
    if distance <= max(float(lookahead), 1e-6):
        return goal_pos.copy()
    return current_pos + goal_rel / distance * float(lookahead)


def sense_visible_obstacles(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    obstacles: list[dict],
    settings: dict | None = None,
) -> tuple[list[dict], dict]:
    settings = default_scheduler_config() if settings is None else settings
    sensor_params = settings["sensor"]
    safe_zone = get_visible_obstacles(robot_pos, robot_vel, obstacles, sensor_params)
    visible_obs: list[dict] = []
    for obstacle in obstacles:
        for ray_idx in range(safe_zone["rays"].shape[1]):
            if not bool(safe_zone["is_blocked"][ray_idx]):
                continue
            hit_point = safe_zone["origin"] + safe_zone["rays"][:, ray_idx] * safe_zone["dists"][ray_idx]
            if abs(np.linalg.norm(hit_point - obstacle["c"]) - obstacle["r"]) < 0.1:
                visible_obs.append(obstacle)
                break
    return visible_obs, safe_zone


def compute_sector_clearances(
    position_world: np.ndarray,
    visible_obstacles: list[dict],
    fish_radius: float,
    rotation_bw: np.ndarray,
    sensor_range: float,
) -> dict[str, float]:
    position_world = np.asarray(position_world, dtype=np.float64).reshape(3)
    rotation_bw = np.asarray(rotation_bw, dtype=np.float64).reshape(3, 3)
    clearances = {
        "left": float(sensor_range),
        "right": float(sensor_range),
        "up": float(sensor_range),
        "down": float(sensor_range),
    }
    for obstacle in visible_obstacles:
        rel_world = np.asarray(obstacle["c"], dtype=np.float64).reshape(3) - position_world
        rel_body = rotate_world_to_body(rel_world, rotation_bw)
        distance = float(np.linalg.norm(rel_body))
        clearance = max(0.0, distance - float(obstacle["r"]) - float(fish_radius))
        if rel_body[1] >= 0.0:
            clearances["left"] = min(clearances["left"], clearance)
        else:
            clearances["right"] = min(clearances["right"], clearance)
        if rel_body[2] >= 0.0:
            clearances["up"] = min(clearances["up"], clearance)
        else:
            clearances["down"] = min(clearances["down"], clearance)
    return clearances


def _safe_obstacle_metrics(sensor_range: float) -> dict:
    inf_vec = np.full(3, sensor_range, dtype=np.float64)
    return {
        "c": np.full(3, np.nan, dtype=np.float64),
        "r": 0.0,
        "v": np.zeros(3, dtype=np.float64),
        "distance": np.inf,
        "clearance": float(sensor_range),
        "rel_world": inf_vec.copy(),
        "rel_body": inf_vec.copy(),
        "rel_vel_world": np.zeros(3, dtype=np.float64),
        "rel_vel_body": np.zeros(3, dtype=np.float64),
        "closing_speed": 0.0,
        "ttc": np.inf,
        "is_static": 1.0,
        "moving_away": 1.0,
        "bearing_alignment": 0.0,
        "collision_risk": 0.0,
    }


def compute_obstacle_metrics(
    position_world: np.ndarray,
    velocity_world: np.ndarray,
    obstacle: dict,
    fish_radius: float,
    rotation_bw: np.ndarray,
    forward_world: np.ndarray,
    settings: dict,
) -> dict:
    position_world = np.asarray(position_world, dtype=np.float64).reshape(3)
    velocity_world = np.asarray(velocity_world, dtype=np.float64).reshape(3)
    rotation_bw = np.asarray(rotation_bw, dtype=np.float64).reshape(3, 3)
    forward_world = _unit(forward_world, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
    obs_velocity = np.asarray(obstacle.get("v", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
    rel_world = center - position_world
    distance = float(np.linalg.norm(rel_world))
    rel_unit = _unit(rel_world, forward_world)
    clearance = distance - float(obstacle["r"]) - float(fish_radius)
    rel_vel_world = obs_velocity - velocity_world
    closing_speed = max(0.0, -float(np.dot(rel_unit, rel_vel_world)))
    ttc = np.inf
    if closing_speed > 1e-6:
        ttc = max(0.0, clearance) / closing_speed
    is_static = float(np.linalg.norm(obs_velocity) < float(settings["static_speed_threshold"]))
    moving_away = float(np.dot(rel_world, rel_vel_world) > 0.0)
    bearing_alignment = max(0.0, float(np.dot(rel_unit, forward_world)))
    risk_clearance = float(
        np.clip((float(settings["risk_clearance_ref"]) - clearance) / float(settings["risk_clearance_ref"]), 0.0, 1.0)
    )
    risk_ttc = 0.0 if not np.isfinite(ttc) else float(
        np.clip((float(settings["risk_ttc_ref"]) - ttc) / float(settings["risk_ttc_ref"]), 0.0, 1.0)
    )
    risk_closing = float(
        np.clip(closing_speed / float(settings["risk_closing_speed_ref"]), 0.0, 1.0)
    )
    risk_dynamic = 0.3 if bool(is_static) else 1.0
    collision_risk = float(
        np.clip(
            float(settings["risk_clearance_weight"]) * risk_clearance
            + float(settings["risk_ttc_weight"]) * risk_ttc
            + float(settings["risk_closing_weight"]) * risk_closing
            + float(settings["risk_heading_weight"]) * bearing_alignment
            + float(settings["risk_dynamic_weight"]) * risk_dynamic
            - (float(settings["risk_away_bonus"]) if bool(moving_away) else 0.0),
            0.0,
            1.0,
        )
    )
    return {
        "c": center.copy(),
        "r": float(obstacle["r"]),
        "v": obs_velocity.copy(),
        "distance": distance,
        "clearance": float(clearance),
        "rel_world": rel_world.copy(),
        "rel_body": rotate_world_to_body(rel_world, rotation_bw),
        "rel_vel_world": rel_vel_world.copy(),
        "rel_vel_body": rotate_world_to_body(rel_vel_world, rotation_bw),
        "closing_speed": float(closing_speed),
        "ttc": float(ttc),
        "is_static": float(is_static),
        "moving_away": float(moving_away),
        "bearing_alignment": float(bearing_alignment),
        "collision_risk": collision_risk,
    }


def determine_target_mode(collision_risk: float, settings: dict | None = None) -> str:
    settings = default_scheduler_config() if settings is None else settings
    if collision_risk < float(settings["target_mode_low"]):
        return "aggressive"
    if collision_risk < float(settings["target_mode_high"]):
        return "normal"
    return "conservative"


def select_dangerous_obstacle(
    position_world: np.ndarray,
    velocity_world: np.ndarray,
    visible_obstacles: list[dict],
    fish_radius: float,
    rotation_bw: np.ndarray,
    forward_world: np.ndarray,
    settings: dict | None = None,
) -> dict:
    settings = default_scheduler_config() if settings is None else settings
    sensor_range = float(settings["sensor"]["range"])
    best = _safe_obstacle_metrics(sensor_range)
    best_risk = -np.inf
    for obstacle in visible_obstacles:
        metrics = compute_obstacle_metrics(
            position_world,
            velocity_world,
            obstacle,
            fish_radius,
            rotation_bw,
            forward_world,
            settings,
        )
        if metrics["collision_risk"] > best_risk:
            best_risk = float(metrics["collision_risk"])
            best = metrics
    return best


def build_scheduler_observation(
    fish_state: np.ndarray,
    goal_world: np.ndarray,
    visible_obstacles: list[dict],
    fish_radius: float,
    prev_mode: str | None,
    settings: dict | None = None,
) -> tuple[np.ndarray, dict]:
    settings = default_scheduler_config() if settings is None else settings
    fish_state = np.asarray(fish_state, dtype=np.float64).reshape(-1)
    goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)
    rotation_bw = rotation_body_to_world(fish_state)
    position_world = fish_state[[PX, PY, PZ]]
    velocity_world = world_velocity(fish_state)
    forward_world = rotation_bw[:, 0]

    goal_rel_world = goal_world - position_world
    goal_rel_body = rotate_world_to_body(goal_rel_world, rotation_bw)
    goal_dist = float(np.linalg.norm(goal_rel_world))
    dangerous = select_dangerous_obstacle(
        position_world,
        velocity_world,
        visible_obstacles,
        fish_radius,
        rotation_bw,
        forward_world,
        settings,
    )
    target_mode = determine_target_mode(dangerous["collision_risk"], settings)
    sector_clearances = compute_sector_clearances(
        position_world,
        visible_obstacles,
        fish_radius,
        rotation_bw,
        float(settings["sensor"]["range"]),
    )
    prev_mode_vec = np.zeros(len(SCHEDULER_MODE_NAMES), dtype=np.float64)
    if prev_mode in SCHEDULER_MODE_TO_INDEX:
        prev_mode_vec[SCHEDULER_MODE_TO_INDEX[prev_mode]] = 1.0
    obs = np.array(
        [
            fish_state[VX],
            fish_state[VY],
            fish_state[VZ],
            fish_state[WX],
            fish_state[WY],
            fish_state[WZ],
            goal_rel_body[0],
            goal_rel_body[1],
            goal_rel_body[2],
            goal_dist,
            dangerous["rel_body"][0],
            dangerous["rel_body"][1],
            dangerous["rel_body"][2],
            dangerous["rel_vel_body"][0],
            dangerous["rel_vel_body"][1],
            dangerous["rel_vel_body"][2],
            dangerous["clearance"],
            dangerous["r"],
            dangerous["ttc"] if np.isfinite(dangerous["ttc"]) else float(settings["risk_ttc_ref"]),
            dangerous["closing_speed"],
            dangerous["is_static"],
            dangerous["moving_away"],
            dangerous["bearing_alignment"],
            float(len(visible_obstacles)),
            sector_clearances["left"],
            sector_clearances["right"],
            sector_clearances["up"],
            sector_clearances["down"],
            prev_mode_vec[0],
            prev_mode_vec[1],
            prev_mode_vec[2],
        ],
        dtype=np.float64,
    )
    obs = np.clip(obs, -50.0, 50.0).astype(np.float32)
    diagnostics = {
        "goal_world": goal_world.copy(),
        "goal_rel_world": goal_rel_world.copy(),
        "goal_dist": goal_dist,
        "dangerous_obstacle": dangerous,
        "collision_risk": float(dangerous["collision_risk"]),
        "target_mode": target_mode,
        "num_visible_obs": int(len(visible_obstacles)),
        "sector_clearances": sector_clearances,
        "forward_world": _unit(forward_world, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        "velocity_world": velocity_world.copy(),
        "position_world": position_world.copy(),
    }
    return obs, diagnostics


def compute_scheduler_reward(
    prev_position_world: np.ndarray,
    current_position_world: np.ndarray,
    goal_world: np.ndarray,
    selected_mode: str,
    target_mode: str,
    clearance: float,
    planner_success: bool,
    prev_mode: str | None,
    reached_goal: bool,
    collided: bool,
    settings: dict | None = None,
) -> tuple[float, dict]:
    settings = default_scheduler_config() if settings is None else settings
    reward_cfg = settings["reward"]
    prev_position_world = np.asarray(prev_position_world, dtype=np.float64).reshape(3)
    current_position_world = np.asarray(current_position_world, dtype=np.float64).reshape(3)
    goal_world = np.asarray(goal_world, dtype=np.float64).reshape(3)
    goal_dir = _safe_goal_direction(goal_world - prev_position_world, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    step_delta = current_position_world - prev_position_world
    direct_progress = float(np.dot(step_delta, goal_dir))
    lateral_delta = step_delta - direct_progress * goal_dir
    detour_penalty = float(np.linalg.norm(lateral_delta))
    clearance_penalty = float(
        max(0.0, (float(settings["risk_clearance_ref"]) - float(clearance)) / float(settings["risk_clearance_ref"]))
    )
    mode_match = 1.0 if selected_mode == target_mode else 0.0
    over_aggressive = 1.0 if target_mode == "conservative" and selected_mode == "aggressive" else 0.0
    over_conservative = 1.0 if target_mode == "aggressive" and selected_mode == "conservative" else 0.0
    switch_penalty = 1.0 if prev_mode is not None and selected_mode != prev_mode else 0.0
    aggressive_detour_scale = (
        float(reward_cfg["aggressive_detour_scale"])
        if target_mode == "aggressive" and selected_mode == "aggressive"
        else 1.0
    )
    goal_bonus = float(reward_cfg["goal_bonus"]) if reached_goal else 0.0
    collision_term = 1.0 if collided else 0.0
    solver_fail = 0.0 if planner_success else 1.0
    reward = (
        float(reward_cfg["progress_weight"]) * direct_progress
        + float(reward_cfg["mode_match_bonus"]) * mode_match
        - float(reward_cfg["over_aggressive_penalty"]) * over_aggressive
        - float(reward_cfg["over_conservative_penalty"]) * over_conservative
        - float(reward_cfg["clearance_penalty_weight"]) * clearance_penalty
        - float(reward_cfg["collision_penalty"]) * collision_term
        - float(reward_cfg["solver_fail_penalty"]) * solver_fail
        - float(reward_cfg["switch_penalty"]) * switch_penalty
        - float(reward_cfg["detour_penalty_weight"]) * aggressive_detour_scale * detour_penalty
        + goal_bonus
    )
    terms = {
        "direct_progress": direct_progress,
        "mode_match": mode_match,
        "over_aggressive": over_aggressive,
        "over_conservative": over_conservative,
        "clearance_penalty": clearance_penalty,
        "collision": collision_term,
        "solver_fail": solver_fail,
        "switch": switch_penalty,
        "detour": aggressive_detour_scale * detour_penalty,
        "goal_bonus": goal_bonus,
    }
    return float(reward), terms
