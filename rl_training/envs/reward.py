from __future__ import annotations

import numpy as np


def compute_reward(
    prev_goal_dist: float,
    goal_dist: float,
    clearance: float,
    direction_error: float,
    angular_rate_norm: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    action_ref: np.ndarray,
    cfg: dict,
) -> tuple[float, dict]:
    reward_cfg = cfg["reward"]
    safe_margin = float(cfg["safe_margin"])
    danger_margin = float(cfg.get("danger_margin", safe_margin))
    unsafe_margin = float(cfg.get("unsafe_margin", danger_margin))

    progress = reward_cfg["progress"] * (prev_goal_dist - goal_dist)
    goal_bonus = reward_cfg["goal"] if goal_dist < cfg["goal_radius"] else 0.0
    collision_penalty = -reward_cfg["collision"] if clearance <= 0.0 else 0.0
    near_gap = max(0.0, safe_margin - clearance)
    danger_gap = max(0.0, danger_margin - clearance)
    unsafe_gap = max(0.0, unsafe_margin - clearance)
    near_obstacle = -reward_cfg["near_obstacle"] * near_gap
    danger_obstacle = -reward_cfg.get("danger_obstacle", 0.0) * (danger_gap**2) / max(danger_margin, 1e-6)
    unsafe_obstacle = -reward_cfg.get("unsafe_obstacle", 0.0) * (unsafe_gap**2) / max(unsafe_margin, 1e-6)
    direction_penalty = -reward_cfg["direction_alignment"] * abs(direction_error)
    angular_rate_penalty = -reward_cfg["angular_rate"] * abs(angular_rate_norm)
    smooth_penalty = -reward_cfg["smooth"] * float(np.sum((action - prev_action) ** 2))
    energy_penalty = -reward_cfg["energy"] * float(np.sum(action_ref ** 2))
    time_penalty = -reward_cfg.get("time", 0.0)

    terms = {
        "progress": progress,
        "goal": goal_bonus,
        "collision": collision_penalty,
        "near_obstacle": near_obstacle,
        "danger_obstacle": danger_obstacle,
        "unsafe_obstacle": unsafe_obstacle,
        "direction_alignment": direction_penalty,
        "angular_rate": angular_rate_penalty,
        "smooth": smooth_penalty,
        "energy": energy_penalty,
        "time": time_penalty,
    }
    return float(sum(terms.values())), terms
