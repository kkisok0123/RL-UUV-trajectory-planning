from __future__ import annotations

import numpy as np


def compute_reward(
    prev_goal_dist: float,
    goal_dist: float,
    clearance: float,
    yaw_err: float,
    wz: float,
    action: np.ndarray,
    prev_action: np.ndarray,
    action_ref: np.ndarray,
    cfg: dict,
) -> tuple[float, dict]:
    reward_cfg = cfg["reward"]
    progress = reward_cfg["progress"] * (prev_goal_dist - goal_dist)
    goal_bonus = reward_cfg["goal"] if goal_dist < cfg["goal_radius"] else 0.0
    collision_penalty = -reward_cfg["collision"] if clearance <= 0.0 else 0.0
    near_obstacle = -reward_cfg["near_obstacle"] * max(0.0, cfg["safe_margin"] - clearance)
    heading_penalty = -reward_cfg["heading"] * abs(yaw_err)
    yaw_rate_penalty = -reward_cfg["yaw_rate"] * abs(wz)
    smooth_penalty = -reward_cfg["smooth"] * float(np.sum((action - prev_action) ** 2))
    energy_penalty = -reward_cfg["energy"] * float(np.sum(action_ref ** 2))

    terms = {
        "progress": progress,
        "goal": goal_bonus,
        "collision": collision_penalty,
        "near_obstacle": near_obstacle,
        "heading": heading_penalty,
        "yaw_rate": yaw_rate_penalty,
        "smooth": smooth_penalty,
        "energy": energy_penalty,
    }
    return float(sum(terms.values())), terms
