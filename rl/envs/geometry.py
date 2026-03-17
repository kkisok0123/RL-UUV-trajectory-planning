from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quaternion(q0: float, q1: float, q2: float, q3: float) -> float:
    siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def rotate_world_to_body_2d(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([c * vec_xy[0] + s * vec_xy[1], -s * vec_xy[0] + c * vec_xy[1]])


def nearest_obstacle_info(position_xy: np.ndarray, obstacles: Iterable[dict], fish_radius: float) -> dict:
    best = None
    best_distance = np.inf
    for obstacle in obstacles:
        rel = obstacle["center"] - position_xy
        dist = float(np.linalg.norm(rel))
        clearance = dist - float(obstacle["radius"]) - fish_radius
        if dist < best_distance:
            best_distance = dist
            best = {
                "center": obstacle["center"],
                "radius": float(obstacle["radius"]),
                "distance": dist,
                "clearance": clearance,
                "rel_world": rel,
            }
    if best is None:
        best = {
            "center": np.array([np.inf, np.inf], dtype=np.float64),
            "radius": 0.0,
            "distance": np.inf,
            "clearance": np.inf,
            "rel_world": np.array([np.inf, np.inf], dtype=np.float64),
        }
    return best
