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


def quaternion_to_rotation_matrix(q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=np.float64,
    )


def rotate_world_to_body(vec_world: np.ndarray, rotation_body_to_world: np.ndarray) -> np.ndarray:
    return rotation_body_to_world.T @ np.asarray(vec_world, dtype=np.float64)


def rotate_body_to_world(vec_body: np.ndarray, rotation_body_to_world: np.ndarray) -> np.ndarray:
    return rotation_body_to_world @ np.asarray(vec_body, dtype=np.float64)


def vector_angle(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
    vec_a = np.asarray(vec_a, dtype=np.float64)
    vec_b = np.asarray(vec_b, dtype=np.float64)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < eps or norm_b < eps:
        return 0.0
    cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def nearest_obstacle_info(
    position_world: np.ndarray,
    velocity_world: np.ndarray,
    obstacles: Iterable[dict],
    fish_radius: float,
) -> dict:
    best = None
    best_distance = np.inf
    for obstacle in obstacles:
        center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
        velocity = np.asarray(obstacle.get("v", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
        rel = center - position_world
        dist = float(np.linalg.norm(rel))
        clearance = dist - float(obstacle["r"]) - fish_radius
        if dist < best_distance:
            best_distance = dist
            best = {
                "c": center,
                "r": float(obstacle["r"]),
                "v": velocity,
                "distance": dist,
                "clearance": clearance,
                "rel_world": rel,
                "rel_vel_world": velocity - velocity_world,
            }
    if best is None:
        best = {
            "c": np.full(3, np.inf, dtype=np.float64),
            "r": 0.0,
            "v": np.zeros(3, dtype=np.float64),
            "distance": np.inf,
            "clearance": np.inf,
            "rel_world": np.full(3, np.inf, dtype=np.float64),
            "rel_vel_world": np.zeros(3, dtype=np.float64),
        }
    return best
