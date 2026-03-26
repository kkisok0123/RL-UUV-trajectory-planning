from __future__ import annotations

import numpy as np


def find_local_target(traj: dict, current_pos: np.ndarray, lookahead_dist: float) -> np.ndarray:
    """Pick a path point ahead of the robot by the requested lookahead distance."""
    xyz = np.asarray(traj["xyz"], dtype=np.float64)
    current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)

    dists_to_traj = np.sum((xyz - current_pos.reshape(1, 3)) ** 2, axis=1)
    idx_curr = int(np.argmin(dists_to_traj))

    idx_tgt = idx_curr
    dist_accum = 0.0
    for idx in range(idx_curr, xyz.shape[0] - 1):
        dist_accum += float(np.linalg.norm(xyz[idx + 1] - xyz[idx]))
        if dist_accum >= lookahead_dist:
            idx_tgt = idx + 1
            break

    return xyz[idx_tgt].copy()


def get_visible_obstacles(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    all_obstacles: list[dict],
    params: dict,
) -> dict:
    """Cast FOV rays and return the blocked rays and hit distances."""
    robot_pos = np.asarray(robot_pos, dtype=np.float64).reshape(3)
    robot_vel = np.asarray(robot_vel, dtype=np.float64).reshape(3)
    range_max = float(params["range"])
    fov_deg = float(params["fov_angle"])

    if np.linalg.norm(robot_vel) > 0.01:
        x_body = robot_vel / np.linalg.norm(robot_vel)
    else:
        x_body = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    up = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(x_body[2]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    y_body = np.cross(up, x_body)
    y_body = y_body / np.linalg.norm(y_body)
    z_body = np.cross(x_body, y_body)
    r_body = np.column_stack([x_body, y_body, z_body])

    num_rays = int(params.get("num_rays", 100))
    indices = np.arange(num_rays)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * indices
    cos_fov = np.cos(np.deg2rad(fov_deg))
    x_s = 1.0 - (1.0 - cos_fov) * indices / max(num_rays - 1, 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - x_s**2))
    y_s = radius * np.cos(phi)
    z_s = radius * np.sin(phi)
    rays_body = np.vstack([x_s, y_s, z_s])
    rays_world = r_body @ rays_body

    dists = np.full(num_rays, range_max, dtype=np.float64)
    is_blocked = np.zeros(num_rays, dtype=bool)

    for ray_idx in range(num_rays):
        ray_dir = rays_world[:, ray_idx]
        min_dist = range_max
        hit = False
        for obstacle in all_obstacles:
            center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
            radius_obs = float(obstacle["r"])
            m_vec = robot_pos - center
            b_val = 2.0 * float(np.dot(m_vec, ray_dir))
            c_val = float(np.dot(m_vec, m_vec) - radius_obs**2)
            delta = b_val * b_val - 4.0 * c_val
            if delta < 0.0:
                continue
            sqrt_delta = np.sqrt(delta)
            d1 = (-b_val - sqrt_delta) / 2.0
            d2 = (-b_val + sqrt_delta) / 2.0
            d_hit = np.inf
            if d1 > 0.0:
                d_hit = d1
            elif d2 > 0.0:
                d_hit = d2
            if d_hit < min_dist:
                min_dist = d_hit
                hit = True
        dists[ray_idx] = min_dist
        is_blocked[ray_idx] = hit

    return {
        "origin": robot_pos,
        "rays": rays_world,
        "dists": dists,
        "is_blocked": is_blocked,
    }
