from __future__ import annotations

import numpy as np


def find_local_target(traj: dict, current_pos: np.ndarray, lookahead_dist: float) -> np.ndarray:
    xyz = np.asarray(traj["xyz"], dtype=np.float64)
    current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)

    dists_to_traj = np.sum((xyz - current_pos.reshape(1, 3)) ** 2, axis=1)
    idx_curr = int(np.argmin(dists_to_traj))

    idx_tgt = idx_curr
    dist_accum = 0.0
    for i in range(idx_curr, xyz.shape[0] - 1):
        dist_accum += float(np.linalg.norm(xyz[i + 1] - xyz[i]))
        if dist_accum >= lookahead_dist:
            idx_tgt = i + 1
            break

    return xyz[idx_tgt].copy()
