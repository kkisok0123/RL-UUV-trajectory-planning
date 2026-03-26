from __future__ import annotations

import numpy as np


def los_guidance_3d(
    pos_re: np.ndarray,
    s_range: tuple[int, int],
    delta_psi: float,
    delta_theta: float,
    fx,
    fy,
    fz,
) -> tuple[float, float, float, float]:
    n = 4000
    pos_re = np.asarray(pos_re, dtype=np.float64).reshape(3)

    s_k = np.linspace(s_range[0], s_range[1], n)
    x_k = fx(s_k)
    y_k = fy(s_k)
    z_k = fz(s_k)

    d2 = (x_k - pos_re[0]) ** 2 + (y_k - pos_re[1]) ** 2 + (z_k - pos_re[2]) ** 2
    k = int(np.argmin(d2))
    x_p = x_k[k]
    y_p = y_k[k]
    z_p = z_k[k]

    k_prev = max(k - 1, 0)
    k_next = min(k + 1, n - 1)
    dx = x_k[k_next] - x_k[k_prev]
    dy = y_k[k_next] - y_k[k_prev]
    dz = z_k[k_next] - z_k[k_prev]

    norm_t = np.sqrt(dx * dx + dy * dy + dz * dz)
    if norm_t < 1e-6:
        dx = x_p - pos_re[0]
        dy = y_p - pos_re[1]
        dz = z_p - pos_re[2]
        norm_t = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)

    tx = dx / norm_t
    ty = dy / norm_t
    tz = dz / norm_t

    psi_p = np.arctan2(ty, tx)
    theta_p = np.arctan2(-tz, np.sqrt(tx * tx + ty * ty))

    e_x = pos_re[0] - x_p
    e_y = pos_re[1] - y_p
    e_z = pos_re[2] - z_p
    y_e = -e_x * np.sin(psi_p) + e_y * np.cos(psi_p)
    z_e = (
        e_x * np.cos(psi_p) * np.sin(theta_p)
        + e_y * np.sin(psi_p) * np.sin(theta_p)
        + e_z * np.cos(theta_p)
    )

    psi_ref = np.arctan2(np.sin(psi_p - np.arctan(y_e / delta_psi)), np.cos(psi_p - np.arctan(y_e / delta_psi)))
    theta_ref = np.arctan2(
        np.sin(theta_p + np.arctan(z_e / delta_theta)),
        np.cos(theta_p + np.arctan(z_e / delta_theta)),
    )
    return float(y_e), float(z_e), float(psi_ref), float(theta_ref)
