from __future__ import annotations

import numpy as np

from dynamics_wrapper.indices import Q0, Q1, Q2, Q3, PZ


def fin_controller(
    cmd_vel: np.ndarray,
    current_state: np.ndarray,
    controller_state: dict,
    dt: float,
    params: dict,
) -> tuple[float, float, float, float, float, dict]:
    current_state = np.asarray(current_state, dtype=np.float64)
    cmd_vel = np.asarray(cmd_vel, dtype=np.float64).reshape(3)

    q0 = current_state[Q0]
    q1 = current_state[Q1]
    q2 = current_state[Q2]
    q3 = current_state[Q3]
    current_pos_z = current_state[PZ]

    psi = np.arctan2(2.0 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)

    vx_des, vy_des, vz_des = cmd_vel
    if np.linalg.norm(cmd_vel[:2]) > 0.05:
        psi_ref = np.arctan2(vy_des, vx_des)
    else:
        psi_ref = psi

    z_ref = current_pos_z + vz_des * 5.0

    e_z = z_ref - current_pos_z
    de_z = (e_z - controller_state["e_z_prev"]) / dt
    alpha5_ref = params["Kp_z"] * e_z + params["Kd_z"] * de_z
    alpha5_ref = float(np.clip(alpha5_ref, params["alpha5_min"], params["alpha5_max"]))
    controller_state["e_z_prev"] = e_z

    e_psi = np.arctan2(np.sin(psi_ref - psi), np.cos(psi_ref - psi))
    de_psi = (e_psi - controller_state["e_psi_prev"]) / dt
    delta_ref = params["Kp_psi"] * e_psi + params["Kd_psi"] * de_psi
    delta_ref = float(np.clip(delta_ref, -params["delta_rot_max"], params["delta_rot_max"]))
    controller_state["e_psi_prev"] = e_psi

    a2_ref = params["A_rot_base"] - delta_ref
    a4_ref = params["A_rot_base"] + delta_ref

    speed_cmd = float(np.linalg.norm(cmd_vel))
    v_max_expected = 0.4
    scaling_factor = min(speed_cmd / (0.5 * v_max_expected), 2.5)
    if speed_cmd < 0.05:
        a1_ref = 0.0
        a3_ref = 0.0
    else:
        a1_ref = params["A_base"] * scaling_factor
        a3_ref = params["A_base"] * scaling_factor

    return a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, controller_state
