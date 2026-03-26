from __future__ import annotations

import numpy as np

from dynamics.indices import Q0, Q1, Q2, Q3


def extract_attitude(current_state: np.ndarray) -> tuple[float, float]:
    """Extract yaw and pitch from the fish quaternion state."""
    current_state = np.asarray(current_state, dtype=np.float64)
    q0 = current_state[Q0]
    q1 = current_state[Q1]
    q2 = current_state[Q2]
    q3 = current_state[Q3]
    psi = np.arctan2(2.0 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    theta = np.arctan2(
        2.0 * (q0 * q2 - q1 * q3),
        np.sqrt(max(0.0, 1.0 - (2.0 * (q0 * q2 - q1 * q3)) ** 2)),
    )
    return float(psi), float(theta)


def velocity_to_attitude_refs(cmd_vel: np.ndarray, current_state: np.ndarray) -> tuple[float, float, float]:
    """Convert a desired world-frame velocity into yaw, pitch, and speed references."""
    cmd_vel = np.asarray(cmd_vel, dtype=np.float64).reshape(3)
    psi, theta = extract_attitude(current_state)

    vx_des, vy_des, vz_des = cmd_vel
    if np.linalg.norm(cmd_vel[:2]) > 0.05:
        psi_ref = np.arctan2(vy_des, vx_des)
    else:
        psi_ref = psi

    speed_xy_des = np.linalg.norm(cmd_vel[:2])
    speed_des = np.linalg.norm(cmd_vel)
    if speed_des > 0.05:
        theta_ref = np.arctan2(-vz_des, speed_xy_des)
    else:
        theta_ref = theta

    return float(psi_ref), float(theta_ref), float(speed_des)


def _smoothstep01(value: float) -> float:
    """Smoothly map a clamped scalar from 0 to 1."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def fin_controller(
    psi_ref: float,
    theta_ref: float,
    cmd_speed: float,
    current_state: np.ndarray,
    controller_state: dict,
    dt: float,
    params: dict,
) -> tuple[float, float, float, float, float, dict]:
    """Apply a PID-like fin controller around yaw and pitch references."""
    current_state = np.asarray(current_state, dtype=np.float64)
    psi_ref = float(psi_ref)
    theta_ref = float(theta_ref)
    cmd_speed = float(max(0.0, cmd_speed))

    psi, theta = extract_attitude(current_state)

    e_theta = np.arctan2(np.sin(theta_ref - theta), np.cos(theta_ref - theta))
    e_theta_prev = controller_state.get("e_theta_prev", controller_state.get("e_z_prev", 0.0))
    theta_int = controller_state.get("e_theta_int", 0.0) + e_theta * dt
    theta_int_limit = float(params.get("e_theta_int_limit", 1.0))
    theta_int = float(np.clip(theta_int, -theta_int_limit, theta_int_limit))
    de_theta = (e_theta - e_theta_prev) / dt
    alpha5_ref = (
        params["Kp_z"] * e_theta
        + params["Ki_z"] * theta_int
        + params["Kd_z"] * de_theta
    )
    alpha5_ref = float(np.clip(alpha5_ref, params["alpha5_min"], params["alpha5_max"]))
    controller_state["e_theta_prev"] = e_theta
    controller_state["e_theta_int"] = theta_int

    e_psi = np.arctan2(np.sin(psi_ref - psi), np.cos(psi_ref - psi))
    e_psi_prev = controller_state.get("e_psi_prev", 0.0)
    psi_int = controller_state.get("e_psi_int", 0.0) + e_psi * dt
    psi_int_limit = float(params.get("e_psi_int_limit", 1.0))
    psi_int = float(np.clip(psi_int, -psi_int_limit, psi_int_limit))
    de_psi = (e_psi - e_psi_prev) / dt
    delta_ref = (
        params["Kp_psi"] * e_psi
        + params.get("Ki_psi", 0.0) * psi_int
        + params["Kd_psi"] * de_psi
    )
    delta_ref = float(np.clip(delta_ref, -params["delta_rot_max"], params["delta_rot_max"]))
    controller_state["e_psi_prev"] = e_psi
    controller_state["e_psi_int"] = psi_int

    a2_ref = params["A_rot_base"] - delta_ref
    a4_ref = params["A_rot_base"] + delta_ref

    a_min = float(params.get("A_min", 0.0))
    a_max = float(params.get("A_max", params["A_base"]))
    speed_sched_min = float(params.get("speed_sched_min", 0.04))
    speed_sched_max = float(params.get("speed_sched_max", 0.30))
    if speed_sched_max <= speed_sched_min + 1e-9:
        a_prop = a_max if cmd_speed >= speed_sched_max else a_min
    else:
        sched = (cmd_speed - speed_sched_min) / (speed_sched_max - speed_sched_min)
        a_prop = a_min + _smoothstep01(sched) * (a_max - a_min)
    a_prop = float(np.clip(a_prop, min(a_min, a_max), max(a_min, a_max)))
    a1_ref = a_prop
    a3_ref = a_prop

    controller_state["theta_ref"] = theta_ref
    controller_state["theta"] = theta
    controller_state["e_theta"] = e_theta
    controller_state["psi_ref"] = psi_ref
    controller_state["psi"] = psi
    controller_state["e_psi"] = e_psi
    controller_state["alpha5_ref"] = alpha5_ref
    controller_state["delta_ref"] = delta_ref
    controller_state["cmd_speed"] = cmd_speed
    controller_state["a_prop_ref"] = a_prop

    fin_output_history = controller_state.setdefault("fin_output_history", [])
    fin_output = {
        "call_idx": len(fin_output_history),
        "a1_ref": float(a1_ref),
        "a2_ref": float(a2_ref),
        "a3_ref": float(a3_ref),
        "a4_ref": float(a4_ref),
        "alpha5_ref": float(alpha5_ref),
    }
    fin_output_history.append(fin_output.copy())
    controller_state["fin_output"] = fin_output

    return a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, controller_state
