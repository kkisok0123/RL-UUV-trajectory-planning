from __future__ import annotations

import numpy as np

from dynamics_wrapper import probe_fin_wrenches
from dynamics_wrapper.indices import A1, A2, A3, A4, A5, Q0, Q1, Q2, Q3, VX, VY, VZ, WX, WY, WZ
from dynamics_wrapper.params import load_fin_params

_RIGHT_IDX = np.array([A1, A2], dtype=np.int64)
_LEFT_IDX = np.array([A3, A4], dtype=np.int64)
_TAIL_IDX = np.array([A5], dtype=np.int64)
_FIN_NAMES = ("right", "left", "tail")
_ACTUATED_WRENCH_IDX = np.array([0, 3, 4, 5], dtype=np.int64)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = np.asarray(q1, dtype=np.float64).reshape(4)
    b0, b1, b2, b3 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        dtype=np.float64,
    )


def _quaternion_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _roll_pitch_yaw_from_quaternion(q0: float, q1: float, q2: float, q3: float) -> tuple[float, float, float]:
    sin_roll = 2.0 * (q0 * q1 + q2 * q3)
    cos_roll = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sin_roll, cos_roll)
    yaw = np.arctan2(2.0 * (q1 * q2 + q0 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
    sin_pitch = 2.0 * (q0 * q2 - q3 * q1)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    return float(roll), float(pitch), float(yaw)


def _as_vec(params: dict, key: str, size: int, default) -> np.ndarray:
    value = params.get(key, default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{key} must have shape ({size},)")
    return arr


def _as_mat(params: dict, key: str, shape: tuple[int, int], default) -> np.ndarray:
    value = params.get(key, default)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}")
    return arr


def _as_wrench_vec(params: dict, key: str, default6) -> np.ndarray:
    value = params.get(key, default6)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (6,):
        return arr
    if arr.shape == (4,):
        out = np.asarray(default6, dtype=np.float64).copy()
        out[_ACTUATED_WRENCH_IDX] = arr
        return out
    raise ValueError(f"{key} must have shape (6,) or legacy shape (4,)")


def _as_wrench_mat(params: dict, key: str, default6: np.ndarray) -> np.ndarray:
    value = params.get(key, default6)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 6):
        return arr
    if arr.shape == (3, 4):
        out = np.asarray(default6, dtype=np.float64).copy()
        out[:, _ACTUATED_WRENCH_IDX] = arr
        return out
    raise ValueError(f"{key} must have shape (3, 6) or legacy shape (3, 4)")


def _get_force_gains(params: dict) -> np.ndarray:
    default = np.array(
        [
            [6.0, 1.5],
            [4.0, 1.5],
            [4.0, 1.5],
        ],
        dtype=np.float64,
    )
    if "backstepping_force_gains" in params:
        return _as_mat(params, "backstepping_force_gains", (3, 2), default)
    gains = default.copy()
    gains[0] = _as_vec(params, "backstepping_surge_gains", 2, default[0])
    return gains


def _current_actuator_state(hist: np.ndarray | None, controller_state: dict, params: dict) -> np.ndarray:
    if hist is not None:
        arr = np.asarray(hist, dtype=np.float64).reshape(5)
    elif "A_now" in controller_state:
        arr = np.asarray(controller_state["A_now"], dtype=np.float64).reshape(5)
    else:
        arr = np.asarray(params.get("trim_refs", np.zeros(5, dtype=np.float64)), dtype=np.float64).reshape(5)
    if not np.isfinite(arr).all():
        return np.asarray(params.get("trim_refs", np.zeros(5, dtype=np.float64)), dtype=np.float64).reshape(5)
    return arr


def _evaluate_joint_kinematics(
    actuator_state: np.ndarray,
    phase_time: float,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    actuator_state = np.asarray(actuator_state, dtype=np.float64).reshape(5)
    fin_f = float(params.get("fin_f", 4.0))
    phase = 2.0 * np.pi * fin_f * float(phase_time)
    sp = np.sin(phase)
    cp = np.cos(phase)
    omega = 2.0 * np.pi * fin_f

    joint_angles = np.array(
        [
            actuator_state[0] * sp,
            -actuator_state[1] * cp,
            -actuator_state[2] * sp,
            -actuator_state[3] * cp,
            actuator_state[4],
        ],
        dtype=np.float64,
    )
    joint_rates = np.array(
        [
            actuator_state[0] * cp * omega,
            actuator_state[1] * sp * omega,
            -actuator_state[2] * cp * omega,
            actuator_state[3] * sp * omega,
            0.0,
        ],
        dtype=np.float64,
    )
    return joint_angles, joint_rates


def _get_actuator_limits(params: dict) -> tuple[np.ndarray, np.ndarray]:
    default_min = np.array([0.0, 0.0, 0.0, 0.0, -np.pi / 4.0], dtype=np.float64)
    default_max = np.array([np.pi / 4.0, np.pi / 6.0, np.pi / 4.0, np.pi / 6.0, np.pi / 4.0], dtype=np.float64)
    lower = _as_vec(params, "actuator_min", 5, default_min)
    upper = _as_vec(params, "actuator_max", 5, default_max)
    return lower, upper


def _project_sum_box(x: np.ndarray, target_sum: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), lower, upper)
    for _ in range(8):
        residual = target_sum - float(np.sum(x))
        if abs(residual) < 1e-8:
            break
        if residual > 0.0:
            free = np.where(x < upper - 1e-10)[0]
        else:
            free = np.where(x > lower + 1e-10)[0]
        if free.size == 0:
            break
        x = x.copy()
        x[free] += residual / float(free.size)
        x = np.clip(x, lower, upper)
    return x


def _solve_channel_qp(
    tau_value: float,
    preferred: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    effort_weight: float,
    smooth_weight: float,
    asym_weight: float,
    iterations: int,
    step_size: float,
) -> np.ndarray:
    x = _project_sum_box(preferred, tau_value, lower, upper)
    for _ in range(max(int(iterations), 1)):
        grad = effort_weight * (x - preferred) + smooth_weight * (x - previous)
        diff = x[0] - x[1]
        grad[0] += asym_weight * diff
        grad[1] -= asym_weight * diff
        x = _project_sum_box(x - step_size * grad, tau_value, lower, upper)
    return x


def _probe_with_realized_actuators(
    current_state: np.ndarray,
    realized_actuators: np.ndarray,
    phase_time: float,
    params: dict,
) -> dict[str, np.ndarray]:
    fin_params = np.asarray(params.get("fin_params", load_fin_params()), dtype=np.float64).reshape(38)
    c_A = float(params.get("c_A", 5.0))
    fin_f = float(params.get("fin_f", fin_params[-4] if fin_params.shape[0] >= 35 else 4.0))
    return probe_fin_wrenches(
        current_state,
        fin_params,
        phase_time,
        realized_actuators,
        realized_actuators,
        c_A,
        fin_f,
    )


def _phase_average_sample_times(phase_time: float, params: dict) -> np.ndarray:
    fin_f = float(params.get("fin_f", 4.0))
    sample_count = max(int(params.get("inverse_phase_average_samples", 9)), 1)
    if sample_count == 1 or fin_f <= 0.0:
        return np.array([float(phase_time)], dtype=np.float64)
    period = 1.0 / fin_f
    window = float(params.get("inverse_phase_average_window", period))
    if window <= 0.0:
        window = period
    offsets = (np.arange(sample_count, dtype=np.float64) + 0.5) * (window / sample_count)
    return float(phase_time) + offsets


def _average_probe_dict(probes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    averaged: dict[str, np.ndarray] = {}
    for key in ("total", "right", "left", "tail"):
        averaged[key] = np.mean(
            [np.asarray(probe[key], dtype=np.float64).reshape(6) for probe in probes],
            axis=0,
        )
    return averaged


def _probe_with_phase_average(
    current_state: np.ndarray,
    realized_actuators: np.ndarray,
    phase_time: float,
    params: dict,
) -> dict[str, np.ndarray]:
    sample_times = _phase_average_sample_times(phase_time, params)
    probes = [
        _probe_with_realized_actuators(current_state, realized_actuators, float(sample_t), params)
        for sample_t in sample_times
    ]
    averaged = _average_probe_dict(probes)
    averaged["sample_times"] = sample_times.copy()
    return averaged


def _build_fin_jacobians(
    current_state: np.ndarray,
    A_now: np.ndarray,
    phase_time: float,
    params: dict,
    base_probe: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    eps = float(params.get("jacobian_epsilon", np.deg2rad(1.0)))
    act_min, act_max = _get_actuator_limits(params)
    jacobians: dict[str, np.ndarray] = {
        "right": np.zeros((6, 2), dtype=np.float64),
        "left": np.zeros((6, 2), dtype=np.float64),
        "tail": np.zeros((6, 1), dtype=np.float64),
    }
    for fin_name, indices in (("right", _RIGHT_IDX), ("left", _LEFT_IDX), ("tail", _TAIL_IDX)):
        base_wrench = np.asarray(base_probe[fin_name], dtype=np.float64).reshape(6)
        J = np.zeros((6, len(indices)), dtype=np.float64)
        for col, idx in enumerate(indices):
            perturbed = np.asarray(A_now, dtype=np.float64).copy()
            candidate = float(np.clip(perturbed[idx] + eps, act_min[idx], act_max[idx]))
            delta = candidate - perturbed[idx]
            if abs(delta) < 1e-8:
                candidate = float(np.clip(perturbed[idx] - eps, act_min[idx], act_max[idx]))
                delta = candidate - perturbed[idx]
            if abs(delta) < 1e-8:
                continue
            perturbed[idx] = candidate
            perturbed_probe = _probe_with_phase_average(current_state, perturbed, phase_time, params)
            J[:, col] = (np.asarray(perturbed_probe[fin_name], dtype=np.float64).reshape(6) - base_wrench) / delta
        jacobians[fin_name] = J
    return jacobians


def _damped_least_squares(J: np.ndarray, residual: np.ndarray, damping: float) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(J.shape[0])
    if J.size == 0:
        return np.zeros(0, dtype=np.float64)
    JTJ = J.T @ J + float(damping) * np.eye(J.shape[1], dtype=np.float64)
    return np.linalg.solve(JTJ, J.T @ residual)


def _stack_total_jacobian(jacobians: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((jacobians["right"], jacobians["left"], jacobians["tail"]))


def _apply_fin_affine_correction(
    A_nom: np.ndarray,
    params: dict,
    controller_state: dict,
) -> np.ndarray:
    act_min, act_max = _get_actuator_limits(params)
    A_nom = np.asarray(A_nom, dtype=np.float64).reshape(5)
    A_cmd = A_nom.copy()
    for fin_name, indices in (("right", _RIGHT_IDX), ("left", _LEFT_IDX), ("tail", _TAIL_IDX)):
        gain = np.asarray(
            controller_state.get(f"{fin_name}_G", np.ones(len(indices), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(len(indices))
        bias = np.asarray(
            controller_state.get(f"{fin_name}_b", np.zeros(len(indices), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(len(indices))
        A_cmd[indices] = np.clip(gain * A_nom[indices] + bias, act_min[indices], act_max[indices])
    return A_cmd


def _update_fin_affine_adaptation(
    fin_targets: np.ndarray,
    achieved_probe: dict[str, np.ndarray],
    jacobians: dict[str, np.ndarray],
    A_cmd: np.ndarray,
    params: dict,
    controller_state: dict,
) -> dict:
    gain_rate = float(params.get("adapt_gain_rate", 0.03))
    bias_rate = float(params.get("adapt_bias_rate", 0.02))
    min_scale = float(params.get("inverse_min_scale", np.deg2rad(2.0)))
    act_min, act_max = _get_actuator_limits(params)

    for row, (fin_name, indices) in enumerate((("right", _RIGHT_IDX), ("left", _LEFT_IDX), ("tail", _TAIL_IDX))):
        gain_key = f"{fin_name}_G"
        bias_key = f"{fin_name}_b"
        gain = np.asarray(
            controller_state.get(gain_key, np.ones(len(indices), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(len(indices))
        bias = np.asarray(
            controller_state.get(bias_key, np.zeros(len(indices), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(len(indices))
        residual = np.asarray(fin_targets[row], dtype=np.float64).reshape(6) - np.asarray(
            achieved_probe[fin_name],
            dtype=np.float64,
        ).reshape(6)
        sensitivity = np.asarray(jacobians[fin_name], dtype=np.float64).T @ residual
        gain = gain + gain_rate * sensitivity * np.maximum(np.abs(A_cmd[indices]), min_scale)
        bias = bias + bias_rate * sensitivity
        gain = np.clip(gain, float(params.get("adapt_gain_min", 0.6)), float(params.get("adapt_gain_max", 1.6)))
        bias = np.clip(
            bias,
            float(params.get("adapt_bias_min", -np.deg2rad(10.0))),
            float(params.get("adapt_bias_max", np.deg2rad(10.0))),
        )
        controller_state[gain_key] = gain
        controller_state[bias_key] = np.clip(bias, act_min[indices] - act_max[indices], act_max[indices] - act_min[indices])
        controller_state[f"{fin_name}_residual"] = residual.copy()

    return controller_state


def build_attitude_reference(
    cmd_vel_body: np.ndarray,
    current_state: np.ndarray,
    params: dict,
) -> np.ndarray:
    current_state = np.asarray(current_state, dtype=np.float64)
    cmd_vel_body = np.asarray(cmd_vel_body, dtype=np.float64).reshape(3)

    _, pitch, yaw = _roll_pitch_yaw_from_quaternion(
        current_state[Q0], current_state[Q1], current_state[Q2], current_state[Q3]
    )
    u_body, v_body, w_body = cmd_vel_body
    speed_cmd = float(np.linalg.norm(cmd_vel_body))
    zero_speed_eps = float(params.get("zero_speed_epsilon", 0.05))
    u_ref = float(np.clip(speed_cmd, 0.0, params.get("u_ref_max", 0.75)))

    if speed_cmd < zero_speed_eps:
        return np.array([0.0, yaw, pitch], dtype=np.float64)

    horizontal_speed = np.hypot(u_body, v_body)
    if horizontal_speed < zero_speed_eps:
        psi_ref = yaw
    else:
        psi_ref = yaw + np.arctan2(v_body, u_body)

    theta_ref = np.arctan2(-w_body, max(horizontal_speed, zero_speed_eps))
    theta_ref = float(np.clip(theta_ref, -params.get("theta_ref_max", np.pi / 5.0), params.get("theta_ref_max", np.pi / 5.0)))
    return np.array([u_ref, psi_ref, theta_ref], dtype=np.float64)


def _layer1_backstepping_eso(
    ref_cmd: np.ndarray,
    current_state: np.ndarray,
    controller_state: dict,
    dt: float,
    params: dict,
) -> tuple[np.ndarray, dict]:
    current_q = _normalize_quaternion(current_state[[Q0, Q1, Q2, Q3]])
    roll, pitch, yaw = _roll_pitch_yaw_from_quaternion(*current_q)
    linear_vel = np.asarray(current_state[[VX, VY, VZ]], dtype=np.float64)
    omega = np.asarray(current_state[[WX, WY, WZ]], dtype=np.float64)

    phi_ref = float(params.get("phi_ref", 0.0))
    theta_ref = float(np.clip(ref_cmd[2], -params["theta_ref_max"], params["theta_ref_max"]))
    psi_ref = float(ref_cmd[1])
    u_ref = float(np.clip(ref_cmd[0], 0.0, params["u_ref_max"]))

    q_d = _normalize_quaternion(_quaternion_from_roll_pitch_yaw(phi_ref, theta_ref, psi_ref))
    q_err = _normalize_quaternion(_quaternion_multiply(q_d, _quaternion_conjugate(current_q)))
    if q_err[0] < 0.0:
        q_err = -q_err
    e_rot = q_err[1:]

    attitude_error = np.array(
        [
            _wrap_angle(phi_ref - roll),
            _wrap_angle(theta_ref - pitch),
            _wrap_angle(psi_ref - yaw)
        ],
        dtype=np.float64,
    )

    k_virtual = _as_vec(params, "backstepping_virtual_rate_gains", 3, np.array([1.5, 1.8, 1.5], dtype=np.float64))
    k_att = _as_vec(params, "backstepping_attitude_gains", 3, np.array([8.0, 10.0, 8.0], dtype=np.float64))
    k_rate = _as_vec(params, "backstepping_rate_gains", 3, np.array([2.0, 2.5, 2.0], dtype=np.float64))
    force_gains = _get_force_gains(params)
    linear_ref = np.array([u_ref, 0.0, 0.0], dtype=np.float64)
    linear_error = linear_ref - linear_vel
    force_bs = force_gains[:, 0] * linear_error - force_gains[:, 1] * linear_vel
    omega_virtual = -k_virtual * e_rot
    omega_error = omega - omega_virtual
    moment_bs = np.array(
        [
            k_att[0] * e_rot[0] - k_rate[0] * omega_error[0],
            k_att[1] * e_rot[1] - k_rate[1] * omega_error[1],
            k_att[2] * e_rot[2] - k_rate[2] * omega_error[2],
        ],
        dtype=np.float64,
    )
    tau_bs = np.concatenate((force_bs, moment_bs))

    y = np.concatenate((linear_vel, np.array([roll, pitch, yaw], dtype=np.float64)))
    z1 = np.asarray(controller_state.get("eso_z1", y.copy()), dtype=np.float64).reshape(6)
    z2 = np.asarray(controller_state.get("eso_z2", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(6)
    tau_prev = np.asarray(controller_state.get("tau_prev", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(6)
    beta1 = _as_wrench_vec(params, "eso_beta1", np.array([8.0, 8.0, 8.0, 10.0, 10.0, 8.0], dtype=np.float64))
    beta2 = _as_wrench_vec(params, "eso_beta2", np.array([20.0, 20.0, 20.0, 25.0, 25.0, 20.0], dtype=np.float64))
    b0 = _as_wrench_vec(params, "eso_b0", np.ones(6, dtype=np.float64))

    obs_error = y - z1
    obs_error[3] = _wrap_angle(obs_error[3])
    obs_error[4] = _wrap_angle(obs_error[4])
    obs_error[5] = _wrap_angle(obs_error[5])
    z1 = z1 + dt * (z2 + b0 * tau_prev + beta1 * obs_error)
    z1[3] = _wrap_angle(z1[3])
    z1[4] = _wrap_angle(z1[4])
    z1[5] = _wrap_angle(z1[5])
    z2 = z2 + dt * (beta2 * obs_error)
    d_hat = z2.copy()

    tau_d = tau_bs - d_hat
    tau_limits = _as_wrench_vec(params, "tau_limits", np.array([6.0, 2.0, 2.0, 1.5, 2.0, 1.5], dtype=np.float64))
    tau_d = np.clip(tau_d, -tau_limits, tau_limits)

    controller_state["eso_z1"] = z1
    controller_state["eso_z2"] = z2
    controller_state["tau_prev"] = tau_d.copy()
    controller_state["tau_bs"] = tau_bs.copy()
    controller_state["d_hat"] = d_hat.copy()
    controller_state["tau_d"] = tau_d.copy()
    controller_state["surge_error"] = float(linear_error[0])
    controller_state["linear_velocity_error"] = linear_error.copy()
    controller_state["force_cmd"] = tau_d[:3].copy()
    controller_state["attitude_error"] = attitude_error.copy()
    controller_state["moment_cmd"] = tau_d[3:].copy()
    return tau_d, controller_state


def _layer2_allocate_fin_wrench(
    tau_d: np.ndarray,
    controller_state: dict,
    params: dict,
) -> tuple[np.ndarray, dict]:
    tau_d = np.asarray(tau_d, dtype=np.float64).reshape(6)
    tau_alloc_request = tau_d.copy()
    shares = _as_wrench_mat(
        params,
        "fin_share_matrix",
        np.array(
            [
                [0.45, 0.50, 0.10, 0.50, 0.10, 0.50],
                [0.45, 0.50, 0.10, 0.50, 0.10, 0.50],
                [0.10, 0.00, 0.80, 0.00, 0.80, 0.00],
            ],
            dtype=np.float64,
        ),
    )
    wrench_max = _as_wrench_mat(
        params,
        "fin_wrench_max",
        np.array(
            [
                [4.0, 1.25, 0.60, 1.25, 0.60, 1.00],
                [4.0, 1.25, 0.60, 1.25, 0.60, 1.00],
                [1.5, 0.30, 2.00, 0.30, 2.00, 0.30],
            ],
            dtype=np.float64,
        ),
    )
    lower = -wrench_max
    upper = wrench_max
    prev = np.asarray(controller_state.get("fin_wrench_prev", np.zeros((3, 6), dtype=np.float64)), dtype=np.float64).reshape(3, 6)
    effort_weights = _as_wrench_vec(params, "alloc_effort_weight", np.ones(6, dtype=np.float64))
    smooth_weights = _as_wrench_vec(params, "alloc_smooth_weight", np.full(6, 0.15, dtype=np.float64))
    asym_weights = _as_wrench_vec(params, "alloc_asym_weight", np.array([0.25, 0.10, 0.50, 0.05, 0.50, 0.05], dtype=np.float64))
    step_size = float(params.get("alloc_step_size", 0.35))
    iterations = int(params.get("alloc_iterations", 12))

    fin_targets = np.zeros((3, 6), dtype=np.float64)
    for channel in range(6):
        preferred = shares[:, channel] * tau_alloc_request[channel]
        fin_targets[:, channel] = _solve_channel_qp(
            tau_alloc_request[channel],
            preferred,
            prev[:, channel],
            lower[:, channel],
            upper[:, channel],
            float(effort_weights[channel]),
            float(smooth_weights[channel]),
            float(asym_weights[channel]),
            iterations,
            step_size,
        )

    controller_state["fin_wrench_prev"] = fin_targets.copy()
    controller_state["tau_d_alloc"] = tau_alloc_request.copy()
    controller_state["w_r_des"] = fin_targets[0].copy()
    controller_state["w_l_des"] = fin_targets[1].copy()
    controller_state["w_t_des"] = fin_targets[2].copy()
    controller_state["tau_alloc"] = np.sum(fin_targets, axis=0)
    return fin_targets, controller_state


def _adaptive_fin_inverse(
    fin_name: str,
    local_indices: np.ndarray,
    desired_wrench: np.ndarray,
    current_wrench: np.ndarray,
    J: np.ndarray,
    A_now: np.ndarray,
    current_state: np.ndarray,
    phase_time: float,
    params: dict,
    controller_state: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    act_min, act_max = _get_actuator_limits(params)
    local_min = act_min[local_indices]
    local_max = act_max[local_indices]
    A_local_now = A_now[local_indices]

    desired_wrench = np.asarray(desired_wrench, dtype=np.float64).reshape(6)
    current_wrench = np.asarray(current_wrench, dtype=np.float64).reshape(6)
    J = np.asarray(J, dtype=np.float64)
    residual = desired_wrench - current_wrench
    damping = float(params.get("inverse_damping", 0.08))
    delta_nom = _damped_least_squares(J, residual, damping)
    A_nom = np.clip(A_local_now + delta_nom, local_min, local_max)

    gain_key = f"{fin_name}_G"
    bias_key = f"{fin_name}_b"
    gain = np.asarray(controller_state.get(gain_key, np.ones(len(local_indices), dtype=np.float64)), dtype=np.float64).reshape(len(local_indices))
    bias = np.asarray(controller_state.get(bias_key, np.zeros(len(local_indices), dtype=np.float64)), dtype=np.float64).reshape(len(local_indices))
    A_des = np.clip(gain * A_nom + bias, local_min, local_max)

    candidate = A_now.copy()
    candidate[local_indices] = A_des
    achieved_probe = _probe_with_realized_actuators(current_state, candidate, phase_time, params)
    achieved_wrench = np.asarray(achieved_probe[fin_name], dtype=np.float64).reshape(6)
    achieved_residual = desired_wrench - achieved_wrench

    sensitivity = J.T @ achieved_residual
    gain_rate = float(params.get("adapt_gain_rate", 0.03))
    bias_rate = float(params.get("adapt_bias_rate", 0.02))
    min_scale = float(params.get("inverse_min_scale", np.deg2rad(2.0)))
    gain = gain + gain_rate * sensitivity * np.maximum(np.abs(A_nom), min_scale)
    bias = bias + bias_rate * sensitivity
    gain = np.clip(gain, float(params.get("adapt_gain_min", 0.6)), float(params.get("adapt_gain_max", 1.6)))
    bias = np.clip(bias, float(params.get("adapt_bias_min", -np.deg2rad(10.0))), float(params.get("adapt_bias_max", np.deg2rad(10.0))))

    controller_state[gain_key] = gain
    controller_state[bias_key] = bias
    controller_state[f"{fin_name}_residual"] = achieved_residual.copy()
    return A_des, achieved_wrench, controller_state


def _solve_actuator_targets(
    tau_target: np.ndarray,
    fin_targets: np.ndarray,
    current_state: np.ndarray,
    A_now: np.ndarray,
    phase_time: float,
    params: dict,
    controller_state: dict,
) -> tuple[np.ndarray, dict]:
    tau_target = np.asarray(tau_target, dtype=np.float64).reshape(6)
    fin_targets = np.asarray(fin_targets, dtype=np.float64).reshape(3, 6)
    base_probe = _probe_with_phase_average(current_state, A_now, phase_time, params)
    jacobians = _build_fin_jacobians(current_state, A_now, phase_time, params, base_probe)

    act_min, act_max = _get_actuator_limits(params)
    damping = float(params.get("inverse_damping", 0.08))
    inverse_iterations = max(int(params.get("inverse_iterations", 2)), 1)
    A_iter = A_now.copy()
    last_probe = base_probe
    last_jacobians = jacobians
    for iteration in range(inverse_iterations):
        if iteration > 0:
            last_probe = _probe_with_phase_average(current_state, A_iter, phase_time, params)
            last_jacobians = _build_fin_jacobians(current_state, A_iter, phase_time, params, last_probe)
        residual = tau_target - np.asarray(last_probe["total"], dtype=np.float64).reshape(6)
        delta_nom = _damped_least_squares(_stack_total_jacobian(last_jacobians), residual, damping)
        A_nom = np.clip(A_iter + delta_nom, act_min, act_max)
        A_iter = _apply_fin_affine_correction(A_nom, params, controller_state)

    rate_limit = _as_vec(
        params,
        "actuator_rate_limit",
        5,
        np.array([0.70, 0.70, 0.70, 0.70, 0.90], dtype=np.float64),
    )
    delta_limit = rate_limit * max(float(params.get("controller_dt", 0.0)), 0.0)
    if float(params.get("controller_dt", 0.0)) <= 0.0:
        delta_limit = rate_limit * 0.2
    A_des = np.clip(A_iter, A_now - delta_limit, A_now + delta_limit)
    A_des = np.clip(A_des, act_min, act_max)
    achieved_probe = _probe_with_phase_average(current_state, A_des, phase_time, params)
    controller_state = _update_fin_affine_adaptation(
        fin_targets,
        achieved_probe,
        last_jacobians,
        A_des,
        params,
        controller_state,
    )

    controller_state["A_now"] = A_now.copy()
    controller_state["A_des"] = A_des.copy()
    controller_state["J_r"] = last_jacobians["right"].copy()
    controller_state["J_l"] = last_jacobians["left"].copy()
    controller_state["J_t"] = last_jacobians["tail"].copy()
    controller_state["J_total"] = _stack_total_jacobian(last_jacobians).copy()
    controller_state["tau_target"] = tau_target.copy()
    controller_state["tau_residual"] = tau_target - np.asarray(achieved_probe["total"], dtype=np.float64).reshape(6)
    controller_state["inverse_phase_average_samples"] = int(len(np.asarray(base_probe["sample_times"], dtype=np.float64)))
    controller_state["inverse_phase_sample_times"] = np.asarray(base_probe["sample_times"], dtype=np.float64).copy()
    controller_state["fin_probe_now"] = {
        "total": np.asarray(base_probe["total"], dtype=np.float64).copy(),
        "right": np.asarray(base_probe["right"], dtype=np.float64).copy(),
        "left": np.asarray(base_probe["left"], dtype=np.float64).copy(),
        "tail": np.asarray(base_probe["tail"], dtype=np.float64).copy(),
    }
    controller_state["fin_probe_des"] = {
        "total": np.asarray(achieved_probe["total"], dtype=np.float64).copy(),
        "right": np.asarray(achieved_probe["right"], dtype=np.float64).copy(),
        "left": np.asarray(achieved_probe["left"], dtype=np.float64).copy(),
        "tail": np.asarray(achieved_probe["tail"], dtype=np.float64).copy(),
    }
    return A_des, controller_state


def _compute_a_refs(
    A_des: np.ndarray,
    A_now: np.ndarray,
    params: dict,
    dt: float,
    controller_state: dict,
) -> tuple[np.ndarray, dict]:
    c_A = float(params.get("c_A", 5.0))
    half_c = 0.5 * c_A
    g_dt = float((1.0 + half_c * dt) * np.exp(-half_c * dt))
    denom = max(1.0 - g_dt, 1e-6)
    A_ref = (np.asarray(A_des, dtype=np.float64).reshape(5) - g_dt * np.asarray(A_now, dtype=np.float64).reshape(5)) / denom
    act_min, act_max = _get_actuator_limits(params)
    A_ref = np.clip(A_ref, act_min, act_max)
    controller_state["A_ref"] = A_ref.copy()
    controller_state["cpg_g_dt"] = g_dt
    return A_ref, controller_state


def _prepare_ref_cmd(ref_cmd: np.ndarray, controller_state: dict, params: dict) -> tuple[np.ndarray, dict]:
    ref_cmd = np.asarray(ref_cmd, dtype=np.float64).reshape(3)
    u_ref = float(np.clip(ref_cmd[0], 0.0, params["u_ref_max"]))
    psi_ref = float(ref_cmd[1])
    theta_ref = float(np.clip(ref_cmd[2], -params["theta_ref_max"], params["theta_ref_max"]))

    ref_filter_alpha = float(params.get("ref_filter_alpha", 1.0))
    if 0.0 < ref_filter_alpha < 1.0:
        prev_ref = np.asarray(
            controller_state.get("filtered_ref_cmd", np.array([u_ref, psi_ref, theta_ref], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        psi_delta = _wrap_angle(psi_ref - float(prev_ref[1]))
        theta_delta = _wrap_angle(theta_ref - float(prev_ref[2]))
        filtered_ref = np.array(
            [
                ref_filter_alpha * u_ref + (1.0 - ref_filter_alpha) * float(prev_ref[0]),
                float(prev_ref[1]) + ref_filter_alpha * psi_delta,
                float(prev_ref[2]) + ref_filter_alpha * theta_delta,
            ],
            dtype=np.float64,
        )
        controller_state["filtered_ref_cmd"] = filtered_ref.copy()
        return filtered_ref, controller_state

    return np.array([u_ref, psi_ref, theta_ref], dtype=np.float64), controller_state


def run_layer1_controller(
    ref_cmd: np.ndarray,
    current_state: np.ndarray,
    controller_state: dict,
    dt: float,
    params: dict,
) -> tuple[np.ndarray, dict]:
    current_state = np.asarray(current_state, dtype=np.float64)
    ref_cmd, controller_state = _prepare_ref_cmd(ref_cmd, controller_state, params)
    params = dict(params)
    params["controller_dt"] = float(dt)

    tau_d, controller_state = _layer1_backstepping_eso(
        ref_cmd,
        current_state,
        controller_state,
        dt,
        params,
    )
    controller_state["layers"] = {
        "layer1": {
            "ref_cmd": ref_cmd.copy(),
            "tau_bs": np.asarray(controller_state["tau_bs"], dtype=np.float64).copy(),
            "tau_d": np.asarray(controller_state["tau_d"], dtype=np.float64).copy(),
            "d_hat": np.asarray(controller_state["d_hat"], dtype=np.float64).copy(),
            "force_cmd": np.asarray(controller_state["force_cmd"], dtype=np.float64).copy(),
            "moment_cmd": np.asarray(controller_state["moment_cmd"], dtype=np.float64).copy(),
            "linear_velocity_error": np.asarray(controller_state["linear_velocity_error"], dtype=np.float64).copy(),
            "attitude_error": np.asarray(controller_state["attitude_error"], dtype=np.float64).copy(),
        }
    }
    return tau_d.copy(), controller_state


def run_layer2_controller(
    ref_cmd: np.ndarray,
    current_state: np.ndarray,
    hist: np.ndarray | None,
    controller_state: dict,
    dt: float,
    params: dict,
) -> tuple[np.ndarray, dict]:
    current_state = np.asarray(current_state, dtype=np.float64)
    ref_cmd, controller_state = _prepare_ref_cmd(ref_cmd, controller_state, params)
    phase_time = float(controller_state.get("phase_time", 0.0))
    A_now = _current_actuator_state(hist, controller_state, params)
    params = dict(params)
    params["controller_dt"] = float(dt)
    params.setdefault("inverse_phase_average_window", 1.0 / max(float(params.get("fin_f", 4.0)), 1e-6))

    tau_d, controller_state = _layer1_backstepping_eso(
        ref_cmd,
        current_state,
        controller_state,
        dt,
        params,
    )
    fin_targets, controller_state = _layer2_allocate_fin_wrench(tau_d, controller_state, params)
    tau_target = np.asarray(controller_state["tau_alloc"], dtype=np.float64).reshape(6)
    A_des, controller_state = _solve_actuator_targets(
        tau_target,
        fin_targets,
        current_state,
        A_now,
        phase_time,
        params,
        controller_state,
    )
    A_ref, controller_state = _compute_a_refs(A_des, A_now, params, dt, controller_state)

    controller_state["phase_time"] = phase_time + dt
    controller_state["tau_target"] = tau_target.copy()
    controller_state["layers"] = {
        "layer1": {
            "ref_cmd": ref_cmd.copy(),
            "tau_d": np.asarray(controller_state["tau_d"], dtype=np.float64).copy(),
            "force_cmd": np.asarray(controller_state["force_cmd"], dtype=np.float64).copy(),
            "moment_cmd": np.asarray(controller_state["moment_cmd"], dtype=np.float64).copy(),
            "d_hat": np.asarray(controller_state["d_hat"], dtype=np.float64).copy(),
        },
        "layer2": {
            "tau_d_alloc": np.asarray(controller_state["tau_d_alloc"], dtype=np.float64).copy(),
            "tau_target": tau_target.copy(),
            "w_r_des": np.asarray(controller_state["w_r_des"], dtype=np.float64).copy(),
            "w_l_des": np.asarray(controller_state["w_l_des"], dtype=np.float64).copy(),
            "w_t_des": np.asarray(controller_state["w_t_des"], dtype=np.float64).copy(),
            "tau_alloc": np.asarray(controller_state["tau_alloc"], dtype=np.float64).copy(),
            "A_now": np.asarray(controller_state["A_now"], dtype=np.float64).copy(),
            "A_des": np.asarray(controller_state["A_des"], dtype=np.float64).copy(),
            "A_ref": np.asarray(controller_state["A_ref"], dtype=np.float64).copy(),
        },
    }
    return A_ref.copy(), controller_state


def fin_controller(
    ref_cmd: np.ndarray,
    current_state: np.ndarray,
    hist: np.ndarray | None,
    controller_state: dict,
    dt: float,
    params: dict,
) -> tuple[float, float, float, float, float, dict]:
    A_ref, controller_state = run_layer2_controller(
        ref_cmd,
        current_state,
        hist,
        controller_state,
        dt,
        params,
    )
    return float(A_ref[0]), float(A_ref[1]), float(A_ref[2]), float(A_ref[3]), float(A_ref[4]), controller_state
