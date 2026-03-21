from __future__ import annotations

import numpy as np

# Sample a vehicle state for fin-wrench probing. The probe only uses body
# velocities, so quaternion is identity and position remains zero.
def sample_body_state(
    rng: np.random.Generator,
    linear_velocity_limits: np.ndarray,
    angular_velocity_limits: np.ndarray,
) -> np.ndarray:
    state = np.zeros(13, dtype=np.float64)
    state[6] = 1.0
    state[:3] = rng.uniform(-linear_velocity_limits, linear_velocity_limits)
    state[3:6] = rng.uniform(-angular_velocity_limits, angular_velocity_limits)
    return state

# Sample joint angles and joint rates from amplitude-phase variables so the
# generated fin motion stays consistent with the sinusoidal actuation model.
def sample_joint_state(
    rng: np.random.Generator,
    amplitude_min: np.ndarray,
    amplitude_max: np.ndarray,
    tail_angle_min: float,
    tail_angle_max: float,
    phase_min: float,
    phase_max: float,
    fin_f: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    amplitudes = rng.uniform(amplitude_min, amplitude_max)
    tail_angle = float(rng.uniform(tail_angle_min, tail_angle_max))
    phase = float(rng.uniform(phase_min, phase_max))
    omega = 2.0 * np.pi * fin_f
    sp = np.sin(phase)
    cp = np.cos(phase)

    joint_angles = np.array(
        [
            amplitudes[0] * sp,
            -amplitudes[1] * cp,
            -amplitudes[2] * sp,
            -amplitudes[3] * cp,
            tail_angle,
        ],
        dtype=np.float64,
    )
    joint_rates = np.array(
        [
            amplitudes[0] * omega * cp,
            amplitudes[1] * omega * sp,
            -amplitudes[2] * omega * cp,
            amplitudes[3] * omega * sp,
            0.0,
        ],
        dtype=np.float64,
    )
    meta = {
        "amplitudes": amplitudes,
        "tail_angle": tail_angle,
        "phase": phase,
    }
    return joint_angles, joint_rates, meta
