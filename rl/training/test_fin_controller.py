from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics_wrapper.indices import Q0, Q1, Q2, Q3, STATE_DIM, VX
from rl.configs.fish_env import build_fish_env_config
from simulation.local_planning.fin_controller import build_attitude_reference, fin_controller, run_layer2_controller


def _make_state(*, yaw: float = 0.0, pitch: float = 0.0, surge: float = 0.0) -> np.ndarray:
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)

    q0 = cy * cp
    q1 = -sy * sp
    q2 = cy * sp
    q3 = sy * cp

    state = np.zeros(STATE_DIM, dtype=np.float64)
    state[VX] = surge
    state[Q0] = q0
    state[Q1] = q1
    state[Q2] = q2
    state[Q3] = q3
    return state


def _run(ref_cmd: np.ndarray, state: np.ndarray, hist: np.ndarray, params: dict, ctrl_state: dict | None = None):
    if ctrl_state is None:
        ctrl_state = {}
    return fin_controller(ref_cmd, state, hist, ctrl_state, 0.2, params)


def main() -> None:
    cfg = build_fish_env_config()
    params = cfg["controller_params"]
    trim = np.asarray(cfg["trim_refs"], dtype=np.float64)

    state = _make_state()
    zero_ref = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    a1, a2, a3, a4, alpha5, ctrl_state = _run(zero_ref, state, trim, params)
    action_ref = np.array([a1, a2, a3, a4, alpha5], dtype=np.float64)
    assert np.isfinite(action_ref).all()
    assert np.isfinite(ctrl_state["tau_d"]).all()
    assert ctrl_state["tau_d"].shape == (6,)
    assert ctrl_state["force_cmd"].shape == (3,)
    assert ctrl_state["moment_cmd"].shape == (3,)
    assert np.isfinite(ctrl_state["tau_alloc"]).all()
    assert ctrl_state["tau_alloc"].shape == (6,)
    assert np.allclose(ctrl_state["tau_alloc"], ctrl_state["w_r_des"] + ctrl_state["w_l_des"] + ctrl_state["w_t_des"])
    assert np.allclose(ctrl_state["tau_d_alloc"], ctrl_state["tau_d"])
    assert np.all(action_ref >= params["actuator_min"] - 1e-9)
    assert np.all(action_ref <= params["actuator_max"] + 1e-9)

    g_dt = float(ctrl_state["cpg_g_dt"])
    reconstructed = action_ref + (trim - action_ref) * g_dt
    unsaturated = (action_ref > params["actuator_min"] + 1e-6) & (action_ref < params["actuator_max"] - 1e-6)
    assert np.any(unsaturated)
    assert np.allclose(reconstructed[unsaturated], ctrl_state["A_des"][unsaturated], atol=1e-6)

    yaw_ref = np.array([0.0, np.deg2rad(15.0), 0.0], dtype=np.float64)
    _, _, _, _, _, yaw_state = _run(yaw_ref, state, trim, params)
    assert yaw_state["tau_d"][5] > 0.0
    assert abs(yaw_state["w_r_des"][5]) > 0.0
    assert abs(yaw_state["w_l_des"][5]) > 0.0
    assert np.isfinite(yaw_state["J_r"]).all()
    assert np.isfinite(yaw_state["J_l"]).all()
    assert yaw_state["inverse_phase_average_samples"] == params["inverse_phase_average_samples"]
    assert len(yaw_state["inverse_phase_sample_times"]) == params["inverse_phase_average_samples"]

    pitch_ref = np.array([0.0, 0.0, np.deg2rad(10.0)], dtype=np.float64)
    _, _, _, _, _, pitch_state = _run(pitch_ref, state, trim, params)
    assert pitch_state["tau_d"][4] > 0.0
    assert abs(pitch_state["w_t_des"][4]) >= abs(pitch_state["w_r_des"][4])
    assert abs(pitch_state["w_t_des"][4]) >= abs(pitch_state["w_l_des"][4])

    low_u = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    high_u = np.array([0.6, 0.0, 0.0], dtype=np.float64)
    _, _, _, _, _, low_state = _run(low_u, state, trim, params)
    _, _, _, _, _, high_state = _run(high_u, state, trim, params)
    assert high_state["tau_d"][0] > low_state["tau_d"][0]
    assert high_state["w_r_des"][0] + high_state["w_l_des"][0] >= low_state["w_r_des"][0] + low_state["w_l_des"][0]

    adaptive_state: dict = {}
    adaptive_probe_state = _make_state(surge=0.6)
    _, _, _, _, _, adaptive_state = _run(pitch_ref, adaptive_probe_state, trim, params, adaptive_state)
    residual_1 = float(np.linalg.norm(adaptive_state["tail_residual"]))
    adaptive_state["phase_time"] = 0.0
    _, _, _, _, _, adaptive_state = _run(pitch_ref, adaptive_probe_state, trim, params, adaptive_state)
    residual_2 = float(np.linalg.norm(adaptive_state["tail_residual"]))
    assert residual_2 <= residual_1 + 1e-6

    layer2_state: dict = {}
    layer2_action_ref, layer2_state = run_layer2_controller(
        yaw_ref,
        adaptive_probe_state,
        trim,
        layer2_state,
        0.1,
        params,
    )
    assert np.isfinite(layer2_action_ref).all()
    assert layer2_action_ref.shape == (5,)
    assert np.all(layer2_action_ref >= params["actuator_min"] - 1e-9)
    assert np.all(layer2_action_ref <= params["actuator_max"] + 1e-9)
    assert np.allclose(layer2_state["tau_target"], layer2_state["tau_alloc"])
    assert np.isclose(
        params["inverse_phase_average_window"],
        1.0 / params["fin_f"],
    )
    assert np.allclose(
        layer2_state["layers"]["layer2"]["A_ref"],
        layer2_action_ref,
    )

    cmd_vel_body = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ref_cmd = build_attitude_reference(cmd_vel_body, state, params)
    assert np.allclose(ref_cmd, np.array([0.0, 0.0, 0.0], dtype=np.float64))

    print("fin_controller smoke tests passed")


if __name__ == "__main__":
    main()
