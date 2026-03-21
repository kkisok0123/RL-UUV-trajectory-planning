from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from .indices import BODY_PARAM_DIM, FIN_PARAM_DIM, HIST_DIM, REF_DIM, STATE_DIM

_BACKEND = None
_BACKEND_NAME = None


def _load_backend():
    global _BACKEND, _BACKEND_NAME
    if _BACKEND is not None:
        return _BACKEND

    try:
        _BACKEND = importlib.import_module(".fish_dynamics", package=__package__)
        _BACKEND_NAME = "pybind11"
        return _BACKEND
    except ImportError:
        pass

    try:
        _BACKEND = importlib.import_module("fish_dynamics")
        _BACKEND_NAME = "pybind11"
        return _BACKEND
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    wrapper_dir = repo_root / "dynamics_wrapper"
    if str(wrapper_dir) not in sys.path:
        sys.path.insert(0, str(wrapper_dir))

    _BACKEND = importlib.import_module("dynamics_step")
    _BACKEND_NAME = "legacy_cpython_extension"
    return _BACKEND


def backend_name() -> str:
    _load_backend()
    return _BACKEND_NAME


def _as_vector(values, size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return np.ascontiguousarray(arr)


def step(
    state,
    body_params,
    fin_params,
    dt: float,
    t_k: float,
    action_ref,
    hist,
    c_A: float,
    fin_f: float,
):
    state = _as_vector(state, STATE_DIM, "state")
    body_params = _as_vector(body_params, BODY_PARAM_DIM, "body_params")
    fin_params = _as_vector(fin_params, FIN_PARAM_DIM, "fin_params")
    action_ref = _as_vector(action_ref, REF_DIM, "action_ref")
    hist = _as_vector(hist, HIST_DIM, "hist")

    backend = _load_backend()
    if hasattr(backend, "step"):
        next_state, next_hist = backend.step(
            state, body_params, fin_params, dt, t_k, action_ref, hist, c_A, fin_f
        )
        return _as_vector(next_state, STATE_DIM, "next_state"), _as_vector(
            next_hist, HIST_DIM, "next_hist"
        )

    result = backend.update(
        state.tolist(),
        body_params.tolist(),
        fin_params.tolist(),
        float(dt),
        float(t_k),
        float(action_ref[0]),
        float(action_ref[1]),
        float(action_ref[2]),
        float(action_ref[3]),
        float(action_ref[4]),
        float(hist[0]),
        float(hist[1]),
        float(hist[2]),
        float(hist[3]),
        float(hist[4]),
        float(c_A),
        float(fin_f),
    )
    next_state, *hist_values = result
    return _as_vector(next_state, STATE_DIM, "next_state"), _as_vector(
        hist_values, HIST_DIM, "next_hist"
    )


def step_with_body_wrench(
    state,
    body_params,
    dt: float,
    body_wrench,
    hist,
):
    state = _as_vector(state, STATE_DIM, "state")
    body_params = _as_vector(body_params, BODY_PARAM_DIM, "body_params")
    body_wrench = _as_vector(body_wrench, 6, "body_wrench")
    hist = _as_vector(hist, HIST_DIM, "hist")

    backend = _load_backend()
    if not hasattr(backend, "step_with_body_wrench"):
        raise NotImplementedError("The active dynamics backend does not expose step_with_body_wrench")

    next_state, next_hist = backend.step_with_body_wrench(
        state,
        body_params,
        float(dt),
        body_wrench,
        hist,
    )
    return _as_vector(next_state, STATE_DIM, "next_state"), _as_vector(
        next_hist, HIST_DIM, "next_hist"
    )


def step_with_joint_rates(
    state,
    body_params,
    fin_params,
    dt: float,
    joint_angles,
    joint_rates,
):
    state = _as_vector(state, STATE_DIM, "state")
    body_params = _as_vector(body_params, BODY_PARAM_DIM, "body_params")
    fin_params = _as_vector(fin_params, FIN_PARAM_DIM, "fin_params")
    joint_angles = _as_vector(joint_angles, HIST_DIM, "joint_angles")
    joint_rates = _as_vector(joint_rates, HIST_DIM, "joint_rates")

    backend = _load_backend()
    if not hasattr(backend, "step_with_joint_rates"):
        raise NotImplementedError("The active dynamics backend does not expose step_with_joint_rates")

    next_state, next_joint_angles = backend.step_with_joint_rates(
        state,
        body_params,
        fin_params,
        float(dt),
        joint_angles,
        joint_rates,
    )
    return _as_vector(next_state, STATE_DIM, "next_state"), _as_vector(
        next_joint_angles, HIST_DIM, "next_joint_angles"
    )


def probe_fin_wrenches(
    state,
    fin_params,
    t_k: float,
    action_ref,
    hist,
    c_A: float,
    fin_f: float,
):
    state = _as_vector(state, STATE_DIM, "state")
    fin_params = _as_vector(fin_params, FIN_PARAM_DIM, "fin_params")
    action_ref = _as_vector(action_ref, REF_DIM, "action_ref")
    hist = _as_vector(hist, HIST_DIM, "hist")

    backend = _load_backend()
    if not hasattr(backend, "probe_fin_wrenches"):
        raise NotImplementedError("The active dynamics backend does not expose probe_fin_wrenches")

    total, right, left, tail = backend.probe_fin_wrenches(
        state,
        fin_params,
        float(t_k),
        action_ref,
        hist,
        float(c_A),
        float(fin_f),
    )
    return {
        "total": _as_vector(total, 6, "total"),
        "right": _as_vector(right, 6, "right"),
        "left": _as_vector(left, 6, "left"),
        "tail": _as_vector(tail, 6, "tail"),
    }


def probe_fin_wrenches_with_joint_rates(
    state,
    fin_params,
    joint_angles,
    joint_rates,
):
    state = _as_vector(state, STATE_DIM, "state")
    fin_params = _as_vector(fin_params, FIN_PARAM_DIM, "fin_params")
    joint_angles = _as_vector(joint_angles, HIST_DIM, "joint_angles")
    joint_rates = _as_vector(joint_rates, HIST_DIM, "joint_rates")

    backend = _load_backend()
    if not hasattr(backend, "probe_fin_wrenches_with_joint_rates"):
        raise NotImplementedError("The active dynamics backend does not expose probe_fin_wrenches_with_joint_rates")

    total, right, left, tail = backend.probe_fin_wrenches_with_joint_rates(
        state,
        fin_params,
        joint_angles,
        joint_rates,
    )
    return {
        "total": _as_vector(total, 6, "total"),
        "right": _as_vector(right, 6, "right"),
        "left": _as_vector(left, 6, "left"),
        "tail": _as_vector(tail, 6, "tail"),
    }
