from __future__ import annotations

import numpy as np

from .indices import BODY_PARAM_DIM, FIN_PARAM_DIM


# These defaults are intentionally centralized so you can replace them with
# identified fish/body parameters without touching the environment code.
DEFAULT_BODY_PARAMS = np.array(
    [
        6.414,
        62.8572,
        62.8572,
        1000.0,
        0.0,
        0.0,
        0.0,
        0.01832,
        0.0,
        0.00228,
        0.0,
        0.0677,
        0.0,
        0.00228,
        0.0,
        0.0736,
        0.02072,
        0.384,
        -0.4,
        -1.3696,
        -0.9643,
        0.8612,
        -0.3902,
        6.1062,
        -3.067,
        -0.0841,
        -1.2335,
        -0.5579,
        1.2328,
        4.2324,
        6.9113,
        0.0036,
        0.1079,
        0.0752,
        -0.3920,
        0.6225,
    ],
    dtype=np.float64,
)

# Legacy layout uses length 38 although only the first 35 entries are consumed.
DEFAULT_FIN_PARAMS = np.array(
    [
        1000.0,
        0.45,
        0.23465,
        0.29992,
        0.001,
        0.05305,
        0.0111,
        0.289,
        -14.843,
        165.97,
        -1061.9,
        3362.9,
        -4217.9,
        -0.227,
        -0.0838,
        24.996,
        -446.53,
        4147.7,
        -21132.0,
        55611.0,
        -58992.0,
        -0.1124,
        2.6572,
        -62.2,
        7081.4,
        0.054429,
        0.1085,
        0.0,
        0.054429,
        -0.1085,
        0.0,
        -0.20855,
        0.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def validate_params(body_params: np.ndarray, fin_params: np.ndarray) -> None:
    if body_params.shape != (BODY_PARAM_DIM,):
        raise ValueError(f"body_params must have shape ({BODY_PARAM_DIM},)")
    if fin_params.shape != (FIN_PARAM_DIM,):
        raise ValueError(f"fin_params must have shape ({FIN_PARAM_DIM},)")
    if body_params[0] <= 0.0:
        raise ValueError("body mass must be positive")
    if body_params[7] <= 0.0 or body_params[11] <= 0.0 or body_params[15] <= 0.0:
        raise ValueError("principal inertias must be positive")
    if fin_params[2] <= 0.0 or fin_params[3] <= 0.0:
        raise ValueError("fin span parameters must be positive")


def load_body_params(overrides: np.ndarray | None = None) -> np.ndarray:
    params = DEFAULT_BODY_PARAMS.copy()
    if overrides is not None:
        overrides = np.asarray(overrides, dtype=np.float64)
        if overrides.shape != (BODY_PARAM_DIM,):
            raise ValueError(f"body overrides must have shape ({BODY_PARAM_DIM},)")
        params[:] = overrides
    return params


def load_fin_params(overrides: np.ndarray | None = None) -> np.ndarray:
    params = DEFAULT_FIN_PARAMS.copy()
    if overrides is not None:
        overrides = np.asarray(overrides, dtype=np.float64)
        if overrides.shape != (FIN_PARAM_DIM,):
            raise ValueError(f"fin overrides must have shape ({FIN_PARAM_DIM},)")
        params[:] = overrides
    return params
