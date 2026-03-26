from __future__ import annotations

import numpy as np


def build_fin_controller_config() -> dict:
    """Build default fin-controller gains and actuator limits."""
    return {
        "Kp_z": -1.2,
        "Ki_z": 0.0,
        "Kd_z": -6.0,
        "Kp_psi": 1.5,
        "Ki_psi": 0.0,
        "Kd_psi": 6.0,
        "A_base": np.deg2rad(45.0),
        "A_rot_base": np.deg2rad(15.0),
        "A_min": np.deg2rad(8.0),
        "alpha5_min": np.deg2rad(-60.0),
        "alpha5_max": np.deg2rad(60.0),
        "delta_rot_max": np.deg2rad(15.0),
        "e_theta_int_limit": 0.75,
        "e_psi_int_limit": 0.75,
        "speed_sched_min": 0.04,
        "speed_sched_max": 0.30,
    }


FIN_CONTROLLER_CONFIG = build_fin_controller_config()
