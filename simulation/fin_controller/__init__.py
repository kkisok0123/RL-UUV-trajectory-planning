"""Fin-controller logic and default controller parameters."""

from simulation.fin_controller.config import FIN_CONTROLLER_CONFIG, build_fin_controller_config
from simulation.fin_controller.controller import extract_attitude, fin_controller, velocity_to_attitude_refs

__all__ = [
    "FIN_CONTROLLER_CONFIG",
    "build_fin_controller_config",
    "extract_attitude",
    "fin_controller",
    "velocity_to_attitude_refs",
]
