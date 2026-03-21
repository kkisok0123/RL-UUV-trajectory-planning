from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.configs.fish_env import build_fish_env_config


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "neural_network"

# Build the default sampling bounds from the current fish-environment settings
# so generated surrogate data stays inside a realistic operating envelope.
def build_surrogate_sampling_config() -> dict:
    env_cfg = build_fish_env_config()
    linear_limits = np.maximum(
        np.asarray(env_cfg["action_velocity_limits"], dtype=np.float64) * 1.75,
        np.array([1.0, 0.8, 0.8], dtype=np.float64),
    )
    angular_limits = np.deg2rad(np.array([120.0, 120.0, 120.0], dtype=np.float64))
    return {
        "seed": 7,
        "num_samples": 50000,
        "linear_velocity_limits": linear_limits,
        "angular_velocity_limits": angular_limits,
        "amplitude_min": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "amplitude_max": np.asarray(env_cfg["ref_max"][:4], dtype=np.float64).copy(),
        "tail_angle_min": float(env_cfg["ref_min"][4]),
        "tail_angle_max": float(env_cfg["ref_max"][4]),
        "phase_min": -np.pi,
        "phase_max": np.pi,
        "fin_f": float(env_cfg["fin_f"]),
        "fin_params": np.asarray(env_cfg["fin_params"], dtype=np.float64).copy(),
    }

# Build the default training hyperparameters and artifact locations used by the
# standalone surrogate training script.
def build_surrogate_training_config() -> dict:
    dataset_path = ARTIFACT_ROOT / "datasets" / "surrogate_joint_wrench_dataset.npz"
    return {
        "seed": 7,
        "device": "auto",
        "dataset_path": dataset_path,
        "save_dir": ARTIFACT_ROOT / "models" / "joint_wrench_surrogate",
        "num_samples": 50000,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "batch_size": 512,
        "epochs": 120,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "hidden_sizes": [256, 256, 128],
        "dropout": 0.05,
        "train_split_seed": 19,
        "log_interval": 10,
    }
