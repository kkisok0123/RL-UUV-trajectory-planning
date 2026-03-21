from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dynamics_wrapper import probe_fin_wrenches_with_joint_rates

from .config import build_surrogate_sampling_config
from .kinematics import sample_body_state, sample_joint_state


FEATURE_NAMES = (
    "u",
    "v",
    "w",
    "p",
    "q",
    "r",
    "alpha1",
    "alpha2",
    "alpha3",
    "alpha4",
    "alpha5",
    "alpha1_dot",
    "alpha2_dot",
    "alpha3_dot",
    "alpha4_dot",
    "alpha5_dot",
)

TARGET_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

# Generate supervised training data by sampling body/joint states and querying
# the physics model for the corresponding total fin wrench.
def generate_surrogate_dataset(
    num_samples: int | None = None,
    seed: int | None = None,
    output_path: str | Path | None = None,
    sampling_cfg: dict | None = None,
) -> dict:
    sampling_cfg = dict(build_surrogate_sampling_config() if sampling_cfg is None else sampling_cfg)
    if num_samples is not None:
        sampling_cfg["num_samples"] = int(num_samples)
    if seed is not None:
        sampling_cfg["seed"] = int(seed)

    rng = np.random.default_rng(int(sampling_cfg["seed"]))
    sample_count = int(sampling_cfg["num_samples"])
    fin_params = np.asarray(sampling_cfg["fin_params"], dtype=np.float64).reshape(-1)

    X = np.zeros((sample_count, len(FEATURE_NAMES)), dtype=np.float64)
    Y = np.zeros((sample_count, len(TARGET_NAMES)), dtype=np.float64)
    amplitudes = np.zeros((sample_count, 4), dtype=np.float64)
    tail_angles = np.zeros(sample_count, dtype=np.float64)
    phases = np.zeros(sample_count, dtype=np.float64)

    for idx in range(sample_count):
        state = sample_body_state(
            rng,
            np.asarray(sampling_cfg["linear_velocity_limits"], dtype=np.float64),
            np.asarray(sampling_cfg["angular_velocity_limits"], dtype=np.float64),
        )
        joint_angles, joint_rates, meta = sample_joint_state(
            rng,
            np.asarray(sampling_cfg["amplitude_min"], dtype=np.float64),
            np.asarray(sampling_cfg["amplitude_max"], dtype=np.float64),
            float(sampling_cfg["tail_angle_min"]),
            float(sampling_cfg["tail_angle_max"]),
            float(sampling_cfg["phase_min"]),
            float(sampling_cfg["phase_max"]),
            float(sampling_cfg["fin_f"]),
        )
        wrench = probe_fin_wrenches_with_joint_rates(
            state=state,
            fin_params=fin_params,
            joint_angles=joint_angles,
            joint_rates=joint_rates,
        )["total"]

        X[idx] = np.concatenate((state[:6], joint_angles, joint_rates))
        Y[idx] = np.asarray(wrench, dtype=np.float64).reshape(6)
        amplitudes[idx] = np.asarray(meta["amplitudes"], dtype=np.float64).reshape(4)
        tail_angles[idx] = float(meta["tail_angle"])
        phases[idx] = float(meta["phase"])

    dataset = {
        "X": X,
        "Y": Y,
        "feature_names": np.asarray(FEATURE_NAMES),
        "target_names": np.asarray(TARGET_NAMES),
        "fin_params": fin_params,
        "sampled_amplitudes": amplitudes,
        "sampled_tail_angles": tail_angles,
        "sampled_phases": phases,
        "sampling_config_json": json.dumps(_json_ready_config(sampling_cfg)),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **dataset)

    return dataset

# Load a saved dataset bundle and compute dataset-level normalization statistics
# that are later used by the PyTorch training pipeline.
def load_dataset_bundle(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        Y = np.asarray(data["Y"], dtype=np.float64)
        feature_names = tuple(str(name) for name in data["feature_names"].tolist())
        target_names = tuple(str(name) for name in data["target_names"].tolist())
        fin_params = np.asarray(data["fin_params"], dtype=np.float64)
        sampling_json = str(data["sampling_config_json"].item())

    x_mean = X.mean(axis=0)
    x_std = np.maximum(X.std(axis=0), 1e-6)
    y_mean = Y.mean(axis=0)
    y_std = np.maximum(Y.std(axis=0), 1e-6)

    return {
        "X": X,
        "Y": Y,
        "feature_names": feature_names,
        "target_names": target_names,
        "fin_params": fin_params,
        "sampling_config": json.loads(sampling_json),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

# Randomly shuffle the full dataset and split it into train/validation/test
# subsets using reproducible index permutations.
def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be in [0, 1) with val_ratio + test_ratio < 1")

    count = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)

    test_count = int(round(count * test_ratio))
    val_count = int(round(count * val_ratio))
    train_count = count - val_count - test_count

    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]

    return {
        "train": (X[train_idx], Y[train_idx]),
        "val": (X[val_idx], Y[val_idx]),
        "test": (X[test_idx], Y[test_idx]),
    }


# Store normalized regression samples in a PyTorch Dataset so training code can
# consume them directly through DataLoader.
class NormalizedArrayDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        # Normalize once up front so each training batch only does tensor reads.
        self.X = ((np.asarray(X, dtype=np.float32) - x_mean.astype(np.float32)) / x_std.astype(np.float32)).astype(
            np.float32
        )
        self.Y = ((np.asarray(Y, dtype=np.float32) - y_mean.astype(np.float32)) / y_std.astype(np.float32)).astype(
            np.float32
        )

    def __len__(self) -> int:
        # Report dataset size to DataLoader.
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return one normalized input-target pair as tensors.
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])


# Convert numpy values into plain Python objects so the config can be serialized
# into JSON and saved inside the dataset bundle.
def _json_ready_config(config: dict) -> dict:
    ready = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            ready[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready
