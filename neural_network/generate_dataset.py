from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.config import build_surrogate_sampling_config
from neural_network.dataset import generate_surrogate_dataset

# Parse command-line options for standalone dataset generation.
def parse_args() -> argparse.Namespace:
    cfg = build_surrogate_sampling_config()
    default_path = ROOT / "artifacts" / "neural_network" / "datasets" / "surrogate_joint_wrench_dataset.npz"
    parser = argparse.ArgumentParser(description="Generate surrogate data for the fin wrench neural network.")
    parser.add_argument("--num-samples", type=int, default=int(cfg["num_samples"]))
    parser.add_argument("--seed", type=int, default=int(cfg["seed"]))
    parser.add_argument("--output", type=Path, default=default_path)
    return parser.parse_args()

# Entry point for generating and saving a dataset file from the terminal.
def main() -> None:
    args = parse_args()
    dataset = generate_surrogate_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Saved dataset to {args.output}")
    print(f"X shape: {dataset['X'].shape}")
    print(f"Y shape: {dataset['Y'].shape}")


if __name__ == "__main__":
    main()
