from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_network.config import build_surrogate_training_config
from neural_network.dataset import (
    FEATURE_NAMES,
    TARGET_NAMES,
    NormalizedArrayDataset,
    generate_surrogate_dataset,
    load_dataset_bundle,
    split_dataset,
)
from neural_network.model import FinForceSurrogate

# Parse command-line options for the standalone training script.
def parse_args() -> argparse.Namespace:
    cfg = build_surrogate_training_config()
    parser = argparse.ArgumentParser(description="Train the fin wrench surrogate neural network.")
    parser.add_argument("--dataset", type=Path, default=cfg["dataset_path"])
    parser.add_argument("--save-dir", type=Path, default=cfg["save_dir"])
    parser.add_argument("--num-samples", type=int, default=int(cfg["num_samples"]))
    parser.add_argument("--epochs", type=int, default=int(cfg["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(cfg["batch_size"]))
    parser.add_argument("--learning-rate", type=float, default=float(cfg["learning_rate"]))
    parser.add_argument("--weight-decay", type=float, default=float(cfg["weight_decay"]))
    parser.add_argument("--seed", type=int, default=int(cfg["seed"]))
    parser.add_argument("--val-ratio", type=float, default=float(cfg["val_ratio"]))
    parser.add_argument("--test-ratio", type=float, default=float(cfg["test_ratio"]))
    parser.add_argument("--regenerate-dataset", action="store_true")
    return parser.parse_args()

# Seed all random-number generators used by data splitting and training.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Resolve the target device. "auto" prefers CUDA if it is available.
def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    # Run one full training pass over the loader and return average loss in
    # normalized output space.
    model.train()
    running = 0.0
    sample_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        batch_size = int(batch_x.shape[0])
        running += float(loss.item()) * batch_size
        sample_count += batch_size
    return running / max(sample_count, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict:
    # Evaluate without gradient updates, then convert predictions back to
    # physical units to report meaningful error metrics.
    model.eval()
    total_loss = 0.0
    sample_count = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        batch_size = int(batch_x.shape[0])
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        preds.append(pred.cpu().numpy())
        targets.append(batch_y.cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0) if preds else np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)
    target_norm = np.concatenate(targets, axis=0) if targets else np.zeros((0, len(TARGET_NAMES)), dtype=np.float32)

    pred_phys = pred_norm * y_std + y_mean
    target_phys = target_norm * y_std + y_mean
    abs_err = np.abs(pred_phys - target_phys)
    mse = np.mean((pred_phys - target_phys) ** 2, axis=0) if sample_count else np.zeros(len(TARGET_NAMES))
    mae = np.mean(abs_err, axis=0) if sample_count else np.zeros(len(TARGET_NAMES))

    return {
        "loss": total_loss / max(sample_count, 1),
        "mae_mean": float(mae.mean()) if sample_count else 0.0,
        "mae_per_target": {name: float(value) for name, value in zip(TARGET_NAMES, mae)},
        "rmse_per_target": {name: float(np.sqrt(value)) for name, value in zip(TARGET_NAMES, mse)},
    }

# Main training entry point: create or load the dataset, split and normalize
# it, train the model, then save the best checkpoint and summary metrics.
def main() -> None:
    default_cfg = build_surrogate_training_config()
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    if args.regenerate_dataset or not dataset_path.exists():
        print(f"Generating dataset at {dataset_path} with {args.num_samples} samples...")
        generate_surrogate_dataset(
            num_samples=args.num_samples,
            seed=args.seed,
            output_path=dataset_path,
        )

    bundle = load_dataset_bundle(dataset_path)
    splits = split_dataset(
        bundle["X"],
        bundle["Y"],
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=int(default_cfg["train_split_seed"]) + int(args.seed),
    )

    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    test_x, test_y = splits["test"]

    x_mean = train_x.mean(axis=0)
    x_std = np.maximum(train_x.std(axis=0), 1e-6)
    y_mean = train_y.mean(axis=0)
    y_std = np.maximum(train_y.std(axis=0), 1e-6)

    train_ds = NormalizedArrayDataset(train_x, train_y, x_mean, x_std, y_mean, y_std)
    val_ds = NormalizedArrayDataset(val_x, val_y, x_mean, x_std, y_mean, y_std)
    test_ds = NormalizedArrayDataset(test_x, test_y, x_mean, x_std, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(default_cfg["device"])
    model = FinForceSurrogate(
        input_dim=len(FEATURE_NAMES),
        output_dim=len(TARGET_NAMES),
        hidden_sizes=tuple(int(v) for v in default_cfg["hidden_sizes"]),
        dropout=float(default_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_path = save_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, y_mean, y_std)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_mae_mean": float(val_metrics["mae_mean"]),
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_names": FEATURE_NAMES,
                    "target_names": TARGET_NAMES,
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "dataset_path": str(dataset_path),
                    "history": history,
                },
                best_path,
            )

        if epoch == 1 or epoch % int(default_cfg["log_interval"]) == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_mae={val_metrics['mae_mean']:.6f}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_epoch(model, test_loader, criterion, device, y_mean, y_std)

    summary = {
        "dataset_path": str(dataset_path),
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "train_size": int(train_x.shape[0]),
        "val_size": int(val_x.shape[0]),
        "test_size": int(test_x.shape[0]),
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
    }
    (save_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics saved to: {save_dir / 'metrics.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
