# Fin Wrench Surrogate

This package trains a neural network surrogate for the mapping

`[u, v, w, p, q, r, alpha1..alpha5, alpha1_dot..alpha5_dot] -> [Fx, Fy, Fz, Mx, My, Mz]`

using labels from `dynamics_wrapper.probe_fin_wrenches_with_joint_rates`.

## Files

- `generate_dataset.py`: samples body states and fin joint states, then saves a supervised dataset.
- `train_surrogate.py`: trains an MLP regressor and saves checkpoints and metrics.
- `dataset.py`: dataset generation, loading, normalization, and splitting.
- `kinematics.py`: realistic joint-angle and joint-rate sampling.
- `model.py`: PyTorch surrogate network.

## Usage

Generate a dataset:

```bash
python neural_network/generate_dataset.py --num-samples 50000
```

Train the surrogate:

```bash
python neural_network/train_surrogate.py --regenerate-dataset --num-samples 50000 --epochs 120
```

Outputs are written under `artifacts/neural_network/`.
