from __future__ import annotations

import torch
from torch import nn

# Define a simple multilayer perceptron for surrogate regression from fin/body
# states to the corresponding body-frame wrench.
class FinForceSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 6,
        hidden_sizes: tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the normalized feature vector through the network.
        return self.network(x)

    def _reset_parameters(self) -> None:
        # Use a stable default initialization for all linear layers.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
