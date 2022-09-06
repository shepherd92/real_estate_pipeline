#!/usr/bin/env python
"""Neural network model for training."""

from torch import Tensor, nn


class NeuralNetwork(nn.Module):
    """Neural network model for training."""

    def __init__(self, num_of_features: int) -> None:
        """Construct a neural network model."""
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(num_of_features, 2 * num_of_features),
            nn.ReLU(),
            nn.Linear(2 * num_of_features, num_of_features),
            nn.ReLU(),
            nn.Linear(num_of_features, num_of_features // 2 + 1),
            nn.ReLU(),
            nn.Linear(num_of_features // 2 + 1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass."""
        output = self._model(input_)
        return output
