"""Configurable MNIST networks with the paper's Gaussian Glorot initialization."""

from __future__ import annotations

from torch import Tensor, nn


ARCHITECTURES: dict[str, tuple[int, int]] = {
    "A": (162, 421),
    "B": (200, 204),
    "C": (231, 77),
    "D": (242, 40),
}

MAIN_ARCHITECTURES = ("A", "B", "C", "D")


class MLP(nn.Module):
    """784 -> hidden1 -> hidden2 -> 10, returning unnormalized logits.

    Gaussian Glorot is Xavier normal with gain=1, including for the ReLU
    hidden layers. Biases start at zero and are kept dense during pruning.
    """

    def __init__(self, hidden1: int = 300, hidden2: int = 100) -> None:
        super().__init__()
        if any(not isinstance(width, int) or isinstance(width, bool) or width <= 0
               for width in (hidden1, hidden2)):
            raise ValueError("Hidden-layer widths must be positive integers.")
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw fresh Glorot-normal weights and zero every bias."""
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.flatten(start_dim=1)
        hidden = self.fc1(inputs).relu()
        hidden = self.fc2(hidden).relu()
        return self.fc3(hidden)


def parameter_count(model: nn.Module) -> int:
    """Count all scalar parameters, including the unpruned biases."""
    return sum(parameter.numel() for parameter in model.parameters())
