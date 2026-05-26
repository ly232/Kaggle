import torch.nn as nn

from jaxtyping import Float
from torch import Tensor


class SurvivalClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: Float[Tensor, "batch_size input_dim"]
    ) -> Float[Tensor, "batch_size 1"]:
        return self.model(x)
