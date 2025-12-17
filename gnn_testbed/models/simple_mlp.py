from __future__ import annotations

import torch.nn as nn


class SimplePointMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, points):
        x = self.point(points)
        x = x.mean(dim=1)
        logits = self.head(x).squeeze(-1)
        return logits


def build_model(model_type: str, **kwargs) -> nn.Module:
    if model_type == "simple_point_mlp":
        return SimplePointMLP(**kwargs)
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["SimplePointMLP", "build_model"]
