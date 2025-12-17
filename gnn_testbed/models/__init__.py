from __future__ import annotations

from gnn_testbed.config import ModelConfig

from .lss_gnn import ChiralLSSClassifier
from .simple_mlp import SimplePointMLP


def build_model(cfg: ModelConfig):
    if cfg.type == "lss_gnn":
        return ChiralLSSClassifier(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            spin_symmetry=cfg.spin_symmetry,
            graph_mode=cfg.graph_mode,
            k=cfg.k,
        )
    if cfg.type == "simple_point_mlp":
        return SimplePointMLP(in_dim=cfg.in_dim, hidden=cfg.hidden)
    raise ValueError(f"Unknown model type: {cfg.type}")


__all__ = ["build_model", "ChiralLSSClassifier", "SimplePointMLP"]
