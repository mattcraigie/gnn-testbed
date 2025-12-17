from __future__ import annotations

import torch
import torch.nn as nn

from .lss_gnn import GraphBuilder


class StandardEGNNLayer(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        message_input_dim = (2 * scalar_dim) + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(message_input_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(scalar_dim + scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )

        self.coord_mlp = nn.Sequential(nn.Linear(scalar_dim, 1), nn.Tanh())
        self.vector_gate = nn.Sequential(nn.Linear(scalar_dim, 1), nn.Sigmoid())

    def forward(
        self,
        h_scalar: torch.Tensor,
        h_vector: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        delta_pos = pos[src] - pos[dst]
        dist_sq = torch.sum(delta_pos**2, dim=1, keepdim=True)

        edge_input = torch.cat([h_scalar[src], h_scalar[dst], dist_sq], dim=-1)
        m_ij = self.edge_mlp(edge_input)

        trans_weights = self.coord_mlp(m_ij)
        weighted_dirs = delta_pos * trans_weights

        agg_vec_update = torch.zeros((h_scalar.shape[0], 2), device=h_scalar.device)
        agg_vec_update.index_add_(0, dst, weighted_dirs)

        agg_scalar = torch.zeros_like(h_scalar)
        agg_scalar.index_add_(0, dst, m_ij)

        h_scalar_new = self.node_mlp(torch.cat([h_scalar, agg_scalar], dim=-1))
        gate = self.vector_gate(h_scalar_new)

        if h_vector.shape[-1] != 2:
            padded = torch.zeros((h_vector.shape[0], 2), device=h_vector.device, dtype=h_vector.dtype)
            padded[:, : min(2, h_vector.shape[-1])] = h_vector[:, : min(2, h_vector.shape[-1])]
            h_vector = padded

        h_vector_new = (h_vector * gate) + agg_vec_update
        return h_scalar + h_scalar_new, h_vector_new


class StandardEGNN(nn.Module):
    def __init__(self, in_scalar: int, in_vector: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.s_dim = hidden_dim
        self.v_dim = in_vector

        self.enc_s = nn.Sequential(nn.Linear(in_scalar, self.s_dim), nn.ReLU())
        self.enc_v = nn.Linear(in_vector, 2, bias=False) if in_vector != 2 else nn.Identity()

        self.layers = nn.ModuleList(
            [StandardEGNNLayer(self.s_dim, self.v_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(self.s_dim, 1)

    def forward(
        self,
        x_s: torch.Tensor,
        x_v: torch.Tensor,
        pos: torch.Tensor,
        orientation: torch.Tensor | None = None,
        *,
        graph_mode: str = "knn",
        k: int = 5,
    ) -> torch.Tensor:
        batch_size, num_points, _ = x_s.shape

        x_s_flat = x_s.view(-1, x_s.shape[-1])
        x_v_flat = x_v.view(-1, x_v.shape[-1])
        pos_flat = pos.view(-1, 2)

        edge_index = GraphBuilder.build_edges(pos, mode=graph_mode, k=k)

        h_s = self.enc_s(x_s_flat)
        h_v = self.enc_v(x_v_flat)

        for layer in self.layers:
            h_s, h_v = layer(h_s, h_v, pos_flat, edge_index)

        h_s = h_s.view(batch_size, num_points, -1)
        return self.head(h_s.mean(dim=1))


class ChiralEGNNClassifier(nn.Module):
    """Wrapper that adapts chiral point clouds into Standard EGNN inputs."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, graph_mode: str = "knn", k: int = 5):
        super().__init__()
        self.graph_mode = graph_mode
        self.k = k
        self.model = StandardEGNN(in_scalar=3, in_vector=2, hidden_dim=hidden_dim, num_layers=num_layers)

    @staticmethod
    def _compute_tangent(points: torch.Tensor) -> torch.Tensor:
        diff = points[:, 1:, :] - points[:, :-1, :]
        prev_step = torch.zeros_like(points)
        next_step = torch.zeros_like(points)
        prev_step[:, 1:, :] = diff
        next_step[:, :-1, :] = diff
        return prev_step + next_step

    def forward(self, points: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tangent = self._compute_tangent(points)
        step_norm = torch.norm(tangent, dim=-1, keepdim=True) + 1e-6
        orientation = torch.atan2(tangent[..., 1], tangent[..., 0]).unsqueeze(-1)

        scalar_feats = torch.cat([step_norm, torch.cos(orientation), torch.sin(orientation)], dim=-1)
        vector_feats = tangent

        logits = self.model(
            scalar_feats,
            vector_feats,
            points,
            orientation,
            graph_mode=self.graph_mode,
            k=self.k,
        )
        return logits.squeeze(-1)


__all__ = ["StandardEGNN", "StandardEGNNLayer", "ChiralEGNNClassifier"]
