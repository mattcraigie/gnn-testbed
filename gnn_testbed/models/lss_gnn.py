from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:  # Optional dependency for graph construction
    from scipy.spatial import Delaunay, cKDTree
except Exception:  # pragma: no cover - only used to provide a clear runtime error
    Delaunay = None
    cKDTree = None


class GraphBuilder:
    """Helper to construct edge indices from positions on the fly."""

    @staticmethod
    def build_edges(pos: torch.Tensor, mode: str = "knn", k: int = 5) -> torch.Tensor:
        """
        Build sparse edge indices for a batch of point clouds.

        Args:
            pos: Tensor of shape ``[B, N, 2]`` or ``[N, 2]`` with point coordinates.
            mode: Either ``"knn"`` or ``"delaunay"``.
            k: Number of neighbors for kNN mode (ignored for delaunay).

        Returns:
            ``edge_index`` with shape ``[2, E]`` following PyG-style (src, dst) ordering.
        """

        if Delaunay is None or cKDTree is None:
            raise ImportError("GraphBuilder requires scipy to be installed for edge construction.")

        device = pos.device
        pos_np = pos.detach().cpu().numpy()

        if pos.dim() == 2:
            pos_np = [pos_np]

        all_edges_src = []
        all_edges_dst = []
        offset = 0

        for p in pos_np:
            num_points = p.shape[0]
            if mode == "knn":
                tree = cKDTree(p)
                _, idx = tree.query(p, k=k + 1)
                src = idx[:, 1:].flatten()
                dst = np.repeat(np.arange(num_points), k)
            elif mode == "delaunay":
                tri = Delaunay(p)
                indices = tri.simplices
                edges = np.concatenate(
                    [indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]]], axis=0
                )
                edges_rev = edges[:, ::-1]
                full_edges = np.concatenate([edges, edges_rev], axis=0)
                full_edges = np.unique(full_edges, axis=0)
                src = full_edges[:, 1]
                dst = full_edges[:, 0]
            else:  # pragma: no cover - validated by caller
                raise ValueError(f"Unknown graph mode: {mode}")

            all_edges_src.append(torch.tensor(src, dtype=torch.long) + offset)
            all_edges_dst.append(torch.tensor(dst, dtype=torch.long) + offset)
            offset += num_points

        edge_index = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)], dim=0).to(device)
        return edge_index


class SparseLocalFrameLayer(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, spin_symmetry: int = 2):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.spin_symmetry = spin_symmetry

        geo_dim = 3
        input_dim = (2 * scalar_dim) + vector_dim + geo_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, scalar_dim + vector_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim + vector_dim, scalar_dim + vector_dim),
        )

    def forward(
        self,
        h_scalar: torch.Tensor,
        h_vector: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        orientation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        delta_pos = pos[src] - pos[dst]

        dist = torch.norm(delta_pos, dim=1, keepdim=True) + 1e-6
        phi_global = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])

        alpha_i = orientation[dst].squeeze(-1)
        delta_phi = phi_global - alpha_i

        geo_feat = torch.cat(
            [
                dist,
                torch.cos(self.spin_symmetry * delta_phi).unsqueeze(-1),
                torch.sin(self.spin_symmetry * delta_phi).unsqueeze(-1),
            ],
            dim=-1,
        )

        beta_j = orientation[src].squeeze(-1)
        rot_angle = (beta_j - alpha_i) * self.spin_symmetry

        c = torch.cos(rot_angle)
        s = torch.sin(rot_angle)

        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)
        R = torch.stack([row1, row2], dim=-2)

        v_j = h_vector[src]
        num_vecs = self.vector_dim // 2
        v_j_reshaped = v_j.view(-1, num_vecs, 2).unsqueeze(-1)
        v_j_rot = torch.matmul(R.unsqueeze(1), v_j_reshaped)
        v_j_rot = v_j_rot.view(-1, self.vector_dim)

        msg_input = torch.cat([h_scalar[src], h_scalar[dst], v_j_rot, geo_feat], dim=-1)
        raw_msg = self.message_mlp(msg_input)

        out_scalar = torch.zeros_like(h_scalar)
        out_vector = torch.zeros_like(h_vector)

        msg_scalar = raw_msg[:, : self.scalar_dim]
        msg_vector = raw_msg[:, self.scalar_dim :]

        out_scalar.index_add_(0, dst, msg_scalar)
        out_vector.index_add_(0, dst, msg_vector)

        return h_scalar + out_scalar, h_vector + out_vector


class LSSGNN(nn.Module):
    def __init__(self, in_scalar: int, in_vector: int, hidden_dim: int, num_layers: int = 2, spin_symmetry: int = 2):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even so it can be split between scalar/vector channels")

        self.s_dim = hidden_dim // 2
        self.v_dim = hidden_dim // 2

        self.enc_s = nn.Sequential(nn.Linear(in_scalar, self.s_dim), nn.ReLU())
        self.enc_v = nn.Linear(in_vector, self.v_dim, bias=False)

        self.layers = nn.ModuleList(
            [SparseLocalFrameLayer(self.s_dim, self.v_dim, spin_symmetry) for _ in range(num_layers)]
        )

        self.head = nn.Linear(self.s_dim, 1)

    def forward(
        self,
        x_s: torch.Tensor,
        x_v: torch.Tensor,
        pos: torch.Tensor,
        orientation: torch.Tensor,
        *,
        graph_mode: str = "knn",
        k: int = 5,
    ) -> torch.Tensor:
        batch_size, num_points, _ = x_s.shape

        x_s_flat = x_s.view(-1, x_s.shape[-1])
        x_v_flat = x_v.view(-1, x_v.shape[-1])
        pos_flat = pos.view(-1, 2)
        ori_flat = orientation.view(-1, 1)

        edge_index = GraphBuilder.build_edges(pos, mode=graph_mode, k=k)

        h_s = self.enc_s(x_s_flat)
        h_v = self.enc_v(x_v_flat)

        for layer in self.layers:
            h_s, h_v = layer(h_s, h_v, edge_index, pos_flat, ori_flat)

        h_s = h_s.view(batch_size, num_points, -1)
        global_pool = h_s.mean(dim=1)
        return self.head(global_pool)


class ChiralLSSClassifier(nn.Module):
    """Wrapper that adapts chiral point clouds into LSS-GNN inputs."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        spin_symmetry: int = 2,
        graph_mode: str = "knn",
        k: int = 5,
    ) -> None:
        super().__init__()
        self.graph_mode = graph_mode
        self.k = k
        self.model = LSSGNN(
            in_scalar=3,
            in_vector=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            spin_symmetry=spin_symmetry,
        )

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


__all__ = ["GraphBuilder", "SparseLocalFrameLayer", "LSSGNN", "ChiralLSSClassifier"]
