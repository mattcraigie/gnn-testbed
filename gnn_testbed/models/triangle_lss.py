from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay, cKDTree


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2D cross product (scalar): a_x b_y - a_y b_x."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass
class TriangleBatch:
    tri_pos: torch.Tensor
    tri_ori: torch.Tensor
    tri_chi: torch.Tensor
    tri_indices: torch.Tensor
    tri_edge_index: torch.Tensor


class TriangleGraphBuilder:
    """Build triangle (triplet) nodes and edges from 2D point clouds (Delaunay)."""

    @staticmethod
    def build_triangles(
        pos: torch.Tensor,
        *,
        tri_graph_mode: str = "adjacent",
        k_tri: int = 6,
        eps: float = 1e-9,
    ) -> TriangleBatch:
        if pos.dim() not in (2, 3):
            raise ValueError("pos must have shape [B, N, 2] or [N, 2].")

        device = pos.device
        pos_np = pos.detach().cpu().numpy()
        if pos.dim() == 2:
            pos_np = [pos_np]

        tri_pos_all = []
        tri_ori_all = []
        tri_chi_all = []
        tri_idx_all = []
        tri_edge_src_all = []
        tri_edge_dst_all = []

        point_offset = 0
        tri_offset = 0

        for p in pos_np:
            n = p.shape[0]
            if n < 3:
                point_offset += n
                continue

            tri = Delaunay(p)
            simplices = tri.simplices
            num_tri = simplices.shape[0]

            if tri_graph_mode == "adjacent":
                edge_map: dict[Tuple[int, int], list[int]] = {}
                for t_id, (i0, i1, i2) in enumerate(simplices):
                    for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                        if a > b:
                            a, b = b, a
                        edge_map.setdefault((a, b), []).append(t_id)

                adj_pairs = set()
                for t_list in edge_map.values():
                    if len(t_list) == 2:
                        a, b = t_list
                        adj_pairs.add((a, b))
                        adj_pairs.add((b, a))

                tri_graph_mode_local = "adjacent" if adj_pairs else "knn"
            else:
                tri_graph_mode_local = "knn"

            simp_t = torch.tensor(simplices, dtype=torch.long, device=device)
            pts_t = torch.tensor(p, dtype=torch.float32, device=device)
            a = pts_t[simp_t[:, 0]]
            b = pts_t[simp_t[:, 1]]
            c = pts_t[simp_t[:, 2]]

            centroid = (a + b + c) / 3.0

            ab2 = torch.sum((a - b) ** 2, dim=-1)
            bc2 = torch.sum((b - c) ** 2, dim=-1)
            ca2 = torch.sum((c - a) ** 2, dim=-1)
            lens2 = torch.stack([ab2, bc2, ca2], dim=-1)
            base_choice = torch.argmax(lens2, dim=-1)

            p_t = torch.empty_like(a)
            q_t = torch.empty_like(a)
            r_t = torch.empty_like(a)

            mask0 = base_choice == 0
            mask1 = base_choice == 1
            mask2 = base_choice == 2

            p_t[mask0], q_t[mask0], r_t[mask0] = a[mask0], b[mask0], c[mask0]
            p_t[mask1], q_t[mask1], r_t[mask1] = b[mask1], c[mask1], a[mask1]
            p_t[mask2], q_t[mask2], r_t[mask2] = c[mask2], a[mask2], b[mask2]

            e = q_t - p_t
            e_norm = torch.norm(e, dim=-1, keepdim=True) + eps
            e_hat = e / e_norm

            s = _cross2(e, (r_t - p_t))
            chi = torch.sign(s).unsqueeze(-1)
            chi = torch.where(chi == 0, torch.ones_like(chi), chi)

            u = chi * e_hat
            theta = torch.atan2(u[:, 1], u[:, 0]).unsqueeze(-1)

            simp_flat = simp_t + point_offset

            if tri_graph_mode_local == "adjacent":
                src_list = []
                dst_list = []
                for s_id, d_id in adj_pairs:
                    src_list.append(s_id + tri_offset)
                    dst_list.append(d_id + tri_offset)
                if src_list:
                    tri_edge_src = torch.tensor(src_list, dtype=torch.long, device=device)
                    tri_edge_dst = torch.tensor(dst_list, dtype=torch.long, device=device)
                else:
                    tri_edge_src = torch.empty((0,), dtype=torch.long, device=device)
                    tri_edge_dst = torch.empty((0,), dtype=torch.long, device=device)
            else:
                cent_np = centroid.detach().cpu().numpy()
                tree = cKDTree(cent_np)
                _, idx = tree.query(cent_np, k=min(k_tri + 1, max(1, num_tri)))
                idx = idx[:, 1:]
                src = idx.reshape(-1)
                dst = np.repeat(np.arange(num_tri), idx.shape[1])

                tri_edge_src = torch.tensor(src, dtype=torch.long, device=device) + tri_offset
                tri_edge_dst = torch.tensor(dst, dtype=torch.long, device=device) + tri_offset

            tri_pos_all.append(centroid)
            tri_ori_all.append(theta)
            tri_chi_all.append(chi)
            tri_idx_all.append(simp_flat)
            tri_edge_src_all.append(tri_edge_src)
            tri_edge_dst_all.append(tri_edge_dst)

            point_offset += n
            tri_offset += num_tri

        if not tri_pos_all:
            empty = torch.empty((0, 2), dtype=torch.float32, device=device)
            empty1 = torch.empty((0, 1), dtype=torch.float32, device=device)
            emptyi = torch.empty((0, 3), dtype=torch.long, device=device)
            emptye = torch.empty((2, 0), dtype=torch.long, device=device)
            return TriangleBatch(empty, empty1, empty1, emptyi, emptye)

        tri_pos = torch.cat(tri_pos_all, dim=0)
        tri_ori = torch.cat(tri_ori_all, dim=0)
        tri_chi = torch.cat(tri_chi_all, dim=0)
        tri_indices = torch.cat(tri_idx_all, dim=0)

        edge_src = (
            torch.cat(tri_edge_src_all, dim=0)
            if tri_edge_src_all
            else torch.empty((0,), device=device, dtype=torch.long)
        )
        edge_dst = (
            torch.cat(tri_edge_dst_all, dim=0)
            if tri_edge_dst_all
            else torch.empty((0,), device=device, dtype=torch.long)
        )
        tri_edge_index = (
            torch.stack([edge_src, edge_dst], dim=0)
            if edge_src.numel()
            else torch.empty((2, 0), device=device, dtype=torch.long)
        )

        return TriangleBatch(tri_pos, tri_ori, tri_chi, tri_indices, tri_edge_index)


class SparseTriangleFrameLayer(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, spin_symmetry: int = 1):
        super().__init__()
        if vector_dim % 2 != 0:
            raise ValueError("vector_dim must be even (packed 2D vectors).")

        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.spin_symmetry = spin_symmetry

        geo_dim = 4
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
        chirality: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        delta_pos = pos[src] - pos[dst]
        dist = torch.norm(delta_pos, dim=1, keepdim=True) + eps

        phi_global = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])
        alpha_i = orientation[dst].squeeze(-1)
        delta_phi = phi_global - alpha_i

        parity_edge = chirality[src] * chirality[dst]

        geo_feat = torch.cat(
            [
                dist,
                torch.cos(self.spin_symmetry * delta_phi).unsqueeze(-1),
                torch.sin(self.spin_symmetry * delta_phi).unsqueeze(-1),
                parity_edge,
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


class TriangleLSSGNN(nn.Module):
    def __init__(
        self,
        in_scalar: int,
        in_vector: int,
        hidden_dim: int,
        num_layers: int = 2,
        spin_symmetry: int = 1,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even so it can be split between scalar/vector channels")

        self.s_dim = hidden_dim // 2
        self.v_dim = hidden_dim // 2

        self.enc_s = nn.Sequential(nn.Linear(in_scalar, self.s_dim), nn.ReLU())
        self.enc_v = nn.Linear(in_vector, self.v_dim, bias=False)

        self.layers = nn.ModuleList(
            [SparseTriangleFrameLayer(self.s_dim, self.v_dim, spin_symmetry) for _ in range(num_layers)]
        )

        self.head = nn.Linear(self.s_dim, 1)

    def forward(
        self,
        x_s: torch.Tensor,
        x_v: torch.Tensor,
        tri_pos: torch.Tensor,
        tri_ori: torch.Tensor,
        tri_chi: torch.Tensor,
        tri_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if x_s.dim() == 3:
            batch_size, num_tri, _ = x_s.shape
            x_s_flat = x_s.reshape(-1, x_s.shape[-1])
            x_v_flat = x_v.reshape(-1, x_v.shape[-1])
            pos_flat = tri_pos.reshape(-1, 2)
            ori_flat = tri_ori.reshape(-1, 1)
            chi_flat = tri_chi.reshape(-1, 1)
        else:
            batch_size = 1
            num_tri = x_s.shape[0]
            x_s_flat = x_s
            x_v_flat = x_v
            pos_flat = tri_pos
            ori_flat = tri_ori
            chi_flat = tri_chi

        h_s = self.enc_s(x_s_flat)
        h_v = self.enc_v(x_v_flat)

        for layer in self.layers:
            h_s, h_v = layer(h_s, h_v, tri_edge_index, pos_flat, ori_flat, chi_flat)

        if x_s.dim() == 3:
            h_s = h_s.view(batch_size, num_tri, -1)
            global_pool = h_s.mean(dim=1)
        else:
            global_pool = h_s.mean(dim=0, keepdim=True)

        return self.head(global_pool)


class ChiralTriangleLSSClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        tri_graph_mode: str = "adjacent",
        k_tri: int = 6,
    ) -> None:
        super().__init__()
        self.tri_graph_mode = tri_graph_mode
        self.k_tri = k_tri

        self.model = TriangleLSSGNN(
            in_scalar=4,
            in_vector=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            spin_symmetry=1,
        )

    @staticmethod
    def _triangle_features_from_points(
        points_flat: torch.Tensor,
        tri_indices: torch.Tensor,
        tri_chi: torch.Tensor,
        eps: float = 1e-9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = points_flat[tri_indices[:, 0]]
        b = points_flat[tri_indices[:, 1]]
        c = points_flat[tri_indices[:, 2]]

        ab2 = torch.sum((a - b) ** 2, dim=-1)
        bc2 = torch.sum((b - c) ** 2, dim=-1)
        ca2 = torch.sum((c - a) ** 2, dim=-1)
        lens2 = torch.stack([ab2, bc2, ca2], dim=-1)
        base_choice = torch.argmax(lens2, dim=-1)

        p = torch.empty_like(a)
        q = torch.empty_like(a)
        r = torch.empty_like(a)

        mask0 = base_choice == 0
        mask1 = base_choice == 1
        mask2 = base_choice == 2

        p[mask0], q[mask0], r[mask0] = a[mask0], b[mask0], c[mask0]
        p[mask1], q[mask1], r[mask1] = b[mask1], c[mask1], a[mask1]
        p[mask2], q[mask2], r[mask2] = c[mask2], a[mask2], b[mask2]

        e = q - p
        b_len = torch.norm(e, dim=-1, keepdim=True) + eps
        e_hat = e / b_len

        s = _cross2(e, (r - p)).unsqueeze(-1)
        h_signed = s / (b_len + eps)
        h_abs = torch.abs(h_signed)

        t_proj = torch.sum((r - p) * (q - p), dim=-1, keepdim=True) / (
            torch.sum((q - p) ** 2, dim=-1, keepdim=True) + eps
        )

        chi = tri_chi
        u = chi * e_hat
        theta = torch.atan2(u[:, 1], u[:, 0]).unsqueeze(-1)

        scalar_feats = torch.cat([b_len, h_abs, t_proj, chi], dim=-1)
        vector_feats = u
        return scalar_feats, vector_feats, theta

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3 or points.size(-1) != 2:
            raise ValueError("points must have shape [B, N, 2].")

        batch_size, num_points, _ = points.shape
        points_flat = points.reshape(-1, 2)

        tri_batch = TriangleGraphBuilder.build_triangles(
            points,
            tri_graph_mode=self.tri_graph_mode,
            k_tri=self.k_tri,
        )

        if tri_batch.tri_pos.numel() == 0:
            return torch.zeros((batch_size,), device=points.device, dtype=points.dtype)

        tri_s, tri_v, tri_theta = self._triangle_features_from_points(
            points_flat,
            tri_batch.tri_indices,
            tri_batch.tri_chi,
        )

        batch_id = (tri_batch.tri_indices[:, 0] // num_points).clamp(0, batch_size - 1)
        counts = torch.bincount(batch_id, minlength=batch_size).tolist()
        max_tri = max(counts)

        def pad_by_batch(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            out = x.new_full((batch_size, max_tri, x.shape[-1]), fill)
            start = 0
            for b in range(batch_size):
                nb = counts[b]
                if nb > 0:
                    out[b, :nb] = x[start : start + nb]
                start += nb
            return out

        tri_s_b = pad_by_batch(tri_s, 0.0)
        tri_v_b = pad_by_batch(tri_v, 0.0)
        tri_pos_b = pad_by_batch(tri_batch.tri_pos, 0.0)
        tri_ori_b = pad_by_batch(tri_theta, 0.0)
        tri_chi_b = pad_by_batch(tri_batch.tri_chi, 1.0)

        logits = self.model(
            tri_s_b,
            tri_v_b,
            tri_pos_b,
            tri_ori_b,
            tri_chi_b,
            tri_batch.tri_edge_index,
        )
        return logits.squeeze(-1)


__all__ = [
    "TriangleGraphBuilder",
    "SparseTriangleFrameLayer",
    "TriangleLSSGNN",
    "ChiralTriangleLSSClassifier",
]
