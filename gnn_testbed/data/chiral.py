from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ChiralChainGenerator:
    """
    Chiral 2D random-walk chains in a periodic box.

    Class 'L' : always turns left  (phi <- phi + alpha)
    Class 'R' : always turns right (phi <- phi - alpha)

    Step lengths r ~ Uniform[rmin, rmax]
    Turn magnitudes alpha ~ Uniform[0, pi]
    Initial heading phi0 ~ Uniform[0, 2*pi]
    Start position x0 ~ Uniform in [-L/2, L/2)^2

    Box size L defaults to box_factor * mean_step_length, where mean_step_length = (rmin+rmax)/2.
    """

    N = 256
    rmin = 0.5
    rmax = 1.5
    box_factor = 10.0
    seed = 0

    def __init__(self, N=256, rmin=0.5, rmax=1.5, box_factor=10.0, seed=0):
        """
        Initialize the generator.

        Args:
            N: Number of points in each chain.
            rmin: Minimum step length.
            rmax: Maximum step length.
            box_factor: Box size multiplier relative to mean step length.
            seed: RNG seed.
        """
        self.N = int(N)
        self.rmin = float(rmin)
        self.rmax = float(rmax)
        self.box_factor = float(box_factor)
        self.seed = seed

        if self.N < 2:
            raise ValueError("N must be >= 2")
        if not (0.0 < self.rmin <= self.rmax):
            raise ValueError("Require 0 < rmin <= rmax")
        if self.box_factor <= 0:
            raise ValueError("box_factor must be > 0")

        self.rng = np.random.default_rng(self.seed)

    @property
    def L(self):
        """
        Compute periodic box size.

        Returns:
            Box size (float).
        """
        r_mean = 0.5 * (self.rmin + self.rmax)
        return self.box_factor * r_mean

    def wrap(self, x):
        """
        Wrap positions into [-L/2, L/2).

        Args:
            x: Array-like (...,2) positions.

        Returns:
            Wrapped positions, same shape as x.
        """
        L = self.L
        return (x + L / 2) % L - L / 2

    def pbc_delta(self, a, b):
        """
        Minimum-image displacement a-b under periodic box.

        Args:
            a: Array (...,2)
            b: Array (...,2)

        Returns:
            Displacement array (...,2)
        """
        L = self.L
        d = a - b
        d -= L * np.round(d / L)
        return d

    def generate(self, cls):
        """
        Generate one chain.

        Args:
            cls: 'L' or 'R'

        Returns:
            pts: (N,2) float64 positions wrapped to [-L/2, L/2)
            L: box size (float)
        """
        if cls not in ("L", "R"):
            raise ValueError("cls must be 'L' or 'R'")

        L = self.L
        pts = np.zeros((self.N, 2), dtype=np.float64)

        pts[0] = self.rng.uniform(-L / 2, L / 2, size=2)
        phi = self.rng.uniform(0.0, 2 * np.pi)

        for t in range(1, self.N):
            r = self.rng.uniform(self.rmin, self.rmax)

            if t > 1:
                alpha = self.rng.uniform(0.0, np.pi)
                phi = phi + alpha if cls == "L" else phi - alpha

            step = r * np.array([np.cos(phi), np.sin(phi)])
            pts[t] = self.wrap(pts[t - 1] + step)

        return pts, L

    def edge_lengths(self, pts):
        """
        Compute edge lengths between consecutive points under PBC.

        Args:
            pts: (N,2) array.

        Returns:
            (N-1,) array of lengths.
        """
        d = self.pbc_delta(pts[1:], pts[:-1])
        return np.linalg.norm(d, axis=1)

    def signed_turns(self, pts):
        """
        Compute signed turn proxy from consecutive displacements.

        Args:
            pts: (N,2) array.

        Returns:
            (N-2,) array of signed cross products in 2D.
        """
        v1 = self.pbc_delta(pts[1:-1], pts[:-2])
        v2 = self.pbc_delta(pts[2:], pts[1:-1])
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]


class ChiralChainDataset(Dataset):
    def __init__(self, generator: ChiralChainGenerator, size: int, seed: int, normalize: str = "box"):
        """
        A PyTorch Dataset that generates chiral chains on the fly.

        Labels:
            'L' -> 0
            'R' -> 1

        Args:
            generator: ChiralChainGenerator (used as a template).
            size: Number of samples in this dataset split.
            seed: Split-specific seed for deterministic generation.
            normalize: "none" or "box".
                - "box": divide coordinates by L so positions roughly lie in [-0.5,0.5).
        """
        self.size = int(size)
        self.normalize = normalize

        self.rng = np.random.default_rng(seed)

        self.N = generator.N
        self.rmin = generator.rmin
        self.rmax = generator.rmax
        self.box_factor = generator.box_factor

        self.sample_cls = self.rng.integers(0, 2, size=self.size)
        self.sample_seed = self.rng.integers(0, 2**31 - 1, size=self.size, dtype=np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    def __getitem__(self, idx: int):  # type: ignore[override]
        cls_id = int(self.sample_cls[idx])
        cls = "L" if cls_id == 0 else "R"

        gen = ChiralChainGenerator(
            N=self.N,
            rmin=self.rmin,
            rmax=self.rmax,
            box_factor=self.box_factor,
            seed=int(self.sample_seed[idx]),
        )
        pts, L = gen.generate(cls)

        if self.normalize == "box":
            pts = pts / float(L)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError("normalize must be 'none' or 'box'")

        points = torch.from_numpy(pts.astype(np.float32))
        label = torch.tensor(float(cls_id), dtype=torch.float32)
        return points, label


def chiral_collate(batch: Tuple[torch.Tensor, torch.Tensor]):
    points = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0).view(-1)
    return points, labels


__all__ = [
    "ChiralChainGenerator",
    "ChiralChainDataset",
    "chiral_collate",
]
