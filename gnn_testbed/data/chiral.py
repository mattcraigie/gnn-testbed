from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------
# Shared base (PBC + utilities)
# ----------------------------

@dataclass(frozen=True)
class ChainConfig:
    N: int = 256
    rmin: float = 0.5
    rmax: float = 1.5
    box_factor: float = 10.0
    seed: int = 0


class BaseChainGenerator:
    """
    Shared implementation:
      - RNG
      - periodic box size L = box_factor * mean_step_length
      - wrap() and pbc_delta()
      - edge_lengths()
    Subclasses only decide how headings/steps are produced.
    """

    def __init__(self, cfg: ChainConfig):
        self.N = int(cfg.N)
        self.rmin = float(cfg.rmin)
        self.rmax = float(cfg.rmax)
        self.box_factor = float(cfg.box_factor)
        self.seed = int(cfg.seed)

        if self.N < 2:
            raise ValueError("N must be >= 2")
        if not (0.0 < self.rmin <= self.rmax):
            raise ValueError("Require 0 < rmin <= rmax")
        if self.box_factor <= 0:
            raise ValueError("box_factor must be > 0")

        self.rng = np.random.default_rng(self.seed)

    @property
    def L(self) -> float:
        r_mean = 0.5 * (self.rmin + self.rmax)
        return self.box_factor * r_mean

    def wrap(self, x: np.ndarray) -> np.ndarray:
        L = self.L
        return (x + L / 2) % L - L / 2

    def pbc_delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        L = self.L
        d = a - b
        d -= L * np.round(d / L)
        return d

    def edge_lengths(self, pts: np.ndarray) -> np.ndarray:
        d = self.pbc_delta(pts[1:], pts[:-1])
        return np.linalg.norm(d, axis=1)

    # --- subclass contract ---
    def generate(self, cls: str):
        raise NotImplementedError


# ----------------------------
# Option 1: chiral (L/R) walks
# ----------------------------

class ChiralChainGenerator(BaseChainGenerator):
    """
    Same behavior as your original:
      - step length r ~ Uniform[rmin, rmax]
      - initial heading phi0 ~ Uniform[0,2pi)
      - for t>1: alpha ~ Uniform[0,pi], phi += alpha (L) or phi -= alpha (R)
      - positions wrapped in periodic box
    """

    def generate(self, cls: Literal["L", "R"]):
        if cls not in ("L", "R"):
            raise ValueError("cls must be 'L' or 'R'")

        L = self.L
        pts = np.zeros((self.N, 2), dtype=np.float64)

        pts[0] = self.rng.uniform(-L / 2, L / 2, size=2)
        phi = self.rng.uniform(0.0, 2.0 * np.pi)

        for t in range(1, self.N):
            r = self.rng.uniform(self.rmin, self.rmax)

            if t > 1:
                alpha = self.rng.uniform(0.0, np.pi)
                phi = (phi + alpha) if cls == "L" else (phi - alpha)

            step = r * np.array([np.cos(phi), np.sin(phi)])
            pts[t] = self.wrap(pts[t - 1] + step)

        return pts, L

    def signed_turns(self, pts: np.ndarray) -> np.ndarray:
        v1 = self.pbc_delta(pts[1:-1], pts[:-2])
        v2 = self.pbc_delta(pts[2:], pts[1:-1])
        return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]


# -----------------------------------------
# Option 2: unbiased direction, short/long
# -----------------------------------------

class DistancePreferenceChainGenerator(BaseChainGenerator):
    """
    No directional preference:
      - direction theta ~ Uniform[0,2pi) each step (independent)
    Distance preference is encoded by choosing different step ranges per class:
      - 'S' (short): r ~ Uniform[short_rmin, short_rmax]
      - 'G' (long):  r ~ Uniform[long_rmin,  long_rmax]
    """

    def __init__(
        self,
        cfg: ChainConfig,
        *,
        short_range: Tuple[float, float],
        long_range: Tuple[float, float],
    ):
        super().__init__(cfg)

        self.short_rmin, self.short_rmax = map(float, short_range)
        self.long_rmin, self.long_rmax = map(float, long_range)

        if not (0.0 < self.short_rmin <= self.short_rmax):
            raise ValueError("Require 0 < short_rmin <= short_rmax")
        if not (0.0 < self.long_rmin <= self.long_rmax):
            raise ValueError("Require 0 < long_rmin <= long_rmax")

    def generate(self, cls: Literal["S", "G"]):
        if cls not in ("S", "G"):
            raise ValueError("cls must be 'S' (short) or 'G' (long)")

        L = self.L
        pts = np.zeros((self.N, 2), dtype=np.float64)
        pts[0] = self.rng.uniform(-L / 2, L / 2, size=2)

        rlo, rhi = (self.short_rmin, self.short_rmax) if cls == "S" else (self.long_rmin, self.long_rmax)

        for t in range(1, self.N):
            r = self.rng.uniform(rlo, rhi)
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            step = r * np.array([np.cos(theta), np.sin(theta)])
            pts[t] = self.wrap(pts[t - 1] + step)

        return pts, L


# ----------------------------
# Dataset that can do either
# ----------------------------

class ChainDataset(Dataset):
    """
    One dataset that can produce either:
      - task='chiral'   with classes 'L'->0, 'R'->1
      - task='distance' with classes 'S'->0, 'G'->1   (short/long)

    normalize:
      - 'box': divide coordinates by L
      - 'none'
    """

    def __init__(
        self,
        *,
        task: Literal["chiral", "distance"],
        size: int,
        seed: int,
        cfg: ChainConfig,
        normalize: Literal["box", "none"] = "box",
        short_range: Tuple[float, float] = (0.2, 0.6),
        long_range: Tuple[float, float] = (1.0, 2.0),
    ):
        self.task = task
        self.size = int(size)
        self.normalize = normalize
        self.cfg = cfg

        self.short_range = short_range
        self.long_range = long_range

        rng = np.random.default_rng(int(seed))
        self.sample_cls = rng.integers(0, 2, size=self.size)
        self.sample_seed = rng.integers(0, 2**31 - 1, size=self.size, dtype=np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    def __getitem__(self, idx: int):  # type: ignore[override]
        cls_id = int(self.sample_cls[idx])
        seed_i = int(self.sample_seed[idx])

        cfg_i = ChainConfig(
            N=self.cfg.N,
            rmin=self.cfg.rmin,
            rmax=self.cfg.rmax,
            box_factor=self.cfg.box_factor,
            seed=seed_i,
        )

        if self.task == "chiral":
            cls = "L" if cls_id == 0 else "R"
            gen: BaseChainGenerator = ChiralChainGenerator(cfg_i)
        elif self.task == "distance":
            cls = "S" if cls_id == 0 else "G"
            gen = DistancePreferenceChainGenerator(cfg_i, short_range=self.short_range, long_range=self.long_range)
        else:
            raise ValueError("task must be 'chiral' or 'distance'")

        pts, L = gen.generate(cls)  # type: ignore[arg-type]

        if self.normalize == "box":
            pts = pts / float(L)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError("normalize must be 'box' or 'none'")

        points = torch.from_numpy(pts.astype(np.float32))
        label = torch.tensor(float(cls_id), dtype=torch.float32)
        return points, label


def chain_collate(batch: Tuple[torch.Tensor, torch.Tensor]):
    points = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0).view(-1)
    return points, labels


__all__ = [
    "ChainConfig",
    "BaseChainGenerator",
    "ChiralChainGenerator",
    "DistancePreferenceChainGenerator",
    "ChainDataset",
    "chain_collate",
]
