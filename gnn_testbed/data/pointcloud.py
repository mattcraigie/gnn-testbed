from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairFieldConfig:
    box_size: float = 10.0
    seed: int = 0


class PairFieldDataset(Dataset):
    """
    Unordered point cloud dataset made of embedded pairs + random noise.

    Each sample:
      - n_pairs = signal_points // 2 pairs
      - each pair has fixed separation `pair_distance`
      - each pair is placed with random orientation and random center in a periodic box
      - plus noise_points random points
      - optional Gaussian jitter applied to ALL points

    Labels:
      - y=0,1 are balanced but (intentionally) NOT parity-identifiable from unordered pairs alone.
        A mirror transform produces an identical distribution for this construction.
        Use this dataset for "non-parity" controls / baselines.
    """

    def __init__(
        self,
        *,
        size: int,
        seed: int,
        cfg: PairFieldConfig,
        signal_points: int,
        noise_points: int,
        pair_distance: float = 1.0,
        normalize: Literal["box", "none"] = "box",
        jitter_std: float = 0.01,
        jitter_clip: Optional[float] = 0.05,
    ):
        self.size = int(size)
        self.cfg = cfg
        self.signal_points = int(signal_points)
        self.noise_points = int(noise_points)
        self.pair_distance = float(pair_distance)
        self.normalize = normalize

        self.jitter_std = float(jitter_std)
        self.jitter_clip = None if jitter_clip is None else float(jitter_clip)

        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.cfg.box_size <= 0:
            raise ValueError("cfg.box_size must be > 0")
        if self.signal_points < 0 or self.noise_points < 0:
            raise ValueError("signal_points and noise_points must be >= 0")
        if self.pair_distance <= 0:
            raise ValueError("pair_distance must be > 0")
        if self.jitter_std < 0:
            raise ValueError("jitter_std must be >= 0")
        if self.jitter_clip is not None and self.jitter_clip <= 0:
            raise ValueError("jitter_clip must be > 0 or None")

        rng = np.random.default_rng(int(seed))
        self.sample_cls = rng.integers(0, 2, size=self.size)
        self.sample_seed = rng.integers(0, 2**31 - 1, size=self.size, dtype=np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    @property
    def L(self) -> float:
        return float(self.cfg.box_size)

    def wrap(self, x: np.ndarray) -> np.ndarray:
        L = self.L
        return (x + L / 2) % L - L / 2

    def _add_jitter(self, rng: np.random.Generator, pts: np.ndarray) -> np.ndarray:
        if self.jitter_std == 0.0 or pts.size == 0:
            return pts
        noise = rng.normal(loc=0.0, scale=self.jitter_std, size=pts.shape)
        if self.jitter_clip is not None:
            noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
        return self.wrap(pts + noise)

    def __getitem__(self, idx: int):  # type: ignore[override]
        y = int(self.sample_cls[idx])
        seed_i = int(self.sample_seed[idx])
        rng = np.random.default_rng(seed_i)

        n_pairs = self.signal_points // 2
        pts_list: list[np.ndarray] = []

        half = 0.5 * self.pair_distance

        for _ in range(n_pairs):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

            center = rng.uniform(-self.L / 2, self.L / 2, size=2)
            p1 = center - half * direction
            p2 = center + half * direction

            pair = np.stack([p1, p2], axis=0)
            pair = self.wrap(pair)
            pts_list.append(pair)

        if self.noise_points > 0:
            noise_pts = rng.uniform(-self.L / 2, self.L / 2, size=(self.noise_points, 2))
            pts_list.append(noise_pts)

        pts = np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 2), dtype=np.float64)

        pts = self._add_jitter(rng, pts)

        if self.normalize == "box":
            pts = pts / self.L
        elif self.normalize == "none":
            pass
        else:
            raise ValueError("normalize must be 'box' or 'none'")

        if pts.shape[0] > 1:
            rng.shuffle(pts, axis=0)

        points = torch.from_numpy(pts.astype(np.float32))
        label = torch.tensor(float(y), dtype=torch.float32)
        return points, label


@dataclass(frozen=True)
class TriangleFieldConfig:
    box_size: float = 10.0
    seed: int = 0


class TriangleFieldDataset(Dataset):
    """
    Unordered point cloud parity task with right triangles + noise.

    Labels:
      - y=0: right-handed triangles
      - y=1: left-handed (parity mirrored)
    """

    def __init__(
        self,
        *,
        size: int,
        seed: int,
        cfg: TriangleFieldConfig,
        signal_points: int,
        noise_points: int,
        normalize: Literal["box", "none"] = "box",
        jitter_std: float = 0.01,
        jitter_clip: Optional[float] = 0.05,
    ):
        self.size = int(size)
        self.cfg = cfg
        self.signal_points = int(signal_points)
        self.noise_points = int(noise_points)
        self.normalize = normalize

        self.jitter_std = float(jitter_std)
        self.jitter_clip = None if jitter_clip is None else float(jitter_clip)

        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.cfg.box_size <= 0:
            raise ValueError("cfg.box_size must be > 0")
        if self.signal_points < 0 or self.noise_points < 0:
            raise ValueError("signal_points and noise_points must be >= 0")
        if self.jitter_std < 0:
            raise ValueError("jitter_std must be >= 0")
        if self.jitter_clip is not None and self.jitter_clip <= 0:
            raise ValueError("jitter_clip must be > 0 or None")

        rng = np.random.default_rng(int(seed))
        self.sample_cls = rng.integers(0, 2, size=self.size)
        self.sample_seed = rng.integers(0, 2**31 - 1, size=self.size, dtype=np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self.size

    @property
    def L(self) -> float:
        return float(self.cfg.box_size)

    def wrap(self, x: np.ndarray) -> np.ndarray:
        L = self.L
        return (x + L / 2) % L - L / 2

    def _add_jitter(self, rng: np.random.Generator, pts: np.ndarray) -> np.ndarray:
        if self.jitter_std == 0.0 or pts.size == 0:
            return pts
        noise = rng.normal(loc=0.0, scale=self.jitter_std, size=pts.shape)
        if self.jitter_clip is not None:
            noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
        return self.wrap(pts + noise)

    def __getitem__(self, idx: int):  # type: ignore[override]
        y = int(self.sample_cls[idx])
        seed_i = int(self.sample_seed[idx])
        rng = np.random.default_rng(seed_i)

        n_tri = self.signal_points // 3
        pts_list: list[np.ndarray] = []

        base = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        )

        for _ in range(n_tri):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)

            tri = base @ R.T
            center = rng.uniform(-self.L / 2, self.L / 2, size=2)
            tri = self.wrap(tri + center)
            pts_list.append(tri)

        if self.noise_points > 0:
            noise_pts = rng.uniform(-self.L / 2, self.L / 2, size=(self.noise_points, 2))
            pts_list.append(noise_pts)

        pts = np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 2), dtype=np.float64)

        if y == 1 and pts.size > 0:
            pts[:, 1] *= -1.0
            pts = self.wrap(pts)

        pts = self._add_jitter(rng, pts)

        if self.normalize == "box":
            pts = pts / self.L
        elif self.normalize == "none":
            pass
        else:
            raise ValueError("normalize must be 'box' or 'none'")

        if pts.shape[0] > 1:
            rng.shuffle(pts, axis=0)

        points = torch.from_numpy(pts.astype(np.float32))
        label = torch.tensor(float(y), dtype=torch.float32)
        return points, label


def pointcloud_collate(batch):
    points = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0).view(-1)
    return points, labels


__all__ = [
    "PairFieldConfig",
    "TriangleFieldConfig",
    "PairFieldDataset",
    "TriangleFieldDataset",
    "pointcloud_collate",
]
