from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

from gnn_testbed.config import ExperimentConfig, load_experiment_config
from gnn_testbed.data import (
    PairFieldConfig,
    PairFieldDataset,
    TriangleFieldConfig,
    TriangleFieldDataset,
    pointcloud_collate,
)
from gnn_testbed.models import build_model
from gnn_testbed.training.trainer import Trainer


def build_dataloader(data_cfg, split_cfg, *, collate_fn=pointcloud_collate) -> DataLoader:
    if data_cfg.task == "pair":
        dataset = PairFieldDataset(
            size=split_cfg.size,
            seed=split_cfg.seed,
            cfg=PairFieldConfig(
                box_size=data_cfg.pair.box_size,
                seed=data_cfg.pair.seed,
            ),
            signal_points=data_cfg.signal_points,
            noise_points=data_cfg.noise_points,
            pair_distance=data_cfg.pair_distance,
            normalize=data_cfg.normalize,
            jitter_std=data_cfg.jitter_std,
            jitter_clip=data_cfg.jitter_clip,
        )
    elif data_cfg.task == "triangle":
        dataset = TriangleFieldDataset(
            size=split_cfg.size,
            seed=split_cfg.seed,
            cfg=TriangleFieldConfig(
                box_size=data_cfg.triangle.box_size,
                seed=data_cfg.triangle.seed,
            ),
            signal_points=data_cfg.signal_points,
            noise_points=data_cfg.noise_points,
            normalize=data_cfg.normalize,
            jitter_std=data_cfg.jitter_std,
            jitter_clip=data_cfg.jitter_clip,
        )
    else:
        raise ValueError("data.task must be 'pair' or 'triangle'")
    return DataLoader(
        dataset,
        batch_size=split_cfg.batch_size,
        shuffle=split_cfg.shuffle,
        num_workers=split_cfg.num_workers,
        pin_memory=split_cfg.pin_memory,
        drop_last=split_cfg.drop_last,
        collate_fn=collate_fn,
    )


def build_components(cfg: ExperimentConfig):
    train_loader = build_dataloader(cfg.data, cfg.data.train)
    val_loader = build_dataloader(cfg.data, cfg.data.val)
    test_loader = build_dataloader(cfg.data, cfg.data.test)

    model = build_model(cfg.model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg.training,
    )

    return trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train point cloud classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    trainer = build_components(cfg)
    history = trainer.fit()
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Test metrics: {history['test']}")


if __name__ == "__main__":
    main()
