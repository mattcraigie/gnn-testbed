from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

from gnn_testbed.config import ExperimentConfig, load_experiment_config
from gnn_testbed.data.chiral import ChainDataset, ChainConfig, chain_collate
from gnn_testbed.models import build_model
from gnn_testbed.training.trainer import Trainer


def build_dataloader(data_cfg, split_cfg, *, collate_fn=chain_collate) -> DataLoader:
    dataset = ChainDataset(
        task=data_cfg.task,
        size=split_cfg.size,
        seed=split_cfg.seed,
        cfg=ChainConfig(
            N=data_cfg.chain.N,
            rmin=data_cfg.chain.rmin,
            rmax=data_cfg.chain.rmax,
            box_factor=data_cfg.chain.box_factor,
            seed=data_cfg.chain.seed,
        ),
        normalize=data_cfg.normalize,
        short_range=tuple(data_cfg.short_range),
        long_range=tuple(data_cfg.long_range),
    )
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
    parser = argparse.ArgumentParser(description="Train chain classifier")
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
