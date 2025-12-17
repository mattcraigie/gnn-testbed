from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent

from gnn_testbed.config import ExperimentConfig, load_experiment_config
from gnn_testbed.data.chiral import ChiralChainDataset, ChiralChainGenerator, chiral_collate
from gnn_testbed.models import build_model
from gnn_testbed.training.trainer import Trainer


def build_dataloader(
    generator: ChiralChainGenerator, split_cfg, normalize: str, *, collate_fn=chiral_collate
) -> DataLoader:
    dataset = ChiralChainDataset(
        generator=generator,
        size=split_cfg.size,
        seed=split_cfg.seed,
        normalize=normalize,
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
    base_gen = ChiralChainGenerator(
        N=cfg.data.generator.N,
        rmin=cfg.data.generator.rmin,
        rmax=cfg.data.generator.rmax,
        box_factor=cfg.data.generator.box_factor,
        seed=cfg.data.generator.seed,
    )

    train_loader = build_dataloader(base_gen, cfg.data.train, cfg.data.normalize)
    val_loader = build_dataloader(base_gen, cfg.data.val, cfg.data.normalize)
    test_loader = build_dataloader(base_gen, cfg.data.test, cfg.data.normalize)

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
    parser = argparse.ArgumentParser(description="Train chiral chain classifier")
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
