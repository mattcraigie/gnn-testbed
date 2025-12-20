from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from torch.utils.data import DataLoader

from gnn_testbed.config import ExperimentConfig, ModelConfig, load_experiment_config
from gnn_testbed.data import (
    PairFieldConfig,
    PairFieldDataset,
    TriangleFieldConfig,
    TriangleFieldDataset,
    pointcloud_collate,
)
from gnn_testbed.models import build_model
from gnn_testbed.training.trainer import Trainer

ROOT = Path(__file__).resolve().parent


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


def _model_field_names() -> set[str]:
    return {field.name for field in fields(ModelConfig)}


def _parse_model_entries(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    comparison = raw.get("comparison", {}) if isinstance(raw, dict) else {}
    entries = comparison.get("models")
    if not entries:
        return [{"type": "lss_gnn"}, {"type": "standard_egnn"}, {"type": "triangle_lss"}]

    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"type": entry})
        elif isinstance(entry, dict):
            normalized.append(entry)
        else:
            raise ValueError("comparison.models entries must be strings or mappings.")
    return normalized


def _build_model_configs(base: ExperimentConfig, entries: Iterable[Dict[str, Any]]):
    model_fields = _model_field_names()
    configs = []
    for entry in entries:
        if entry.get("type") == "simple_point_mlp":
            raise ValueError("comparison.models must not include simple_point_mlp.")

        label = entry.get("name", entry.get("type", "model"))
        overrides = {k: v for k, v in entry.items() if k in model_fields}
        if "type" not in overrides:
            raise ValueError("Each comparison model entry must include a 'type'.")
        model_cfg = replace(base.model, **overrides)
        configs.append((label, model_cfg))
    return configs


def run_model(label: str, cfg: ExperimentConfig) -> Dict[str, Any]:
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

    history = trainer.fit()
    return {
        "label": label,
        "model_type": cfg.model.type,
        "best_epoch": history.get("best_epoch"),
        "test": history.get("test"),
        "work_dir": cfg.training.work_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GNN model variants")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "comparison.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    base_cfg = load_experiment_config(config_path)
    model_entries = _parse_model_entries(raw)

    results = []
    for label, model_cfg in _build_model_configs(base_cfg, model_entries):
        work_dir = Path(base_cfg.training.work_dir) / label
        run_cfg = ExperimentConfig(
            data=base_cfg.data,
            model=model_cfg,
            training=replace(base_cfg.training, work_dir=str(work_dir)),
        )
        results.append(run_model(label, run_cfg))

    summary = {"results": results}
    summary_path = Path(base_cfg.training.work_dir) / "comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
