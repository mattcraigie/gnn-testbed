from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from gnn_testbed.config import TrainingConfig


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, mode: str):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        value = float(value)
        if self.best is None:
            self.best = value
            return False

        if self.mode == "min":
            improved = (self.best - value) > self.min_delta
        else:
            improved = (value - self.best) > self.min_delta

        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True

        return self.should_stop


def sigmoid_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    return torch.sigmoid(logits)


def binary_accuracy_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).to(targets.dtype)
    return (preds == targets).float().mean().item()


def safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num) / (float(den) + float(eps))


def binary_precision_recall_f1_from_probs(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float]:
    preds = (probs >= threshold).long()
    t = targets.long()

    tp = ((preds == 1) & (t == 1)).sum().item()
    fp = ((preds == 1) & (t == 0)).sum().item()
    fn = ((preds == 0) & (t == 1)).sum().item()

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: TrainingConfig,
        criterion=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = config

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(self.cfg.amp))
        self.scheduler = self._build_scheduler()

        self.early_stopper = EarlyStopping(
            patience=self.cfg.early_stop_patience,
            min_delta=self.cfg.early_stop_min_delta,
            mode=self.cfg.mode,
        )

        os.makedirs(self.cfg.work_dir, exist_ok=True)
        self.best_ckpt_path = os.path.join(self.cfg.work_dir, "best.pt")
        self.last_ckpt_path = os.path.join(self.cfg.work_dir, "last.pt")

        self.writer = SummaryWriter(log_dir=self.cfg.work_dir, flush_secs=self.cfg.tb_flush_secs)

        self.history = {"train": [], "val": [], "test": None, "best_epoch": None}

        self._log_config()

    def _log_config(self):
        cfg_path = os.path.join(self.cfg.work_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        cfg_lines = "\n".join([f"{k}: {v}" for k, v in asdict(self.cfg).items()])
        self.writer.add_text("config", cfg_lines, global_step=0)

    def _build_scheduler(self):
        if self.cfg.scheduler == "none":
            return None

        if self.cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.cfg.epochs - self.cfg.warmup_epochs),
                eta_min=self.cfg.min_lr,
            )

        if self.cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.step_size,
                gamma=self.cfg.gamma,
            )

        raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr)

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict) -> None:
        payload = {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "config": asdict(self.cfg),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if load_optimizer and "scaler" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        return ckpt

    def _unpack_batch(self, batch):
        points, labels = batch
        points = points.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        if labels.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            labels = labels.float()
        return points, labels

    def _forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.model(points)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return self.criterion(logits, labels)

    def _step_lr(self, epoch: int) -> None:
        if self.cfg.warmup_epochs and epoch < self.cfg.warmup_epochs:
            frac = (epoch + 1) / float(self.cfg.warmup_epochs)
            self._set_lr(self.cfg.lr * frac)
            return
        if self.scheduler is not None:
            self.scheduler.step()

    def _log_epoch_scalars(self, split: str, metrics: Dict, epoch: int):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{split}/{k}", float(v), int(epoch))

    def _log_optional_histograms(self, epoch: int) -> None:
        if self.cfg.tb_log_histograms:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(f"params/{name}", p.detach().float().cpu(), int(epoch))

        if self.cfg.tb_log_grads:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                self.writer.add_histogram(f"grads/{name}", p.grad.detach().float().cpu(), int(epoch))

    def train_one_epoch(self, epoch: int) -> Dict:
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        total_prec = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        n_batches = 0

        start = time.time()

        for i, batch in enumerate(self.train_loader):
            points, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(self.cfg.amp)):
                logits = self._forward(points)
                loss = self._compute_loss(logits, labels)

            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip_norm is not None and self.cfg.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            probs = sigmoid_logits_to_probs(logits.detach())
            acc = binary_accuracy_from_probs(probs, labels.detach(), threshold=self.cfg.threshold)
            prec, rec, f1 = binary_precision_recall_f1_from_probs(
                probs, labels.detach(), threshold=self.cfg.threshold
            )

            total_loss += float(loss.detach().item())
            total_acc += float(acc)
            total_prec += float(prec)
            total_rec += float(rec)
            total_f1 += float(f1)
            n_batches += 1

            global_step = epoch * len(self.train_loader) + i
            self.writer.add_scalar("train/loss_step", float(loss.detach().item()), int(global_step))

        self._step_lr(epoch)

        elapsed = time.time() - start
        metrics = {
            "loss": total_loss / max(1, n_batches),
            "acc": total_acc / max(1, n_batches),
            "precision": total_prec / max(1, n_batches),
            "recall": total_rec / max(1, n_batches),
            "f1": total_f1 / max(1, n_batches),
            "lr": self._current_lr(),
            "time_sec": elapsed,
        }

        self._log_epoch_scalars("train", metrics, epoch)
        self._log_optional_histograms(epoch)
        return metrics

    def validate(self, epoch: int) -> Dict:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_prec = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        n_batches = 0

        start = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                points, labels = self._unpack_batch(batch)

                with torch.cuda.amp.autocast(enabled=bool(self.cfg.amp)):
                    logits = self._forward(points)
                    loss = self._compute_loss(logits, labels)

                probs = sigmoid_logits_to_probs(logits)
                acc = binary_accuracy_from_probs(probs, labels, threshold=self.cfg.threshold)
                prec, rec, f1 = binary_precision_recall_f1_from_probs(
                    probs, labels, threshold=self.cfg.threshold
                )

                total_loss += float(loss.item())
                total_acc += float(acc)
                total_prec += float(prec)
                total_rec += float(rec)
                total_f1 += float(f1)
                n_batches += 1

        elapsed = time.time() - start
        metrics = {
            "loss": total_loss / max(1, n_batches),
            "acc": total_acc / max(1, n_batches),
            "precision": total_prec / max(1, n_batches),
            "recall": total_rec / max(1, n_batches),
            "f1": total_f1 / max(1, n_batches),
            "time_sec": elapsed,
        }

        self._log_epoch_scalars("val", metrics, epoch)
        return metrics

    def test(self) -> Dict:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_prec = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        n_batches = 0

        start = time.time()

        with torch.no_grad():
            for batch in self.test_loader:
                points, labels = self._unpack_batch(batch)

                with torch.cuda.amp.autocast(enabled=bool(self.cfg.amp)):
                    logits = self._forward(points)
                    loss = self._compute_loss(logits, labels)

                probs = sigmoid_logits_to_probs(logits)
                acc = binary_accuracy_from_probs(probs, labels, threshold=self.cfg.threshold)
                prec, rec, f1 = binary_precision_recall_f1_from_probs(
                    probs, labels, threshold=self.cfg.threshold
                )

                total_loss += float(loss.item())
                total_acc += float(acc)
                total_prec += float(prec)
                total_rec += float(rec)
                total_f1 += float(f1)
                n_batches += 1

        elapsed = time.time() - start
        metrics = {
            "loss": total_loss / max(1, n_batches),
            "acc": total_acc / max(1, n_batches),
            "precision": total_prec / max(1, n_batches),
            "recall": total_rec / max(1, n_batches),
            "f1": total_f1 / max(1, n_batches),
            "time_sec": elapsed,
        }

        self._log_epoch_scalars("test", metrics, 0)
        return metrics

    def _get_monitor_value(self, val_metrics: Dict) -> float:
        if self.cfg.monitor == "val_loss":
            return float(val_metrics["loss"])
        if self.cfg.monitor == "val_acc":
            return float(val_metrics["acc"])
        if self.cfg.monitor == "val_f1":
            return float(val_metrics["f1"])
        raise ValueError(f"Unsupported monitor: {self.cfg.monitor}")

    def fit(self) -> Dict:
        best_epoch = None
        best_monitor = None

        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            monitor_val = self._get_monitor_value(val_metrics)
            self.writer.add_scalar("val/monitor", float(monitor_val), int(epoch))

            is_best = False
            if best_monitor is None:
                is_best = True
            else:
                if self.cfg.mode == "min":
                    is_best = monitor_val < (best_monitor - self.cfg.early_stop_min_delta)
                else:
                    is_best = monitor_val > (best_monitor + self.cfg.early_stop_min_delta)

            if is_best:
                best_monitor = float(monitor_val)
                best_epoch = int(epoch)
                self.history["best_epoch"] = best_epoch

                metrics_payload = {
                    "epoch": int(epoch),
                    "train": train_metrics,
                    "val": val_metrics,
                    "monitor": float(monitor_val),
                }
                self.save_checkpoint(self.best_ckpt_path, epoch, metrics_payload)
                self.writer.add_text("best", f"epoch={epoch}, monitor={monitor_val}", int(epoch))

            if not self.cfg.save_best_only:
                metrics_payload = {
                    "epoch": int(epoch),
                    "train": train_metrics,
                    "val": val_metrics,
                    "monitor": float(monitor_val),
                }
                self.save_checkpoint(self.last_ckpt_path, epoch, metrics_payload)

            if self.early_stopper.step(monitor_val):
                self.writer.add_text("early_stop", f"stopped at epoch={epoch}", int(epoch))
                break

        if os.path.exists(self.best_ckpt_path):
            self.load_checkpoint(self.best_ckpt_path, load_optimizer=False)

        test_metrics = self.test()
        self.history["test"] = test_metrics

        summary = {"best_epoch": best_epoch, "best_monitor": best_monitor, "test": test_metrics}
        with open(os.path.join(self.cfg.work_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        self.writer.add_text("summary", json.dumps(summary, indent=2), 0)
        self.writer.flush()
        self.writer.close()

        return self.history


__all__ = [
    "Trainer",
    "EarlyStopping",
    "binary_accuracy_from_probs",
    "binary_precision_recall_f1_from_probs",
    "sigmoid_logits_to_probs",
]
