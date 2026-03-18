from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .data import DatasetBundle, PairwiseDataset, build_eval_dict, load_dataset_bundle
from .metrics import MetricSummary
from .model import TCSEModel


@dataclass
class TrainerConfig:
    data_root: str
    output_dir: str
    embedding_dim: int = 64
    neg_sample_rate: int = 4
    time_splits: int = 4
    use_temporal_prototypes: bool = True
    temporal_weight_mode: str = "linear"
    temporal_weight_alpha: float = 0.5
    temporal_weight: float = 0.05
    int_weight: float = 1.0
    pop_weight: float = 0.5
    discrepancy_penalty: float = 0.01
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    eval_interval: int = 5
    topk: List[int] = None
    monitor_metric: str = "recall@20"
    patience: int = 5
    use_gpu: bool = True
    device: str = "cuda:0"
    item_text_emb_path: str = ""


def load_config(path: str) -> TrainerConfig:
    with open(path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    raw.setdefault("topk", [20, 50])
    raw.setdefault("monitor_metric", "recall@20")
    return TrainerConfig(**raw)


class TCSETrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = self._resolve_device()

        data_root = Path(cfg.data_root)
        self.bundle: DatasetBundle = load_dataset_bundle(data_root, cfg.time_splits)
        self.val_dict = build_eval_dict(self.bundle.val)
        self.test_dict = build_eval_dict(self.bundle.test)

        item_text_tensor = None
        if cfg.item_text_emb_path:
            path = Path(cfg.item_text_emb_path)
            if path.exists():
                arr = np.load(path)
                item_text_tensor = torch.from_numpy(arr).float()

        self.model = TCSEModel(
            num_users=self.bundle.num_users,
            num_items=self.bundle.num_items,
            embedding_dim=cfg.embedding_dim,
            int_weight=cfg.int_weight,
            pop_weight=cfg.pop_weight,
            discrepancy_penalty=cfg.discrepancy_penalty,
            temporal_weight=cfg.temporal_weight,
            temporal_weight_mode=cfg.temporal_weight_mode,
            temporal_weight_alpha=cfg.temporal_weight_alpha,
            time_splits=cfg.time_splits,
            use_temporal_prototypes=cfg.use_temporal_prototypes,
            item_text_tensor=item_text_tensor,
        ).to(self.device)

        dataset = PairwiseDataset(
            self.bundle.train,
            num_items=self.bundle.num_items,
            time_bins=self.bundle.time_bins,
            neg_sample_rate=cfg.neg_sample_rate,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_path = self.output_dir / "tcse_best.pt"
        self.epochs_no_improve = 0
        self._log_setup()

    def _resolve_device(self) -> torch.device:
        if self.cfg.use_gpu and torch.cuda.is_available():
            return torch.device(self.cfg.device)
        return torch.device("cpu")

    def _collate(self, batch):
        user, pos, neg, mask, pos_period, neg_period = zip(*batch)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(pos_period, dtype=torch.long),
            torch.tensor(neg_period, dtype=torch.long),
        )

    def train(self):
        print("[TCSE] ===== Training started =====")
        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\n[TCSE] Epoch {epoch}/{self.cfg.epochs}")
            epoch_loss = self._run_epoch()
            log = {"epoch": epoch, "loss": epoch_loss}
            print(f"[TCSE] epoch {epoch} loss: {epoch_loss:.4f}")
            if epoch % self.cfg.eval_interval == 0 or epoch == self.cfg.epochs:
                metrics = self.evaluate(split="val")
                log.update({f"val_{k}": v for k, v in metrics.items()})
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[TCSE] validation @ epoch {epoch}: {metrics_str}")
                self._maybe_update_best(epoch, metrics)
                if self.epochs_no_improve >= self.cfg.patience:
                    print(
                        f"[TCSE] Early stopping triggered at epoch {epoch} "
                        f"(no improvement for {self.cfg.patience} evals)."
                    )
                    break
            self.history.append(log)
            self._write_history()
        self._load_best_checkpoint()
        print(
            f"[TCSE] Training finished. Best epoch={self.best_epoch}"
            f" with {self.cfg.monitor_metric}={self.best_metric:.4f}."
        )

    def _run_epoch(self) -> float:
        self.model.train()
        running = 0.0
        steps = 0
        for batch in tqdm(self.loader, desc="training", leave=False):
            batch = tuple(t.to(self.device) for t in batch)
            loss = self.model(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running += loss.item()
            steps += 1
        return running / max(steps, 1)

    def evaluate(self, split: str = "val") -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            scores = self.model.full_scores().detach().cpu().numpy()
        data = self.val_dict if split == "val" else self.test_dict
        metrics = MetricSummary(tuple(self.cfg.topk))
        max_k = max(self.cfg.topk)
        for user, positives in data.items():
            ranked = self._topk(scores[user], max_k)
            metrics.update(ranked, positives)
        summary = metrics.aggregate()
        split_name = "validation" if split == "val" else "test"
        summary_str = ", ".join(f"{k}={v:.4f}" for k, v in summary.items())
        print(f"[TCSE] {split_name} metrics: {summary_str}")
        return summary

    @staticmethod
    def _topk(user_scores: np.ndarray, k: int) -> List[int]:
        if k >= len(user_scores):
            order = np.argsort(-user_scores)
            return order.tolist()
        idx = np.argpartition(-user_scores, k)[:k]
        idx = idx[np.argsort(-user_scores[idx])]
        return idx.tolist()

    def _maybe_update_best(self, epoch: int, metrics: Dict[str, float]) -> None:
        monitor_value = metrics.get(self.cfg.monitor_metric)
        if monitor_value is None:
            print(f"[TCSE] Warning: metric {self.cfg.monitor_metric} not found.")
            return
        if monitor_value > self.best_metric:
            self.best_metric = monitor_value
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), self.best_path)
            print(
                f"[TCSE] New best {self.cfg.monitor_metric}={monitor_value:.4f} "
                f"at epoch {epoch}, checkpoint saved."
            )
        else:
            self.epochs_no_improve += 1

    def _load_best_checkpoint(self) -> None:
        if self.best_path.exists():
            state = torch.load(self.best_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(
                f"[TCSE] Loaded best checkpoint from epoch {self.best_epoch} "
                f"({self.cfg.monitor_metric}={self.best_metric:.4f})."
            )

    def _write_history(self) -> None:
        path = self.output_dir / "training_log.json"
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.history, fp, indent=2)

    def _log_setup(self) -> None:
        train_size = len(self.bundle.train)
        val_size = len(self.bundle.val)
        test_size = len(self.bundle.test)
        print("[TCSE] ===== Run configuration =====")
        print(f"[TCSE] data_root: {self.cfg.data_root}")
        print(f"[TCSE] output_dir: {self.cfg.output_dir}")
        print(
            f"[TCSE] users={self.bundle.num_users}, items={self.bundle.num_items},"
            f" train={train_size}, val={val_size}, test={test_size}"
        )
        print(
            f"[TCSE] time_splits={self.cfg.time_splits}, neg_sample_rate={self.cfg.neg_sample_rate}"
        )
        print(
            f"[TCSE] weights -> int={self.cfg.int_weight}, pop={self.cfg.pop_weight},"
            f" temporal={self.cfg.temporal_weight} ({self.cfg.temporal_weight_mode})"
        )
        print(
            f"[TCSE] optimizer -> lr={self.cfg.lr}, batch_size={self.cfg.batch_size},"
            f" epochs={self.cfg.epochs}, eval_interval={self.cfg.eval_interval}"
        )
        print(
            f"[TCSE] early-stop -> monitor={self.cfg.monitor_metric}, patience={self.cfg.patience}"
        )
        print("[TCSE] =================================")
