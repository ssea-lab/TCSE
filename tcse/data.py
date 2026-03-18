from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


COLUMNS = ["uid", "iid", "ts"]


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    num_users: int
    num_items: int
    time_bins: Optional[np.ndarray]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[COLUMNS].copy()


def build_time_bins(timestamps: np.ndarray, splits: int) -> Optional[np.ndarray]:
    if splits <= 1 or timestamps.size == 0:
        return None
    lo, hi = float(timestamps.min()), float(timestamps.max())
    if math.isclose(lo, hi):
        return None
    return np.linspace(lo, hi, splits + 1, dtype=np.float64)


def assign_time_period(ts: float, bins: Optional[np.ndarray]) -> int:
    if bins is None:
        return -1
    idx = np.searchsorted(bins, ts, side="right") - 1
    idx = max(0, min(len(bins) - 2, idx))
    return int(idx)


def load_dataset_bundle(data_root: Path, time_splits: int) -> DatasetBundle:
    train = load_csv(data_root / "train_record.csv")
    val = load_csv(data_root / "val_record.csv")
    test = load_csv(data_root / "test_record.csv")

    num_users = int(max(train["uid"].max(), val["uid"].max(), test["uid"].max())) + 1
    num_items = int(max(train["iid"].max(), val["iid"].max(), test["iid"].max())) + 1

    time_bins = build_time_bins(train["ts"].to_numpy(), time_splits)
    return DatasetBundle(train, val, test, num_users, num_items, time_bins)


class PairwiseDataset(IterableDataset):
    """Generates (user, pos, neg, mask, pos_period, neg_period) tuples."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        num_items: int,
        time_bins: Optional[np.ndarray],
        neg_sample_rate: int = 4,
        interest_quantile: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.num_items = num_items
        self.time_bins = time_bins
        self.neg_sample_rate = max(1, neg_sample_rate)
        self.seed = seed

        grouped = interactions.groupby("uid")
        self.user_items: Dict[int, List[Tuple[int, float]]] = {
            int(uid): list(zip(group["iid"].tolist(), group["ts"].tolist()))
            for uid, group in grouped
        }
        self.user_item_sets: Dict[int, set] = {
            uid: {item for item, _ in pairs} for uid, pairs in self.user_items.items()
        }

        counts = interactions.groupby("iid").size()
        threshold = counts.quantile(interest_quantile) if not counts.empty else 0
        self.item_interest_mask: Dict[int, bool] = {
            int(iid): freq <= threshold for iid, freq in counts.items()
        }
        self.timestamps = interactions["ts"].to_numpy()

    def __iter__(self) -> Iterable[Tuple[int, int, int, float, int, int]]:
        worker = torch.utils.data.get_worker_info()
        seed = self.seed if worker is None else self.seed + worker.id
        rng = random.Random(seed)
        users = list(self.user_items.keys())
        rng.shuffle(users)
        for uid in users:
            history = sorted(self.user_items[uid], key=lambda x: x[1])
            seen = self.user_item_sets[uid]
            for item, ts in history:
                pos_period = assign_time_period(ts, self.time_bins)
                mask = 1.0 if self.item_interest_mask.get(item, True) else 0.0
                for _ in range(self.neg_sample_rate):
                    neg = self._sample_negative(rng, seen)
                    neg_ts = self._sample_timestamp(rng)
                    neg_period = assign_time_period(neg_ts, self.time_bins)
                    yield uid, item, neg, mask, pos_period, neg_period

    def _sample_negative(self, rng: random.Random, seen: set) -> int:
        while True:
            candidate = rng.randint(0, self.num_items - 1)
            if candidate not in seen:
                return candidate

    def _sample_timestamp(self, rng: random.Random) -> float:
        if len(self.timestamps) == 0:
            return 0.0
        idx = rng.randint(0, len(self.timestamps) - 1)
        return float(self.timestamps[idx])


def build_eval_dict(df: pd.DataFrame) -> Dict[int, List[int]]:
    grouped = df.groupby("uid")["iid"].apply(list)
    return {int(uid): [int(i) for i in items] for uid, items in grouped.items()}
