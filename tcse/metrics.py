from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], positives: Sequence[int], k: int) -> float:
    if not positives:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for item in positives if item in topk)
    return hits / float(len(positives))


def ndcg_at_k(ranked: Sequence[int], positives: Sequence[int], k: int) -> float:
    if not positives:
        return 0.0
    gain = 0.0
    hits = 0
    for idx, item in enumerate(ranked[:k]):
        if item in positives:
            hits += 1
            gain += 1.0 / np.log2(idx + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(positives), k)))
    if ideal == 0.0:
        return 0.0
    return gain / ideal


@dataclass
class MetricSummary:
    ks: Sequence[int]
    recall: Dict[int, List[float]] = field(default_factory=dict)
    ndcg: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in self.ks:
            self.recall.setdefault(k, [])
            self.ndcg.setdefault(k, [])

    def update(self, ranked: Sequence[int], positives: Sequence[int]) -> None:
        for k in self.ks:
            self.recall[k].append(recall_at_k(ranked, positives, k))
            self.ndcg[k].append(ndcg_at_k(ranked, positives, k))

    def aggregate(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in self.ks:
            out[f"recall@{k}"] = float(np.mean(self.recall[k])) if self.recall[k] else 0.0
            out[f"ndcg@{k}"] = float(np.mean(self.ndcg[k])) if self.ndcg[k] else 0.0
        return out
