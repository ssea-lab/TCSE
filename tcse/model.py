from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCSEModel(nn.Module):
    """
    Temporal Causal Sequential Embedding.

    The model keeps two channels per entity:
    - interest channel (captures intent-driven signals)
    - exposure channel (captures popularity bias)

    Optional text features are injected via small projection heads.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        int_weight: float = 1.0,
        pop_weight: float = 1.0,
        discrepancy_penalty: float = 0.01,
        temporal_weight: float = 0.0,
        temporal_weight_mode: str = "none",
        temporal_weight_alpha: float = 0.0,
        time_splits: int = 0,
        use_temporal_prototypes: bool = False,
        item_text_tensor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.users_int = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.users_pop = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.items_int = nn.Parameter(torch.empty(num_items, embedding_dim))
        self.items_pop = nn.Parameter(torch.empty(num_items, embedding_dim))

        self.int_weight = int_weight
        self.pop_weight = pop_weight
        self.discrepancy_penalty = discrepancy_penalty

        self.temporal_weight = temporal_weight
        self.temporal_weight_mode = temporal_weight_mode
        self.temporal_weight_alpha = temporal_weight_alpha
        self.time_splits = time_splits

        self.use_item_text = item_text_tensor is not None
        if self.use_item_text:
            if item_text_tensor.dim() != 2:
                raise ValueError("Item text embeddings must be 2-D")
            if item_text_tensor.size(0) != num_items:
                raise ValueError("Item text embeddings must align with num_items")
            self.item_text_dim = item_text_tensor.size(1)
            self.item_text_embedding = nn.Embedding.from_pretrained(
                item_text_tensor.float(), freeze=False
            )
            self.item_text_proj_int = nn.Linear(self.item_text_dim, embedding_dim)
            self.item_text_proj_pop = nn.Linear(self.item_text_dim, embedding_dim)
            nn.init.xavier_uniform_(self.item_text_proj_int.weight)
            nn.init.zeros_(self.item_text_proj_int.bias)
            nn.init.xavier_uniform_(self.item_text_proj_pop.weight)
            nn.init.zeros_(self.item_text_proj_pop.bias)
        else:
            self.item_text_embedding = None

        self.use_temporal_prototypes = use_temporal_prototypes and time_splits > 0
        if self.use_temporal_prototypes:
            self.proto_int = nn.Embedding(time_splits, embedding_dim)
            self.proto_pop = nn.Embedding(time_splits, embedding_dim)
            nn.init.zeros_(self.proto_int.weight)
            nn.init.zeros_(self.proto_pop.weight)

        self._init_params()

    def _init_params(self) -> None:
        stdv = 1.0 / math.sqrt(self.users_int.size(1))
        for param in [self.users_int, self.users_pop, self.items_int, self.items_pop]:
            param.data.uniform_(-stdv, stdv)

    def forward(self, batch):
        user, pos, neg, mask, pos_period, neg_period = batch
        users_int = self.users_int[user]
        users_pop = self.users_pop[user]
        pos_int = self.items_int[pos]
        pos_pop = self.items_pop[pos]
        neg_int = self.items_int[neg]
        neg_pop = self.items_pop[neg]

        if self.use_item_text:
            pos_text = self.item_text_embedding(pos)
            neg_text = self.item_text_embedding(neg)
            pos_int = pos_int + self.item_text_proj_int(pos_text)
            pos_pop = pos_pop + self.item_text_proj_pop(pos_text)
            neg_int = neg_int + self.item_text_proj_int(neg_text)
            neg_pop = neg_pop + self.item_text_proj_pop(neg_text)

        pos_weight = self._temporal_weight(pos_period)
        neg_weight = self._temporal_weight(neg_period)

        p_int = torch.sum(users_int * pos_int, dim=-1)
        n_int = torch.sum(users_int * neg_int, dim=-1)
        p_pop = torch.sum(users_pop * pos_pop, dim=-1)
        n_pop = torch.sum(users_pop * neg_pop, dim=-1)

        mask = mask.float()
        loss_int = self._weighted_bpr(p_int, n_int, mask, pos_weight, neg_weight)
        loss_pop = self._weighted_bpr(
            n_pop, p_pop, 1.0 - mask, pos_weight, neg_weight
        ) + self._weighted_bpr(p_pop, n_pop, mask, pos_weight, neg_weight)


        loss = self.int_weight * loss_int + self.pop_weight * loss_pop

        if self.use_temporal_prototypes and self.temporal_weight > 0.0:
            proto_loss = self._prototype_loss(pos_int, pos_pop, pos_period)
            loss = loss + self.temporal_weight * proto_loss

        return loss

    def _temporal_weight(self, period: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            self.temporal_weight_mode == "none"
            or self.time_splits <= 1
            or period is None
        ):
            return None
        period = period.float()
        norm = period / max(self.time_splits - 1, 1)
        centered = norm - 0.5
        if self.temporal_weight_mode == "linear":
            weights = 1.0 + self.temporal_weight_alpha * centered
        elif self.temporal_weight_mode == "exp":
            weights = torch.exp(self.temporal_weight_alpha * centered)
        else:
            return None
        return torch.clamp(weights, min=0.0)

    def _weighted_bpr(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        mask: torch.Tensor,
        pos_weight: Optional[torch.Tensor],
        neg_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        logits = pos_score - neg_score
        if pos_weight is not None and neg_weight is not None:
            logits = logits * (pos_weight + 1e-6) / (neg_weight + 1e-6)
        mask = torch.clamp(mask, min=0.0, max=1.0)
        loss = -torch.log(torch.sigmoid(logits) + 1e-8)
        loss = loss * mask
        return loss.mean()

    def _prototype_loss(
        self, pos_int: torch.Tensor, pos_pop: torch.Tensor, period: torch.Tensor
    ) -> torch.Tensor:
        valid = period >= 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=pos_int.device)
        period = period[valid].long()
        proto_int = self.proto_int(period)
        proto_pop = self.proto_pop(period)
        int_loss = F.mse_loss(pos_int[valid], proto_int)
        pop_loss = F.mse_loss(pos_pop[valid], proto_pop)
        return int_loss + pop_loss

    def user_embedding(self) -> torch.Tensor:
        return torch.cat([self.users_int, self.users_pop], dim=-1)

    def item_embedding(self) -> torch.Tensor:
        items_int = self.items_int
        items_pop = self.items_pop
        if self.use_item_text:
            text = self.item_text_embedding.weight
            items_int = items_int + self.item_text_proj_int(text)
            items_pop = items_pop + self.item_text_proj_pop(text)
        return torch.cat([items_int, items_pop], dim=-1)

    def full_scores(self) -> torch.Tensor:
        return torch.matmul(self.user_embedding(), self.item_embedding().T)
