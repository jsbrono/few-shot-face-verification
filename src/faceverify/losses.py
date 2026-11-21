"""Metric-learning losses over a batch of embeddings with identity labels.

Both losses use every pair in the batch rather than pre-built pairs or triplets, so a batch of
P identities x K images (see :class:`faceverify.data.IdentityBatchSampler`) gives
O((PK)^2) pairs for the price of PK forward passes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Euclidean distance matrix, safe to backpropagate through at zero distance.

    Differences are computed directly rather than as ``|a|^2 + |b|^2 - 2ab``, which loses
    precision to cancellation for near-identical embeddings (batches are small, so the
    ``(n, n, dim)`` intermediate is cheap). ``sqrt`` has an infinite gradient at 0, which the
    diagonal always hits, so squared distances are clamped away from zero first.
    """
    dist_sq = (embeddings.unsqueeze(1) - embeddings.unsqueeze(0)).pow(2).sum(dim=-1)
    return dist_sq.clamp_min(1e-12).sqrt()


def _pair_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    return same & ~eye, ~same


class ContrastiveLoss(nn.Module):
    """Contrastive loss (Hadsell, Chopra & LeCun, 2006) over all pairs in the batch.

    Same-identity pairs are pulled together (``d^2``); different-identity pairs are pushed at
    least ``margin`` apart (``max(0, margin - d)^2``). The two terms are averaged separately and
    added, because a P x K batch has far more negative pairs than positive ones.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = pairwise_distances(embeddings)
        positive, negative = _pair_masks(labels)
        loss = embeddings.new_zeros(())
        if positive.any():
            loss = loss + dist[positive].pow(2).mean()
        if negative.any():
            loss = loss + F.relu(self.margin - dist[negative]).pow(2).mean()
        return loss


class BatchHardTripletLoss(nn.Module):
    """Triplet loss with batch-hard mining (Hermans, Beyer & Leibe, 2017).

    Each image in the batch is an anchor. Its triplet uses the *furthest* image of the same
    identity and the *closest* image of another identity, and costs
    ``max(0, d(a, p) - d(a, n) + margin)``. Anchors without a positive or a negative in the
    batch are skipped.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = pairwise_distances(embeddings)
        positive, negative = _pair_masks(labels)
        hardest_positive = dist.masked_fill(~positive, float("-inf")).amax(dim=1)
        hardest_negative = dist.masked_fill(~negative, float("inf")).amin(dim=1)

        valid = positive.any(dim=1) & negative.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0  # zero, but still connected to the graph
        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        return losses[valid].mean()
