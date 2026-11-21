"""Embedding networks and verification heads.

A :class:`Verifier` answers "is the query the same person as these reference images?". It embeds
every image with a shared :class:`EmbeddingNet` (the Siamese part), then scores the query
against the references with one of two heads:

- ``"cosine"``: average the reference embeddings and take the cosine similarity with the query.
  No learned parameters; used with metric-learning objectives (contrastive, triplet).
- ``"differential"``: a small MLP on two summary features (see :class:`DifferentialHead`),
  trained end to end with binary cross-entropy.

Both accept a variable number of references per sample, given as a padded tensor and a mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

Backbone = Literal["resnet18", "small_cnn"]
Head = Literal["cosine", "differential"]


@dataclass
class ModelConfig:
    backbone: Backbone = "resnet18"
    pretrained: bool = True
    embed_dim: int = 128
    head: Head = "cosine"
    head_hidden_dim: int = 256

    def __post_init__(self) -> None:
        if self.backbone not in ("resnet18", "small_cnn"):
            raise ValueError(f"unknown backbone {self.backbone!r}")
        if self.head not in ("cosine", "differential"):
            raise ValueError(f"unknown head {self.head!r}")
        if self.pretrained and self.backbone == "small_cnn":
            raise ValueError("there are no pretrained weights for small_cnn")


def small_cnn() -> tuple[nn.Module, int]:
    """A lightweight four-block CNN (about 100k parameters, vs 11M for ResNet-18)."""
    layers: list[nn.Module] = []
    in_channels = 3
    for out_channels in (16, 32, 64, 128):
        layers += [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        in_channels = out_channels
    layers += [nn.AdaptiveAvgPool2d(4), nn.Flatten()]
    return nn.Sequential(*layers), 128 * 4 * 4


def resnet18(pretrained: bool) -> tuple[nn.Module, int]:
    """ResNet-18 without its classifier, optionally with ImageNet weights."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    out_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, out_dim


class EmbeddingNet(nn.Module):
    """Backbone, then a linear projection to ``embed_dim``, L2-normalised."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.backbone == "resnet18":
            self.backbone, features = resnet18(config.pretrained)
        else:
            self.backbone, features = small_cnn()
        self.projection = nn.Linear(features, config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(images)), dim=-1)


def mean_reference(references: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of ``(batch, refs, dim)`` reference embeddings, re-normalised to unit length."""
    weights = mask.unsqueeze(-1).to(references.dtype)
    mean = (references * weights).sum(dim=1) / weights.sum(dim=1)
    return F.normalize(mean, dim=-1)


class DifferentialHead(nn.Module):
    """Learned verification from how far the query is from the references, relative to how much
    the references vary among themselves.

    Two features, both element-wise absolute differences averaged over valid entries:

    - ``spread``: over every pair of references ``i < j``, ``|r_i - r_j|``. How much this
      person's own images vary. Zero when there is only one reference.
    - ``distance``: over every reference, ``|q - r_i|``. How far the query is from them.

    An MLP maps ``[spread, distance]`` to a logit.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, references: torch.Tensor, query: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        valid = mask.to(references.dtype)
        num_refs = references.shape[1]

        distance = (query.unsqueeze(1) - references).abs() * valid.unsqueeze(-1)
        distance = distance.sum(dim=1) / valid.sum(dim=1, keepdim=True)

        upper = torch.ones(num_refs, num_refs, device=mask.device).triu(diagonal=1)
        pair_mask = valid.unsqueeze(2) * valid.unsqueeze(1) * upper
        pairs = (references.unsqueeze(2) - references.unsqueeze(1)).abs()
        spread = (pairs * pair_mask.unsqueeze(-1)).sum(dim=(1, 2))
        spread = spread / pair_mask.sum(dim=(1, 2)).clamp_min(1).unsqueeze(-1)

        return self.mlp(torch.cat([spread, distance], dim=-1)).squeeze(-1)


class Verifier(nn.Module):
    """Scores query images against sets of reference images; higher means "same person".

    Args to :meth:`forward`:
        references: ``(batch, max_refs, 3, H, W)``, padded where a sample has fewer references.
        query: ``(batch, 3, H, W)``.
        mask: ``(batch, max_refs)``, 1 for real references and 0 for padding. Every sample needs
            at least one reference.

    Returns ``(batch,)`` scores: cosine similarities for the cosine head, logits for the
    differential head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedder = EmbeddingNet(config)
        self.head = (
            DifferentialHead(config.embed_dim, config.head_hidden_dim)
            if config.head == "differential"
            else None
        )

    def embed_references(self, references: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embeds only the real references; padded slots get zero vectors."""
        batch, max_refs = mask.shape
        valid = mask.reshape(-1).bool()
        embedded = self.embedder(references.flatten(0, 1)[valid])
        out = embedded.new_zeros(batch * max_refs, embedded.shape[-1])
        out[valid] = embedded
        return out.view(batch, max_refs, -1)

    def forward(
        self, references: torch.Tensor, query: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if (mask.sum(dim=1) < 1).any():
            raise ValueError("every sample needs at least one reference")
        refs = self.embed_references(references, mask)
        query_embedding = self.embedder(query)
        if self.head is None:
            return (mean_reference(refs, mask) * query_embedding).sum(dim=-1)
        return self.head(refs, query_embedding, mask)
