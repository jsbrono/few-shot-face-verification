"""Experiment configuration (YAML) and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
import yaml

from faceverify.models import ModelConfig, Verifier

Objective = Literal["contrastive", "triplet", "bce"]


@dataclass
class DataConfig:
    casia_dir: str = "data/casia-webface"
    lfw_dir: str = "data/lfw-deepfunneled"
    lfw_pairs: str = "data/pairs.csv"
    image_size: int = 112
    # LFW images are 250x250 with the face in the middle; crop before resizing so faces are a
    # similar scale to CASIA-WebFace's tighter crops. null to disable.
    lfw_center_crop: int | None = 150
    val_fraction: float = 0.05
    min_images: int = 2


@dataclass
class TrainConfig:
    """``contrastive`` / ``triplet`` train the embedding network on P x K identity batches;
    ``bce`` trains the whole verifier (with the differential head) on episodes."""

    objective: Objective = "triplet"
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    margin: float = 0.2
    # contrastive / triplet
    identities_per_batch: int = 32
    images_per_identity: int = 4
    batches_per_epoch: int | None = None  # default: one pass over the images
    # bce
    batch_size: int = 32
    episodes_per_epoch: int = 20_000
    # both: references per episode for bce training and for validation
    max_refs: int = 5
    val_episodes: int = 2_000
    num_workers: int = 4
    mixed_precision: bool = True
    seed: int = 0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        objective, head = self.train.objective, self.model.head
        if objective not in ("contrastive", "triplet", "bce"):
            raise ValueError(f"unknown objective {objective!r}")
        if objective == "bce" and head != "differential":
            raise ValueError("objective 'bce' trains a learned head; set model.head: differential")
        if objective != "bce" and head != "cosine":
            raise ValueError(f"objective {objective!r} trains embeddings; set model.head: cosine")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        unknown = set(raw) - {"model", "data", "train"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
        )


def load_config(path: str | Path) -> Config:
    return Config.from_dict(yaml.safe_load(Path(path).read_text()) or {})


def save_checkpoint(model: Verifier, config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "state_dict": model.state_dict()}, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[Verifier, Config]:
    """Rebuilds the model in eval mode. Pretrained weights are not re-downloaded, since the
    checkpoint replaces them."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    config = Config.from_dict(checkpoint["config"])
    model = Verifier(replace(config.model, pretrained=False))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), config
