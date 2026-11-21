"""Face datasets, identity splits and samplers.

Training data is a folder of identities, ``root/<identity>/<image>`` (CASIA-WebFace's layout).
Evaluation uses LFW's standard pairs protocol (:class:`LFWPairs`) and multi-reference episodes
on held-out identities (:class:`EpisodeDataset`).
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

Identities = dict[str, list[Path]]


def build_transform(
    image_size: int, train: bool, center_crop: int | None = None
) -> transforms.Compose:
    """Resize (optionally after a centre crop), light augmentation for training, ImageNet
    normalisation (the pretrained backbone expects it; harmless for the small CNN)."""
    ops: list = []
    if center_crop:
        ops.append(transforms.CenterCrop(center_crop))
    ops.append(transforms.Resize((image_size, image_size)))
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def load_image(path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def scan_identities(root: str | Path, min_images: int = 1) -> Identities:
    """Maps each identity folder under ``root`` to its sorted image paths.

    Identities with fewer than ``min_images`` images are dropped. Sorting makes the result
    independent of filesystem order.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no such directory: {root}")
    identities = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) >= min_images:
            identities[folder.name] = images
    if not identities:
        raise ValueError(f"no identities with at least {min_images} images in {root}")
    return identities


def split_identities(
    identities: Identities, val_fraction: float, seed: int = 0
) -> tuple[Identities, Identities]:
    """Splits by identity, so no person appears in both halves. Deterministic for a seed."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    names = sorted(identities)
    random.Random(seed).shuffle(names)
    num_val = max(1, round(len(names) * val_fraction))
    val, train = names[:num_val], names[num_val:]
    if not train:
        raise ValueError("val_fraction leaves no training identities")
    return {n: identities[n] for n in sorted(train)}, {n: identities[n] for n in sorted(val)}


class ImageDataset(Dataset):
    """Every image of every identity, as ``(image, label)`` with labels ``0 .. n-1``."""

    def __init__(self, identities: Identities, transform: transforms.Compose):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.indices_by_label: list[list[int]] = []
        for label, paths in enumerate(identities.values()):
            start = len(self.samples)
            self.samples += [(path, label) for path in paths]
            self.indices_by_label.append(list(range(start, len(self.samples))))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return load_image(path, self.transform), label


class IdentityBatchSampler(Sampler[list[int]]):
    """Yields batches of ``identities_per_batch`` (P) identities x ``images_per_identity`` (K)
    images, the batch structure metric-learning losses need: every image has positives and
    negatives in its batch.

    Identities with fewer than K images contribute repeated images. Call :meth:`set_epoch`
    before each epoch to get a new, reproducible order.
    """

    def __init__(
        self,
        indices_by_label: list[list[int]],
        identities_per_batch: int,
        images_per_identity: int,
        num_batches: int | None = None,
        seed: int = 0,
    ):
        if identities_per_batch > len(indices_by_label):
            raise ValueError(
                f"identities_per_batch ({identities_per_batch}) exceeds the number of "
                f"identities ({len(indices_by_label)})"
            )
        if images_per_identity < 2:
            raise ValueError("images_per_identity must be at least 2 to form positive pairs")
        self.indices_by_label = indices_by_label
        self.p = identities_per_batch
        self.k = images_per_identity
        total_images = sum(len(indices) for indices in indices_by_label)
        self.num_batches = num_batches or max(1, total_images // (self.p * self.k))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        for _ in range(self.num_batches):
            labels = torch.randperm(len(self.indices_by_label), generator=generator)[: self.p]
            batch = []
            for label in labels.tolist():
                indices = self.indices_by_label[label]
                if len(indices) >= self.k:
                    picks = torch.randperm(len(indices), generator=generator)[: self.k]
                else:
                    picks = torch.randint(len(indices), (self.k,), generator=generator)
                batch += [indices[i] for i in picks.tolist()]
            yield batch


@dataclass(frozen=True)
class Episode:
    references: list[Path]
    query: Path
    label: int  # 1: query is the same identity as the references


class EpisodeDataset(Dataset):
    """Few-shot verification episodes: between ``min_refs`` and ``max_refs`` reference images
    of one identity, and a query that is either another image of that identity (label 1) or an
    image of a different identity (label 0). Even indices are positive, odd negative.

    Items are ``(references, query, label, mask)``, with references zero-padded to
    ``max_refs``. Each episode is generated from ``(seed, epoch, index)``: reproducible, but
    different every epoch after :meth:`set_epoch`.
    """

    def __init__(
        self,
        identities: Identities,
        transform: transforms.Compose,
        num_episodes: int,
        min_refs: int = 1,
        max_refs: int = 5,
        seed: int = 0,
    ):
        if not 1 <= min_refs <= max_refs:
            raise ValueError("need 1 <= min_refs <= max_refs")
        # A positive needs min_refs references plus a distinct query image.
        self.identities = [paths for paths in identities.values() if len(paths) > min_refs]
        if len(self.identities) < 2:
            raise ValueError(f"need at least 2 identities with more than {min_refs} images")
        self.transform = transform
        self.num_episodes = num_episodes
        self.min_refs = min_refs
        self.max_refs = max_refs
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_episodes

    def episode(self, index: int) -> Episode:
        rng = random.Random(f"{self.seed}-{self.epoch}-{index}")
        identity = rng.randrange(len(self.identities))
        images = rng.sample(self.identities[identity], len(self.identities[identity]))
        positive = index % 2 == 0

        if positive:
            query, pool = images[0], images[1:]
        else:
            other = rng.choice([i for i in range(len(self.identities)) if i != identity])
            query, pool = rng.choice(self.identities[other]), images
        num_refs = rng.randint(self.min_refs, min(self.max_refs, len(pool)))
        return Episode(pool[:num_refs], query, int(positive))

    def __getitem__(self, index: int):
        episode = self.episode(index)
        query = load_image(episode.query, self.transform)
        references = torch.zeros(self.max_refs, *query.shape)
        mask = torch.zeros(self.max_refs)
        for i, path in enumerate(episode.references):
            references[i] = load_image(path, self.transform)
            mask[i] = 1
        return references, query, torch.tensor(float(episode.label)), mask


@dataclass(frozen=True)
class Pair:
    first: Path
    second: Path
    label: int
    fold: int


def parse_lfw_pairs(pairs_file: str | Path, image_root: str | Path) -> list[Pair]:
    """Reads LFW's View 2 pairs list: 10 folds of 300 matched then 300 mismatched pairs.

    Accepts the official whitespace-separated ``pairs.txt`` (first line ``"10 300"``) and the
    comma-separated ``pairs.csv`` from the Kaggle mirror (a header row, possibly trailing
    commas). Matched rows are ``name i j``, mismatched rows ``name1 i name2 j``; image ``i`` of
    ``name`` is ``<name>/<name>_<i:04d>.jpg``.
    """
    image_root = Path(image_root)
    rows = []
    for line in Path(pairs_file).read_text().splitlines():
        fields = [f for f in re.split(r"[,\s]+", line.strip()) if f]
        if fields:
            rows.append(fields)

    pairs_per_fold = 300
    if len(rows[0]) == 2 and all(f.isdigit() for f in rows[0]):
        pairs_per_fold = int(rows[0][1])
        rows = rows[1:]
    rows = [r for r in rows if r[1].isdigit()]  # drops a CSV header

    def image(name: str, number: str) -> Path:
        return image_root / name / f"{name}_{int(number):04d}.jpg"

    pairs = []
    for i, row in enumerate(rows):
        fold = i // (2 * pairs_per_fold)
        if len(row) == 3:
            pairs.append(Pair(image(row[0], row[1]), image(row[0], row[2]), 1, fold))
        elif len(row) == 4:
            pairs.append(Pair(image(row[0], row[1]), image(row[2], row[3]), 0, fold))
        else:
            raise ValueError(f"malformed pairs row {i + 1}: {row}")
    if len(pairs) % (2 * pairs_per_fold) != 0:
        raise ValueError(f"{len(pairs)} pairs is not a whole number of folds")
    return pairs


class LFWPairs(Dataset):
    """LFW pairs as single-reference episodes: ``(references, query, label, mask)``, with
    ``references`` of shape ``(1, 3, H, W)``."""

    def __init__(
        self, image_root: str | Path, pairs_file: str | Path, transform: transforms.Compose
    ):
        self.pairs = parse_lfw_pairs(pairs_file, image_root)
        missing = [p for pair in self.pairs for p in (pair.first, pair.second) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} LFW images not found, e.g. {missing[0]}")
        self.transform = transform

    @property
    def folds(self) -> torch.Tensor:
        return torch.tensor([pair.fold for pair in self.pairs])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        references = load_image(pair.first, self.transform).unsqueeze(0)
        query = load_image(pair.second, self.transform)
        return references, query, torch.tensor(float(pair.label)), torch.ones(1)
