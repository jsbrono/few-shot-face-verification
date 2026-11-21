"""Synthetic datasets in the CASIA-WebFace and LFW folder layouts.

Each identity's images share a distinct colour (plus noise), so they are easy to tell apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

IMAGE_SIZE = 24


def _write_image(path: Path, colour: np.ndarray, rng: np.random.Generator) -> None:
    pixels = colour + rng.normal(0, 10, (IMAGE_SIZE, IMAGE_SIZE, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(path)


def _colour(index: int) -> np.ndarray:
    return np.random.default_rng(index).uniform(0, 255, 3)


def make_identity_folder(root: Path, image_counts: list[int], seed: int = 0) -> Path:
    """``root/person_<i>/<j>.png`` with ``image_counts[i]`` images for person ``i``."""
    rng = np.random.default_rng(seed)
    for person, count in enumerate(image_counts):
        for j in range(count):
            _write_image(root / f"person_{person:03d}" / f"{j}.png", _colour(person), rng)
    return root


def make_lfw(root: Path, num_people: int = 6, images_per_person: int = 3) -> tuple[Path, Path]:
    """LFW-style images (``<name>/<name>_0001.jpg``) and a 2-fold ``pairs.txt`` with 3 matched
    and 3 mismatched pairs per fold."""
    rng = np.random.default_rng(1)
    names = [f"Person_{chr(65 + i)}" for i in range(num_people)]
    for i, name in enumerate(names):
        for j in range(1, images_per_person + 1):
            _write_image(root / "images" / name / f"{name}_{j:04d}.jpg", _colour(100 + i), rng)

    lines = ["2\t3"]
    for fold in range(2):
        for k in range(3):
            name = names[(fold * 3 + k) % num_people]
            lines.append(f"{name}\t1\t{2 + k % 2}")
        for k in range(3):
            a, b = names[k], names[(k + 1 + fold) % num_people]
            lines.append(f"{a}\t1\t{b}\t{1 + k % images_per_person}")
    pairs = root / "pairs.txt"
    pairs.write_text("\n".join(lines) + "\n")
    return root / "images", pairs


@pytest.fixture
def identity_root(tmp_path: Path) -> Path:
    return make_identity_folder(tmp_path / "casia", [6, 5, 4, 4, 3, 3, 2, 1])


@pytest.fixture
def lfw(tmp_path: Path) -> tuple[Path, Path]:
    return make_lfw(tmp_path / "lfw")
