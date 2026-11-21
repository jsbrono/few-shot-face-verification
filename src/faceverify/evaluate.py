"""Verification metrics and the evaluation CLI.

Usage::

    python -m faceverify.evaluate runs/resnet18-triplet-seed0/model.pt
    python -m faceverify.evaluate runs/resnet18-triplet-seed0/model.pt --errors results/errors.png

Reports, as JSON (also written next to the checkpoint as ``eval.json``):

- LFW: 10-fold accuracy (the threshold for each fold is chosen on the other nine), ROC AUC
  and TAR at 1% FAR.
- Held-out CASIA identities: ROC AUC with 1, 3 and 5 reference images, which shows what
  averaging several references gains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from faceverify.config import Config, load_checkpoint
from faceverify.data import (
    EpisodeDataset,
    LFWPairs,
    build_transform,
    scan_identities,
    split_identities,
)
from faceverify.models import Verifier

# --- metrics -------------------------------------------------------------------------------


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The threshold ``t`` maximising the accuracy of ``score >= t``.

    Every distinct cut of the sorted scores is tried; the threshold is the midpoint between the
    scores either side of the best cut.
    """
    order = np.argsort(scores, kind="stable")
    s, y = scores[order], labels[order].astype(bool)
    # Predicting "positive" from sorted position k onwards: correct = negatives before k +
    # positives from k on.
    negatives_before = np.concatenate([[0], np.cumsum(~y)])
    positives_from = np.concatenate([np.cumsum(y[::-1])[::-1], [0]])
    correct = negatives_before + positives_from
    # A cut between two equal scores cannot be expressed as a threshold.
    realisable = np.concatenate([[True], s[1:] != s[:-1], [True]])
    k = int(np.argmax(np.where(realisable, correct, -1)))
    if k == 0:
        return float(s[0]) - 1.0
    if k == len(s):
        return float(s[-1]) + 1.0
    return float((s[k - 1] + s[k]) / 2)


def accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    return float(((scores >= threshold) == labels.astype(bool)).mean())


def k_fold_accuracy(
    scores: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> tuple[float, float]:
    """Mean and standard deviation of per-fold accuracy, each fold's threshold chosen on the
    remaining folds (the LFW protocol). Choosing it on the test fold itself would overstate
    accuracy."""
    accuracies = []
    for fold in np.unique(folds):
        test = folds == fold
        threshold = best_threshold(scores[~test], labels[~test])
        accuracies.append(accuracy(scores[test], labels[test], threshold))
    return float(np.mean(accuracies)), float(np.std(accuracies))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: the probability that a random positive outscores a random
    negative (ties count half). Computed from ranks (Mann-Whitney U)."""
    labels = labels.astype(bool)
    num_pos, num_neg = labels.sum(), (~labels).sum()
    if num_pos == 0 or num_neg == 0:
        raise ValueError("ROC AUC needs both positive and negative examples")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inverse, weights=ranks) / counts)[inverse]  # average ranks of ties
    return float((ranks[labels].sum() - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg))


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float) -> float:
    """True accept rate at the strictest threshold that accepts at most ``far`` of negatives."""
    labels = labels.astype(bool)
    negatives = np.sort(scores[~labels])[::-1]
    allowed = int(np.floor(far * len(negatives)))
    threshold = negatives[allowed] if allowed < len(negatives) else -np.inf
    return float((scores[labels] > threshold).mean())


# --- scoring ---------------------------------------------------------------------------------


@torch.no_grad()
def compute_scores(
    model: Verifier,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scores every ``(references, query, label, mask)`` item; returns ``(scores, labels)``."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    scores, labels = [], []
    for references, query, label, mask in loader:
        scores.append(model(references.to(device), query.to(device), mask.to(device)).cpu())
        labels.append(label)
    return torch.cat(scores).float().numpy(), torch.cat(labels).numpy()


def lfw_metrics(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict:
    mean, std = k_fold_accuracy(scores, labels, folds)
    return {
        "accuracy": mean,
        "accuracy_std": std,
        "auc": roc_auc(scores, labels),
        "tar_at_far_1pct": tar_at_far(scores, labels, 0.01),
    }


def evaluate_lfw(model: Verifier, dataset: LFWPairs, device: torch.device, **kwargs) -> dict:
    scores, labels = compute_scores(model, dataset, device, **kwargs)
    return lfw_metrics(scores, labels, dataset.folds.numpy())


def evaluate_episodes(
    model: Verifier,
    identities: dict,
    config: Config,
    device: torch.device,
    reference_counts: tuple[int, ...] = (1, 3, 5),
    num_episodes: int = 2_000,
    **kwargs,
) -> dict:
    """ROC AUC on held-out identities, for each fixed number of references."""
    transform = build_transform(config.data.image_size, train=False)
    results = {}
    for k in reference_counts:
        try:
            episodes = EpisodeDataset(
                identities, transform, num_episodes, k, k, seed=config.train.seed
            )
        except ValueError:
            continue  # too few identities with more than k images
        scores, labels = compute_scores(model, episodes, device, **kwargs)
        results[f"{k}_refs"] = {"auc": roc_auc(scores, labels)}
    return results


def plot_errors(
    dataset: LFWPairs, scores: np.ndarray, threshold: float, out: Path, count: int = 8
) -> None:
    """The ``count`` most confident mistakes on LFW, each pair side by side."""
    labels = np.array([pair.label for pair in dataset.pairs])
    wrong = np.flatnonzero((scores >= threshold) != labels.astype(bool))
    worst = wrong[np.argsort(-np.abs(scores[wrong] - threshold))][:count]
    if len(worst) == 0:
        return

    fig = Figure(figsize=(4, 2 * len(worst)))
    axes = fig.subplots(len(worst), 2, squeeze=False)
    for row, index in enumerate(worst):
        pair = dataset.pairs[index]
        verdict = "missed match" if pair.label else "false match"
        for col, path in enumerate((pair.first, pair.second)):
            axes[row, col].imshow(read_rgb(path))
            axes[row, col].axis("off")
        axes[row, 0].set_title(f"{verdict}, score {scores[index]:.2f}", fontsize=8, loc="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# --- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained verifier.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--lfw-dir", help="overrides data.lfw_dir")
    parser.add_argument("--lfw-pairs", help="overrides data.lfw_pairs")
    parser.add_argument("--casia-dir", help="overrides data.casia_dir")
    parser.add_argument("--skip-episodes", action="store_true", help="LFW only")
    parser.add_argument("--errors", type=Path, help="save the worst LFW mistakes to this image")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    model, config = load_checkpoint(args.checkpoint)
    data = config.data
    device = torch.device(args.device)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    transform = build_transform(data.image_size, train=False, center_crop=data.lfw_center_crop)
    lfw = LFWPairs(args.lfw_dir or data.lfw_dir, args.lfw_pairs or data.lfw_pairs, transform)
    scores, labels = compute_scores(model, lfw, device, **loader_args)
    results = {"lfw": lfw_metrics(scores, labels, lfw.folds.numpy())}
    if args.errors:
        plot_errors(lfw, scores, best_threshold(scores, labels), args.errors)

    if not args.skip_episodes:
        # Recreate the training run's split, so these identities were never trained on.
        identities = scan_identities(args.casia_dir or data.casia_dir, data.min_images)
        _, val = split_identities(identities, data.val_fraction, config.train.seed)
        results["held_out_identities"] = evaluate_episodes(
            model, val, config, device, **loader_args
        )

    text = json.dumps(results, indent=2)
    print(text)
    (args.checkpoint.parent / "eval.json").write_text(text)


if __name__ == "__main__":
    main()
