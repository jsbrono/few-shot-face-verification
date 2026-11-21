"""Train a face verifier on an identity folder (CASIA-WebFace).

Usage::

    python -m faceverify.train --config configs/resnet18-triplet.yaml
    python -m faceverify.train --config configs/resnet18-differential.yaml --epochs 2

Writes to ``runs/<config>-seed<seed>/`` (or ``--out``): the resolved ``config.yaml``,
``metrics.json`` (per epoch: training loss and validation AUC on held-out identities) and
``model.pt`` (the epoch with the best validation AUC). If the LFW files exist, the saved model
is then evaluated (see :mod:`faceverify.evaluate`).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from faceverify import evaluate
from faceverify.config import Config, load_config, save_checkpoint
from faceverify.data import (
    EpisodeDataset,
    Identities,
    IdentityBatchSampler,
    ImageDataset,
    build_transform,
    scan_identities,
    split_identities,
)
from faceverify.losses import BatchHardTripletLoss, ContrastiveLoss
from faceverify.models import Verifier


def build_train_loader(config: Config, identities: Identities) -> tuple[DataLoader, object]:
    """Returns the loader and the object whose ``set_epoch`` reshuffles it each epoch."""
    train, data = config.train, config.data
    transform = build_transform(data.image_size, train=True)
    common = {"num_workers": train.num_workers, "pin_memory": torch.cuda.is_available()}

    if train.objective == "bce":
        episodes = EpisodeDataset(
            identities, transform, train.episodes_per_epoch, 1, train.max_refs, seed=train.seed
        )
        # No persistent workers: each worker holds its own copy of the dataset, so set_epoch
        # only reaches them if they are recreated every epoch.
        loader = DataLoader(episodes, batch_size=train.batch_size, drop_last=True, **common)
        return loader, episodes

    images = ImageDataset(identities, transform)
    sampler = IdentityBatchSampler(
        images.indices_by_label,
        train.identities_per_batch,
        train.images_per_identity,
        train.batches_per_epoch,
        seed=train.seed,
    )
    # The sampler runs in the main process, so persistent workers are safe here.
    loader = DataLoader(
        images, batch_sampler=sampler, persistent_workers=train.num_workers > 0, **common
    )
    return loader, sampler


def make_loss(config: Config) -> nn.Module:
    objective, margin = config.train.objective, config.train.margin
    if objective == "triplet":
        return BatchHardTripletLoss(margin)
    if objective == "contrastive":
        return ContrastiveLoss(margin)
    return nn.BCEWithLogitsLoss()


def train_one_epoch(
    model: Verifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    """Returns the mean training loss."""
    model.train()
    total, batches = 0.0, 0
    for batch in loader:
        with torch.autocast(device.type, enabled=mixed_precision):
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                references, query, labels, mask = (t.to(device) for t in batch)
                logits = model(references, query, mask)
                loss = loss_fn(logits.float(), labels)
            else:
                images, labels = (t.to(device) for t in batch)
                loss = loss_fn(model.embedder(images).float(), labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        batches += 1
    return total / batches


def fit(
    model: Verifier,
    config: Config,
    train_identities: Identities,
    val_identities: Identities,
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, float]]:
    """Trains, validates on held-out identities after every epoch and keeps the best model in
    ``out_dir/model.pt``. Returns per-epoch metrics."""
    train = config.train
    loader, reshuffle = build_train_loader(config, train_identities)
    val_set = EpisodeDataset(
        val_identities,
        build_transform(config.data.image_size, train=False),
        train.val_episodes,
        1,
        train.max_refs,
        seed=train.seed,
    )
    loss_fn = make_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.lr, weight_decay=train.weight_decay)
    mixed_precision = train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)
    model.to(device)

    history: list[dict[str, float]] = []
    best_auc = -1.0
    for epoch in range(1, train.epochs + 1):
        start = time.perf_counter()
        reshuffle.set_epoch(epoch)
        loss = train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, mixed_precision)
        scores, labels = evaluate.compute_scores(
            model, val_set, device, batch_size=train.batch_size, num_workers=train.num_workers
        )
        val_auc = evaluate.roc_auc(scores, labels)

        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_auc": val_auc,
                "seconds": time.perf_counter() - start,
            }
        )
        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            save_checkpoint(model, config, out_dir / "model.pt")
        print(
            f"epoch {epoch:3d}/{train.epochs}  loss {loss:.4f}  val AUC {val_auc:.4f}"
            f"{'  (saved)' if improved else ''}  ({history[-1]['seconds']:.0f}s)"
        )
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    return history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a face verifier.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--seed", type=int, help="overrides train.seed")
    parser.add_argument("--epochs", type=int, help="overrides train.epochs")
    parser.add_argument("--out", type=Path, help="output directory (default: runs/<name>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overrides = {"seed": args.seed, "epochs": args.epochs}
    config.train = replace(config.train, **{k: v for k, v in overrides.items() if v is not None})

    out_dir = args.out or Path("runs") / f"{args.config.stem}-seed{config.train.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))

    torch.manual_seed(config.train.seed)
    identities = scan_identities(config.data.casia_dir, config.data.min_images)
    train_ids, val_ids = split_identities(identities, config.data.val_fraction, config.train.seed)
    print(
        f"{len(train_ids)} training and {len(val_ids)} validation identities; writing to {out_dir}"
    )

    model = Verifier(config.model)
    fit(model, config, train_ids, val_ids, torch.device(args.device), out_dir)

    if Path(config.data.lfw_dir).is_dir() and Path(config.data.lfw_pairs).is_file():
        evaluate.main(
            [
                str(out_dir / "model.pt"),
                "--device",
                args.device,
                "--num-workers",
                str(config.train.num_workers),
            ]
        )
    else:
        print("LFW not found; skipping evaluation (see README for where to put it)")


if __name__ == "__main__":
    main()
