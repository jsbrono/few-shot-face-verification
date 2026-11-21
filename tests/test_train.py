import json
from pathlib import Path

import pytest
import yaml

from faceverify.config import load_checkpoint, load_config
from faceverify.models import Verifier
from faceverify.train import main

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_are_valid(path):
    config = load_config(path)
    assert config.train.epochs > 0
    if config.model.pretrained:
        config.model.pretrained = False  # don't download weights in tests
    Verifier(config.model)


def write_config(tmp_path: Path, identity_root: Path, lfw: tuple[Path, Path], **train) -> Path:
    images, pairs = lfw
    objective = train.get("objective", "triplet")
    config = {
        "model": {
            "backbone": "small_cnn",
            "pretrained": False,
            "embed_dim": 8,
            "head": "differential" if objective == "bce" else "cosine",
            "head_hidden_dim": 8,
        },
        "data": {
            "casia_dir": str(identity_root),
            "lfw_dir": str(images),
            "lfw_pairs": str(pairs),
            "image_size": 16,
            "lfw_center_crop": None,
            "val_fraction": 0.3,
        },
        "train": {
            "epochs": 2,
            "identities_per_batch": 2,
            "images_per_identity": 2,
            "batches_per_epoch": 3,
            "batch_size": 4,
            "episodes_per_epoch": 8,
            "max_refs": 2,
            "val_episodes": 8,
            "num_workers": 0,
            **train,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("objective", ["triplet", "contrastive", "bce"])
def test_training_cli_end_to_end(tmp_path, identity_root, lfw, objective):
    config_path = write_config(tmp_path, identity_root, lfw, objective=objective)
    out = tmp_path / "run"
    main(["--config", str(config_path), "--out", str(out), "--device", "cpu"])

    history = json.loads((out / "metrics.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]
    assert all(0 <= h["val_auc"] <= 1 for h in history)

    model, config = load_checkpoint(out / "model.pt")
    assert config.train.objective == objective
    assert yaml.safe_load((out / "config.yaml").read_text())["train"]["objective"] == objective

    results = json.loads((out / "eval.json").read_text())
    assert 0 <= results["lfw"]["accuracy"] <= 1
    assert "1_refs" in results["held_out_identities"]


def test_cli_overrides_and_missing_lfw(tmp_path, identity_root, lfw, capsys):
    config_path = write_config(tmp_path, identity_root, (tmp_path / "none", tmp_path / "none.txt"))
    out = tmp_path / "run"
    main(["--config", str(config_path), "--out", str(out), "--epochs", "1", "--seed", "3"])

    assert len(json.loads((out / "metrics.json").read_text())) == 1
    _, config = load_checkpoint(out / "model.pt")
    assert config.train.seed == 3
    assert "skipping evaluation" in capsys.readouterr().out


@pytest.mark.parametrize("objective", ["triplet", "bce"])
def test_training_learns_to_verify_synthetic_identities(tmp_path, identity_root, lfw, objective):
    # Identities differ only by colour, so a working pipeline separates unseen ones quickly. A
    # flipped sign or a broken head would leave AUC near 0 or 0.5.
    config_path = write_config(
        tmp_path,
        identity_root,
        lfw,
        objective=objective,
        epochs=8,
        episodes_per_epoch=64,
        batch_size=16,
        lr=3e-3,
    )
    out = tmp_path / "run"
    main(["--config", str(config_path), "--out", str(out), "--device", "cpu"])

    history = json.loads((out / "metrics.json").read_text())
    assert max(h["val_auc"] for h in history) > 0.9
    assert json.loads((out / "eval.json").read_text())["lfw"]["auc"] > 0.9
