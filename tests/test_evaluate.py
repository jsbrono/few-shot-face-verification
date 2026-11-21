import itertools

import numpy as np
import pytest
import torch

from faceverify.config import Config, load_checkpoint, save_checkpoint
from faceverify.data import LFWPairs, build_transform, scan_identities
from faceverify.evaluate import (
    accuracy,
    best_threshold,
    evaluate_episodes,
    evaluate_lfw,
    k_fold_accuracy,
    plot_errors,
    roc_auc,
    tar_at_far,
)
from faceverify.models import ModelConfig, Verifier

SMALL_MODEL = ModelConfig(backbone="small_cnn", pretrained=False, embed_dim=8)


def brute_force_best_accuracy(scores, labels):
    candidates = np.concatenate([scores, [scores.min() - 1, scores.max() + 1]])
    return max(accuracy(scores, labels, t) for t in candidates)


@pytest.mark.parametrize("seed", range(5))
def test_best_threshold_is_optimal(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 200)
    scores = np.round(labels + rng.normal(0, 0.8, 200), 1)  # rounding creates ties
    threshold = best_threshold(scores, labels)
    assert accuracy(scores, labels, threshold) == pytest.approx(
        brute_force_best_accuracy(scores, labels)
    )


def test_best_threshold_separable_and_degenerate_cases():
    assert accuracy(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]), 0.5) == 1.0
    assert best_threshold(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 0.5
    # All positive: the best threshold accepts everything.
    scores = np.array([0.3, 0.1])
    assert accuracy(scores, np.array([1, 1]), best_threshold(scores, np.array([1, 1]))) == 1.0


def test_k_fold_uses_held_out_thresholds():
    # Fold 0 alone would pick a threshold near 0.5; the other folds put it near 0.0, so fold 0
    # is only 50% accurate under the protocol. Tuning on fold 0 itself would report 100%.
    scores = np.array([0.4, 0.6, -0.1, 0.1, -0.1, 0.1])
    labels = np.array([0, 1, 0, 1, 0, 1])
    folds = np.array([0, 0, 1, 1, 2, 2])
    mean, std = k_fold_accuracy(scores, labels, folds)
    assert mean == pytest.approx((0.5 + 1 + 1) / 3)
    assert std > 0


def brute_force_auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = [1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg)]
    return np.mean(wins)


@pytest.mark.parametrize("seed", range(5))
def test_roc_auc_matches_pair_counting(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 60)
    scores = np.round(labels * 0.5 + rng.normal(0, 1, 60), 1)
    assert roc_auc(scores, labels) == pytest.approx(brute_force_auc(scores, labels))


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError):
        roc_auc(np.array([0.1, 0.2]), np.array([1, 1]))


def test_tar_at_far():
    negatives = np.linspace(0, 0.99, 100)  # 1% FAR allows exactly one negative through
    positives = np.array([0.5, 0.985, 0.995, 1.5])
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros(100), np.ones(4)])
    # Threshold is the second-highest negative (0.98): three positives are above it.
    assert tar_at_far(scores, labels, 0.01) == pytest.approx(0.75)
    assert tar_at_far(scores, labels, 0.0) == pytest.approx(0.5)


def test_lfw_and_episode_evaluation_run(lfw, identity_root, tmp_path):
    torch.manual_seed(0)
    model = Verifier(SMALL_MODEL)
    images, pairs = lfw
    dataset = LFWPairs(images, pairs, build_transform(16, train=False))

    results = evaluate_lfw(model, dataset, torch.device("cpu"), batch_size=4)
    assert set(results) == {"accuracy", "accuracy_std", "auc", "tar_at_far_1pct"}
    assert 0 <= results["accuracy"] <= 1

    config = Config(model=SMALL_MODEL)
    config.data.image_size = 16
    episodes = evaluate_episodes(
        model, scan_identities(identity_root), config, torch.device("cpu"), num_episodes=20
    )
    # Only person_000 and person_001 have more than 3 images, so there are no 5-ref episodes.
    assert set(episodes) == {"1_refs", "3_refs"}

    out = tmp_path / "errors.png"
    scores = np.linspace(-1, 1, len(dataset))  # every matched pair scores low: all wrong
    plot_errors(dataset, scores, threshold=0.5, out=out, count=3)
    assert out.stat().st_size > 0


def test_config_rejects_mismatched_objective_and_head():
    with pytest.raises(ValueError, match="differential"):
        Config.from_dict({"train": {"objective": "bce"}})
    with pytest.raises(ValueError, match="cosine"):
        Config.from_dict({"model": {"head": "differential"}, "train": {"objective": "triplet"}})
    with pytest.raises(ValueError, match="sections"):
        Config.from_dict({"modle": {}})


def test_checkpoint_round_trip(tmp_path):
    config = Config(model=SMALL_MODEL)
    model = Verifier(config.model).eval()
    save_checkpoint(model, config, tmp_path / "run" / "model.pt")
    restored, restored_config = load_checkpoint(tmp_path / "run" / "model.pt")

    assert restored_config == config
    refs, query, mask = torch.randn(2, 1, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.ones(2, 1)
    torch.testing.assert_close(restored(refs, query, mask), model(refs, query, mask))
