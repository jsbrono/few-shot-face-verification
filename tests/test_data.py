import pytest
import torch

from faceverify.data import (
    EpisodeDataset,
    IdentityBatchSampler,
    ImageDataset,
    LFWPairs,
    build_transform,
    parse_lfw_pairs,
    scan_identities,
    split_identities,
)

TRANSFORM = build_transform(16, train=False)


def test_scan_identities_filters_and_sorts(identity_root):
    identities = scan_identities(identity_root, min_images=2)
    assert list(identities) == sorted(identities)
    assert len(identities) == 7  # person_007 has a single image
    for paths in identities.values():
        assert paths == sorted(paths)


def test_scan_identities_errors(tmp_path, identity_root):
    with pytest.raises(FileNotFoundError):
        scan_identities(tmp_path / "missing")
    with pytest.raises(ValueError):
        scan_identities(identity_root, min_images=100)


def test_split_is_identity_disjoint_and_deterministic(identity_root):
    identities = scan_identities(identity_root)
    train, val = split_identities(identities, val_fraction=0.25, seed=3)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(identities)
    assert len(val) == 2
    assert split_identities(identities, 0.25, seed=3) == (train, val)
    assert split_identities(identities, 0.25, seed=4) != (train, val)


def test_image_dataset_labels(identity_root):
    dataset = ImageDataset(scan_identities(identity_root), TRANSFORM)
    image, label = dataset[0]
    assert image.shape == (3, 16, 16)
    assert label == 0
    for label, indices in enumerate(dataset.indices_by_label):
        assert all(dataset.samples[i][1] == label for i in indices)


def test_identity_batch_sampler_structure(identity_root):
    dataset = ImageDataset(scan_identities(identity_root), TRANSFORM)
    sampler = IdentityBatchSampler(dataset.indices_by_label, 3, 4, num_batches=5, seed=0)
    batches = list(sampler)
    assert len(batches) == 5
    for batch in batches:
        labels = [dataset.samples[i][1] for i in batch]
        assert len(batch) == 12
        assert len(set(labels)) == 3
        assert all(labels.count(label) == 4 for label in set(labels))

    sampler.set_epoch(1)
    assert list(sampler) != batches
    sampler.set_epoch(0)
    assert list(sampler) == batches


def test_identity_batch_sampler_validation(identity_root):
    dataset = ImageDataset(scan_identities(identity_root), TRANSFORM)
    with pytest.raises(ValueError):
        IdentityBatchSampler(dataset.indices_by_label, 100, 4)
    with pytest.raises(ValueError):
        IdentityBatchSampler(dataset.indices_by_label, 2, 1)


def test_episodes_are_valid_and_balanced(identity_root):
    identities = scan_identities(identity_root)
    dataset = EpisodeDataset(identities, TRANSFORM, num_episodes=200, min_refs=1, max_refs=3)
    owner = {path: name for name, paths in identities.items() for path in paths}

    for i in range(len(dataset)):
        episode = dataset.episode(i)
        assert 1 <= len(episode.references) <= 3
        assert len({owner[p] for p in episode.references}) == 1
        assert episode.query not in episode.references
        same = owner[episode.query] == owner[episode.references[0]]
        assert same == bool(episode.label) == (i % 2 == 0)


def test_episodes_change_with_epoch_but_are_reproducible(identity_root):
    dataset = EpisodeDataset(scan_identities(identity_root), TRANSFORM, num_episodes=20)
    first = [dataset.episode(i) for i in range(20)]
    dataset.set_epoch(1)
    assert [dataset.episode(i) for i in range(20)] != first
    dataset.set_epoch(0)
    assert [dataset.episode(i) for i in range(20)] == first


def test_episode_tensors_are_padded(identity_root):
    dataset = EpisodeDataset(scan_identities(identity_root), TRANSFORM, 10, min_refs=1, max_refs=4)
    for i in range(10):
        refs, query, label, mask = dataset[i]
        assert refs.shape == (4, 3, 16, 16)
        assert query.shape == (3, 16, 16)
        num_refs = len(dataset.episode(i).references)
        assert mask.tolist() == [1.0] * num_refs + [0.0] * (4 - num_refs)
        assert (refs[num_refs:] == 0).all()
        assert label.item() == dataset.episode(i).label


def test_fixed_reference_count_filters_small_identities(identity_root):
    dataset = EpisodeDataset(scan_identities(identity_root), TRANSFORM, 50, min_refs=3, max_refs=3)
    assert all(len(paths) > 3 for paths in dataset.identities)
    assert all(len(dataset.episode(i).references) == 3 for i in range(50))


def test_parse_lfw_pairs_txt(lfw):
    images, pairs_file = lfw
    pairs = parse_lfw_pairs(pairs_file, images)
    assert len(pairs) == 12
    assert [p.fold for p in pairs] == [0] * 6 + [1] * 6
    assert [p.label for p in pairs[:6]] == [1, 1, 1, 0, 0, 0]
    assert pairs[0].first == images / "Person_A" / "Person_A_0001.jpg"
    assert pairs[0].second == images / "Person_A" / "Person_A_0002.jpg"


def test_parse_lfw_pairs_csv(tmp_path):
    # Kaggle's pairs.csv: header row, comma-separated, trailing empty field on matched pairs.
    csv = tmp_path / "pairs.csv"
    rows = ["name,imagenum1,imagenum2,"] + ["Ann,1,2,"] * 300 + ["Ann,1,Bob,3"] * 300
    csv.write_text("\n".join(rows))
    pairs = parse_lfw_pairs(csv, tmp_path)
    assert len(pairs) == 600
    assert pairs[0].label == 1
    assert pairs[-1].second == tmp_path / "Bob" / "Bob_0003.jpg"
    assert {p.fold for p in pairs} == {0}


def test_parse_lfw_pairs_rejects_partial_folds(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("1\t2\nAnn\t1\t2\n")
    with pytest.raises(ValueError, match="whole number of folds"):
        parse_lfw_pairs(path, tmp_path)


def test_lfw_dataset(lfw):
    images, pairs_file = lfw
    dataset = LFWPairs(images, pairs_file, TRANSFORM)
    refs, query, label, mask = dataset[0]
    assert refs.shape == (1, 3, 16, 16)
    assert query.shape == (3, 16, 16)
    assert label.item() == 1.0
    assert mask.tolist() == [1.0]
    assert torch.equal(dataset.folds, torch.tensor([0] * 6 + [1] * 6))


def test_lfw_dataset_reports_missing_images(lfw):
    images, pairs_file = lfw
    (images / "Person_A" / "Person_A_0001.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        LFWPairs(images, pairs_file, TRANSFORM)
