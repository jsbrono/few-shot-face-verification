import pytest
import torch
import torch.nn.functional as F

from faceverify.losses import BatchHardTripletLoss, ContrastiveLoss, pairwise_distances


def test_pairwise_distances_match_cdist():
    x = torch.randn(6, 5)
    torch.testing.assert_close(pairwise_distances(x), torch.cdist(x, x), atol=1e-5, rtol=0)


def test_pairwise_distance_gradient_is_finite_for_duplicates():
    x = torch.randn(1, 4).repeat(3, 1).requires_grad_()
    pairwise_distances(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def brute_force_triplet(embeddings, labels, margin):
    dist = torch.cdist(embeddings, embeddings)
    losses = []
    for a in range(len(labels)):
        positives = [dist[a, p] for p in range(len(labels)) if p != a and labels[p] == labels[a]]
        negatives = [dist[a, n] for n in range(len(labels)) if labels[n] != labels[a]]
        if positives and negatives:
            losses.append(F.relu(max(positives) - min(negatives) + margin))
    return torch.stack(losses).mean()


def test_batch_hard_triplet_matches_brute_force():
    torch.manual_seed(0)
    embeddings = F.normalize(torch.randn(12, 8), dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4])  # identity 4 has no positive
    loss = BatchHardTripletLoss(margin=0.3)(embeddings, labels)
    expected = brute_force_triplet(embeddings, labels, 0.3)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=0)


def test_well_separated_identities_have_zero_loss():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert BatchHardTripletLoss(margin=0.5)(embeddings, labels).item() == pytest.approx(0, abs=1e-5)
    assert ContrastiveLoss(margin=1.0)(embeddings, labels).item() == pytest.approx(0, abs=1e-5)


def test_contrastive_loss_terms():
    # Positive pair at distance 0.6 -> 0.36. Negative pairs at distance 0.5 -> (1 - 0.5)^2 = 0.25.
    embeddings = torch.tensor([[0.0, 0.0], [0.6, 0.0], [0.0, 0.5]])
    labels = torch.tensor([0, 0, 1])
    # Negatives: (0, 2) at 0.5 and (1, 2) at sqrt(0.61).
    negatives = torch.tensor([0.5, 0.61**0.5])
    expected = 0.36 + F.relu(1.0 - negatives).pow(2).mean()
    torch.testing.assert_close(ContrastiveLoss(margin=1.0)(embeddings, labels), expected)


@pytest.mark.parametrize("loss_fn", [ContrastiveLoss(), BatchHardTripletLoss()])
def test_batch_of_one_identity_gives_zero_loss_with_a_graph(loss_fn):
    embeddings = torch.randn(4, 3, requires_grad=True)
    loss = loss_fn(embeddings, torch.zeros(4, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss)
    if isinstance(loss_fn, BatchHardTripletLoss):
        assert loss.item() == 0.0
