import itertools

import pytest
import torch
import torch.nn.functional as F

from faceverify.models import ModelConfig, Verifier, mean_reference

SIZE = 32
VARIANTS = [
    ModelConfig(backbone=backbone, pretrained=False, embed_dim=16, head=head, head_hidden_dim=8)
    for backbone, head in itertools.product(("resnet18", "small_cnn"), ("cosine", "differential"))
]


def batch(batch_size: int = 3, max_refs: int = 4):
    refs = torch.randn(batch_size, max_refs, 3, SIZE, SIZE)
    query = torch.randn(batch_size, 3, SIZE, SIZE)
    mask = torch.zeros(batch_size, max_refs)
    for i in range(batch_size):
        mask[i, : 1 + i % max_refs] = 1  # 1, 2, 3, ... real references
    return refs, query, mask


@pytest.fixture(params=VARIANTS, ids=lambda c: f"{c.backbone}-{c.head}")
def model(request):
    torch.manual_seed(0)
    return Verifier(request.param).eval()


def test_scores_have_one_value_per_sample(model):
    refs, query, mask = batch()
    scores = model(refs, query, mask)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_embeddings_are_unit_length(model):
    embeddings = model.embedder(torch.randn(5, 3, SIZE, SIZE))
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(5))


def test_padded_references_are_ignored(model):
    refs, query, mask = batch()
    scores = model(refs, query, mask)
    refs[mask == 0] = torch.randn_like(refs[mask == 0]) * 100
    torch.testing.assert_close(model(refs, query, mask), scores)


def test_reference_order_does_not_matter(model):
    refs, query, _ = batch(max_refs=4)
    mask = torch.ones(3, 4)
    permuted = refs[:, torch.tensor([2, 0, 3, 1])]
    torch.testing.assert_close(model(permuted, query, mask), model(refs, query, mask))


def test_cosine_head_with_one_reference_is_cosine_similarity():
    model = Verifier(ModelConfig(backbone="small_cnn", pretrained=False, embed_dim=16)).eval()
    refs, query, _ = batch(max_refs=1)
    expected = F.cosine_similarity(model.embedder(refs[:, 0]), model.embedder(query))
    torch.testing.assert_close(model(refs, query, torch.ones(3, 1)), expected)


def test_differential_head_learns_from_a_single_reference():
    # The original model compared the query only with references after the first, so with one
    # reference its input was all zeros. Here the head must depend on the query.
    torch.manual_seed(0)
    config = ModelConfig(backbone="small_cnn", pretrained=False, embed_dim=16, head="differential")
    model = Verifier(config).eval()
    refs, query, _ = batch(max_refs=1)
    mask = torch.ones(3, 1)
    assert not torch.allclose(model(refs, query, mask), model(refs, torch.randn_like(query), mask))


def test_every_parameter_gets_a_gradient(model):
    model.train()
    refs, query, mask = batch()
    model(refs, query, mask).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_sample_without_references_is_rejected(model):
    refs, query, mask = batch()
    mask[1] = 0
    with pytest.raises(ValueError, match="at least one reference"):
        model(refs, query, mask)


def test_mean_reference_ignores_padding():
    refs = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-5.0, -5.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    expected = torch.tensor([[1.0, 1.0]]) / 2**0.5
    torch.testing.assert_close(mean_reference(refs, mask), expected)


@pytest.mark.parametrize(
    "kwargs", [{"backbone": "vgg"}, {"head": "mlp"}, {"backbone": "small_cnn", "pretrained": True}]
)
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
