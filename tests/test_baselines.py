import torch

from latent_nca.models import GeneCA, GenePropCA, GrowingNCA


def test_growing_shape_and_seed_channels():
    model = GrowingNCA()
    state = model.seed(2, 16, 16, torch.device("cpu"))
    assert state.shape == (2, 16, 16, 16)
    assert torch.all(state[:, 3:, 8, 8] == 1)
    assert model(state).shape == state.shape


def test_engram_private_channels_are_preserved_by_gene_ca():
    model = GeneCA(visible_channels=4, public_hidden_channels=4, private_channels=3)
    state = torch.randn(2, 11, 16, 16)
    private = state[:, -3:].clone()
    result = model(state, update_probability=1.0)
    assert result.shape == state.shape
    assert torch.equal(result[:, -3:], private)


def test_engram_propagation_updates_private_channels():
    model = GenePropCA(visible_channels=4, public_hidden_channels=4, private_channels=3)
    state = torch.randn(2, 11, 16, 16)
    result = model(state, update_probability=1.0)
    assert result.shape == state.shape
