"""Section 8 tests: the market dynamics encoder, on real S&P 500 windows.

Every forward pass uses real ``(h, g)`` windows from ``real_dataset``
(L = 256, F = 5, N = 224).  The only injected input is the single-step
perturbation in the TCN locality test, which probes the receptive field and
needs no market meaning.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from diffolio.data import window_dataset
from diffolio.model import DilatedTCN, MarketEncoder


@pytest.fixture(scope="module")
def batch(real_dataset):
    ds = window_dataset(real_dataset, "train")
    b = ds.stack([0, len(ds) // 2, len(ds) - 1])
    return b.h.float(), b.g.float()


@pytest.fixture(scope="module")
def encoder(real_dataset):
    torch.manual_seed(0)
    return MarketEncoder.from_config(real_dataset.config).eval()


def test_shapes_and_derived_width(real_dataset, encoder, batch):
    h, g = batch
    config = real_dataset.config
    assert encoder.embed_dim == config.embed_dim == 1280
    out = encoder(h, g)
    assert out.z_merged.shape == (3, 1280)
    assert out.asset_attention.shape == (3, real_dataset.n_assets)
    assert out.time_attention.shape == (3, config.lookback)
    assert torch.isfinite(out.z_merged).all()
    torch.testing.assert_close(out.asset_attention.sum(-1), torch.ones(3))
    torch.testing.assert_close(out.time_attention.sum(-1), torch.ones(3))


def test_forward_matches_the_longhand_equations(encoder, batch):
    h, g = batch
    h, g = h[:1], g[:1]
    d = encoder.embed_dim
    with torch.no_grad():
        n = h.shape[1]
        q, k, v = (t(h).reshape(n, d) for t in (encoder.tcn_q, encoder.tcn_k, encoder.tcn_v))
        s = torch.softmax(q @ k.T / d**0.5, dim=-1)  # 8.2
        h_prime = encoder.norm_attn(h[0] + (s @ v).reshape_as(h[0]))  # 8.3
        h_final = encoder.norm_out(h_prime + encoder.tcn_h(h_prime[None])[0])
        u, _ = encoder.index.lstm(g)  # 8.4
        psi = torch.softmax(encoder.index.score(u[0]).squeeze(-1), dim=0)
        g_final = encoder.index.project(psi @ u[0])
        z = h_final.reshape(n, d) @ encoder.w_h.weight.T + g_final  # 8.5
        alpha = torch.softmax(encoder.asset_score(z).squeeze(-1), dim=0)  # 8.6
        out = encoder(h, g)
    torch.testing.assert_close(out.z_merged[0], alpha @ z, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out.asset_attention[0], alpha, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(out.time_attention[0], psi)


def test_output_is_invariant_to_asset_order(encoder, batch):
    h, g = batch
    perm = torch.randperm(h.shape[1], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out, shuffled = encoder(h, g), encoder(h[:, perm], g)
    torch.testing.assert_close(shuffled.z_merged, out.z_merged, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(shuffled.asset_attention, out.asset_attention[:, perm])


def test_the_same_weights_accept_any_number_of_assets(encoder, batch):
    h, g = batch
    with torch.no_grad():
        for n in (1, 10, h.shape[1]):
            assert encoder(h[:, :n], g).z_merged.shape == (3, encoder.embed_dim)


def test_samples_in_a_batch_are_independent(encoder, batch):
    h, g = batch
    with torch.no_grad():
        together = encoder(h, g).z_merged
        alone = encoder(h[1:2], g[1:2]).z_merged
    torch.testing.assert_close(together[1:2], alone, rtol=1e-4, atol=1e-5)


def test_tcn_preserves_length_and_has_the_expected_receptive_field(batch):
    h, _ = batch
    torch.manual_seed(0)
    tcn = DilatedTCN(channels=5, kernel_size=3, dilations=(1, 2))
    assert tcn.receptive_radius == 3
    x = h[:1, :4]
    assert tcn(x).shape == x.shape

    # Injected probe: nudge one step and see which output steps move.
    t = 100
    bumped = x.clone()
    bumped[:, :, t] += 1.0
    with torch.no_grad():
        moved = (tcn(bumped) - tcn(x)).abs().amax(dim=(0, 1, 3)) > 0
    assert moved.nonzero().flatten().tolist() == list(range(t - 3, t + 4))

    # Assets are convolved independently: asset 0's output ignores asset 1.
    other = x.clone()
    other[:, 1] = x[:, 2]
    with torch.no_grad():
        torch.testing.assert_close(tcn(other)[:, 0], tcn(x)[:, 0])


def test_gradients_reach_every_parameter(real_dataset, batch):
    h, g = batch
    torch.manual_seed(0)
    encoder = MarketEncoder.from_config(real_dataset.config)
    encoder(h[:2, :32], g[:2]).z_merged.pow(2).mean().backward()
    for name, p in encoder.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_attention_is_not_collapsed_at_initialisation(encoder, batch):
    # With LayerNorm'd inputs the pooling starts near uniform, not on one asset.
    h, g = batch
    with torch.no_grad():
        alpha = encoder(h, g).asset_attention
    assert alpha.max() < 10.0 / h.shape[1]


def test_rejects_mis_shaped_inputs(encoder, batch):
    h, g = batch
    with pytest.raises(ValueError, match="h must be"):
        encoder(h.transpose(2, 3), g)
    with pytest.raises(ValueError, match="g must be"):
        encoder(h, g[:2])
    with pytest.raises(ValueError, match="g must be"):
        encoder(h, g.transpose(1, 2))


def test_config_validation(real_dataset):
    from diffolio.config import DiffolioConfig

    config = DiffolioConfig()
    config.model.tcn_kernel_size = 4
    with pytest.raises(ValueError, match="odd"):
        config.validate()
    config = DiffolioConfig()
    config.model.tcn_dilations = []
    with pytest.raises(ValueError, match="dilations"):
        config.validate()
