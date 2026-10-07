"""Section 10 tests: the joint objective, on real S&P 500 batches.

Inputs are real windows, section-6 targets and realised next-step returns
from the build (N = 224), with the section-7 schedule fitted on the real
training split.  The one injected edge case is an invalid return in the
masking test, which the real build does not contain.
"""

from __future__ import annotations

import pytest
import torch

from diffolio.config import DiffolioConfig
from diffolio.data.targets import portfolio_dataset
from diffolio.loss import DiffolioLoss, auxiliary_return, denoising_loss
from diffolio.model import DiffolioModel
from diffolio.portfolio import l1_normalize


@pytest.fixture(scope="module")
def batch(real_dataset, real_targets):
    ds = portfolio_dataset(real_dataset, "train", real_targets)
    return ds.stack(range(0, len(ds), len(ds) // 8)[:8])


@pytest.fixture(scope="module")
def z(batch, real_dataset):
    # A real market summary: the (untrained) encoder applied to real windows.
    torch.manual_seed(0)
    model = DiffolioModel.from_config(real_dataset.config, real_dataset.n_assets, 0.006)
    with torch.no_grad():
        return model.encode(batch.h, batch.g).z_merged


def _loss(real_dataset, seed=0, **kwargs) -> DiffolioLoss:
    torch.manual_seed(seed)
    defaults = dict(n_assets=real_dataset.n_assets, embed_dim=real_dataset.config.embed_dim)
    return DiffolioLoss(**{**defaults, **kwargs})


def test_denoising_loss_is_the_batch_mean_squared_l2_error(batch):
    x0 = batch.x0[:, 2]  # (B, N), gamma = 2
    x_hat = batch.x_base  # a different real portfolio as the "prediction"
    expected = sum(((x_hat[i] - x0[i]) ** 2).sum() for i in range(len(x0))) / len(x0)
    torch.testing.assert_close(denoising_loss(x_hat, x0), expected)
    # With a risk axis, the mean also runs over gamma.
    all_levels = denoising_loss(batch.x_base[:, None].expand_as(batch.x0), batch.x0)
    per_level = torch.stack([denoising_loss(batch.x_base, batch.x0[:, g]) for g in range(5)])
    torch.testing.assert_close(all_levels, per_level.mean())
    assert denoising_loss(x0, x0) == 0
    with pytest.raises(ValueError):
        denoising_loss(x0, batch.x0)


def test_objective_matches_eq_18(real_dataset, batch, z):
    loss = _loss(real_dataset, aux_weight=0.7)
    x0, x_hat = batch.x0[:, 0], batch.x_base
    out = loss(x_hat, x0, z, batch.r, batch.r_valid)

    weights = l1_normalize(z @ loss.projection.w_p.weight.T)  # f(W_p z), Eq. 12
    aux = (weights * batch.r).sum(-1).mean()  # <., r_tau>, Eq. 14
    torch.testing.assert_close(out.aux_weights, weights)
    torch.testing.assert_close(out.aux_return, aux)
    torch.testing.assert_close(out.denoise, denoising_loss(x_hat, x0))
    torch.testing.assert_close(out.total, out.denoise - 0.7 * aux)


def test_auxiliary_portfolio_is_a_unit_l1_portfolio_with_bounded_return(real_dataset, batch, z):
    out = _loss(real_dataset)(batch.x_base, batch.x0[:, 0], z, batch.r, batch.r_valid)
    torch.testing.assert_close(out.aux_weights.abs().sum(-1), torch.ones(8))
    per_sample = auxiliary_return(out.aux_weights, batch.r, batch.r_valid)
    # With sum |w| = 1, no portfolio beats putting everything on the largest move.
    assert (per_sample.abs() <= batch.r.abs().amax(-1) + 1e-7).all()
    assert torch.isclose(per_sample.mean(), out.aux_return)


def test_minimising_the_loss_maximises_the_auxiliary_return(real_dataset, batch, z):
    # Only W_p trains, on fixed real z and r: the auxiliary return must rise
    # towards its ceiling, the batch-mean largest |r|.
    loss = _loss(real_dataset)
    optimiser = torch.optim.Adam(loss.parameters(), lr=1e-2)
    x = batch.x0[:, 0]
    start = loss(x, x, z, batch.r).aux_return.item()
    for _ in range(100):
        out = loss(x, x, z, batch.r)
        optimiser.zero_grad()
        out.total.backward()
        optimiser.step()
    end = loss(x, x, z, batch.r).aux_return.item()
    ceiling = batch.r.abs().amax(-1).mean().item()
    assert end > start
    assert end > 0.5 * ceiling


def test_invalid_returns_get_no_auxiliary_weight(real_dataset, batch, z):
    loss = _loss(real_dataset)
    # Injected: asset 7's return in sample 0 is a stale fill carrying a huge value.
    r, valid = batch.r.clone(), batch.r_valid.clone()
    valid[0, 7] = False
    r[0, 7] = 5.0
    out = loss(batch.x_base, batch.x0[:, 0], z, r, valid)
    assert out.aux_weights[0, 7] == 0
    torch.testing.assert_close(out.aux_weights.abs().sum(-1), torch.ones(8))
    clean = r.clone()
    clean[0, 7] = -5.0
    torch.testing.assert_close(loss(batch.x_base, batch.x0[:, 0], z, clean, valid).total, out.total)


def test_aux_weight_zero_leaves_only_denoising(real_dataset, batch, z):
    out = _loss(real_dataset, aux_weight=0.0)(batch.x_base, batch.x0[:, 0], z, batch.r)
    assert out.total == out.denoise


def test_gradient_clip_bounds_only_the_auxiliary_gradient(real_dataset, batch, z):
    x0 = batch.x0[:, 0]
    head = torch.nn.Linear(z.shape[1], x0.shape[1])  # a stand-in denoiser reading z

    def grads(clip, with_denoise):
        loss = _loss(real_dataset, seed=3, aux_weight=1.0, aux_grad_clip=clip)
        zz = z.clone().requires_grad_(True)
        x_hat = head(zz) if with_denoise else x0
        loss(x_hat, x0, zz, batch.r).total.backward()
        return zz.grad

    aux_only = grads(None, False)
    clip = 0.1 * aux_only.norm().item()
    clipped = grads(clip, False)
    assert clipped.norm().item() == pytest.approx(clip, rel=1e-5)
    torch.testing.assert_close(clipped / clipped.norm(), aux_only / aux_only.norm())
    # The denoising gradient passes unclipped: total = denoise grad + clipped aux grad.
    denoise_grad = grads(None, True) - aux_only
    torch.testing.assert_close(grads(clip, True), denoise_grad + clipped)
    # A clip above the gradient's norm changes nothing.
    torch.testing.assert_close(grads(10 * aux_only.norm().item(), False), aux_only)


def test_end_to_end_gradients_reach_encoder_head_and_projection(
    real_dataset, real_schedule, batch
):
    torch.manual_seed(0)
    config = real_dataset.config
    model = DiffolioModel.from_config(config, real_dataset.n_assets, float(real_schedule.sigma_x))
    loss = DiffolioLoss.from_config(config, real_dataset.n_assets)
    gamma = torch.arange(8) % 5
    x0 = batch.x0[torch.arange(8), gamma]
    t = real_schedule.sample_timesteps(8, generator=torch.Generator().manual_seed(1))
    x_t = real_schedule.q_sample(x0, t, generator=torch.Generator().manual_seed(2))

    pred = model(x_t, t, batch.h, batch.g, gamma)
    out = loss(pred.x_hat, x0, pred.encoding.z_merged, batch.r, batch.r_valid)
    out.total.backward()

    assert torch.isfinite(out.total)
    for name, p in [*model.named_parameters(), *loss.named_parameters()]:
        if name == "head.risk_embedding.weight":
            continue  # only the rows of the sampled levels are used
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_from_config_and_validation(real_dataset):
    config = DiffolioConfig.from_dict(real_dataset.config.to_dict())
    config.training.aux_weight = 0.5
    config.training.aux_grad_clip = 2.0
    loss = DiffolioLoss.from_config(config, real_dataset.n_assets)
    assert (loss.aux_weight, loss.aux_grad_clip) == (0.5, 2.0)
    assert loss.projection.w_p.weight.shape == (real_dataset.n_assets, config.embed_dim)
    assert loss.projection.w_p.bias is None

    for key, value in (("aux_weight", -1.0), ("aux_grad_clip", 0.0)):
        bad = DiffolioConfig()
        setattr(bad.training, key, value)
        with pytest.raises(ValueError, match=key):
            bad.validate()
    with pytest.raises(ValueError, match="r must be"):
        loss(torch.zeros(2, 224), torch.zeros(2, 224), torch.zeros(2, 1280), torch.zeros(2, 10))
