"""Section 12 tests: covariance, risk guidance and Algorithm 2, on real data.

Windows, returns and covariances are real (the S&P 500 build, N = 224,
L = 256), and sigma_x is fitted on the real training split.  The network is
untrained, built from the real config: these tests check the sampler's
arithmetic, not a trained model's quality.  Short schedules (T = 1, 2 or 20)
keep it fast on CPU.  The injected parts are the return perturbations in the
leakage test and the reordered ticker list in the compatibility test.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from diffolio.config import DiffolioConfig, SamplingConfig
from diffolio.data.windows import stack_windows
from diffolio.diffusion import DiffusionSchedule
from diffolio.model import DiffolioModel
from diffolio.portfolio import l1_normalize
from diffolio.sampling import (
    CovarianceModel,
    RiskGuidedSampler,
    SampleSet,
    covariance,
    ledoit_wolf,
    proxy_risk,
    risk_gradient,
    sample_split,
    zeta,
)
from diffolio.training import TrainedModel

G = 5


@pytest.fixture(scope="module")
def real(real_dataset, real_schedule):
    tensors = real_dataset.panel.torch()
    test_tau = real_dataset.splits.test.tau
    return {
        "dataset": real_dataset,
        "tensors": tensors,
        "sigma_x": float(real_schedule.sigma_x),
        "tau": test_tau[[0, len(test_tau) // 2]],
        "train_tau": real_dataset.splits.train.tau,
        "lookback": real_dataset.config.lookback,
    }


@pytest.fixture(scope="module")
def model(real):
    torch.manual_seed(0)
    return DiffolioModel.from_config(
        real["dataset"].config, real["dataset"].n_assets, real["sigma_x"]
    ).eval()


@pytest.fixture(scope="module")
def batch(real):
    window = stack_windows(real["tensors"], real["tau"], real["lookback"])
    cov = CovarianceModel(
        real["tensors"].returns, real["tensors"].return_valid, SamplingConfig(), real["lookback"]
    )(real["tau"])
    return window, cov


def _schedule(real, steps: int) -> DiffusionSchedule:
    config = copy.deepcopy(real["dataset"].config.diffusion)
    config.num_steps = steps
    return DiffusionSchedule.from_config(config, real["sigma_x"])


# -- covariance ---------------------------------------------------------------


def test_ledoit_wolf_matches_scikit_learn_on_real_returns(real):
    sk = pytest.importorskip("sklearn.covariance")
    tau = int(real["tau"][0])
    x = real["tensors"].returns[tau - 255 : tau].double()
    expected, expected_shrinkage = sk.ledoit_wolf(x.numpy())
    actual, shrinkage = ledoit_wolf(x)
    assert float(shrinkage) == pytest.approx(expected_shrinkage, rel=1e-10)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-16)
    assert 0 < float(shrinkage) < 1


def test_rolling_covariance_uses_only_returns_known_at_tau(real):
    tau = int(real["tau"][0])
    returns, valid = real["tensors"].returns.clone(), real["tensors"].return_valid
    model = CovarianceModel(returns, valid, SamplingConfig(), real["lookback"])
    cov = model([tau])[0]
    # The L - 1 returns between the window's L opens: rows tau-255 .. tau-1.
    torch.testing.assert_close(cov, covariance(returns[tau - 255 : tau]).float()[0:])
    assert torch.linalg.eigvalsh(cov.double()).min() > 0

    # Injected: corrupt the target return and everything after it - no change.
    future = returns.clone()
    future[tau:] *= 5.0
    torch.testing.assert_close(
        CovarianceModel(future, valid, SamplingConfig(), real["lookback"])([tau])[0], cov
    )
    # Injected: corrupt the last return known at tau - it must matter.
    past = returns.clone()
    past[tau - 1] *= 5.0
    changed = CovarianceModel(past, valid, SamplingConfig(), real["lookback"])([tau])[0]
    assert not torch.allclose(changed, cov)


def test_train_covariance_is_one_matrix_from_training_returns(real):
    settings = SamplingConfig(covariance="train", shrinkage="none")
    tensors = real["tensors"]
    model = CovarianceModel(
        tensors.returns, tensors.return_valid, settings, real["lookback"], real["train_tau"]
    )
    out = model(real["tau"])
    expected = torch.from_numpy(
        np.cov(tensors.returns[real["train_tau"]].double().numpy(), rowvar=False, bias=True)
    ).float()
    torch.testing.assert_close(out[0], expected)
    assert torch.equal(out[0], out[1])


# -- risk guidance --------------------------------------------------------------


def test_zeta_runs_from_full_guidance_to_none():
    torch.testing.assert_close(zeta(torch.arange(G), G), torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_risk_gradient_matches_closed_form(real, batch):
    _, cov = batch
    cov = cov.double()
    gen = torch.Generator().manual_seed(0)
    x = real["sigma_x"] * torch.randn((2, 224), generator=gen, dtype=torch.float64)
    risk, grad = risk_gradient(x, cov)

    w = l1_normalize(x)
    s = x.abs().sum(-1, keepdim=True)
    g = 2 * (cov @ w.unsqueeze(-1)).squeeze(-1)  # d rho / d w
    closed = (g - x.sign() * (g * w).sum(-1, keepdim=True)) / s
    torch.testing.assert_close(grad, closed)
    torch.testing.assert_close(risk, proxy_risk(w, cov))
    # f is scale-invariant, so rho' cannot change along x itself.
    assert (x * grad).sum(-1).abs().max() < 1e-12 * grad.norm()
    # Finite differences along a random direction.
    v = torch.randn(x.shape, generator=gen, dtype=torch.float64)
    eps = 1e-7 * real["sigma_x"]
    fd = (proxy_risk(l1_normalize(x + eps * v), cov) - proxy_risk(l1_normalize(x - eps * v), cov)) / (2 * eps)
    torch.testing.assert_close(fd, (grad * v).sum(-1), rtol=1e-5, atol=0)


# -- Algorithm 2 ------------------------------------------------------------------


def test_two_reverse_steps_follow_eq_5_and_eq_20(real, model, batch):
    window, cov = batch
    schedule = _schedule(real, 2)
    sampler = RiskGuidedSampler(model, schedule, G, guidance_scale=40.0)
    out = sampler.sample(window.h, window.g, cov, 3, generator=torch.Generator().manual_seed(1))

    # Replay by hand with the closed-form gradient: x_2 -> x_1 -> x_0.
    gen = torch.Generator().manual_seed(1)
    sigma, n = real["sigma_x"], 224
    x = sigma * torch.randn((2, G, 3, n), generator=gen)
    gamma = torch.arange(G).view(1, G, 1).expand(2, G, 3)
    step = 40.0 * torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]).view(1, G, 1, 1)
    c = cov.view(2, 1, 1, n, n)
    with torch.no_grad():
        z = model.encode(window.h, window.g).z_merged
        for t in (2, 1):
            w, s = l1_normalize(x), x.abs().sum(-1, keepdim=True)
            gw = 2 * (c @ w.unsqueeze(-1)).squeeze(-1)
            grad = (gw - x.sign() * (gw * w).sum(-1, keepdim=True)) / s
            # Eq. 5's coefficients, precomputed in float64 by the schedule
            # (1 - abar_1 ~ 1e-4 loses ~1e-3 to cancellation in float32);
            # their values are tested in test_diffusion.py.
            mean = (
                schedule.posterior_coef_x0[t] * model.denoise(x, t, gamma, z)
                + schedule.posterior_coef_xt[t] * x
                - step * grad
            )
            if t > 1:
                std = schedule.posterior_variance_unit[t].sqrt() * sigma
                x = mean + std * torch.randn(x.shape, generator=gen)
            else:
                x = mean
    torch.testing.assert_close(out.x0, x, rtol=1e-4, atol=1e-7)


def test_samples_are_k_gamma_unit_l1_portfolios(real, model, batch):
    window, cov = batch
    sampler = RiskGuidedSampler(model, _schedule(real, 20), G)
    out = sampler.sample(window.h, window.g, cov, 4, generator=torch.Generator().manual_seed(0))
    assert out.weights.shape == (2, G, 4, 224)
    held = (out.weights != 0).sum(-1)
    assert held.eq(torch.tensor([220, 176, 132, 88, 44]).view(1, G, 1)).all()
    torch.testing.assert_close(out.weights.abs().sum(-1), torch.ones(2, G, 4))
    # Signs pass through from x_0 to the weights (short positions survive).
    kept = out.weights != 0
    assert torch.equal(out.weights[kept].sign(), out.x0[kept].sign())
    assert (out.weights < 0).any()


def test_guidance_only_changes_the_guided_levels(real, model, batch):
    window, cov = batch
    schedule = _schedule(real, 20)

    def run(guidance: bool):
        sampler = RiskGuidedSampler(model, schedule, G, guidance=guidance)
        return sampler.sample(window.h, window.g, cov, 4, generator=torch.Generator().manual_seed(3))

    guided, unguided = run(True), run(False)
    # Same seed, same x_T and z: zeta = 0 at the top level makes them identical.
    assert torch.equal(guided.x0[:, -1], unguided.x0[:, -1])
    for level in range(G - 1):
        assert not torch.equal(guided.x0[:, level], unguided.x0[:, level])
    again = run(True)
    assert torch.equal(again.weights, guided.weights)  # seeded runs repeat exactly


# -- the split-level driver ------------------------------------------------------


def _trained(real, model, steps=2, tickers=None):
    config = copy.deepcopy(real["dataset"].config)
    config.diffusion.num_steps = steps
    meta = {
        "fingerprint": real["dataset"].panel.metadata["fingerprint"],
        "tickers": list(real["dataset"].panel.tickers) if tickers is None else tickers,
    }
    return TrainedModel(config, model, _schedule(real, steps), None, {"epoch": 7, "metadata": meta})


def test_sample_split_writes_every_level_and_round_trips(real, model, tmp_path):
    settings = SamplingConfig(num_samples=3, batch_size=2)
    tau = real["dataset"].splits.test.tau[:3]
    samples = sample_split(_trained(real, model), real["dataset"], "test", settings, "cpu", tau=tau)

    assert samples.weights.shape == (3, G, 3, 224) and samples.risk.shape == (3, G, 3)
    cov = CovarianceModel(
        real["tensors"].returns, real["tensors"].return_valid, settings, real["lookback"]
    )(tau)
    risk = proxy_risk(torch.from_numpy(samples.weights), cov[:, None, None])
    np.testing.assert_allclose(samples.risk, risk.numpy(), rtol=1e-5)
    assert samples.metadata["checkpoint_epoch"] == 7
    assert len(samples.metadata["trace"]["guidance_norm"]) == 2

    samples.save(tmp_path)
    loaded = SampleSet.load(tmp_path)
    assert np.array_equal(loaded.weights, samples.weights) and np.array_equal(loaded.tau, tau)
    assert loaded.metadata["settings"]["num_samples"] == 3


def test_sample_split_refuses_a_mismatched_universe_or_split(real, model):
    tickers = list(real["dataset"].panel.tickers)
    swapped = [tickers[1], tickers[0], *tickers[2:]]  # injected: two assets swapped
    with pytest.raises(ValueError, match="order"):
        sample_split(_trained(real, model, tickers=swapped), real["dataset"], "test", device="cpu")
    with pytest.raises(ValueError, match="split"):
        sample_split(
            _trained(real, model), real["dataset"], "test", device="cpu", tau=real["train_tau"][:2]
        )


def test_configs_without_a_sampling_section_load_with_defaults():
    config = DiffolioConfig.from_dict({"name": "old"})
    assert config.sampling == SamplingConfig()
    for key, value in [("num_samples", 0), ("covariance", "ewma"), ("shrinkage", "oas"),
                       ("guidance_scale", -1.0), ("covariance_window", 1)]:
        bad = DiffolioConfig()
        setattr(bad.sampling, key, value)
        with pytest.raises(ValueError, match=key):
            bad.validate()
    assert math.isclose(SamplingConfig().guidance_scale, 1.0)
