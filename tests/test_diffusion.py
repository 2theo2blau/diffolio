"""Section 7 tests.

Schedule identities are pure arithmetic and need no market data.  Everything
touching sigma_x or the forward/posterior distributions runs on the real S&P
500 portfolios (``real_dataset`` / ``real_targets`` from conftest).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from diffolio.config import DiffolioConfig, DiffusionConfig
from diffolio.diffusion import DiffusionSchedule, estimate_sigma_x, fit_schedule, make_betas

T = 500


def _schedule(sigma_x=0.0064, **kwargs) -> DiffusionSchedule:
    return DiffusionSchedule.from_config(DiffusionConfig(num_steps=T, **kwargs), sigma_x)


# ---------------------------------------------------------------------------
# Schedule arithmetic (plan 7.1, 7.4).


def test_linear_betas_span_the_configured_range():
    betas = make_betas(T, "linear", 1e-4, 0.02)
    assert betas.shape == (T,)
    assert betas[0] == pytest.approx(1e-4) and betas[-1] == pytest.approx(0.02)
    assert (np.diff(betas) > 0).all()


def test_cosine_betas_follow_the_closed_form_alpha_bar():
    betas = make_betas(T, "cosine", cosine_offset=0.008)
    assert ((betas > 0) & (betas <= 0.999)).all()
    abar = np.cumprod(1 - betas)
    s = 0.008
    f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
    for t in (1, T // 2, T - 10):
        assert abar[t - 1] == pytest.approx(f(t / T) / f(0.0), rel=1e-9)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_buffers_are_one_based_with_a_t0_boundary(schedule):
    sched = _schedule(beta_schedule=schedule)
    betas = torch.tensor(make_betas(T, schedule), dtype=torch.float64)

    assert sched.num_steps == T
    assert sched.betas.shape == (T + 1,)
    assert float(sched.alpha_bars[0]) == 1.0
    torch.testing.assert_close(sched.betas[1:].double(), betas, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        sched.alpha_bars[1:].double(), torch.cumprod(1 - betas, 0), rtol=1e-5, atol=0
    )
    torch.testing.assert_close(sched.sqrt_alpha_bars**2, sched.alpha_bars)
    torch.testing.assert_close(sched.sqrt_one_minus_alpha_bars**2, 1 - sched.alpha_bars)


def test_posterior_coefficients_satisfy_the_ddpm_identities():
    sched = _schedule()
    t = torch.arange(1, T + 1)
    abar, abar_prev = sched.alpha_bars[t].double(), sched.alpha_bars[t - 1].double()
    beta, alpha = sched.betas[t].double(), sched.alphas[t].double()

    # A noiseless x_t = sqrt(abar_t) x0 must map back to sqrt(abar_{t-1}) x0.
    c0, ct = sched.posterior_coef_x0[t].double(), sched.posterior_coef_xt[t].double()
    torch.testing.assert_close(c0 + ct * abar.sqrt(), abar_prev.sqrt(), rtol=1e-5, atol=1e-7)

    # Bayes: 1 / var = alpha / beta + 1 / (1 - abar_{t-1})   (unit sigma_x).
    var = sched.posterior_variance_unit[t].double()
    # (t = 1 has 1 - abar_0 = 0, i.e. a point mass, so start at t = 2.)
    bayes = 1.0 / (alpha[1:] / beta[1:] + 1.0 / (1.0 - abar_prev[1:]))
    torch.testing.assert_close(var[1:], bayes, rtol=1e-4, atol=0)
    assert float(var[0]) == 0.0  # t = 1: the posterior collapses onto x_0


def test_posterior_variance_carries_sigma_x_exactly_once():
    sched = _schedule(sigma_x=0.01)
    x = torch.zeros(3, 7)
    t = torch.tensor([1, 2, T])
    var = sched.posterior_variance(t, x)
    assert var.shape == (3, 1)
    expected = sched.posterior_variance_unit[t] * 0.01**2
    torch.testing.assert_close(var[:, 0], expected)
    torch.testing.assert_close(sched.posterior_std(t, x) ** 2, var)


def test_timesteps_outside_one_to_T_are_rejected():
    sched = _schedule()
    x = torch.zeros(2, 4)
    for bad in ([0, 1], [1, T + 1]):
        with pytest.raises(IndexError):
            sched.q_sample(x, torch.tensor(bad))
    t = sched.sample_timesteps(10_000, generator=torch.Generator().manual_seed(0))
    assert int(t.min()) == 1 and int(t.max()) == T


def test_a_short_linear_schedule_warns_about_the_prior(caplog):
    with caplog.at_level("WARNING"):
        DiffusionSchedule.from_config(DiffusionConfig(num_steps=100), 0.0064)
    assert "sqrt(abar_T)" in caplog.text


def test_schedule_travels_with_the_state_dict():
    sched = _schedule(sigma_x=0.0123)
    clone = _schedule(sigma_x=1.0)
    clone.load_state_dict(sched.state_dict())
    assert float(clone.sigma_x) == pytest.approx(0.0123)


def test_config_rejects_bad_diffusion_settings():
    for key, value in (
        ("num_steps", 0),
        ("beta_end", 1.5),
        ("beta_schedule", "quadratic"),
        ("sigma_x_source", "val"),
    ):
        config = DiffolioConfig()
        setattr(config.diffusion, key, value)
        with pytest.raises(ValueError):
            config.validate()


# ---------------------------------------------------------------------------
# sigma_x and the forward/posterior distributions on the real S&P 500 data.


def test_sigma_x_is_the_rms_of_the_training_base_portfolios(real_dataset, real_targets):
    train = real_targets.select(real_dataset.splits.train)
    expected = float(np.sqrt(np.mean(np.asarray(train.base, dtype=np.float64) ** 2)))

    sched = fit_schedule(real_dataset.config.diffusion, real_targets, real_dataset.splits.train)
    assert float(sched.sigma_x) == pytest.approx(expected, rel=1e-6)

    # A base portfolio has sum |x| = 1 over N assets, so its RMS entry sits
    # between the equal-weight 1/N and the one-asset 1/sqrt(N).
    n = real_targets.n_assets
    assert 1.0 / n < expected < 1.0 / math.sqrt(n)
    # mu_x = 0 is the paper's simplification; check it is near zero here too.
    assert abs(float(np.mean(train.base))) < 0.1 * expected


def test_sigma_x_is_fitted_on_training_steps_only(real_dataset, real_targets):
    config = real_dataset.config.diffusion
    with pytest.raises(ValueError, match="training split"):
        fit_schedule(config, real_targets, real_dataset.splits.test)

    # Estimating it on the test split would give a different number - which
    # is exactly the leak the guard above prevents.
    test = real_targets.select(real_dataset.splits.test)
    train = real_targets.select(real_dataset.splits.train)
    assert estimate_sigma_x(test.base) != pytest.approx(estimate_sigma_x(train.base), rel=1e-3)


def test_risk_target_pooling_is_the_flagged_deviation(real_dataset, real_targets, caplog):
    train = real_targets.select(real_dataset.splits.train)
    with caplog.at_level("WARNING"):
        pooled = estimate_sigma_x(train.base, train.targets, source="risk_targets")
    assert "deviates from the paper" in caplog.text
    # Concentrated risk levels hold fewer, larger weights, so pooling raises sigma.
    assert pooled > estimate_sigma_x(train.base)


def test_q_sample_has_the_forward_marginal_moments(real_dataset, real_targets):
    sched = fit_schedule(real_dataset.config.diffusion, real_targets, real_dataset.splits.train)
    train = real_targets.select(real_dataset.splits.train)
    gen = torch.Generator().manual_seed(0)

    x0 = torch.from_numpy(np.array(train.targets[:64, 2]))  # (64, N), gamma = 2
    reps = 400
    x0_rep = x0.repeat(reps, 1)
    for step in (1, 100, T):
        t = torch.full((x0_rep.shape[0],), step)
        x_t = sched.q_sample(x0_rep, t, generator=gen).view(reps, *x0.shape)
        mean = x_t.mean(0)
        std = x_t.std(0)

        want_mean = float(sched.sqrt_alpha_bars[step]) * x0
        want_std = float(sched.sqrt_one_minus_alpha_bars[step]) * float(sched.sigma_x)
        # Standard error of the mean is want_std / sqrt(reps).
        assert (mean - want_mean).abs().max() < 5 * want_std / math.sqrt(reps) + 1e-9
        assert float(std.mean()) == pytest.approx(want_std, rel=0.02)


def test_q_sample_is_exact_given_the_noise(real_dataset, real_targets):
    sched = fit_schedule(real_dataset.config.diffusion, real_targets, real_dataset.splits.train)
    x0 = torch.from_numpy(np.array(real_targets.targets[:8, 0]))
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1))
    t = torch.arange(1, 9) * 50
    got = sched.q_sample(x0, t, noise=eps)
    ab = sched.alpha_bars[t].unsqueeze(-1)
    torch.testing.assert_close(got, ab.sqrt() * x0 + (1 - ab).sqrt() * sched.sigma_x * eps)


def test_the_posterior_is_the_bayes_posterior_of_the_forward_chain(real_dataset, real_targets):
    """Sample x_{t-1} ~ q(.|x0), then x_t = sqrt(alpha_t) x_{t-1} + sqrt(beta_t) sigma_x eps.
    Conditioned on (x_t, x0), x_{t-1} must have the closed-form posterior mean
    and variance; checked by linear regression on real portfolio targets."""
    sched = fit_schedule(real_dataset.config.diffusion, real_targets, real_dataset.splits.train)
    sigma = float(sched.sigma_x)
    gen = torch.Generator().manual_seed(2)
    x0 = torch.from_numpy(np.array(real_targets.targets[:4, 4])).double()  # (4, N)

    for step in (2, 50, 400):
        reps = 20_000
        x0_rep = x0.repeat(reps, 1)
        prev = sched.q_sample(x0_rep, torch.full((x0_rep.shape[0],), step - 1), generator=gen)
        noise = torch.randn(prev.shape, generator=gen, dtype=prev.dtype)
        a, b = float(sched.alphas[step]), float(sched.betas[step])
        x_t = math.sqrt(a) * prev + math.sqrt(b) * sigma * noise

        mean = sched.posterior_mean(x0_rep, x_t, torch.full((x0_rep.shape[0],), step))
        resid = prev - mean
        # Residual must be zero-mean, uncorrelated with x_t, with the posterior variance.
        want_var = float(sched.posterior_variance_unit[step]) * sigma**2
        assert abs(float(resid.mean())) < 5 * math.sqrt(want_var / resid.numel())
        assert float(resid.var()) == pytest.approx(want_var, rel=0.02)
        centred = x_t - x_t.mean()
        corr = float((resid * centred).mean() / (resid.std() * centred.std()))
        assert abs(corr) < 0.01


def test_prior_sample_matches_sigma_x(real_dataset, real_targets):
    sched = fit_schedule(real_dataset.config.diffusion, real_targets, real_dataset.splits.train)
    x_T = sched.prior_sample((2000, real_targets.n_assets), torch.Generator().manual_seed(3))
    assert float(x_T.std()) == pytest.approx(float(sched.sigma_x), rel=0.01)
    assert abs(float(x_T.mean())) < 0.01 * float(sched.sigma_x) * 5
