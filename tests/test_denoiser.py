"""Section 9 tests: the risk-aware denoising head and the joint model.

Inputs are real: windows and section-6 targets from the S&P 500 build
(N = 224, L = 256, F = 5, gamma_max = 5), noised with the section-7 schedule
fitted on the real training split.  The sinusoidal-embedding checks are pure
arithmetic and use no market data.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffolio.config import DiffolioConfig
from diffolio.data.targets import portfolio_dataset
from diffolio.model import DenoisingHead, DiffolioModel, sinusoidal_embedding


@pytest.fixture(scope="module")
def schedule(real_schedule):
    return real_schedule


@pytest.fixture(scope="module")
def batch(real_dataset, real_targets):
    ds = portfolio_dataset(real_dataset, "train", real_targets)
    return ds.stack([0, len(ds) // 3, len(ds) - 1])


def _model(real_dataset, schedule, seed=0, **model):
    config = DiffolioConfig.from_dict(real_dataset.config.to_dict())
    for key, value in model.items():
        setattr(config.model, key, value)
    torch.manual_seed(seed)
    return DiffolioModel.from_config(config, real_dataset.n_assets, float(schedule.sigma_x))


@pytest.fixture(scope="module")
def model(real_dataset, schedule):
    return _model(real_dataset, schedule).eval()


@pytest.fixture(scope="module")
def noised(batch, schedule):
    """One risk level per sample, noised at mixed timesteps."""
    gamma = torch.tensor([0, 2, 4])
    t = torch.tensor([1, 250, schedule.num_steps])
    x0 = batch.x0[torch.arange(3), gamma]
    x_t = schedule.q_sample(x0, t, generator=torch.Generator().manual_seed(0))
    return x_t, t, gamma, x0


# -- sinusoidal embedding (pure arithmetic) -------------------------------------


def test_sinusoidal_embedding_values():
    emb = sinusoidal_embedding(torch.tensor([0, 1, 7]), 8)
    assert emb.shape == (3, 8)
    torch.testing.assert_close(emb[0], torch.tensor([0.0] * 4 + [1.0] * 4))
    freqs = [10_000 ** (-i / 4) for i in range(4)]
    expected = [math.sin(7 * f) for f in freqs] + [math.cos(7 * f) for f in freqs]
    torch.testing.assert_close(emb[2], torch.tensor(expected))
    assert sinusoidal_embedding(torch.tensor([3]), 7).shape == (1, 7)  # odd d padded


def test_every_timestep_gets_a_distinct_embedding():
    emb = sinusoidal_embedding(torch.arange(1, 501), 1280)
    distance = torch.cdist(emb, emb) + torch.eye(500) * 1e9
    assert distance.min() > 0.1


# -- the head on real inputs -------------------------------------------------------


def test_shapes_and_components(real_dataset, model):
    head = model.head
    n, d = real_dataset.n_assets, real_dataset.config.embed_dim
    assert (head.n_assets, head.embed_dim) == (224, 1280)
    assert head.w_omega.weight.shape == (d, d)
    assert head.risk_embedding.weight.shape == (5, d)
    assert head.w_x.weight.shape == (d, n) and head.w_x.bias is None
    linears = [m for m in head.mlp if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(2 * d, d), (d, d), (d, n)]
    assert isinstance(head.mlp[-1], torch.nn.Linear)  # no output activation


def test_forward_matches_the_longhand_equations(model, batch, noised):
    x_t, t, gamma, _ = noised
    head = model.head
    with torch.no_grad():
        z = model.encode(batch.h, batch.g).z_merged
        omega = torch.sigmoid(head.w_omega(sinusoidal_embedding(t, 1280)))  # 9.1
        v = head.risk_embedding.weight[gamma]  # 9.2
        d_t = (x_t @ head.w_x.weight.T) * (omega + v)  # 9.3, Eq. 17
        expected = head.mlp(torch.cat([z, d_t], dim=-1))  # 9.4-9.5
        out = model(x_t, t, batch.h, batch.g, gamma)
    torch.testing.assert_close(out.x_hat, expected)
    torch.testing.assert_close(out.encoding.z_merged, z)
    assert out.x_hat.shape == (3, 224)


def test_output_depends_on_every_conditioning_input(model, batch, noised):
    x_t, t, gamma, _ = noised
    with torch.no_grad():
        z = model.encode(batch.h, batch.g).z_merged
        base = model.denoise(x_t, t, gamma, z)
        for changed in (
            model.denoise(x_t, t, (gamma + 1) % 5, z),
            model.denoise(x_t, t % 500 + 1, gamma, z),
            model.denoise(x_t.roll(1, 0), t, gamma, z),
            model.denoise(x_t, t, gamma, z.roll(1, 0)),
        ):
            assert (changed - base).abs().amax(-1).min() > 0


def test_every_risk_level_of_a_sample_shares_one_encoding(model, batch, schedule):
    # (B, G, N): all five risk levels of each window at once, encoder run once.
    x0 = batch.x0
    t = torch.full((3, 5), 100)
    x_t = schedule.q_sample(x0, t[:, 0], generator=torch.Generator().manual_seed(1))
    gamma = torch.arange(5).expand(3, 5)
    with torch.no_grad():
        z = model.encode(batch.h, batch.g).z_merged
        all_levels = model.denoise(x_t, t, gamma, z)
        for level in range(5):
            one = model.denoise(x_t[:, level], t[:, 0], level, z)
            torch.testing.assert_close(all_levels[:, level], one)


def test_scaling_by_sigma_x_is_a_reparametrisation(real_dataset, schedule, batch, noised):
    # Moving sigma_x into W and the last layer reproduces the unscaled head exactly.
    x_t, t, gamma, _ = noised
    plain = _model(real_dataset, schedule, scale_by_sigma_x=False).eval()
    scaled = _model(real_dataset, schedule, scale_by_sigma_x=True).eval()
    scaled.load_state_dict(plain.state_dict())
    sigma = float(schedule.sigma_x)
    with torch.no_grad():
        scaled.head.w_x.weight.mul_(sigma)
        scaled.head.mlp[-1].weight.div_(sigma)
        scaled.head.mlp[-1].bias.div_(sigma)
        a = plain(x_t, t, batch.h, batch.g, gamma).x_hat
        b = scaled(x_t, t, batch.h, batch.g, gamma).x_hat
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_gradients_flow_into_the_encoder_and_the_head_jointly(real_dataset, schedule, batch, noised):
    model = _model(real_dataset, schedule)
    x_t, t, gamma, x0 = noised
    loss = (model(x_t, t, batch.h, batch.g, gamma).x_hat - x0).pow(2).sum(-1).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if name == "head.risk_embedding.weight":
            # Only the rows of the risk levels in the batch are used.
            used = p.grad.abs().sum(-1) > 0
            assert used.tolist() == [True, False, True, False, True]
            continue
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_the_joint_model_fits_a_fixed_real_batch(real_dataset, schedule, batch, noised):
    # Sanity check that the wiring trains: 60 Adam steps on one batch at small t.
    model = _model(real_dataset, schedule, seed=1)
    _, _, gamma, x0 = noised
    t = torch.tensor([5, 5, 5])
    x_t = schedule.q_sample(x0, t, generator=torch.Generator().manual_seed(2))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(60):
        loss = (model(x_t, t, batch.h, batch.g, gamma).x_hat - x0).pow(2).sum(-1).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * x0.pow(2).sum(-1).mean().item()


def test_sigma_x_travels_with_the_weights(model, schedule):
    state = model.state_dict()
    assert float(state["head.sigma_x"]) == pytest.approx(float(schedule.sigma_x))


def test_rejects_bad_inputs(model, batch, noised):
    x_t, t, gamma, _ = noised
    z = torch.zeros(3, 1280)
    with pytest.raises(IndexError, match="gamma"):
        model.denoise(x_t, t, torch.tensor([0, 1, 5]), z)
    with pytest.raises(IndexError, match="gamma"):
        model.denoise(x_t, t, -1, z)
    with pytest.raises(IndexError, match="timesteps"):
        model.denoise(x_t, torch.tensor([0, 1, 2]), gamma, z)
    with pytest.raises(ValueError, match="N=224"):
        model.denoise(x_t[:, :10], t, gamma, z)
    with pytest.raises(ValueError, match="z_merged"):
        model.denoise(x_t, t, gamma, z[:2])
    with pytest.raises(ValueError, match="assets"):
        model.encode(batch.h[:, :10], batch.g)


def test_config_validation():
    for key, value, match in (
        ("mlp_layers", 1, "mlp_layers"),
        ("mlp_activation", "tanh", "mlp_activation"),
        ("time_embedding_activation", "relu", "time_embedding_activation"),
    ):
        config = DiffolioConfig()
        setattr(config.model, key, value)
        with pytest.raises(ValueError, match=match):
            config.validate()
    with pytest.raises(ValueError):
        DenoisingHead(10, 16, 5, sigma_x=0.0)
