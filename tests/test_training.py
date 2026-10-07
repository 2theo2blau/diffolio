"""Section 11 tests: the training loop, on real S&P 500 data.

Every batch is built from the real build (N = 224, L = 256): real windows,
section-6 targets and the section-7 schedule fitted on the real training
split.  To keep the tests fast on CPU they train on a handful of real
training steps (and validate on a few real validation steps) with small
batches.  The injected parts are the validation-loss sequences in the
early-stopping tests, which are made-up numbers driving the bookkeeping.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from diffolio.data.windows import stack_windows
from diffolio.training import (
    EarlyStopping,
    LossMetrics,
    Trainer,
    load_checkpoint,
    load_trained,
)

N_TRAIN, N_VAL = 8, 4


@pytest.fixture(scope="module")
def real(real_dataset, real_targets, real_schedule):
    train = real_dataset.splits.train.tau
    val = real_dataset.splits.val.tau
    return {
        "dataset": real_dataset,
        "targets": real_targets,
        "schedule": real_schedule,
        "tensors": real_dataset.panel.torch(),
        "train_tau": train[:: len(train) // N_TRAIN][:N_TRAIN],
        "val_tau": val[:: len(val) // N_VAL][:N_VAL],
    }


@pytest.fixture
def make_trainer(real):
    def make(output_dir=None, **training) -> Trainer:
        config = copy.deepcopy(real["dataset"].config)
        settings = dict(batch_size=4, max_epochs=2, patience=50, val_repeats=2)
        for key, value in {**settings, **training}.items():
            setattr(config.training, key, value)
        return Trainer(
            config,
            real["tensors"],
            real["targets"],
            real["train_tau"],
            real["val_tau"],
            copy.deepcopy(real["schedule"]),
            output_dir=output_dir,
            device="cpu",
        )

    return make


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _assert_same_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)


def test_batch_follows_algorithm_1(make_trainer, real):
    trainer = make_trainer()
    positions = torch.tensor([0, 3, 5, 7])
    state = trainer.train_generator.get_state()
    batch = trainer.make_batch(positions)

    # Replay the training stream: gamma, then t, then eps.
    gen = torch.Generator().set_state(state)
    gamma = torch.randint(0, 5, (4,), generator=gen)
    t = torch.randint(1, 501, (4,), generator=gen)
    eps = torch.randn((4, real["dataset"].n_assets), generator=gen)
    assert torch.equal(batch.gamma, gamma) and torch.equal(batch.t, t)

    tau = real["train_tau"][positions.numpy()]
    x0 = torch.from_numpy(np.array(real["targets"].select(tau).targets))[torch.arange(4), gamma]
    torch.testing.assert_close(batch.x0, x0)
    s = real["schedule"]
    expected = (
        s.alpha_bars[t, None].sqrt() * x0
        + (1 - s.alpha_bars[t, None]).sqrt() * s.sigma_x * eps
    )
    torch.testing.assert_close(batch.x_t, expected)

    window = stack_windows(real["tensors"], tau, real["dataset"].config.lookback)
    assert torch.equal(batch.h, window.h) and torch.equal(batch.g, window.g)
    assert torch.equal(batch.r, window.r) and torch.equal(batch.r_valid, window.r_valid)


def test_steps_lower_the_denoising_loss_on_a_real_batch(make_trainer):
    trainer = make_trainer()
    batch = trainer.make_batch(torch.arange(N_TRAIN))
    losses = [float(trainer.train_step(batch).denoise.detach()) for _ in range(30)]
    assert losses[-1] < 0.5 * losses[0], losses


def test_one_step_updates_the_encoder_head_and_w_p(make_trainer):
    trainer = make_trainer()
    in_optimiser = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    w_p = trainer.loss.projection.w_p.weight
    assert id(w_p) in in_optimiser

    before = {
        "w_p": w_p.detach().clone(),
        "encoder": trainer.model.encoder.w_h.weight.detach().clone(),
        "head": trainer.model.head.w_x.weight.detach().clone(),
    }
    trainer.train_step(trainer.make_batch(torch.arange(4)))
    after = {
        "w_p": w_p,
        "encoder": trainer.model.encoder.w_h.weight,
        "head": trainer.model.head.w_x.weight,
    }
    for name in before:
        assert not torch.equal(before[name], after[name]), name


def test_validation_is_deterministic_and_leaves_training_untouched(make_trainer):
    trainer = make_trainer()
    params, stream = _state(trainer.model), trainer.train_generator.get_state()
    first = trainer.evaluate()
    assert first == trainer.evaluate()  # the same fixed draws every call
    _assert_same_state(params, _state(trainer.model))
    assert torch.equal(stream, trainer.train_generator.get_state())
    assert all(np.isfinite(first))
    # More draws per sample change the estimate, not its scale.
    other = make_trainer(val_repeats=3).evaluate()
    assert other != first and 0.5 < other.denoise / first.denoise < 2


def test_validation_scores_every_risk_level(make_trainer, real):
    trainer = make_trainer(val_repeats=2)
    k, g_max, n = 2, 5, real["dataset"].n_assets
    # Rebuild the validation draws (all four steps fit one batch of 4) and
    # score each risk level separately with its own gamma.
    gen = torch.Generator().manual_seed(trainer.config.training.seed + 2)
    t = torch.randint(1, 501, (N_VAL, k, g_max), generator=gen)
    eps = torch.randn((N_VAL, k, g_max, n), generator=gen)
    x0 = trainer.val_x0[:, None].expand(N_VAL, k, g_max, n)
    s = real["schedule"]
    abar = s.alpha_bars[t][..., None]
    x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * s.sigma_x * eps
    window = stack_windows(real["tensors"], real["val_tau"], real["dataset"].config.lookback)
    trainer.model.eval()
    with torch.no_grad():
        z = trainer.model.encode(window.h, window.g).z_merged
        errors = [
            (trainer.model.denoise(x_t[:, :, g], t[:, :, g], g, z) - x0[:, :, g]).pow(2).sum(-1)
            for g in range(g_max)
        ]
    expected = torch.stack(errors).mean()
    assert trainer.evaluate().denoise == pytest.approx(float(expected), rel=1e-5)


def test_early_stopping_counts_epochs_without_strict_improvement():
    stopper = EarlyStopping(patience=2)
    # Injected validation losses: improve, tie, worsen -> stop.
    assert stopper.update(1.0, 0) and stopper.update(0.5, 1)
    assert not stopper.update(0.5, 2) and not stopper.should_stop
    assert not stopper.update(0.9, 3) and stopper.should_stop
    assert (stopper.best, stopper.best_epoch) == (0.5, 1)
    assert not EarlyStopping(1).update(float("nan"), 0)
    restored = EarlyStopping(2)
    restored.load_state_dict(stopper.state_dict())
    assert restored.state_dict() == stopper.state_dict()


def test_fit_keeps_the_best_epoch_and_stops_early(make_trainer, tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, patience=2, max_epochs=10)
    # Training is real; the validation losses are injected so the best epoch
    # (1) and the stop (epoch 3, two epochs without improvement) are known.
    values, snapshots = iter([1.0, 0.5, 0.7, 0.8]), []

    def fake_evaluate():
        snapshots.append(_state(trainer.model))
        value = next(values)
        return LossMetrics(value, value, 0.0)

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)
    history = trainer.fit()

    assert [r["epoch"] for r in history] == [0, 1, 2, 3]
    assert [r["best"] for r in history] == [True, True, False, False]
    assert history[0]["train_total"] is None and history[1]["train_total"] is not None
    best = load_checkpoint(tmp_path / "best.pt")
    assert best["epoch"] == 1 and best["best_value"] == 0.5 and "optimizer" not in best
    _assert_same_state(best["model"], snapshots[1])
    assert load_checkpoint(tmp_path / "last.pt")["epoch"] == 3
    assert (tmp_path / "history.json").exists() and (tmp_path / "config.yaml").exists()


def test_checkpoint_rebuilds_the_trained_model(make_trainer, tmp_path, real):
    trainer = make_trainer(tmp_path, max_epochs=1)
    trainer.fit()
    rebuilt = load_trained(tmp_path / "last.pt")

    assert rebuilt.config.to_dict() == trainer.config.to_dict()
    assert float(rebuilt.schedule.sigma_x) == float(real["schedule"].sigma_x)
    _assert_same_state(rebuilt.loss.state_dict(), trainer.loss.state_dict())
    batch = trainer.make_batch(torch.arange(4))
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(batch.x_t, batch.t, batch.h, batch.g, batch.gamma).x_hat
        actual = rebuilt.model(batch.x_t, batch.t, batch.h, batch.g, batch.gamma).x_hat
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_resume_reproduces_an_uninterrupted_run(make_trainer, tmp_path):
    straight = make_trainer(tmp_path / "straight")
    straight.fit()

    interrupted = make_trainer(tmp_path / "interrupted")
    interrupted.fit(max_epochs=1)
    assert _same_losses(interrupted.history, straight.history[:2])  # same seed, same run
    resumed = make_trainer(tmp_path / "interrupted")
    resumed.resume(tmp_path / "interrupted" / "last.pt")
    resumed.fit()

    assert resumed.epoch == straight.epoch == 2
    assert _same_losses(resumed.history, straight.history)
    _assert_same_state(_state(resumed.model), _state(straight.model))
    _assert_same_state(_state(resumed.loss), _state(straight.loss))


def _same_losses(a: list[dict], b: list[dict]) -> bool:
    keys = [k for k in a[0] if k.startswith(("train_", "val_"))]
    return len(a) == len(b) and all(x[k] == y[k] for x, y in zip(a, b) for k in keys)


def test_bf16_autocast_trains_fp32_weights(make_trainer):
    trainer = make_trainer(precision="bf16")
    out = trainer.train_step(trainer.make_batch(torch.arange(4)))
    assert torch.isfinite(out.total) and out.total.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in trainer.parameters)


def test_from_dataset_trains_on_the_real_splits(real):
    dataset = real["dataset"]
    trainer = Trainer.from_dataset(dataset.config, dataset, real["targets"], device="cpu")
    assert np.array_equal(trainer.train_tau, dataset.splits.train.tau)
    assert np.array_equal(trainer.val_tau, dataset.splits.val.tau)
    assert trainer.sigma_x == pytest.approx(float(real["schedule"].sigma_x))
    assert trainer.metadata["tickers"] == list(dataset.panel.tickers)

    mismatched = copy.deepcopy(dataset.config)
    mismatched.window.lookback = 128
    with pytest.raises(ValueError, match="data sections"):
        Trainer.from_dataset(mismatched, dataset, real["targets"], device="cpu")


@pytest.mark.parametrize(
    "key, value",
    [
        ("precision", "fp8"),
        ("monitor", "aux"),
        ("lr_schedule", "step"),
        ("val_repeats", 0),
        ("grad_clip", 0.0),
        ("learning_rate", 0.0),
        ("patience", 0),
    ],
)
def test_config_rejects_bad_training_settings(real, key, value):
    config = copy.deepcopy(real["dataset"].config)
    setattr(config.training, key, value)
    with pytest.raises(ValueError, match=key):
        config.validate()
