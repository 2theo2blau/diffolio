"""Section 3.4 feature tests, on real cached yfinance bars."""

from __future__ import annotations

import numpy as np

from diffolio.config import FeatureConfig
from diffolio.data.features import FeatureBuilder, FeatureStats, build_feature_tensors


def test_shapes_and_warmup(real_frames):
    frame = real_frames(["AAPL"], "2018-01-01", "2018-03-31")["AAPL"].iloc[:60]
    builder = FeatureBuilder(FeatureConfig(ref_window=20))

    values = builder.transform(frame)

    assert values.shape == (60, 5)
    assert values.dtype == np.float32
    assert builder.warmup == 19
    assert np.isnan(values[:19]).any(axis=1).all()
    assert np.isfinite(values[19:]).all()


def test_normalisation_puts_assets_on_a_comparable_scale(real_frames):
    # F traded around $7 and NVR around $2,900 in 2018: a 400x price gap.
    frames = real_frames(["F", "NVR"], "2018-01-01", "2018-12-31")
    builder = FeatureBuilder(FeatureConfig(ref_window=20))

    cheap = builder.transform(frames["F"])[20:, :4]  # OHLC features
    pricey = builder.transform(frames["NVR"])[20:, :4]
    assert frames["NVR"]["close"].median() / frames["F"]["close"].median() > 100

    # Prices are expressed relative to their own trailing mean, so both land
    # in the same small range around zero; only volatility differs.
    for values in (cheap, pricey):
        assert abs(values.mean()) < 0.05
        assert 0.005 < values.std() < 0.2
    assert 1 / 3 < cheap.std() / pricey.std() < 3


def test_transform_is_causal(real_frames):
    frame = real_frames(["MSFT"], "2018-01-01", "2018-03-31")["MSFT"].iloc[:60]
    builder = FeatureBuilder(FeatureConfig(ref_window=10))
    baseline = builder.transform(frame)

    perturbed = frame.copy()
    perturbed.iloc[40:] *= 3.0  # rewrite the future
    after = builder.transform(perturbed)

    np.testing.assert_allclose(baseline[:40], after[:40], rtol=0, atol=0, equal_nan=True)
    assert not np.allclose(baseline[40:], after[40:], equal_nan=True)


def test_prev_close_and_raw_modes(real_frames):
    frame = real_frames(["JPM"], "2018-01-01", "2018-02-28")["JPM"].iloc[:30]

    prev = FeatureBuilder(FeatureConfig(price_normalization="prev_close", volume_transform="log"))
    assert prev.warmup == 1
    values = prev.transform(frame)
    expected = frame["close"].iloc[5] / frame["close"].iloc[4] - 1.0
    assert abs(float(values[5, 3]) - expected) < 1e-6
    assert abs(float(values[5, 4]) - np.log1p(frame["volume"].iloc[5])) < 1e-4

    raw = FeatureBuilder(FeatureConfig(price_normalization="none", volume_transform="none"))
    assert raw.warmup == 0
    assert abs(float(raw.transform(frame)[0, 0]) - float(frame["open"].iloc[0])) < 1e-3


def test_feature_stats_fit_and_freeze(real_frames):
    # Raw (pre-standardisation) features of real assets, split 2010-2017 / 2018+.
    tickers = ["AAPL", "MSFT", "JPM", "XOM"]
    frames = real_frames(tickers + ["^GSPC"])
    features, _, warmup = build_feature_tensors(
        {t: frames[t] for t in tickers}, tickers, frames["^GSPC"], FeatureConfig()
    )
    features = features[warmup:]
    cut = int(0.7 * len(features))
    stats = FeatureStats.fit(features[:cut], clip_sigma=0.0)

    standardized = stats.transform(features[:cut])
    assert np.allclose(standardized.mean(axis=(0, 1)), 0.0, atol=1e-3)
    assert np.allclose(standardized.std(axis=(0, 1)), 1.0, atol=1e-2)

    # Later data is transformed with the frozen training statistics, so it is
    # *not* re-centred: shifting it moves its mean by shift / std exactly.
    later = features[cut:]
    shifted = stats.transform(later + 1.0).mean(axis=(0, 1)) - stats.transform(later).mean(
        axis=(0, 1)
    )
    np.testing.assert_allclose(shifted, 1.0 / stats.std, rtol=1e-3)

    restored = FeatureStats.from_dict(stats.to_dict())
    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.std, stats.std)


def test_constant_feature_does_not_divide_by_zero():
    # Pure edge case: real features are never exactly constant.
    stats = FeatureStats.fit(np.ones((50, 2), dtype="float32"))
    out = stats.transform(np.ones((50, 2), dtype="float32"))
    assert np.isfinite(out).all()


def test_build_feature_tensors_uses_canonical_ticker_order(real_frames):
    frames = real_frames(["AAPL", "MSFT", "JPM", "^GSPC"], "2018-01-01", "2018-03-31")
    frames = {t: f.iloc[:50] for t, f in frames.items()}
    config = FeatureConfig(ref_window=5)

    features, index_features, warmup = build_feature_tensors(
        {t: frames[t] for t in ("AAPL", "MSFT", "JPM")},
        ["JPM", "AAPL", "MSFT"],
        frames["^GSPC"],
        config,
    )

    assert features.shape == (50, 3, 5)
    assert index_features.shape == (50, 5)
    assert warmup == 4
    builder = FeatureBuilder(config)
    np.testing.assert_allclose(features[:, 0], builder.transform(frames["JPM"]))
    np.testing.assert_allclose(features[:, 1], builder.transform(frames["AAPL"]))
    np.testing.assert_allclose(index_features, builder.transform(frames["^GSPC"]))
