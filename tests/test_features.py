from __future__ import annotations

import numpy as np

from diffolio.config import FeatureConfig
from diffolio.data.features import FeatureBuilder, FeatureStats, build_feature_tensors
from synthetic import make_ohlcv, make_universe_frames, trading_days


def test_shapes_and_warmup():
    index = trading_days(periods=60)
    frame = make_ohlcv(index, seed=0)
    builder = FeatureBuilder(FeatureConfig(ref_window=20))

    values = builder.transform(frame)

    assert values.shape == (60, 5)
    assert values.dtype == np.float32
    assert builder.warmup == 19
    assert np.isnan(values[:19]).any(axis=1).all()
    assert np.isfinite(values[19:]).all()


def test_normalisation_puts_assets_on_a_comparable_scale():
    index = trading_days(periods=80)
    cheap = make_ohlcv(index, seed=1, price0=3.0, volume=1.0e5)
    pricey = make_ohlcv(index, seed=2, price0=3000.0, volume=1.0e9)
    builder = FeatureBuilder(FeatureConfig(ref_window=20))

    a = builder.transform(cheap)[20:]
    b = builder.transform(pricey)[20:]

    # Prices are expressed relative to their own trailing mean, so a $3 stock
    # and a $3000 stock land in the same numeric range.
    assert abs(a.std() - b.std()) < 0.05
    assert abs(a.mean()) < 0.2 and abs(b.mean()) < 0.2


def test_transform_is_causal():
    index = trading_days(periods=60)
    frame = make_ohlcv(index, seed=3)
    builder = FeatureBuilder(FeatureConfig(ref_window=10))
    baseline = builder.transform(frame)

    perturbed = frame.copy()
    perturbed.iloc[40:] *= 3.0
    after = builder.transform(perturbed)

    np.testing.assert_allclose(baseline[:40], after[:40], rtol=0, atol=0, equal_nan=True)
    assert not np.allclose(baseline[40:], after[40:], equal_nan=True)


def test_prev_close_and_raw_modes():
    index = trading_days(periods=30)
    frame = make_ohlcv(index, seed=4)

    prev = FeatureBuilder(FeatureConfig(price_normalization="prev_close", volume_transform="log"))
    assert prev.warmup == 1
    values = prev.transform(frame)
    expected = frame["close"].iloc[5] / frame["close"].iloc[4] - 1.0
    assert abs(float(values[5, 3]) - expected) < 1e-6

    raw = FeatureBuilder(FeatureConfig(price_normalization="none", volume_transform="none"))
    assert raw.warmup == 0
    assert abs(float(raw.transform(frame)[0, 0]) - float(frame["open"].iloc[0])) < 1e-3


def test_feature_stats_fit_and_freeze():
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, size=(200, 4, 3)).astype("float32")
    stats = FeatureStats.fit(train, clip_sigma=10.0)

    standardized = stats.transform(train)
    assert np.allclose(standardized.mean(axis=(0, 1)), 0.0, atol=1e-4)
    assert np.allclose(standardized.std(axis=(0, 1)), 1.0, atol=1e-3)

    # A shifted "test" block must be transformed with the frozen statistics,
    # so its mean is *not* re-centred on zero.
    test = train + 3.0
    assert stats.transform(test).mean() > 1.0

    restored = FeatureStats.from_dict(stats.to_dict())
    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.std, stats.std)


def test_constant_feature_does_not_divide_by_zero():
    stats = FeatureStats.fit(np.ones((50, 2), dtype="float32"))
    out = stats.transform(np.ones((50, 2), dtype="float32"))
    assert np.isfinite(out).all()


def test_build_feature_tensors_uses_canonical_ticker_order():
    index = trading_days(periods=50)
    frames = make_universe_frames(["A", "B", "C"], index)
    benchmark = make_ohlcv(index, seed=9)
    config = FeatureConfig(ref_window=5)

    features, index_features, warmup = build_feature_tensors(
        frames, ["C", "A", "B"], benchmark, config
    )

    assert features.shape == (50, 3, 5)
    assert index_features.shape == (50, 5)
    assert warmup == 4
    builder = FeatureBuilder(config)
    np.testing.assert_allclose(features[:, 0], builder.transform(frames["C"]))
