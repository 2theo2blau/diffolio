from __future__ import annotations

import tempfile
from pathlib import Path

from diffolio.config import DiffolioConfig, merge_overrides


def test_embed_dim_is_derived_from_lookback_and_features():
    config = DiffolioConfig()
    assert config.lookback == 256
    assert config.num_features == 5
    assert config.embed_dim == 256 * 5 == 1280

    config.window.lookback = 128
    assert config.embed_dim == 640


def test_yaml_roundtrip_preserves_every_section():
    config = DiffolioConfig()
    config.name = "roundtrip"
    config.universe.target_size = 42
    config.features.columns = ["open", "close"]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        config.to_yaml(path)
        restored = DiffolioConfig.from_yaml(path)

    assert restored.to_dict() == config.to_dict()
    assert restored.embed_dim == 256 * 2


def test_repo_config_loads_and_validates():
    config = DiffolioConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs/us_sp500.yaml")
    config.validate()
    assert config.universe.benchmark == "^GSPC"
    assert config.embed_dim == 1280


def test_unknown_key_is_rejected():
    try:
        DiffolioConfig.from_dict({"universe": {"not_a_field": 1}})
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("expected unknown keys to raise")


def test_validate_rejects_inconsistent_settings():
    config = DiffolioConfig()
    config.split.train_frac = 0.95
    config.split.val_frac = 0.1
    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected train+val >= 1 to raise")

    config = DiffolioConfig()
    config.features.columns = ["high", "low", "volume"]
    try:
        config.validate()
    except ValueError as exc:
        assert "open" in str(exc)
    else:
        raise AssertionError("expected missing open/close to raise")


def test_fingerprint_tracks_data_fields_only():
    config = DiffolioConfig()
    baseline = config.dataset_fingerprint()

    config.training.learning_rate = 1e-5
    config.model.hidden_dim = 512
    assert config.dataset_fingerprint() == baseline, "model/training must not force a rebuild"

    config.window.lookback = 64
    assert config.dataset_fingerprint() != baseline


def test_cli_overrides_are_typed():
    config = merge_overrides(
        DiffolioConfig(),
        ["data.end=2020-12-31", "universe.target_size=50", "features.standardize=false"],
    )
    assert config.data.end == "2020-12-31"
    assert config.universe.target_size == 50
    assert config.features.standardize is False
