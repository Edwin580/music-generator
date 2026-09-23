import pytest

from musegen.config import PRESET_DIR, ExperimentConfig


def test_overrides_parse_values():
    cfg = ExperimentConfig().with_overrides(["train.epochs=3", "model.dropout=0.5",
                                             "generation.key=A minor", "generation.top_k=null"])
    assert cfg.train.epochs == 3 and cfg.model.dropout == 0.5
    assert cfg.generation.key == "A minor" and cfg.generation.top_k is None


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        ExperimentConfig().with_overrides(["train.epoch=3"])
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({"trian": {}})


def test_arch_representation_mismatch_is_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"model": {"arch": "pianoroll_lstm"}})


def test_yaml_round_trip(tmp_path):
    cfg = ExperimentConfig().with_overrides(["model.arch=lstm", "name=x"])
    cfg.save_yaml(tmp_path / "c.yaml")
    assert ExperimentConfig.from_yaml(tmp_path / "c.yaml") == cfg


@pytest.mark.parametrize("name", ["transformer", "lstm", "pianoroll_lstm", "smoke"])
def test_shipped_configs_are_valid(name):
    assert ExperimentConfig.load(name) == ExperimentConfig.load(PRESET_DIR / f"{name}.yaml")


def test_unknown_preset():
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.load("nope")
