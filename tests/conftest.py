import pytest

from musegen.config import ExperimentConfig
from musegen.data.sources import generate_synthetic_corpus


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic")
    generate_synthetic_corpus(path, n_pieces=12, seed=1)
    return path


@pytest.fixture
def tiny_config(tmp_path, synthetic_dir):
    def make(arch: str = "transformer") -> ExperimentConfig:
        rep = "pianoroll" if arch == "pianoroll_lstm" else "tokens"
        return ExperimentConfig.from_dict({
            "name": f"test-{arch}",
            "data": {"midi_dir": str(synthetic_dir), "cache_dir": str(tmp_path / "cache"),
                     "representation": rep, "seq_len": 64, "stride": 32},
            "model": {"arch": arch, "d_model": 32, "n_layers": 1, "n_heads": 2, "max_len": 128},
            "train": {"output_dir": str(tmp_path / "run"), "epochs": 1, "batch_size": 8,
                      "max_steps_per_epoch": 3, "device": "cpu", "warmup_steps": 1},
            "generation": {"bars": 2},
        })
    return make
