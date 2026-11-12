import json

import numpy as np
import pytest

from src.experiments import runner
from src.experiments.grid import expand_runs
from src.options import ConfigurationError, Options

FONT_SOURCE = (
    "   {0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00},   // 0x60, `\n"
    "   {0x00, 0x0e, 0x01, 0x0d, 0x13, 0x13, 0x0d},   // 0x61, a\n"
    "   {0x10, 0x10, 0x10, 0x1c, 0x12, 0x12, 0x1c},   // 0x62, b\n"
)


def test_grid_expands_cartesian_product_with_seeds():
    runs = expand_runs(
        {
            "model": {"latent_size": 2},
            "grid": {
                "model.hidden_layers": [[8], [16, 8]],
                "training.optimizer.type": ["sgd", "adam"],
            },
            "seeds": [1, 2, 3],
        }
    )
    assert len(runs) == 12
    assert len({run.label for run in runs}) == 12
    first = runs[0]
    assert first.configuration == "hidden_layers=8__type=sgd"
    assert first.config["model"] == {"latent_size": 2, "hidden_layers": [8]}
    assert first.config["training"]["optimizer"]["type"] == "sgd"
    assert "grid" not in first.config and "seeds" not in first.config


def test_grid_runs_do_not_share_mutable_values():
    runs = expand_runs({"grid": {"model.hidden_layers": [[8]]}, "seeds": [1, 2]})
    runs[0].config["model"]["hidden_layers"].append(99)
    assert runs[1].config["model"]["hidden_layers"] == [8]


def test_grid_rejects_seed_conflicts():
    with pytest.raises(ConfigurationError):
        expand_runs({"seed": 1, "seeds": [1, 2]})
    with pytest.raises(ConfigurationError):
        expand_runs({"grid": {"seed": [1, 2]}})


def test_options_reject_unknown_keys():
    options = Options({"epochs": 10, "epocs": 20}, "training")
    options.take("epochs", 1)
    with pytest.raises(ConfigurationError, match="epocs"):
        options.ensure_consumed()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Proyecto mínimo aislado: un font de tres caracteres y resultados en tmp_path."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "font.h").write_text(FONT_SOURCE, encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_config(project, content):
    path = project / "experiment.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _read_summary(experiment_directory, run_label):
    path = experiment_directory / run_label / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_experiment_learns_trains_once_and_reevaluates(project):
    config_path = _write_config(
        project,
        {
            "seed": 3,
            "model": {"hidden_layers": [8]},
            "training": {"epochs": 600, "optimizer": {"type": "adam", "learning_rate": 0.02}},
            "evaluation": {
                "reconstruction": {},
                "latent": {},
                "latent_grid": {"size": 3},
                "interpolation": {"pairs": [["a", "b"]], "steps": 3},
                "sampling": {"count": 4},
                "denoising": {"levels": [0.0, 0.1], "trials": 2},
            },
        },
    )
    output = runner.run_experiment(config_path)
    run_directory = output / "base__seed=3"
    summary = _read_summary(output, "base__seed=3")
    assert summary["max_pixel_errors"] == 0
    assert summary["denoising_output_mean_error@0"] == 0.0
    for produced in ("latent.png", "latent_grid.png", "interpolation.png", "samples.png"):
        assert (run_directory / produced).is_file()

    model, dataset = runner.load_run(run_directory)
    assert np.array_equal(model.reconstruct(dataset.samples) >= 0.5, dataset.samples == 1)

    trained_at = (run_directory / "model.npz").stat().st_mtime_ns
    runner.run_experiment(config_path)  # ya terminada: no debe reentrenar
    assert (run_directory / "model.npz").stat().st_mtime_ns == trained_at

    (run_directory / "latent.png").unlink()
    runner.run_experiment(config_path, evaluate_only=True)
    assert (run_directory / "latent.png").is_file()
    reevaluated = _read_summary(output, "base__seed=3")
    assert reevaluated["training_seconds"] == summary["training_seconds"]


@pytest.mark.parametrize("noise", [None, {"type": "flip", "level": 0.1}])
def test_same_seed_reproduces_variational_training(project, noise):
    training = {"epochs": 30} if noise is None else {"epochs": 30, "noise": noise}
    config = {"training": training, "model": {"type": "variational"}, "evaluation": {}}
    first = runner.run_experiment(_write_config(project, {**config, "name": "first"}))
    second = runner.run_experiment(_write_config(project, {**config, "name": "second"}))
    first_loss = _read_summary(first, "base__seed=0")["final_loss"]
    assert first_loss == _read_summary(second, "base__seed=0")["final_loss"]


def test_typo_in_config_fails_before_training(project):
    config_path = _write_config(project, {"training": {"epocs": 10}})
    with pytest.raises(ConfigurationError, match="epocs"):
        runner.run_experiment(config_path)


def test_parallel_runs_match_serial(tmp_path):
    # Sin el fixture `project`: los procesos hijos no heredan el monkeypatch,
    # así que se usa el font real y solo la salida va a tmp_path.
    config = {"seeds": [1, 2], "training": {"epochs": 30}, "evaluation": {}}
    outputs = {}
    for jobs in (1, 2):
        path = tmp_path / f"jobs{jobs}.json"
        path.write_text(json.dumps({**config, "output": str(tmp_path / f"jobs{jobs}")}))
        outputs[jobs] = runner.run_experiment(path, jobs=jobs)
    for label in ("base__seed=1", "base__seed=2"):
        serial = _read_summary(outputs[1], label)["final_loss"]
        assert _read_summary(outputs[2], label)["final_loss"] == serial


def test_divergence_is_recorded_instead_of_aborting(project):
    config_path = _write_config(
        project,
        {
            "seeds": [1, 2],
            "model": {"precision": "float32"},
            "training": {"epochs": 50, "log_every": 1, "optimizer": {"type": "sgd", "learning_rate": 1e38}},
            "evaluation": {},
        },
    )
    output = runner.run_experiment(config_path)
    for label in ("base__seed=1", "base__seed=2"):
        assert runner.DIVERGED_KEY in _read_summary(output, label)
        assert (output / label / "config.json").is_file()
    assert runner.run_experiment(config_path, evaluate_only=True) == output
