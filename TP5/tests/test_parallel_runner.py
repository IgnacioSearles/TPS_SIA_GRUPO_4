"""--jobs: corridas en procesos separados con los mismos resultados que en serie."""

import csv
import json

import pytest

from src.experiments import runner


def _config(output):
    return {"output": str(output), "seeds": [7, 17],
            "dataset": {"type": "synthetic_notes", "pitch_range": [59, 61],
                        "pitch_study": {"targets": [60], "omit_targets": True},
                        "spectrogram": {"mel_bands": 8, "fft_size": 256, "hop_length": 128,
                                        "griffin_lim_iterations": 1, "mel_inversion_iterations": 2}},
            "model": {"type": "variational", "hidden_layers": [4], "latent_size": 2, "loss": "mse"},
            "training": {"epochs": 2}, "evaluation": {"pitch_interpolation": {}}}


def _rows(directory):
    with (directory / "summary.csv").open() as stream:
        return {row["run"]: row for row in csv.DictReader(stream)}


def test_parallel_runs_match_serial(tmp_path):
    paths = []
    for mode in ("serial", "parallel"):
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(_config(tmp_path / mode)))
        paths.append(path)
    serial = runner.run_experiments([paths[0]], jobs=1)[0]
    parallel = runner.run_experiments([paths[1]], jobs=2)[0]
    serial_rows, parallel_rows = _rows(serial), _rows(parallel)
    assert set(serial_rows) == set(parallel_rows) == {"base__seed=7", "base__seed=17"}
    for run, row in serial_rows.items():
        assert float(parallel_rows[run]["final_loss"]) == pytest.approx(float(row["final_loss"]),
                                                                        rel=1e-4)
        assert (parallel / run / "model.npz").is_file()
