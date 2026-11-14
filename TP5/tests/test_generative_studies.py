import csv
import json
from pathlib import Path
import hashlib

import numpy as np
import pytest

from src.analysis.generative_suite import compare_suite, CONFIGS
from src.data.factory import build_dataset
from src.data.shapes import render_shape
from src.evaluation.generative import nearest_training, pairwise_mse
from src.experiments import runner
from src.experiments.grid import expand_runs
from src.options import ConfigurationError, Options


def build(config, root):
    return build_dataset(Options(config, "dataset"), root)


def test_shapes_are_deterministic_unique_and_held_out(tmp_path):
    config = {"type": "geometric_shapes", "image_size": 16,
              "train_count": 20, "test_count": 8, "data_seed": 7}
    first, second = build(config, tmp_path), build(config, tmp_path)
    assert first.samples.shape == (20, 256)
    study = first.generative_study
    assert study.test_samples.shape == (8, 256)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(study.test_samples, second.generative_study.test_samples)
    all_samples = np.vstack([first.samples, study.test_samples])
    assert len({row.tobytes() for row in all_samples}) == 28
    assert not set(first.labels).intersection(study.test_labels)
    assert set(first.groups) == {"ellipse", "rectangle"}
    assert all_samples.min() >= 0 and all_samples.max() <= 1
    assert np.any((all_samples > 0) & (all_samples < 1))  # Antialias real.
    assert first.audio_codec is None and not first.is_binary
    subset = first.select([3, 1])
    assert [r["label"] for r in subset.generative_study.train_records] == subset.labels
    assert subset.generative_study.metadata["train_sha256"] == hashlib.sha256(subset.samples.tobytes()).hexdigest()


def test_renderer_preserves_basic_shape_properties():
    ellipse = render_shape(32, "ellipse", 8, 5, 0)
    rectangle = render_shape(32, "rectangle", 8, 5, 0)
    np.testing.assert_array_equal(ellipse, ellipse[::-1])
    np.testing.assert_array_equal(ellipse, ellipse[:, ::-1])
    assert 0 < ellipse.sum() < rectangle.sum()


def timbre_config():
    return {"type": "fixed_pitch_timbres", "train_count": 12, "test_count": 4,
            "data_seed": 8, "frequency_hz": 220,
            "spectrogram": {"duration_seconds": .5, "fft_size": 256, "hop_length": 128,
                            "mel_bands": 8, "griffin_lim_iterations": 1, "mel_inversion_iterations": 2}}


def test_timbres_vary_real_audio_without_changing_pitch(tmp_path):
    config = timbre_config()
    first, second = build(config, tmp_path), build(config, tmp_path)
    np.testing.assert_array_equal(first.samples, second.samples)
    study = first.generative_study
    assert first.samples.shape[1] == np.prod(first.image_shape)
    assert study.test_waveforms.shape == (4, 8000)
    assert len({r["amplitude_2"] for r in study.train_records}) == 12
    assert all(r["frequency_hz"] == 220 for r in study.train_records + study.test_records)
    assert not set(first.labels).intersection(study.test_labels)
    for record, waveform in zip(study.test_records, study.test_waveforms):
        assert sum(record[f"amplitude_{i}"] for i in range(1, 7)) == pytest.approx(1)
        assert np.max(abs(waveform)) == pytest.approx(1)
        power = abs(np.fft.rfft(waveform * np.hanning(len(waveform))))**2
        frequencies = np.fft.rfftfreq(len(waveform), 1 / first.audio_codec.sample_rate)
        harmonic = np.zeros_like(power, dtype=bool)
        for k in range(1, 7):
            harmonic |= abs(frequencies - 220 * k) < 15
        assert power[harmonic].sum() / power.sum() > .95


@pytest.mark.parametrize("data", [
    {"type": "geometric_shapes", "families": ["triangle"]},
    {"type": "geometric_shapes", "image_size": 4},
    {"type": "geometric_shapes", "train_count": True},
    {"type": "fixed_pitch_timbres", "frequency_hz": float("nan")},
    {"type": "fixed_pitch_timbres", "frequency_hz": 2000},
    {"type": "geometric_shapes", "pitch_study": {}},
])
def test_invalid_options_fail(data, tmp_path):
    with pytest.raises(ConfigurationError):
        build(data, tmp_path)


def test_nearest_and_diversity_metrics_match_brute_force():
    rng = np.random.default_rng(0)
    training = rng.random((300, 7))
    samples = np.vstack([training[10], rng.random((3, 7))])
    distances, indices = nearest_training(samples, training)
    brute = np.mean((samples[:, None, :] - training[None, :, :])**2, axis=2)
    np.testing.assert_allclose(distances, brute.min(axis=1), atol=1e-15)
    np.testing.assert_array_equal(indices, brute.argmin(axis=1))
    off_diagonal = [np.mean((a - b)**2) for i, a in enumerate(samples)
                    for j, b in enumerate(samples) if i != j]
    assert pairwise_mse(samples) == pytest.approx(np.mean(off_diagonal))
    assert pairwise_mse(np.ones((3, 7))) == 0


def _smoke_config(kind, model_type="variational"):
    data = ({"type": "geometric_shapes", "image_size": 16, "train_count": 12, "test_count": 4}
            if kind == "shapes" else timbre_config())
    return {"seed": 7, "dataset": data,
            "model": {"type": model_type, "hidden_layers": [4], "latent_size": 2,
                      "loss": "mse", "precision": "float32"},
            "training": {"epochs": 2},
            "evaluation": {"generative": {"count": 4, "test_examples": 2, "audio_examples": 2, "seed": 10}}}


@pytest.mark.parametrize("kind,model_type", [("shapes", "autoencoder"), ("shapes", "variational"),
                                              ("timbres", "variational")])
def test_end_to_end_new_studies_reuse_and_evaluate_only(tmp_path, monkeypatch, kind, model_type):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = _smoke_config(kind, model_type)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(config))
    directory = runner.run_experiment(path) / "base__seed=7"
    manifest = json.loads((directory / "generative_manifest.json").read_text())
    assert manifest["parameters_given_to_model"] is False
    assert manifest["train_count"] == 12 and manifest["test_count"] == 4
    assert not {r["label"] for r in manifest["train"]}.intersection(r["label"] for r in manifest["test"])
    summary = json.loads((directory / "summary.json").read_text())
    assert np.isfinite(summary["heldout_mean_squared_error"])
    assert (directory / "test_reconstruction.png").is_file()
    assert (directory / "generated_nearest.png").is_file()
    with np.load(directory / "generated.npz") as generated:
        expected = np.random.default_rng(10).standard_normal((4, 2)).astype(np.float32)
        np.testing.assert_array_equal(generated["coordinates"], expected)
    if kind == "timbres":
        assert (directory / "audio/test/0000__raw.wav").is_file()
        assert (directory / "audio/test/0000__codec.wav").is_file()
        assert (directory / "audio/generative/sample_00.wav").is_file()
        assert "generated_fixed_pitch_hit_rate" in summary
    saved = (directory / "model.npz").stat().st_mtime_ns
    runner.run_experiment(path)
    runner.run_experiment(path, evaluate_only=True)
    assert (directory / "model.npz").stat().st_mtime_ns == saved


def test_configs_have_eighteen_runs_without_duplicating_beta_one():
    root = Path(__file__).resolve().parents[1]
    runs = []
    for name in sum(CONFIGS.values(), []):
        config = json.loads((root / f"configs/{name}.json").read_text())
        runs.extend((name, r) for r in expand_runs(config))
        assert "interpolation" not in config["training"]
    assert len(runs) == 18
    shapes = [(r.config["model"]["type"], r.config["model"].get("kl", {}).get("weight"), r.seed)
              for name, r in runs if "shapes" in name]
    assert len(set(shapes)) == 15
    assert sum(t == "variational" and beta == 1 for t, beta, seed in shapes) == 3


def test_comparison_requires_complete_runs(tmp_path):
    source = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in CONFIGS["shapes"]:
        (tmp_path / f"configs/{name}.json").write_text((source / f"configs/{name}.json").read_text())
    with pytest.raises(ConfigurationError, match="Completar las corridas"):
        compare_suite(tmp_path, "shapes")


def test_comparison_groups_ae_and_vae_without_mixing_datasets(tmp_path):
    source = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    manifests = []
    for name in CONFIGS["shapes"]:
        config = json.loads((source / f"configs/{name}.json").read_text())
        (tmp_path / f"configs/{name}.json").write_text(json.dumps(config))
        for run in expand_runs(config):
            directory = tmp_path / "results" / name / run.label
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(run.config))
            summary = {"mean_squared_error": .01, "heldout_mean_squared_error": .02,
                       "generated_pairwise_mse": .03, "generated_nearest_train_mse_mean": .04}
            (directory / "summary.json").write_text(json.dumps(summary))
            manifest = {"model_seed": run.seed, "generation_seed": 1234, "generation_count": 64,
                        "parameters_given_to_model": False, "data": {"kind": "shapes", "train_sha256": "a", "test_sha256": "b"},
                        "train_count": 1024, "test_count": 256}
            path = directory / "generative_manifest.json"
            path.write_text(json.dumps(manifest)); manifests.append(path)
            (directory / "model.npz").touch()
            (directory / "history.csv").write_text("epoch\n300\n")
    validated = compare_suite(tmp_path, "shapes", write_outputs=False)
    assert not (tmp_path / "results/generative_shapes_report").exists()
    grouped = compare_suite(tmp_path, "shapes")
    assert validated == grouped
    assert len(grouped) == 5 and all(r["runs"] == 3 for r in grouped)
    assert {r["model"] for r in grouped} == {"AE", "VAE β=0", "VAE β=0.1", "VAE β=1", "VAE β=4"}
    assert (tmp_path / "results/generative_shapes_report/shapes.png").is_file()
    # No atribuir a KL una diferencia que en realidad proviene de otro dataset.
    manifest = json.loads(manifests[0].read_text())
    manifest["data"]["train_sha256"] = "changed"
    manifests[0].write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="cambió datos"):
        compare_suite(tmp_path, "shapes")
