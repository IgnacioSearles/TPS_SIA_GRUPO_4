import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.pitch_interploss import compare_interpolation_weights, prepare_final
from src.data.audio import SpectrogramCodec
from src.data.notes import StudySettings
from src.data.synthetic_notes import load_synthetic_notes
from src.experiments import runner
from src.experiments.grid import expand_runs
from src.models.factory import build_model
from src.options import ConfigurationError, Options
from src.training.interpolation import InterpolationSettings, build_interpolation_triplets, triplet_records
from src.training.trainer import Trainer, build_training_plan


def model(precision="float64"):
    return build_model(Options({"type": "variational", "hidden_layers": [3], "latent_size": 2,
                                "loss": "mse", "precision": precision}, "model"),
                       4, np.random.default_rng(0), np.random.default_rng(1))


@pytest.mark.parametrize("combined", [False, True])
def test_interpolation_and_combined_gradients_match_finite_differences(combined):
    net = model()
    rng = np.random.default_rng(42)
    lower, upper, target = [rng.random((3, 4)) for _ in range(3)]
    noise = rng.standard_normal((3, 2))

    def evaluate():
        if combined:
            terms, base = net.objective(lower, lower, noise, .5)
            base = [g.copy() for g in base]
        value, gradients = net.interpolation_loss_and_gradients(lower, upper, target)
        if combined:
            return terms["loss"] + .3 * value, [a + .3 * b for a, b in zip(base, gradients)]
        return value, gradients

    _, gradients = evaluate()
    gradients = [g.copy() for g in gradients]
    for parameter, analytic in zip(net.parameters(), gradients):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = evaluate()[0]
            parameter[index] = original - 1e-6
            minus = evaluate()[0]
            parameter[index] = original
            assert analytic[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-7)


def test_auxiliary_gradient_preserves_precision_and_does_not_touch_logvariance_outputs():
    net = model("float32")
    x = np.random.default_rng(3).random((3, 4)).astype(np.float32)
    _, gradients = net.interpolation_loss_and_gradients(x, 1 - x, x)
    assert all(g.dtype == np.float32 for g in gradients)
    # Última capa del encoder: columnas [mu, logvar].
    np.testing.assert_array_equal(gradients[2][:, 2:], 0)
    np.testing.assert_array_equal(gradients[3][2:], 0)


def test_lambda_zero_has_identical_weights_and_does_not_draw_extra_randomness():
    samples = np.random.default_rng(9).random((6, 4)).astype(np.float32)
    base, control = model("float32"), model("float32")
    rng = np.random.default_rng(99)
    previous = copy.deepcopy(rng.bit_generator.state)
    for net, enabled in [(base, False), (control, True)]:
        config = {"epochs": 3, "batch_size": 2}
        if enabled:
            config["interpolation"] = {"weight": 0, "batch_size": 2}
        plan = build_training_plan(Options(config, "training"))
        trainer = Trainer(net, plan, lambda _: {}, np.random.default_rng(10), np.random.default_rng(11),
                          np.array([[0, 1, 2], [3, 4, 5]]) if enabled else None, rng if enabled else None)
        trainer.train(samples)
    assert rng.bit_generator.state == previous
    for first, second in zip(base.parameters(), control.parameters()):
        np.testing.assert_array_equal(first, second)
    # Agregar un stream no cambia los cinco que ya existían en el ejecutor.
    first = np.random.SeedSequence(7).spawn(5)
    second = np.random.SeedSequence(7).spawn(6)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.generate_state(4), b.generate_state(4))


@pytest.mark.parametrize("omitted,expected", [(True, 280), (False, 560)])
def test_triplets_use_only_training_notes_and_matching_variants(omitted, expected):
    codec = SpectrogramCodec(mel_bands=8, fft_size=256, hop_length=256)
    settings = StudySettings(omitted, (51, 57, 63), ("v00", "v20", "v39"), .75, (55, 61, 67))
    dataset = load_synthetic_notes(codec, settings=settings, variants_per_pitch=40,
                                  microtones_per_semitone=4, microtone_variants=10)
    triplets = build_interpolation_triplets(dataset)
    assert triplets.shape == (expected, 3)
    notes = {n.label: n for n in dataset.pitch_study.notes}
    for row in triplet_records(dataset, triplets):
        lower, center, upper = [notes[row[k]] for k in ("lower", "center", "upper")]
        assert center.pitch - lower.pitch == upper.pitch - center.pitch == 1
        assert lower.parameters == center.parameters == upper.parameters
        assert lower.variant == center.variant == upper.variant
        assert all(n.label in dataset.labels for n in (lower, center, upper))
        blocked = (55, 61, 67) + ((51, 57, 63) if omitted else ())
        assert not any(abs(n.pitch - p) <= .75 for n in (lower, center, upper) for p in blocked)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True])
def test_invalid_settings(weight):
    with pytest.raises(ConfigurationError):
        InterpolationSettings(weight)


def _small_config(weight, output=None):
    config = {"name": "interploss_smoke", "seed": 7,
              "dataset": {"type": "synthetic_notes", "pitch_range": [59, 68],
                          "pitch_study": {"targets": [60], "reserved_targets": [66]},
                          "spectrogram": {"duration_seconds": .5, "mel_bands": 8, "fft_size": 256,
                                          "hop_length": 128, "griffin_lim_iterations": 1,
                                          "mel_inversion_iterations": 2}},
              "model": {"type": "variational", "hidden_layers": [4], "latent_size": 4,
                        "loss": "mse", "precision": "float32"},
              "training": {"epochs": 2, "batch_size": 3,
                           "interpolation": {"weight": weight, "batch_size": 2}},
              "evaluation": {"pitch_interpolation": {}}}
    if output is not None:
        config["output"] = str(output)
    return config


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_end_to_end_manifests_and_evaluate_only(tmp_path, monkeypatch, weight):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(_small_config(weight)))
    output = runner.run_experiment(path)
    directory = output / "base__seed=7"
    training = json.loads((directory / "training.json").read_text())
    assert training["interpolation_triplets"] == 3
    assert training["interpolation_pitch_supervision"] == (weight > 0)
    assert training["final_interpolation_weight"] == weight
    with (directory / "interpolation_triplets.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert {float(row["center_pitch"]) for row in records} == {62, 63, 64}
    with (directory / "history.csv").open() as stream:
        for row in csv.DictReader(stream):
            assert float(row["loss"]) == pytest.approx(float(row["elbo_loss"]) + weight * float(row["interpolation"]))
    saved = (directory / "model.npz").stat().st_mtime_ns
    runner.run_experiment(path, evaluate_only=True)
    assert (directory / "model.npz").stat().st_mtime_ns == saved


def test_missing_triplets_and_wrong_model_fail_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = _small_config(.1)
    config.pop("name")
    config["dataset"]["pitch_range"] = [59, 61]
    config["dataset"]["pitch_study"].pop("reserved_targets")
    with pytest.raises(ConfigurationError, match="ternas completas"):
        runner._prepare(config)
    config = _small_config(.1)
    config.pop("name")
    config["model"]["type"] = "autoencoder"
    with pytest.raises(ConfigurationError, match="requiere model.type variational"):
        runner._prepare(config)


def test_new_configs_change_only_weight_with_z4_and_fixed_data():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "configs/ej2_pitch_latent_validation_omitted.json").read_text())
    for condition in ("complete", "omitted"):
        config = json.loads((root / f"configs/ej2_pitch_interploss_validation_{condition}.json").read_text())
        assert config["model"] == {**source["model"], "latent_size": 4}
        assert config["training"] == {**source["training"], "interpolation": {"weight": 0, "batch_size": 32}}
        assert config["grid"] == {"training.interpolation.weight": [0, .1, 1]}
        assert len(expand_runs(config)) == 9


def _fake_validation(root):
    source = Path(__file__).resolve().parents[1]
    (root / "configs").mkdir()
    for condition in ("complete", "omitted"):
        name = f"ej2_pitch_interploss_validation_{condition}"
        config = json.loads((source / f"configs/{name}.json").read_text())
        (root / f"configs/{name}.json").write_text(json.dumps(config))
        for run in expand_runs(config):
            directory = root / "results" / name / run.label
            directory.mkdir(parents=True)
            for filename, value in [("config.json", run.config), ("summary.json", {}),
                                    ("training.json", {"samples": 3, "interpolation_triplets": 1}),
                                    ("pitch_summary.json", {"pitch_calibration_reliable": True})]:
                (directory / filename).write_text(json.dumps(value))
            notes = [{"label": str(p), "pitch": p, "variant": "v00", "instrument": "harmonic",
                      "velocity": 100, "split": "training"} for p in (48, 49, 50)]
            manifest = {"targets": [51, 57, 63], "reserved_targets": [55, 61, 67],
                        "seed": run.seed, "codec": {}, "notes": notes}
            (directory / "pitch_manifest.json").write_text(json.dumps(manifest))
            (directory / "model.npz").touch()
            (directory / "history.csv").write_text("epoch\n200\n")
            (directory / "interpolation_triplets.csv").write_text("lower,center,upper\n48,49,50\n")
            methods = ["vae_interpolation", "spectrogram_mix", "lower_neighbor"]
            if condition == "complete":
                methods.append("target_reconstruction")
            with (directory / "pitch_interpolation.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["target_pitch", "variant", "method", "fraction",
                                                           "hit", "spectral_hit", "spectral_mse"])
                writer.writeheader()
                for p in (51, 57, 63):
                    for variant in ("v00", "v20", "v39"):
                        for method in methods:
                            win = method == "vae_interpolation" and run.config["training"]["interpolation"]["weight"] == .1
                            writer.writerow({"target_pitch": p, "variant": variant, "method": method,
                                             "fraction": .5, "hit": True, "spectral_hit": win, "spectral_mse": .1})


def test_weight_selection_and_final_configs_do_not_use_final_results(tmp_path):
    _fake_validation(tmp_path)
    report = compare_interpolation_weights(tmp_path)
    assert report["selected_interpolation_weight"] == .1
    prepare_final(tmp_path, report)
    for condition in ("complete", "omitted"):
        config = json.loads((tmp_path / f"configs/ej2_pitch_interploss_final_{condition}.json").read_text())
        assert config["training"]["interpolation"]["weight"] == .1
        assert config["dataset"]["pitch_study"]["targets"] == [55, 61, 67]
        assert config["dataset"]["pitch_study"]["reserved_targets"] == [51, 57, 63]


def test_comparison_rejects_a_reserved_label_in_triplet_manifest(tmp_path):
    _fake_validation(tmp_path)
    path = next((tmp_path / "results").rglob("interpolation_triplets.csv"))
    path.write_text("lower,center,upper\n48,held_out,50\n")
    with pytest.raises(ConfigurationError, match="referencias reservadas"):
        compare_interpolation_weights(tmp_path)


def test_active_interpolation_matches_in_serial_and_parallel(tmp_path):
    outputs = []
    for name, jobs in [("serial", 1), ("parallel", 2)]:
        config = _small_config(.1, tmp_path / name)
        config.pop("seed")
        config["seeds"] = [7, 17]
        config["evaluation"] = {}  # El test compara entrenamiento, no exportación de WAV.
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))
        outputs.append(runner.run_experiments([path], jobs=jobs)[0])
    for seed in (7, 17):
        with np.load(outputs[0] / f"base__seed={seed}/model.npz") as serial, np.load(
                outputs[1] / f"base__seed={seed}/model.npz") as parallel:
            assert serial.files == parallel.files
            for key in serial.files:
                np.testing.assert_allclose(serial[key], parallel[key], rtol=1e-6, atol=1e-7)
