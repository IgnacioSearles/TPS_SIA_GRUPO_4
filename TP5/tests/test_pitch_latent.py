import csv
import json
from pathlib import Path

import pytest

from src.analysis.pitch_latent import compare_latents, prepare_final, VALIDATION_TARGETS, FINAL_TARGETS
from src.data.audio import SpectrogramCodec
from src.data.notes import StudySettings
from src.data.synthetic_notes import load_synthetic_notes
from src.experiments import runner
from src.experiments.grid import expand_runs
from src.options import ConfigurationError, Options


def test_suite_changes_only_latent_dimension_and_has_eighteen_runs():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/ej2_pitch_microtonal_strict_omitted.json").read_text())
    for condition in ("complete", "omitted"):
        config = json.loads((root / f"configs/ej2_pitch_latent_validation_{condition}.json").read_text())
        assert config["training"] == baseline["training"]
        assert config["model"] == baseline["model"]
        assert config["grid"] == {"model.latent_size": [4, 2, 16]}
        assert len(expand_runs(config)) == 9
        study = config["dataset"]["pitch_study"]
        assert study["targets"] == VALIDATION_TARGETS and study["reserved_targets"] == FINAL_TARGETS


@pytest.mark.parametrize("omit", [True, False])
def test_reserved_notes_and_entire_exclusion_windows_never_enter_training(omit):
    codec = SpectrogramCodec(mel_bands=8, fft_size=256, hop_length=256)
    settings = StudySettings(omit, tuple(VALIDATION_TARGETS), ("v00",), .75, tuple(FINAL_TARGETS))
    dataset = load_synthetic_notes(codec, settings=settings, variants_per_pitch=40,
                                  microtones_per_semitone=4, microtone_variants=10)
    training = set(dataset.labels)
    for note in dataset.pitch_study.notes:
        blocked = FINAL_TARGETS + (VALIDATION_TARGETS if omit else [])
        if any(abs(note.pitch - p) <= .75 for p in blocked):
            assert note.label not in training
    assert len(dataset) == (1120 if omit else 1420)
    with pytest.raises(ConfigurationError, match="disjuntos"):
        StudySettings.parse(Options({"targets": [55], "reserved_targets": [55]}, "study"))


def test_reserved_notes_are_not_calibrated_in_smoke_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = {"name": "reserved_smoke", "seed": 7,
              "dataset": {"type": "synthetic_notes", "pitch_range": [59, 64],
                          "pitch_study": {"targets": [60], "reserved_targets": [63]},
                          "spectrogram": {"mel_bands": 8, "fft_size": 256, "hop_length": 128,
                                          "griffin_lim_iterations": 1, "mel_inversion_iterations": 2}},
              "model": {"type": "variational", "latent_size": 2, "hidden_layers": [4], "loss": "mse"},
              "training": {"epochs": 1}, "evaluation": {"pitch_interpolation": {}}}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(config))
    directory = runner.run_experiment(path) / "base__seed=7"
    manifest = json.loads((directory / "pitch_manifest.json").read_text())
    assert manifest["protocol_version"] == 4 and manifest["reserved_targets"] == [63]
    with (directory / "pitch_calibration.csv").open() as stream:
        pitches = {int(r["pitch"]) for r in csv.DictReader(stream)}
    assert 63 not in pitches and 60 in pitches
    # La configuración anterior sin reserved_targets sigue siendo válida en el resto de tests.


def _validation_results(tmp_path):
    source = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for condition in ("complete", "omitted"):
        path = source / f"configs/ej2_pitch_latent_validation_{condition}.json"
        config = json.loads(path.read_text())
        (tmp_path / "configs" / path.name).write_text(json.dumps(config))
        for run in expand_runs(config):
            directory = tmp_path / "results" / path.stem / run.label
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(run.config))
            (directory / "training.json").write_text(json.dumps({"samples": 1120}))
            (directory / "summary.json").write_text("{}")
            (directory / "model.npz").touch()
            (directory / "pitch_summary.json").write_text(json.dumps({"pitch_calibration_reliable": True}))
            manifest = {"targets": VALIDATION_TARGETS, "reserved_targets": FINAL_TARGETS,
                        "seed": run.seed, "codec": {}, "notes": []}
            (directory / "pitch_manifest.json").write_text(json.dumps(manifest))
            (directory / "history.csv").write_text("epoch\n200\n")
            with (directory / "pitch_interpolation.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["target_pitch", "variant", "method", "fraction",
                                                           "hit", "spectral_hit", "spectral_mse"])
                writer.writeheader()
                methods = ["vae_interpolation", "spectrogram_mix", "lower_neighbor"]
                if condition == "complete":
                    methods.append("target_reconstruction")
                for target in VALIDATION_TARGETS:
                    for variant in ["v00", "v20", "v39"]:
                        for method in methods:
                            # Z=16 logra afinación aparente perfecta pero falla en espectro.
                            hit = method != "lower_neighbor"
                            spectral = method == "vae_interpolation" and run.config["model"]["latent_size"] == 4
                            writer.writerow({"target_pitch": target, "variant": variant, "method": method,
                                             "fraction": .5, "hit": hit, "spectral_hit": spectral,
                                             "spectral_mse": .1})


def test_selection_prefers_joint_metric_and_prepares_only_chosen_final(tmp_path):
    _validation_results(tmp_path)
    # Archivos finales deliberadamente inválidos: comparar validación no debe leerlos.
    (tmp_path / "results/irrelevant_final").mkdir()
    (tmp_path / "results/irrelevant_final/summary.json").write_text("not json")
    report = compare_latents(tmp_path)
    assert report["selected_latent_size"] == 4
    prepare_final(tmp_path, report)
    for condition in ("complete", "omitted"):
        config = json.loads((tmp_path / f"configs/ej2_pitch_latent_final_{condition}.json").read_text())
        assert config["model"]["latent_size"] == 4 and "grid" not in config
        assert config["dataset"]["pitch_study"]["targets"] == FINAL_TARGETS
        assert config["dataset"]["pitch_study"]["reserved_targets"] == VALIDATION_TARGETS
        assert len(expand_runs(config)) == 3
    prepare_final(tmp_path, report)  # Idempotente.
    changed = dict(report, selected_latent_size=2)
    with pytest.raises(ConfigurationError, match="no se sobrescribe"):
        prepare_final(tmp_path, changed)


def test_missing_or_reconfigured_runs_cannot_select_model(tmp_path):
    _validation_results(tmp_path)
    path = tmp_path / "results/ej2_pitch_latent_validation_omitted/latent_size=4__seed=7/config.json"
    actual = json.loads(path.read_text())
    actual["training"]["epochs"] = 300
    path.write_text(json.dumps(actual))
    with pytest.raises(ConfigurationError, match="configuración guardada"):
        compare_latents(tmp_path)
