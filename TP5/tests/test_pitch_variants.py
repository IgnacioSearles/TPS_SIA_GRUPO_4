"""Variantes auténticas, separación por altura y compatibilidad con el estudio pequeño."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.analysis.pitch_study import aggregate_study
from src.data.audio import SpectrogramCodec
from src.data.notes import StudySettings
from src.data.synthetic_notes import harmonic_note, load_synthetic_notes
from src.experiments import runner
from src.options import ConfigurationError


def test_all_1000_variants_are_real_and_all_reserved_pitches_are_excluded():
    # Una representación pequeña para verificar el dataset completo sin entrenar.
    codec = SpectrogramCodec(mel_bands=8, fft_size=256, hop_length=256)
    settings = StudySettings(True, (54, 60, 66), ("v00", "v20", "v39"))
    dataset = load_synthetic_notes(codec, settings=settings, variants_per_pitch=40)
    assert len(dataset) == 880 and len(dataset.pitch_study.notes) == 1000
    assert len(set(dataset.labels)) == 880
    encoded_first_pitch = dataset.samples[:40]
    assert len({hashlib.sha256(row.tobytes()).digest() for row in encoded_first_pitch}) == 40
    reserved = [n for n in dataset.pitch_study.notes if n.pitch in settings.targets]
    assert len(reserved) == 120 and not set(n.label for n in reserved).intersection(dataset.labels)
    for pitch in range(48, 73):
        notes = [n for n in dataset.pitch_study.notes if n.pitch == pitch]
        assert len(notes) == 40
        assert len({hashlib.sha256(n.waveform.tobytes()).digest() for n in notes}) == 40
        # El timbre y las envolventes disponibles son los mismos en todas las alturas.
        for note in notes:
            np.testing.assert_array_equal(note.waveform, harmonic_note(
                pitch, codec, **note.parameters).astype(np.float32))
    selected = dataset.pitch_study.evaluation_variants
    assert selected == settings.evaluation_variants
    for target in settings.targets:
        for variant in selected:
            center = dataset.pitch_study.note(target, variant)
            for neighbor in (target - 1, target + 1):
                note = dataset.pitch_study.note(neighbor, variant)
                assert note.parameters == center.parameters
                assert note.label in dataset.labels
    # Ni el control ni la evaluación tienen una configuración de envolventes diferente.
    complete = load_synthetic_notes(codec, settings=replace(settings, omit_targets=False),
                                     variants_per_pitch=40)
    assert len(complete) == 1000
    assert dataset.samples.shape[1] == np.prod(codec.image_shape)
    assert dataset.samples.min() >= 0 and dataset.samples.max() <= 1


def test_variant_validation_and_original_defaults():
    codec = SpectrogramCodec(mel_bands=8)
    original = load_synthetic_notes(codec)
    assert len(original) == 25
    for note in original.labels:
        assert "v00" not in note
    with pytest.raises(ConfigurationError, match="variants_per_pitch"):
        load_synthetic_notes(codec, variants_per_pitch=2)
    with pytest.raises(ConfigurationError, match="inexistentes"):
        load_synthetic_notes(codec, settings=StudySettings(True, (60,), ("absent",)))


@pytest.mark.parametrize("omit", [True, False])
def test_variant_evaluation_and_report_keep_matched_cases(tmp_path, monkeypatch, omit):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = {"name": "variants_smoke", "seed": 7,
              "dataset": {"type": "synthetic_notes", "pitch_range": [59, 61],
                          "variants_per_pitch": 40,
                          "pitch_study": {"targets": [60], "omit_targets": omit,
                                          "evaluation_variants": ["v00", "v39"]},
                          "spectrogram": {"mel_bands": 8, "fft_size": 256, "hop_length": 128,
                                          "griffin_lim_iterations": 1, "mel_inversion_iterations": 2}},
              "model": {"type": "variational", "hidden_layers": [4], "latent_size": 2, "loss": "mse"},
              "training": {"epochs": 1}, "evaluation": {"pitch_interpolation": {}}}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(config))
    experiment = runner.run_experiment(path)
    directory = experiment / "base__seed=7"
    manifest = json.loads((directory / "pitch_manifest.json").read_text())
    assert manifest["protocol_version"] == 2
    assert len(manifest["notes"]) == 120
    with (directory / "pitch_interpolation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * (13 if omit else 14)
    assert {r["variant"] for r in rows} == {"v00", "v39"}
    for variant in ("v00", "v39"):
        assert (directory / f"audio/pitch/60_{variant}/midpoint.wav").is_file()
    with (directory / "pitch_calibration.csv").open() as stream:
        calibration = list(csv.DictReader(stream))
    assert len(calibration) == 3 * 2 * 2  # Todas las alturas, dos variantes, original/códec.
    assert {r["variant"] for r in calibration} == {"v00", "v39"}
    mtime = (directory / "model.npz").stat().st_mtime_ns
    runner.run_experiment(path, evaluate_only=True)
    assert (directory / "model.npz").stat().st_mtime_ns == mtime
    report = aggregate_study([experiment], tmp_path / "report")["synthetic_notes"]
    assert report["status"] == "incomplete"
    assert set(report["per_variant"]) == {"v00", "v39"}


def test_expanded_configs_have_new_outputs_and_consistent_settings():
    root = Path(__file__).resolve().parents[1]
    for mode in ("complete", "omitted"):
        new_path = root / f"configs/ej2_pitch_expanded_synthetic_{mode}.json"
        expanded = json.loads(new_path.read_text())
        small = json.loads((root / f"configs/ej2_pitch_synthetic_{mode}.json").read_text())
        assert "variants_per_pitch" not in small["dataset"]
        assert expanded["dataset"]["variants_per_pitch"] == 40
        assert expanded["dataset"]["pitch_study"]["evaluation_variants"] == ["v00", "v20", "v39"]
        assert expanded["model"] == small["model"] and expanded["training"] == small["training"]
        assert expanded["seeds"] == [7, 17, 27]


@pytest.mark.parametrize("hits_by_variant,status", [([8, 8, 8], "favorable"),
                                                     ([9, 9, 6], "not_favorable")])
def test_aggregate_requires_eight_of_nine_in_every_variant(tmp_path, hits_by_variant, status):
    variants = ["v00", "v20", "v39"]
    experiments = [tmp_path / "complete", tmp_path / "omitted"]
    for omit in (True, False):
        for seed_index, seed in enumerate((7, 17, 27)):
            directory = (experiments[1] if omit else experiments[0]) / str(seed)
            directory.mkdir(parents=True)
            manifest = {"protocol_version": 2, "source": "synthetic_notes", "seed": seed,
                        "omit_targets": omit, "codec": {}, "targets": [54, 60, 66],
                        "evaluation_variants": variants, "notes": []}
            (directory / "pitch_manifest.json").write_text(json.dumps(manifest))
            (directory / "config.json").write_text("{}")
            (directory / "pitch_summary.json").write_text(json.dumps({"pitch_calibration_reliable": True}))
            methods = ["vae_interpolation", "spectrogram_mix", "lower_neighbor"]
            if not omit:
                methods.append("target_reconstruction")
            with (directory / "pitch_interpolation.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["target_pitch", "variant", "method",
                                                           "fraction", "hit", "valid", "error_cents"])
                writer.writeheader()
                for variant, successes in zip(variants, hits_by_variant):
                    for target_index, target in enumerate((54, 60, 66)):
                        for method in methods:
                            hit = method == "vae_interpolation" and seed_index * 3 + target_index < successes
                            writer.writerow({"target_pitch": target, "variant": variant, "method": method,
                                             "fraction": 0.5, "hit": hit, "valid": hit,
                                             "error_cents": 10 if hit else ""})
    result = aggregate_study(experiments, tmp_path / "report")["synthetic_notes"]
    assert result["complete"] and result["status"] == status
    assert result["methods"]["vae_interpolation"]["trials"] == 27
    assert result["methods"]["vae_interpolation"]["hits"] == 24
