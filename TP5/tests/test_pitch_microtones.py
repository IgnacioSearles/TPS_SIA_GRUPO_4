"""Microtonos sintéticos: ventana de exclusión honesta y métrica espectral."""

import csv
import json

import numpy as np
import pytest

from src.data.audio import SpectrogramCodec
from src.data.notes import StudySettings
from src.data.synthetic_notes import harmonic_note, load_synthetic_notes, microtone_variant_indices
from src.evaluation.pitch_interpolation import spectral_measurement
from src.experiments import runner
from src.options import ConfigurationError, Options

CODEC = SpectrogramCodec(mel_bands=8, fft_size=256, hop_length=256)
TARGETS = (54, 60, 66)


def _dataset(omit, radius=0.5):
    settings = StudySettings(omit, TARGETS, ("v00", "v20", "v39"), radius)
    return load_synthetic_notes(CODEC, settings=settings, variants_per_pitch=40,
                                microtones_per_semitone=4, microtone_variants=10)


def _training_pitches(dataset):
    by_label = {n.label: n.pitch for n in dataset.pitch_study.notes}
    return {by_label[label] for label in dataset.labels}


def test_microtonal_grid_keeps_integer_protocol():
    complete = _dataset(False)
    notes = complete.pitch_study.notes
    assert len(notes) == 25 * 40 + 24 * 3 * 10 == len(complete)
    assert sorted({n.pitch for n in notes})[:5] == [48, 48.25, 48.5, 48.75, 49]
    # Las alturas enteras mantienen etiquetas y las 40 envolventes.
    assert complete.pitch_study.note(60, "v20").label == "harmonic-060-v20"
    assert sum(n.pitch == 60 for n in notes) == 40
    fractional = [n for n in notes if n.pitch == 53.25]
    assert len(fractional) == 10 and fractional[0].label.startswith("harmonic-053.25-")
    np.testing.assert_array_equal(fractional[0].waveform, harmonic_note(
        53.25, CODEC, **fractional[0].parameters).astype(np.float32))
    # Entre los tres microtonos de cada semitono se cubren las 40 envolventes... de a 10, sin repetir.
    covered = [i for step in (1, 2, 3) for i in microtone_variant_indices(step, 10)]
    assert len(set(covered)) == 30


@pytest.mark.parametrize("radius,nearest", [(0.5, 0.75), (0.75, 1.0)])
def test_exclusion_window_leaves_no_training_note_inside_hit_tolerance(radius, nearest):
    dataset = _dataset(True, radius)
    pitches = _training_pitches(dataset)
    for target in TARGETS:
        distances = [abs(p - target) for p in pitches]
        assert min(distances) == pytest.approx(nearest)
        assert target - 1 in pitches and target + 1 in pitches
    excluded_per_target = 40 + 10 * (4 if radius == 0.5 else 6)
    assert len(dataset) == len(_dataset(False)) - 3 * excluded_per_target


def test_settings_and_generator_validation():
    with pytest.raises(ConfigurationError, match="exclusion_radius"):
        StudySettings.parse(Options({"exclusion_radius": 0.25}, "pitch_study"))
    with pytest.raises(ConfigurationError, match="exclusion_radius"):
        StudySettings.parse(Options({"exclusion_radius": 1.0}, "pitch_study"))
    assert StudySettings.parse(Options({}, "pitch_study")).exclusion_radius == 0.5
    with pytest.raises(ConfigurationError, match="microtone_variants"):
        load_synthetic_notes(CODEC, variants_per_pitch=40, microtones_per_semitone=2,
                             microtone_variants=3)
    with pytest.raises(ConfigurationError, match="Los microtonos requieren"):
        load_synthetic_notes(CODEC, microtones_per_semitone=2)


def test_spectral_measurement_rejects_two_note_mix():
    codec = SpectrogramCodec(mel_bands=64, fft_size=1024, hop_length=256)
    references = {p: codec.encode(harmonic_note(p, codec)).ravel() for p in (53, 54, 55)}
    mix = (references[53] + references[55]) / 2
    assert spectral_measurement(references[54], references, 54)["spectral_hit"]
    result = spectral_measurement(mix, references, 54)
    assert result["spectral_mse"] > 0 and not result["spectral_hit"]


def test_microtonal_smoke_run_writes_spectral_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = {"name": "micro_smoke", "seed": 7,
              "dataset": {"type": "synthetic_notes", "pitch_range": [59, 61],
                          "variants_per_pitch": 40, "microtones_per_semitone": 2,
                          "microtone_variants": 4,
                          "pitch_study": {"targets": [60], "omit_targets": True,
                                          "evaluation_variants": ["v00"]},
                          "spectrogram": {"mel_bands": 8, "fft_size": 256, "hop_length": 128,
                                          "griffin_lim_iterations": 1, "mel_inversion_iterations": 2}},
              "model": {"type": "variational", "hidden_layers": [4], "latent_size": 2, "loss": "mse"},
              "training": {"epochs": 1}, "evaluation": {"pitch_interpolation": {}}}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(config))
    directory = runner.run_experiment(path) / "base__seed=7"
    manifest = json.loads((directory / "pitch_manifest.json").read_text())
    assert manifest["protocol_version"] == 3
    # 59.5 y 60.5 caen dentro de la ventana de 0.5 semitonos: no se entrenan.
    training = {n["pitch"] for n in manifest["notes"] if n["split"] == "training"}
    assert training == {59, 61}
    with (directory / "pitch_interpolation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {"spectral_mse", "spectral_nearest_pitch", "spectral_hit"} <= set(rows[0])
    with (directory / "pitch_calibration.csv").open() as stream:
        assert {float(r["pitch"]) for r in csv.DictReader(stream)} == {59, 60, 61}
    summary = json.loads((directory / "pitch_summary.json").read_text())
    assert "pitch_midpoint_spectral_hit_rate" in summary
    assert summary["midpoint_hits_by_variant"]["v00"]["total"] == 1


def test_microtonal_configs_differ_only_in_dataset():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/ej2_pitch_expanded_synthetic_omitted.json").read_text())
    for name, omit, radius in [("omitted", True, 0.5), ("strict_omitted", True, 0.75),
                               ("complete", False, 0.5)]:
        config = json.loads((root / f"configs/ej2_pitch_microtonal_{name}.json").read_text())
        assert config["model"] == baseline["model"] and config["training"] == baseline["training"]
        assert config["dataset"]["microtones_per_semitone"] == 4
        assert config["dataset"]["pitch_study"]["omit_targets"] is omit
        assert config["dataset"]["pitch_study"]["exclusion_radius"] == radius
