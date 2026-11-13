import csv
import io
import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest

from src.analysis.pitch_study import aggregate_study
from src.data.audio import SpectrogramCodec, waveform_to_wav_bytes
from src.data.factory import build_dataset
from src.data.notes import StudySettings, midi_frequency
from src.data.nsynth import NoteSelection, _choose_notes, _fingerprint
from src.data.nsynth_study import choose_chromatic_notes, load_nsynth_study
from src.data.synthetic_notes import harmonic_note, load_synthetic_notes
from src.evaluation.pitch import estimate_pitch, pitch_measurement
from src.evaluation.pitch_interpolation import latent_trajectory
from src.experiments import runner
from src.options import ConfigurationError, Options


@pytest.fixture
def codec():
    return SpectrogramCodec(duration_seconds=0.5, fft_size=256, hop_length=128,
                            mel_bands=16, griffin_lim_iterations=2, mel_inversion_iterations=5)


def test_synthetic_is_deterministic_and_holds_out_all_targets(codec):
    settings = StudySettings(True, (54, 60, 66))
    first = load_synthetic_notes(codec, settings=settings)
    second = load_synthetic_notes(codec, settings=settings)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert len(first) == 22
    assert first.samples.shape[1] == np.prod(codec.image_shape)
    assert first.samples.min() >= 0 and first.samples.max() <= 1
    assert len(first.pitch_study.notes) == 25
    held = {first.pitch_study.note(p).label for p in settings.targets}
    assert not held.intersection(first.labels)
    full = load_synthetic_notes(codec, settings=replace(settings, omit_targets=False))
    assert len(full) == 25 and held.issubset(full.labels)
    assert first.pitch_study.targets == full.pitch_study.targets
    for note in first.pitch_study.notes:
        assert note.waveform[0] == 0 and abs(note.waveform[-1]) < 1e-12
        assert max(abs(note.waveform)) == pytest.approx(1)


def note_names(instrument="keyboard_acoustic_001", start=48, stop=61, velocity=100):
    return [f"nsynth-test/audio/{instrument}-{p:03d}-{velocity:03d}.wav" for p in range(start, stop)]


def test_nsynth_coverage_ties_and_group_integrity():
    selection = NoteSelection(pitch_range=(48, 72))
    names = note_names("z_synthetic_001") + note_names("a_acoustic_001")
    names += note_names("a_acoustic_001", 48, 60, 50)  # Más corto, otra velocidad.
    selected = choose_chromatic_notes(list(reversed(names)), selection)
    assert selected == note_names("a_acoustic_001")
    assert choose_chromatic_notes(names, replace(selection, exclude_pitches=(54,))) == selected
    with pytest.raises(ConfigurationError, match="13 notas"):
        choose_chromatic_notes(note_names(stop=60), selection)
    with pytest.raises(ConfigurationError, match="max_per_family"):
        choose_chromatic_notes(names, replace(selection, max_per_family=5))


def test_generic_nsynth_excludes_before_random_cap_and_cache_changes(codec):
    selection = NoteSelection(instrument="keyboard_acoustic_001", velocity=100,
                              exclude_pitches=(54,), max_per_family=100)
    names = note_names() + note_names(velocity=50) + note_names("flute_acoustic_001")
    chosen = _choose_notes(names, selection)
    assert len(chosen) == 12
    assert all("-054-" not in name and "keyboard_acoustic_001" in name
               and "-100.wav" in name for name in chosen)
    from pathlib import Path
    for changed in [replace(selection, velocity=50), replace(selection, exclude_pitches=()),
                    replace(selection, instrument="flute_acoustic_001")]:
        assert _fingerprint(Path("a.tar.gz"), selection, codec) != _fingerprint(Path("a.tar.gz"), changed, codec)


def test_nsynth_small_archive_and_study_cache(tmp_path, codec):
    archive = tmp_path / "nsynth.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for pitch, name in zip(range(48, 61), note_names()):
            content = waveform_to_wav_bytes(harmonic_note(pitch, codec), codec.sample_rate)
            member = tarfile.TarInfo(name)
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    cache = tmp_path / "cache"
    args = (archive, NoteSelection(pitch_range=(48, 72)), codec, cache)
    omitted = load_nsynth_study(*args, StudySettings(True))
    assert omitted.pitch_study.targets == (51, 54, 57)
    assert len(omitted) == 10
    assert len({n.instrument for n in omitted.pitch_study.notes}) == 1
    assert len({n.velocity for n in omitted.pitch_study.notes}) == 1
    again = load_nsynth_study(*args, StudySettings(True))
    np.testing.assert_array_equal(again.samples, omitted.samples)
    assert len(load_nsynth_study(*args, StudySettings(False))) == 13
    assert len(list(cache.glob("*.npz"))) == 2
    with pytest.raises(FileNotFoundError, match="download_nsynth"):
        load_nsynth_study(tmp_path / "absent", *args[1:], StudySettings(True))


@pytest.mark.parametrize("pitch", [48, 54, 60, 66, 72])
@pytest.mark.parametrize("harmonic", [True, False])
def test_fundamental_estimation_on_known_tones(codec, pitch, harmonic):
    time = np.arange(codec.sample_count) / codec.sample_rate
    waveform = (harmonic_note(pitch, codec) if harmonic
                else np.sin(2 * np.pi * midi_frequency(pitch) * time))
    measurement = pitch_measurement(waveform, codec.sample_rate, pitch)
    assert measurement["valid"] and measurement["hit"]
    assert measurement["error_cents"] < 5
    octave = pitch_measurement(waveform, codec.sample_rate, pitch + 12)
    assert not octave["hit"] and octave["error_cents"] > 1100


def test_silence_short_audio_noise_and_two_note_mixture(codec):
    assert estimate_pitch(np.zeros(16000), 16000).frequency is None
    assert estimate_pitch(np.ones(4096), 16000).frequency is None
    assert estimate_pitch(np.random.default_rng(4).normal(size=16000), 16000).frequency is None
    # Una mezcla puede aparentar una fundamental intermedia: no certifica monofonía.
    mixed = harmonic_note(59, codec) + harmonic_note(61, codec)
    result = pitch_measurement(mixed, codec.sample_rate, 60)
    assert result["valid"]
    assert midi_frequency(59) < result["frequency_hz"] < midi_frequency(61)


def test_interpolation_encodes_only_neighbors_and_prescribes_midpoint():
    class SpyModel:
        def encode(self, samples):
            np.testing.assert_array_equal(samples, [[1, 2], [3, 4]])
            return samples

        def decode(self, codes):
            return codes

    fractions, decoded = latent_trajectory(SpyModel(), np.array([[1, 2], [3, 4]]))
    assert len(fractions) == 11 and fractions[5] == 0.5
    np.testing.assert_array_equal(decoded[5], [2, 3])


def test_invalid_study_fails_before_training(tmp_path):
    with pytest.raises(ConfigurationError, match="vecinos"):
        build_dataset(Options({"type": "synthetic_notes", "pitch_range": [59, 61],
                               "pitch_study": {"targets": [59]}}, "dataset"), tmp_path)
    with pytest.raises(ConfigurationError, match="subset"):
        build_dataset(Options({"type": "synthetic_notes", "subset": [0],
                               "pitch_study": {}}, "dataset"), tmp_path)
    with pytest.raises(ConfigurationError, match="MIDI"):
        NoteSelection(exclude_pitches=(False,))


@pytest.mark.parametrize("omit", [True, False])
def test_end_to_end_negative_outcome_reuse_and_evaluate_only(tmp_path, monkeypatch, omit):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = {"name": "smoke", "seed": 7,
              "dataset": {"type": "synthetic_notes", "pitch_range": [59, 61],
                          "pitch_study": {"targets": [60], "omit_targets": omit},
                          "spectrogram": {"duration_seconds": 0.5, "fft_size": 256,
                                          "hop_length": 128, "mel_bands": 16,
                                          "griffin_lim_iterations": 2, "mel_inversion_iterations": 5}},
              "model": {"type": "variational", "hidden_layers": [4], "latent_size": 2,
                        "loss": "mse", "precision": "float32"},
              "training": {"epochs": 2}, "evaluation": {"pitch_interpolation": {}}}
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(config))
    directory = runner.run_experiment(path)
    run = directory / "base__seed=7"
    manifest = json.loads((run / "pitch_manifest.json").read_text())
    assert manifest["seed"] == 7
    assert [n["pitch"] for n in manifest["notes"] if n["split"] == "reference"] == ([60] if omit else [])
    with (run / "pitch_interpolation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == (13 if omit else 14) and {r["seed"] for r in rows} == {"7"}
    assert (run / "audio/pitch/60/midpoint.wav").is_file()
    assert (run / "audio/pitch/references/60_original.wav").is_file()
    assert (run / "pitch_spectrograms.png").is_file()
    mtime = (run / "model.npz").stat().st_mtime_ns
    runner.run_experiment(path)
    runner.run_experiment(path, evaluate_only=True)
    assert (run / "model.npz").stat().st_mtime_ns == mtime
    result = aggregate_study([directory], tmp_path / "report")
    assert result["synthetic_notes"]["status"] == "incomplete"
    assert not result["synthetic_notes"]["favorable"]
    manifest["notes"][0]["waveform_sha256"] = "different-reference"
    (run / "pitch_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="manifiesto"):
        runner.run_experiment(path, evaluate_only=True)


def test_all_four_configs_preserve_model_and_expand_to_twelve_runs():
    from pathlib import Path
    from src.experiments.grid import expand_runs
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/ej2_vae_nsynth.json").read_text())
    original_names = {f"ej2_pitch_{dataset}_{condition}"
                      for dataset in ("synthetic", "nsynth")
                      for condition in ("complete", "omitted")}
    paths = sorted(p for p in (root / "configs").glob("ej2_pitch_*.json")
                   if p.stem in original_names)
    assert len(paths) == 4
    for path in paths:
        config = json.loads(path.read_text())
        assert config["model"] == original["model"]
        assert config["training"] == original["training"]
        assert len(expand_runs(config)) == 3


def _aggregate_fixture(tmp_path, *, hits=8, calibrated=True):
    """Tablas mínimas del protocolo: las nueve predicciones incluyen un fallo."""
    for omit in (True, False):
        for seed_index, seed in enumerate((7, 17, 27)):
            run = tmp_path / ("omitted" if omit else "complete") / str(seed)
            run.mkdir(parents=True)
            manifest = {"protocol_version": 1, "source": "synthetic_notes", "seed": seed,
                        "omit_targets": omit, "codec": {}, "targets": [54, 60, 66],
                        "notes": [{"pitch": p, "split": "reference" if omit else "training"}
                                  for p in (54, 60, 66)]}
            (run / "pitch_manifest.json").write_text(json.dumps(manifest))
            (run / "config.json").write_text("{}")
            (run / "pitch_summary.json").write_text(json.dumps({"pitch_calibration_reliable": calibrated}))
            methods = ["vae_interpolation", "spectrogram_mix", "lower_neighbor"]
            if not omit:
                methods.append("target_reconstruction")
            with (run / "pitch_interpolation.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["target_pitch", "method", "fraction",
                                                           "hit", "valid", "error_cents"])
                writer.writeheader()
                for target_index, target in enumerate((54, 60, 66)):
                    for method in methods:
                        hit = (seed_index * 3 + target_index < hits if method == "vae_interpolation"
                               else method == "target_reconstruction")
                        writer.writerow({"target_pitch": target, "method": method, "fraction": 0.5,
                                         "hit": hit, "valid": hit, "error_cents": 10 if hit else ""})
    return [tmp_path / "omitted", tmp_path / "complete"]


@pytest.mark.parametrize("hits,calibrated,status", [(8, True, "favorable"),
                                                    (7, True, "not_favorable"),
                                                    (9, False, "inconclusive_codec")])
def test_aggregate_criterion_counts_all_failures(tmp_path, hits, calibrated, status):
    paths = _aggregate_fixture(tmp_path, hits=hits, calibrated=calibrated)
    result = aggregate_study(paths, tmp_path / "report")["synthetic_notes"]
    assert result["complete"]
    assert result["status"] == status
    assert result["methods"]["vae_interpolation"]["trials"] == 9
    assert result["methods"]["vae_interpolation"]["hits"] == hits


def test_aggregate_rejects_mixed_protocols_and_duplicate_seeds(tmp_path):
    paths = _aggregate_fixture(tmp_path)
    run = paths[0] / "7"
    manifest_path = run / "pitch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["codec"] = {"mel_bands": 16}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="mismo protocolo"):
        aggregate_study(paths, tmp_path / "report")
    manifest["codec"] = {}
    manifest["seed"] = 17
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="Dos corridas"):
        aggregate_study(paths, tmp_path / "report")


def test_originals_through_actual_study_codec():
    codec = SpectrogramCodec(mel_bands=128)
    for pitch in (48, 60, 72):
        recovered = codec.decode(codec.encode(harmonic_note(pitch, codec)))
        assert pitch_measurement(recovered, codec.sample_rate, pitch)["hit"]
