import numpy as np
import pytest

from src.data.audio import SpectrogramCodec
from src.data.font import load_font
from src.data.noise import Noise
from src.data.nsynth import NoteSelection, _choose_notes
from src.options import ConfigurationError


def test_font_bit_orientation_and_labels(tmp_path):
    path = tmp_path / "font.h"
    path.write_text(
        "int const Font[2][7] = {\n"
        "   {0x10, 0x01, 0x00, 0x1f, 0x00, 0x00, 0x00},   // 0x61, a\n"
        "   {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f}};  // 0x7f, DEL\n",
        encoding="utf-8",
    )
    dataset = load_font(path)
    assert dataset.labels == ["a", "DEL"]
    assert dataset.image_shape == (7, 5)
    np.testing.assert_array_equal(dataset.samples[0, :10], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(dataset.samples[0, 15:20], [1, 1, 1, 1, 1])


def test_font_rejects_rows_wider_than_five_bits(tmp_path):
    path = tmp_path / "font.h"
    path.write_text("{0x20, 0, 0, 0, 0, 0, 0}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_font(path)


def test_noise_extremes_and_purity():
    samples = np.eye(4)
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(Noise("flip", 0.0).apply(samples, rng), samples)
    np.testing.assert_array_equal(Noise("flip", 1.0).apply(samples, rng), 1 - samples)
    np.testing.assert_array_equal(Noise("mask", 1.0).apply(samples, rng), np.zeros((4, 4)))
    noisy = Noise("gaussian", 5.0).apply(samples, rng)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    np.testing.assert_array_equal(samples, np.eye(4))  # no modifica la entrada


def test_noise_rejects_invalid_configuration():
    with pytest.raises(ConfigurationError):
        Noise("flip", 1.5)
    with pytest.raises(ConfigurationError):
        Noise("desconocido", 0.1)


def test_codec_preserves_pitch_through_griffin_lim():
    codec = SpectrogramCodec(duration_seconds=0.5, griffin_lim_iterations=30)
    time = np.arange(codec.sample_count) / codec.sample_rate
    frequency = 440.0
    spectrogram = codec.encode(np.sin(2 * np.pi * frequency * time))
    assert spectrogram.shape == codec.image_shape
    assert 0.0 <= spectrogram.min() and spectrogram.max() <= 1.0

    recovered = codec.decode(spectrogram)
    spectrum = np.abs(np.fft.rfft(recovered))
    dominant = np.fft.rfftfreq(len(recovered), 1 / codec.sample_rate)[np.argmax(spectrum)]
    assert dominant == pytest.approx(frequency, abs=30.0)


def test_codec_decodes_silence_without_dividing_by_zero():
    codec = SpectrogramCodec(duration_seconds=0.25, griffin_lim_iterations=2)
    waveform = codec.decode(np.zeros(codec.image_shape))
    assert np.all(waveform == 0.0)


def test_note_selection_filters_and_caps_per_family():
    names = [
        "nsynth-test/audio/flute_acoustic_001-060-050.wav",
        "nsynth-test/audio/flute_acoustic_001-090-050.wav",
        "nsynth-test/audio/flute_synthetic_002-060-050.wav",
        "nsynth-test/audio/guitar_acoustic_003-055-050.wav",
        "nsynth-test/audio/guitar_acoustic_003-056-050.wav",
        "nsynth-test/audio/guitar_acoustic_003-057-050.wav",
        "nsynth-test/examples.json",
    ]
    selection = NoteSelection(sources=("acoustic",), pitch_range=(48, 72), max_per_family=2)
    chosen = _choose_notes(names, selection)
    assert sum("flute" in name for name in chosen) == 1
    assert sum("guitar" in name for name in chosen) == 2
    assert chosen == _choose_notes(names, selection)  # reproducible
