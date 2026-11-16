import numpy as np
import pytest
from scipy.signal import istft, stft

from src.data.phase import phase_from_magnitude

FFT_SIZE = 1024
HOP = 256
RATE = 16000


def spectrum(waveform):
    return stft(waveform, nperseg=FFT_SIZE, noverlap=FFT_SIZE - HOP)[2]


def inconsistency(magnitudes, phase):
    """Cuánto cambia la magnitud al pasar por audio y volver: 0 si la fase es coherente."""
    _, waveform = istft(magnitudes * np.exp(1j * phase), nperseg=FFT_SIZE, noverlap=FFT_SIZE - HOP)
    rebuilt = np.abs(spectrum(waveform[:RATE]))
    return np.linalg.norm(rebuilt - magnitudes) / np.linalg.norm(magnitudes)


def harmonic_note(fundamental=220.0, partials=8):
    time = np.arange(RATE) / RATE
    envelope = np.minimum(1.0, time / 0.02) * np.exp(-2.0 * time)
    return envelope * sum(np.sin(2 * np.pi * fundamental * k * time) / k for k in range(1, partials + 1))


def test_harmonic_note_phase_is_far_more_consistent_than_random():
    magnitudes = np.abs(spectrum(harmonic_note()))
    random_phase = np.random.default_rng(0).uniform(0, 2 * np.pi, magnitudes.shape)
    assert inconsistency(magnitudes, phase_from_magnitude(magnitudes, HOP, FFT_SIZE)) < 0.5 * inconsistency(
        magnitudes, random_phase)


def test_pure_tone_phase_advances_at_its_frequency():
    frequency = 1000.3
    waveform = np.sin(2 * np.pi * frequency * np.arange(RATE) / RATE)
    magnitudes = np.abs(spectrum(waveform))
    phase = phase_from_magnitude(magnitudes, HOP, FFT_SIZE)
    peak = int(round(frequency * FFT_SIZE / RATE))
    advance = np.angle(np.exp(1j * np.diff(phase[peak, 5:-5])))
    expected = np.angle(np.exp(2j * np.pi * frequency * HOP / RATE))
    assert np.allclose(advance, expected, atol=0.1)


def test_rejects_magnitudes_with_wrong_bin_count():
    with pytest.raises(ValueError, match="bins"):
        phase_from_magnitude(np.ones((10, 5)), HOP, FFT_SIZE)
