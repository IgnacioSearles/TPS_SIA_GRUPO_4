"""Estimación monofónica por autocorrelación; no corrige octavas ni adivina silencios."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, find_peaks

from src.data.notes import midi_frequency


@dataclass(frozen=True)
class PitchEstimate:
    frequency: float | None
    valid_frames: int
    total_frames: int


def estimate_pitch(waveform, sample_rate):
    """4096 muestras, salto 1024; primer pico >= .6 cercano al mejor pico.

    Elegir el primer pico casi máximo evita seleccionar múltiplos del período
    por diferencias numéricas mínimas. No usa la nota objetivo como pista.
    Las mezclas polifónicas pueden presentar períodos comunes engañosos.
    """
    waveform = np.asarray(waveform, dtype=float)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError("Se requiere audio mono finito")
    size, hop = 4096, 1024
    low, high = int(np.ceil(sample_rate / 1200)), int(np.floor(sample_rate / 80))
    frequencies, total = [], 0
    for start in range(0, len(waveform) - size + 1, hop):
        total += 1
        frame = waveform[start:start + size].copy()
        frame -= frame.mean()
        if np.sqrt(np.mean(frame**2)) < 1e-4:
            continue
        correlation = correlate(frame, frame, mode="full", method="fft")[size - 1:]
        energy = np.concatenate(([0.0], np.cumsum(frame**2)))
        lags = np.arange(high + 2)
        normalizer = np.sqrt(energy[size - lags] * (energy[size] - energy[lags]))
        normalized = correlation[:high + 2] / np.maximum(normalizer, 1e-12)
        peaks, _ = find_peaks(normalized)
        peaks = peaks[(peaks >= low) & (peaks <= high)]
        if not len(peaks):
            continue
        maximum = normalized[peaks].max()
        candidates = peaks[normalized[peaks] >= max(0.6, maximum - 0.02)]
        if not len(candidates):
            continue
        lag = candidates[0]
        left, center, right = normalized[lag - 1:lag + 2]
        denominator = left - 2 * center + right
        offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (left - right) / denominator
        frequency = sample_rate / (lag + np.clip(offset, -0.5, 0.5))
        if 80 <= frequency <= 1200:
            frequencies.append(frequency)
    return PitchEstimate(float(np.median(frequencies)) if len(frequencies) >= 3 else None,
                         len(frequencies), total)


def pitch_measurement(waveform, sample_rate, target):
    estimate = estimate_pitch(waveform, sample_rate)
    error = None if estimate.frequency is None else float(
        abs(1200 * np.log2(estimate.frequency / midi_frequency(target))))
    return {"frequency_hz": estimate.frequency, "error_cents": error,
            "valid": estimate.frequency is not None, "valid_frames": estimate.valid_frames,
            "total_frames": estimate.total_frames, "hit": error is not None and error <= 50}
