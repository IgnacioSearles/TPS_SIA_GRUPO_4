"""Conversión entre audio y espectrogramas mel normalizados a [0, 1].

La red solo ve magnitudes; para volver a audio se estima la fase con
Griffin-Lim. Ese ida y vuelta pierde calidad aun sin red de por medio, por lo
que conviene comparar siempre contra el original pasado por el códec.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft

from src.options import ConfigurationError, Options

Array = np.ndarray
_AMPLITUDE_FLOOR = 1e-10
_MEL_BREAK_FREQUENCY = 700.0
_MEL_SCALE = 2595.0
_GRIFFIN_LIM_PHASE_SEED = 0
_INT16_PEAK = 32767


@dataclass(frozen=True)
class SpectrogramCodec:
    sample_rate: int = 16000
    duration_seconds: float = 1.0
    fft_size: int = 1024
    hop_length: int = 256
    mel_bands: int = 64
    dynamic_range_db: float = 70.0
    griffin_lim_iterations: int = 60

    def __post_init__(self) -> None:
        if self.hop_length < 1 or self.fft_size % self.hop_length != 0:
            raise ConfigurationError("fft_size debe ser múltiplo de hop_length")
        if self.mel_bands < 1 or self.mel_bands > self.fft_size // 2:
            raise ConfigurationError("mel_bands debe estar entre 1 y fft_size / 2")
        if self.duration_seconds <= 0 or self.dynamic_range_db <= 0:
            raise ConfigurationError("duration_seconds y dynamic_range_db deben ser positivos")

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_seconds * self.sample_rate))

    @cached_property
    def image_shape(self) -> tuple[int, int]:
        """(bandas mel, cuadros temporales) de un espectrograma codificado."""
        return self.encode(np.zeros(self.sample_count)).shape

    def encode(self, waveform: Array) -> Array:
        """Audio (se recorta o rellena a la duración fija) -> espectrograma en [0, 1]."""
        magnitudes = np.abs(self._transform(self._fit_duration(waveform)))
        mel_magnitudes = self._filterbank @ magnitudes
        decibels = 20.0 * np.log10(np.maximum(mel_magnitudes, _AMPLITUDE_FLOOR))
        return np.clip(1.0 + decibels / self.dynamic_range_db, 0.0, 1.0)

    def decode(self, spectrogram: Array) -> Array:
        """Espectrograma en [0, 1] -> audio normalizado a pico 1 (fase por Griffin-Lim)."""
        if spectrogram.shape != self.image_shape:
            raise ValueError(f"Se esperaba un espectrograma de forma {self.image_shape}")
        decibels = (spectrogram - 1.0) * self.dynamic_range_db
        mel_magnitudes = 10.0 ** (decibels / 20.0)
        # El piso del rango dinámico representa silencio, no -dynamic_range dB.
        mel_magnitudes[spectrogram <= 0.0] = 0.0
        magnitudes = np.maximum(self._inverse_filterbank @ mel_magnitudes, 0.0)
        waveform = self._griffin_lim(magnitudes)
        peak = np.max(np.abs(waveform))
        return waveform / peak if peak > 0 else waveform

    def _transform(self, waveform: Array) -> Array:
        _, _, spectrum = stft(waveform, nperseg=self.fft_size, noverlap=self._overlap)
        return spectrum

    def _inverse_transform(self, spectrum: Array) -> Array:
        _, waveform = istft(spectrum, nperseg=self.fft_size, noverlap=self._overlap)
        return self._fit_duration(waveform)

    def _griffin_lim(self, magnitudes: Array) -> Array:
        rng = np.random.default_rng(_GRIFFIN_LIM_PHASE_SEED)
        spectrum = magnitudes * np.exp(2j * np.pi * rng.random(magnitudes.shape))
        for _ in range(self.griffin_lim_iterations):
            estimate = self._transform(self._inverse_transform(spectrum))
            spectrum = magnitudes * np.exp(1j * np.angle(estimate))
        return self._inverse_transform(spectrum)

    def _fit_duration(self, waveform: Array) -> Array:
        fitted = np.zeros(self.sample_count)
        length = min(len(waveform), self.sample_count)
        fitted[:length] = waveform[:length]
        return fitted

    @property
    def _overlap(self) -> int:
        return self.fft_size - self.hop_length

    @cached_property
    def _filterbank(self) -> Array:
        """Filtros triangulares equiespaciados en escala mel, de 0 Hz a Nyquist."""
        bin_frequencies = np.linspace(0.0, self.sample_rate / 2, self.fft_size // 2 + 1)
        edges = _mel_to_hertz(
            np.linspace(0.0, _hertz_to_mel(self.sample_rate / 2), self.mel_bands + 2)
        )
        lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        rising = (bin_frequencies - lower) / (center - lower)
        falling = (upper - bin_frequencies) / (upper - center)
        return np.maximum(0.0, np.minimum(rising, falling))

    @cached_property
    def _inverse_filterbank(self) -> Array:
        return np.linalg.pinv(self._filterbank)


def build_codec(options: Options) -> SpectrogramCodec:
    """Sección `spectrogram`: cualquier campo de SpectrogramCodec, todos opcionales."""
    fields = {key: options.take(key, None) for key in options.keys()}
    try:
        return SpectrogramCodec(**fields)
    except TypeError as error:
        raise ConfigurationError(f"Parámetros de espectrograma inválidos: {error}") from error


def read_waveform(stream: object, expected_sample_rate: int) -> Array:
    """Lee un WAV mono (ruta u objeto archivo) como flotantes con pico 1."""
    sample_rate, samples = wavfile.read(stream)
    if sample_rate != expected_sample_rate:
        raise ValueError(f"Se esperaban {expected_sample_rate} Hz y el audio tiene {sample_rate}")
    if samples.ndim != 1:
        raise ValueError("Se esperaba audio mono")
    waveform = samples.astype(float)
    peak = np.max(np.abs(waveform))
    return waveform / peak if peak > 0 else waveform


def write_waveform(path: Path, waveform: Array, sample_rate: int) -> None:
    wavfile.write(path, sample_rate, _to_int16(waveform))


def waveform_to_wav_bytes(waveform: Array, sample_rate: int) -> bytes:
    """Contenido de un archivo WAV en memoria (para enviarlo sin tocar el disco)."""
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, _to_int16(waveform))
    return buffer.getvalue()


def _to_int16(waveform: Array) -> Array:
    return np.round(np.clip(waveform, -1.0, 1.0) * _INT16_PEAK).astype(np.int16)


def _hertz_to_mel(frequency: float | Array) -> float | Array:
    return _MEL_SCALE * np.log10(1.0 + frequency / _MEL_BREAK_FREQUENCY)


def _mel_to_hertz(mel: float | Array) -> float | Array:
    return _MEL_BREAK_FREQUENCY * (10.0 ** (mel / _MEL_SCALE) - 1.0)
