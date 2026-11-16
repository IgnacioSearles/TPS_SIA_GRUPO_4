"""Fase de un STFT a partir de su magnitud, sin iterar: Phase Gradient Heap Integration (PGHI).

Prusa, Balazs y Søndergaard (2017). Para una ventana gaussiana g(t) = exp(-π t² / λ), las
derivadas de la fase (referida al centro de cada ventana) se deducen de las de log|STFT|:

    ∂φ/∂t = 2π ω + (1/λ) ∂(log s)/∂ω      (frecuencia instantánea)
    ∂φ/∂ω = -λ ∂(log s)/∂t                 (retardo de grupo)

con t en muestras y ω en ciclos por muestra. La ventana de Hann se aproxima con la gaussiana
de λ = 0,25645 L². La fase se integra desde los coeficientes más fuertes hacia sus vecinos,
para que los errores se acumulen donde la magnitud es baja y se oye menos.

Convención de salida: la de scipy.signal.stft con boundary="zeros" (cuadro n centrado en
la muestra n·hop, FFT referida al inicio del segmento).
"""

from __future__ import annotations

import heapq

import numpy as np

Array = np.ndarray
HANN_GAUSSIAN_RATIO = 0.25645
_LOG_FLOOR = 1e-12


def phase_from_magnitude(magnitudes: Array, hop_length: int, fft_size: int,
                         tolerance_db: float = 50.0, seed: int = 0) -> Array:
    """Fase (bins × cuadros) coherente con `magnitudes`, en la convención de scipy.signal.stft."""
    if magnitudes.ndim != 2 or magnitudes.shape[0] != fft_size // 2 + 1:
        raise ValueError(f"Se esperaban {fft_size // 2 + 1} bins de frecuencia; forma {magnitudes.shape}")
    if hop_length <= 0 or tolerance_db <= 0:
        raise ValueError("hop_length y tolerance_db deben ser positivos")
    log_magnitude = np.log(np.maximum(magnitudes, _LOG_FLOOR))
    time_step, frequency_step = _phase_steps(log_magnitude, hop_length, fft_size)
    reliable = log_magnitude > log_magnitude.max() - tolerance_db * np.log(10) / 20
    phase = _integrate(log_magnitude, time_step, frequency_step, reliable,
                       np.random.default_rng(seed))
    # Del centro de la ventana al inicio del segmento, media ventana antes: 2π k (L/2) / L = π k.
    bins = np.arange(magnitudes.shape[0])[:, None]
    return phase - np.pi * bins


def _phase_steps(log_magnitude: Array, hop_length: int, fft_size: int) -> tuple[Array, Array]:
    """Avance de fase de un cuadro al siguiente y de un bin al siguiente, en cada coeficiente."""
    spread = HANN_GAUSSIAN_RATIO * fft_size ** 2
    bins = np.arange(log_magnitude.shape[0])[:, None]
    along_frequency = np.gradient(log_magnitude, axis=0) * fft_size  # ∂ log s / ∂ω, ω en ciclos/muestra
    along_time = np.gradient(log_magnitude, axis=1) / hop_length     # ∂ log s / ∂t, t en muestras
    time_step = hop_length * (2 * np.pi * bins / fft_size + along_frequency / spread)
    frequency_step = -spread * along_time / fft_size
    return time_step, frequency_step


def _integrate(log_magnitude: Array, time_step: Array, frequency_step: Array,
               reliable: Array, rng: np.random.Generator) -> Array:
    bins, frames = log_magnitude.shape
    phase = rng.uniform(0, 2 * np.pi, log_magnitude.shape)
    pending = reliable.copy()
    order = np.argsort(log_magnitude, axis=None)[::-1]
    for start in order:
        bin_index, frame = divmod(int(start), frames)
        if not pending[bin_index, frame]:
            continue
        phase[bin_index, frame] = 0.0
        pending[bin_index, frame] = False
        heap = [(-log_magnitude[bin_index, frame], bin_index, frame)]
        while heap:
            _, bin_index, frame = heapq.heappop(heap)
            for neighbor_bin, neighbor_frame, steps, sign in (
                    (bin_index, frame + 1, time_step, 1.0), (bin_index, frame - 1, time_step, -1.0),
                    (bin_index + 1, frame, frequency_step, 1.0), (bin_index - 1, frame, frequency_step, -1.0)):
                if not (0 <= neighbor_bin < bins and 0 <= neighbor_frame < frames):
                    continue
                if not pending[neighbor_bin, neighbor_frame]:
                    continue
                average_step = (steps[bin_index, frame] + steps[neighbor_bin, neighbor_frame]) / 2
                phase[neighbor_bin, neighbor_frame] = phase[bin_index, frame] + sign * average_step
                pending[neighbor_bin, neighbor_frame] = False
                heapq.heappush(heap, (-log_magnitude[neighbor_bin, neighbor_frame], neighbor_bin, neighbor_frame))
    return phase
