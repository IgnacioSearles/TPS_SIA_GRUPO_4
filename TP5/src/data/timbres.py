"""Timbres nuevos a frecuencia fija; el decoder sigue siendo un MLP libre."""

from dataclasses import asdict
import numpy as np

from src.data.audio import build_codec
from src.data.generative import assemble_generative, split_options
from src.options import ConfigurationError, Options


def synthesize_timbre(codec, frequency, amplitudes, attack, release, duration):
    time = np.arange(codec.sample_count) / codec.sample_rate
    waveform = sum(amplitude * np.sin(2 * np.pi * (index + 1) * frequency * time)
                   for index, amplitude in enumerate(amplitudes))
    length = round(duration * codec.sample_count)
    envelope = np.zeros(codec.sample_count)
    envelope[:length] = 1
    a = min(round(attack * codec.sample_rate), length // 2)
    r = min(round(release * codec.sample_rate), length - a)
    if a:
        envelope[:a] = np.linspace(0, 1, a)
    if r:
        envelope[length - r:length] = np.linspace(1, 0, r)
    waveform *= envelope
    return (waveform / np.max(np.abs(waveform))).astype(np.float32)


def load_timbres(options: Options):
    train_count, test_count, seed = split_options(options)
    frequency = options.take("frequency_hz", 220.0)
    codec = build_codec(options.section("spectrogram"))
    options.ensure_consumed()
    if (type(frequency) not in (int, float) or not np.isfinite(frequency)
            or not 80 <= frequency <= 1200 or 6 * frequency >= codec.sample_rate / 2):
        raise ConfigurationError("frequency_hz debe estar en [80,1200] y sus seis armónicos debajo de Nyquist")
    if codec.duration_seconds < 0.5:
        raise ConfigurationError("Los timbres requieren al menos 0.5 segundos")
    rng = np.random.default_rng(seed)
    samples, records, test_waveforms = [], [], []
    for index in range(train_count + test_count):
        amplitudes = rng.dirichlet(np.full(6, 1.5))
        amplitudes[0] += 0.5  # Fundamental presente en todos los ejemplos, sin octavas ambiguas.
        amplitudes /= amplitudes.sum()
        attack, release, duration = float(rng.uniform(.01, .12)), float(rng.uniform(.05, .4)), float(rng.uniform(.8, 1))
        waveform = synthesize_timbre(codec, frequency, amplitudes, attack, release, duration)
        samples.append(codec.encode(waveform).ravel())
        records.append({"label": f"timbre_{index:05d}", "family": "harmonic", "frequency_hz": frequency,
                        "attack_seconds": attack, "release_seconds": release, "duration_fraction": duration,
                        **{f"amplitude_{i + 1}": float(a) for i, a in enumerate(amplitudes)}})
        if index >= train_count:
            test_waveforms.append(waveform)
    return assemble_generative(samples, records, train_count, codec.image_shape,
                                {"kind": "timbres", "data_seed": seed, "frequency_hz": frequency,
                                 "harmonics": 6, "codec": asdict(codec)}, codec,
                                np.asarray(test_waveforms))
