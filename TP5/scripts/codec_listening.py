"""Comparación auditiva de formas de volver de espectrograma a audio.

Seis notas reales en MIDI 60, una por familia, y tres muestras del VAE de control,
cada una decodificada de varias maneras. No cambia el códec del proyecto: solo escribe
WAV y métricas en reports/codec_listening/ para elegir de oído.

Uso, desde TP5:  python -m scripts.codec_listening
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.data.audio import SpectrogramCodec, read_waveform, write_waveform
from src.data.phase import phase_from_magnitude
from src.experiments.runner import load_run

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "nsynth" / "nsynth-test.jsonwav.tar.gz"
VAE_RUN = ROOT / "results" / "ej2_nsynth_sampling_control_vae" / "base__seed=7"
OUTPUT = ROOT / "reports" / "codec_listening"
FAMILIES = ("bass", "brass", "guitar", "keyboard", "string", "vocal")
PITCH = 60
GENERATED_COUNT = 3
GENERATION_SEED = 2026111900  # los mismos vectores base del control de muestreo
REFINEMENT_ITERATIONS = 60
DECIBEL_FLOOR = -70.0


def chosen_notes() -> dict[str, np.ndarray]:
    """La primera nota (en orden alfabético) de cada familia en MIDI 60."""
    pattern = re.compile(rf"/({'|'.join(FAMILIES)})_\w+-{PITCH:03d}-\d+\.wav$")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        members = sorted((m for m in archive.getmembers() if pattern.search(m.name)), key=lambda m: m.name)
        first = {}
        for member in members:
            first.setdefault(pattern.search(member.name).group(1), member)
        missing = set(FAMILIES) - set(first)
        if missing:
            raise FileNotFoundError(f"No hay notas en MIDI {PITCH} para {sorted(missing)}")
        return {Path(first[family].name).stem: read_waveform(io.BytesIO(archive.extractfile(first[family]).read()),
                                                            16000) for family in FAMILIES}


def mel_to_linear(codec: SpectrogramCodec, spectrogram: np.ndarray) -> np.ndarray:
    magnitudes = 10.0 ** ((spectrogram - 1.0) * codec.dynamic_range_db / 20.0)
    magnitudes[spectrogram <= 0.0] = 0.0
    return codec._mel_to_linear(magnitudes)


def pghi_decode(codec: SpectrogramCodec, spectrogram: np.ndarray) -> np.ndarray:
    """Magnitudes por NNLS, fase inicial por PGHI y unas iteraciones de Griffin-Lim desde ahí."""
    magnitudes = mel_to_linear(codec, spectrogram)
    spectrum = magnitudes * np.exp(1j * phase_from_magnitude(magnitudes, codec.hop_length, codec.fft_size))
    for _ in range(REFINEMENT_ITERATIONS):
        spectrum = magnitudes * np.exp(1j * np.angle(codec._transform(codec._inverse_transform(spectrum))))
    waveform = codec._inverse_transform(spectrum)
    return waveform / max(np.max(np.abs(waveform)), 1e-12)


def log_spectral_distance(codec: SpectrogramCodec, reference: np.ndarray, estimate: np.ndarray) -> float:
    def decibels(waveform):
        magnitudes = np.abs(codec._transform(codec._fit_duration(waveform)))
        return np.maximum(20 * np.log10(np.maximum(magnitudes / magnitudes.max(), 1e-12)), DECIBEL_FLOOR)
    return float(np.sqrt(np.mean((decibels(reference) - decibels(estimate)) ** 2, axis=0)).mean())


def decoders(codec: SpectrogramCodec) -> dict[str, tuple[SpectrogramCodec, callable]]:
    wide = replace(codec, mel_bands=256)
    return {"actual": (codec, codec.decode),
            "pghi": (codec, lambda spectrogram: pghi_decode(codec, spectrogram)),
            "256_bandas": (wide, wide.decode),
            "256_bandas_pghi": (wide, lambda spectrogram: pghi_decode(wide, spectrogram))}


def render_real_notes(codec: SpectrogramCodec) -> list[dict]:
    rows = []
    for name, waveform in chosen_notes().items():
        fitted = codec._fit_duration(waveform)
        write_waveform(OUTPUT / f"{name}__original.wav", fitted / np.max(np.abs(fitted)), codec.sample_rate)
        row = {"note": name, "distortion_db": {}}
        for method, (method_codec, decode) in decoders(codec).items():
            decoded = decode(method_codec.encode(waveform))
            write_waveform(OUTPUT / f"{name}__{method}.wav", decoded, codec.sample_rate)
            row["distortion_db"][method] = log_spectral_distance(codec, fitted, decoded)
        rows.append(row)
        print(f"  {name}: " + ", ".join(f"{k} {v:.1f} dB" for k, v in row["distortion_db"].items()))
    return rows


def render_generated(codec: SpectrogramCodec) -> list[str]:
    """Muestras del VAE de control: solo existen en 128 bandas, así que solo cambia la fase."""
    model, dataset = load_run(VAE_RUN)
    epsilon = np.random.default_rng(GENERATION_SEED).standard_normal((GENERATED_COUNT, model.latent_size))
    names = []
    for index, values in enumerate(model.decode(epsilon.astype(model.precision))):
        spectrogram = values.reshape(dataset.image_shape)
        name = f"vae_muestra_{index}"
        write_waveform(OUTPUT / f"{name}__actual.wav", codec.decode(spectrogram), codec.sample_rate)
        write_waveform(OUTPUT / f"{name}__pghi.wav", pghi_decode(codec, spectrogram), codec.sample_rate)
        names.append(name)
    return names


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, dataset = load_run(VAE_RUN)
    codec = dataset.audio_codec
    manifest = {"real_notes": render_real_notes(codec), "generated": render_generated(codec),
                "refinement_iterations": REFINEMENT_ITERATIONS,
                "distortion": "log-spectral distance to the original, 70 dB floor, mean over frames"}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
