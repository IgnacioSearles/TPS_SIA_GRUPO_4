"""Subconjunto de notas de NSynth convertido a espectrogramas mel.

Las notas se leen directamente del .tar.gz oficial (sin extraerlo) y el
resultado se guarda en una caché, porque recorrer el archivo lleva tiempo.
El nombre de cada nota codifica sus atributos:
    <familia>_<fuente>_<instrumento>-<pitch MIDI>-<velocidad>.wav
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.data.audio import SpectrogramCodec, read_waveform
from src.data.dataset import Dataset
from src.options import ConfigurationError, Options

_NOTE_PATTERN = re.compile(
    r"(?P<family>[a-z]+)_(?P<source>[a-z]+)_(?P<instrument>\d+)-(?P<pitch>\d+)-(?P<velocity>\d+)\.wav$"
)
_FULL_MIDI_RANGE = (0, 127)


@dataclass(frozen=True)
class NoteSelection:
    """Qué notas tomar. `None` en familias o fuentes significa todas."""

    families: tuple[str, ...] | None = None
    sources: tuple[str, ...] | None = None
    pitch_range: tuple[int, int] = _FULL_MIDI_RANGE
    max_per_family: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        low, high = self.pitch_range
        if low > high:
            raise ConfigurationError(f"pitch_range inválido: {self.pitch_range}")
        if self.max_per_family is not None and self.max_per_family < 1:
            raise ConfigurationError("max_per_family debe ser positivo")

    def accepts(self, family: str, source: str, pitch: int) -> bool:
        return (
            (self.families is None or family in self.families)
            and (self.sources is None or source in self.sources)
            and self.pitch_range[0] <= pitch <= self.pitch_range[1]
        )


def build_selection(options: Options) -> NoteSelection:
    families = options.take("families", None)
    sources = options.take("sources", None)
    selection = NoteSelection(
        families=None if families is None else tuple(families),
        sources=None if sources is None else tuple(sources),
        pitch_range=tuple(options.take("pitch_range", list(_FULL_MIDI_RANGE))),
        max_per_family=options.take("max_per_family", None),
        seed=options.take("selection_seed", 0),
    )
    return selection


def load_nsynth(
    archive: Path, selection: NoteSelection, codec: SpectrogramCodec, cache_directory: Path
) -> Dataset:
    cache_path = cache_directory / f"nsynth_{_fingerprint(archive, selection, codec)}.npz"
    if not cache_path.is_file():
        if not archive.is_file():
            raise FileNotFoundError(
                f"No se encontró {archive}. Descargarlo con: python scripts/download_nsynth.py"
            )
        spectrograms, labels = _encode_selected_notes(archive, selection, codec)
        cache_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, spectrograms=spectrograms, labels=np.asarray(labels))
    with np.load(cache_path, allow_pickle=False) as cached:
        spectrograms = cached["spectrograms"].astype(float)
        labels = [str(label) for label in cached["labels"]]
    return Dataset(
        samples=spectrograms,
        labels=labels,
        groups=[label.split("_")[0] for label in labels],
        image_shape=codec.image_shape,
        is_binary=False,
        audio_codec=codec,
    )


def _encode_selected_notes(
    archive: Path, selection: NoteSelection, codec: SpectrogramCodec
) -> tuple[np.ndarray, list[str]]:
    with tarfile.open(archive, "r:gz") as tar:
        chosen = _choose_notes(tar.getnames(), selection)
    if not chosen:
        raise ConfigurationError("Ninguna nota de NSynth cumple la selección pedida")
    # Un gzip no admite acceso aleatorio eficiente: se recorre una sola vez, en
    # el orden del archivo, leyendo únicamente las notas elegidas.
    encoded: dict[str, np.ndarray] = {}
    pending = set(chosen)
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name not in pending:
                continue
            audio = io.BytesIO(stream.extractfile(member).read())
            encoded[member.name] = codec.encode(read_waveform(audio, codec.sample_rate)).ravel()
            pending.discard(member.name)
            if not pending:
                break
    spectrograms = np.asarray([encoded[name] for name in chosen], dtype=np.float32)
    return spectrograms, [Path(name).stem for name in chosen]


def _choose_notes(member_names: list[str], selection: NoteSelection) -> list[str]:
    """Notas aceptadas, con tope por familia elegido al azar de forma reproducible."""
    by_family: dict[str, list[str]] = {}
    for name in sorted(member_names):
        match = _NOTE_PATTERN.search(name)
        if match and selection.accepts(match["family"], match["source"], int(match["pitch"])):
            by_family.setdefault(match["family"], []).append(name)
    rng = np.random.default_rng(selection.seed)
    chosen: list[str] = []
    for family in sorted(by_family):
        names = by_family[family]
        if selection.max_per_family is not None and len(names) > selection.max_per_family:
            kept = rng.choice(len(names), selection.max_per_family, replace=False)
            names = [names[index] for index in sorted(kept)]
        chosen.extend(names)
    return chosen


def _fingerprint(archive: Path, selection: NoteSelection, codec: SpectrogramCodec) -> str:
    description = {
        "archive": archive.name,
        "selection": asdict(selection),
        # Griffin-Lim solo afecta la decodificación, no los espectrogramas guardados.
        "codec": {k: v for k, v in asdict(codec).items() if k != "griffin_lim_iterations"},
    }
    encoded = json.dumps(description, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
