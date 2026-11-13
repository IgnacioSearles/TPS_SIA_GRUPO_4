"""Selección determinista de una escala NSynth, sin mezclar timbres ni velocidades."""

from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from src.data.audio import read_waveform
from src.data.notes import AudioNote, assemble_notes
from src.data.nsynth import _NOTE_PATTERN, accepts_note
from src.options import ConfigurationError


def choose_chromatic_notes(member_names, selection):
    """Mayor tramo consecutivo; desempate por grupo y comienzo, ambos ascendentes."""
    if selection.max_per_family is not None:
        raise ConfigurationError("pitch_study no admite max_per_family: debe conservar la escala completa")
    groups = {}
    # Los huecos del experimento se aplican DESPUÉS de elegir la escala completa.
    complete = replace(selection, exclude_pitches=())
    for name in sorted(member_names):
        match = _NOTE_PATTERN.search(name)
        if match is None or not accepts_note(match, complete):
            continue
        instrument = f"{match['family']}_{match['source']}_{match['instrument']}"
        group = (instrument, int(match["velocity"]))
        groups.setdefault(group, {}).setdefault(int(match["pitch"]), name)
    runs = []
    for group, notes in groups.items():
        block = []
        for pitch in sorted(notes):
            if block and pitch != block[-1] + 1:
                runs.append((group, block, notes))
                block = []
            block.append(pitch)
        if block:
            runs.append((group, block, notes))
    if not runs:
        raise ConfigurationError("NSynth no tiene notas para la selección pedida")
    group, block, notes = min(runs, key=lambda run: (-len(run[1]), run[0], run[1][0]))
    if len(block) < 13:
        raise ConfigurationError(f"NSynth necesita al menos 13 notas consecutivas del mismo "
                                 f"instrumento y velocidad; máximo encontrado: {len(block)} ({group})")
    return [notes[pitch] for pitch in block]


def load_nsynth_study(archive, selection, codec, cache_directory, settings):
    if not archive.is_file():
        raise FileNotFoundError(f"No se encontró {archive}. Descargarlo con: "
                                "python scripts/download_nsynth.py")
    description = {"version": 1, "archive": str(archive.resolve()),
                   "size": archive.stat().st_size, "mtime": archive.stat().st_mtime_ns,
                   "selection": asdict(selection), "study": asdict(settings),
                   "codec": codec.encoding_parameters()}
    key = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:16]
    cache = cache_directory / f"nsynth_study_{key}.npz"
    if not cache.is_file():
        with tarfile.open(archive, "r:gz") as stream:
            chosen = choose_chromatic_notes([m.name for m in stream if m.isfile()], selection)
        pending = set(chosen)
        waveforms = {}
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                if member.name not in pending:
                    continue
                waveform = read_waveform(io.BytesIO(stream.extractfile(member).read()), codec.sample_rate)
                fitted = np.zeros(codec.sample_count)
                length = min(len(waveform), len(fitted))
                fitted[:length] = waveform[:length]
                waveforms[member.name] = fitted
                pending.remove(member.name)
                if not pending:
                    break
        cache_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, labels=np.asarray([Path(n).stem for n in chosen]),
                            waveforms=np.asarray([waveforms[n] for n in chosen]))
    with np.load(cache, allow_pickle=False) as saved:
        notes = []
        for label, waveform in zip(saved["labels"], saved["waveforms"]):
            label = str(label)
            match = _NOTE_PATTERN.search(label + ".wav")
            instrument = f"{match['family']}_{match['source']}_{match['instrument']}"
            notes.append(AudioNote(label, int(match["pitch"]), instrument,
                                   int(match["velocity"]), waveform.copy()))
    return assemble_notes(tuple(notes), "nsynth", codec, selection.exclude_pitches, settings)
