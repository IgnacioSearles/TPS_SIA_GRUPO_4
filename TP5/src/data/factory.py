"""Construcción del dataset a partir de la sección `dataset`."""

from __future__ import annotations

from pathlib import Path

from src.data.audio import build_codec
from src.data.dataset import Dataset
from src.data.font import load_font
from src.data.nsynth import build_selection, load_nsynth
from src.data.nsynth_study import load_nsynth_study
from src.data.notes import StudySettings, midi_list
from src.data.synthetic_notes import load_synthetic_notes
from src.options import ConfigurationError, Options

DEFAULT_FONT_PATH = "data/font.h"
DEFAULT_NSYNTH_ARCHIVE = "data/nsynth/nsynth-test.jsonwav.tar.gz"
CACHE_DIRECTORY = "data/cache"


def build_dataset(options: Options, project_root: Path) -> Dataset:
    """Las rutas relativas se resuelven contra la raíz del proyecto (TP5).

    `subset` (lista de etiquetas o índices) restringe cualquier dataset.
    """
    dataset_type = options.take("type", "font")
    subset = options.take("subset", None)
    study_options = options.optional_section("pitch_study")
    study = None if study_options is None else StudySettings.parse(study_options)
    if study is not None and (dataset_type == "font" or subset is not None):
        raise ConfigurationError("pitch_study requiere audio y no admite subset")
    if dataset_type == "font":
        dataset = load_font(project_root / options.take("path", DEFAULT_FONT_PATH))
    elif dataset_type == "nsynth":
        loader = load_nsynth if study is None else load_nsynth_study
        extra = {} if study is None else {"settings": study}
        dataset = loader(
            archive=project_root / options.take("archive", DEFAULT_NSYNTH_ARCHIVE),
            selection=build_selection(options),
            codec=build_codec(options.section("spectrogram")),
            cache_directory=project_root / CACHE_DIRECTORY,
            **extra,
        )
    elif dataset_type == "synthetic_notes":
        dataset = load_synthetic_notes(
            codec=build_codec(options.section("spectrogram")),
            pitch_range=options.take("pitch_range", [48, 72]),
            exclude_pitches=midi_list(options.take("exclude_pitches", []), "exclude_pitches"),
            settings=study,
            variants_per_pitch=options.take("variants_per_pitch", 1),
            microtones_per_semitone=options.take("microtones_per_semitone", 1),
            microtone_variants=options.take("microtone_variants", 10),
        )
    else:
        raise ConfigurationError(f"dataset.type debe ser 'font', 'nsynth' o 'synthetic_notes', no {dataset_type!r}")
    options.ensure_consumed()
    if subset is None:
        return dataset
    if not isinstance(subset, list) or not subset:
        raise ConfigurationError("dataset.subset debe ser una lista no vacía de etiquetas o índices")
    return dataset.select(subset)
