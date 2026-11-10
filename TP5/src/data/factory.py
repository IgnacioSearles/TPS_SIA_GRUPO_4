"""Construcción del dataset a partir de la sección `dataset`."""

from __future__ import annotations

from pathlib import Path

from src.data.audio import build_codec
from src.data.dataset import Dataset
from src.data.font import load_font
from src.data.nsynth import build_selection, load_nsynth
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
    if dataset_type == "font":
        dataset = load_font(project_root / options.take("path", DEFAULT_FONT_PATH))
    elif dataset_type == "nsynth":
        dataset = load_nsynth(
            archive=project_root / options.take("archive", DEFAULT_NSYNTH_ARCHIVE),
            selection=build_selection(options),
            codec=build_codec(options.section("spectrogram")),
            cache_directory=project_root / CACHE_DIRECTORY,
        )
    else:
        raise ConfigurationError(f"dataset.type debe ser 'font' o 'nsynth', no {dataset_type!r}")
    options.ensure_consumed()
    if subset is None:
        return dataset
    if not isinstance(subset, list) or not subset:
        raise ConfigurationError("dataset.subset debe ser una lista no vacía de etiquetas o índices")
    return dataset.select(subset)
