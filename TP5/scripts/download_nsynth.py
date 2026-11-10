"""Descarga el split de test de NSynth (4096 notas, ~349 MB) sin descomprimirlo.

Uso, desde TP5:  python scripts/download_nsynth.py
El dataset se lee directamente del .tar.gz, así que no hace falta extraerlo.
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

NSYNTH_TEST_URL = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
)
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parent.parent / "data" / "nsynth" / "nsynth-test.jsonwav.tar.gz"
)
CHUNK_BYTES = 1 << 20


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Ya existe {destination}; no se descarga de nuevo.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, CHUNK_BYTES)
    partial.replace(destination)
    print(f"Descargado en {destination} ({destination.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    arguments = parser.parse_args()
    download(NSYNTH_TEST_URL, arguments.destination)
