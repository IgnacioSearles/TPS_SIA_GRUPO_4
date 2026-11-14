"""Figuras y audios de la presentación, a partir de las corridas guardadas en results/.

No entrena nada. Escribe en reports/presentacion/; después,
`python presentation/sync_assets.py` copia a presentation/assets/ lo que usa el deck.

Uso, desde TP5:  python -m scripts.presentacion [font] [audio] [redes]
"""

from __future__ import annotations

import sys

from scripts.presentacion import audio, diagrams, font

PARTS = {"font": font.make_all, "audio": audio.make_all, "redes": diagrams.make_all}


def main() -> None:
    chosen = sys.argv[1:] or list(PARTS)
    unknown = [name for name in chosen if name not in PARTS]
    if unknown:
        raise SystemExit(f"Partes desconocidas: {unknown}. Opciones: {list(PARTS)}")
    for name in chosen:
        print(f"[{name}]")
        PARTS[name]()


if __name__ == "__main__":
    main()
