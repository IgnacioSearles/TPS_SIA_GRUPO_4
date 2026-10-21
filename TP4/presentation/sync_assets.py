"""Copia a assets/ los gráficos que usa index.html y anota su origen en FUENTES.json.

Cada asset se llama como su ruta desde TP4 con "__" en lugar de "/":
assets/reports__ejercicio2__crosstalk.png sale de reports/ejercicio2/crosstalk.png.
Así la carpeta presentation/ se puede compartir sola y funciona sin conexión.

Uso desde TP4: python presentation/sync_assets.py
"""
import json
from pathlib import Path
import re
import shutil

PRESENTATION = Path(__file__).resolve().parent
ROOT = PRESENTATION.parent
ASSETS = PRESENTATION / 'assets'
ASSET_REFERENCE = re.compile(r'assets/([\w.-]+\.png)')


def referenced_assets(html: str) -> list[str]:
    return sorted(set(ASSET_REFERENCE.findall(html)))


def source_of(asset: str) -> Path:
    return ROOT / asset.replace('__', '/')


def main():
    assets = referenced_assets((PRESENTATION / 'index.html').read_text(encoding='utf-8'))
    missing = [str(source_of(asset).relative_to(ROOT)) for asset in assets if not source_of(asset).exists()]
    if missing:
        raise FileNotFoundError('Faltan gráficos; regenerarlos antes de copiar:\n' + '\n'.join(missing))
    ASSETS.mkdir(exist_ok=True)
    for stale in set(path.name for path in ASSETS.glob('*.png')) - set(assets):
        (ASSETS / stale).unlink()
    sources = {}
    for asset in assets:
        shutil.copy2(source_of(asset), ASSETS / asset)
        sources[asset] = source_of(asset).relative_to(ROOT).as_posix()
    (PRESENTATION / 'FUENTES.json').write_text(json.dumps(sources, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{len(assets)} gráficos copiados a {ASSETS.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
