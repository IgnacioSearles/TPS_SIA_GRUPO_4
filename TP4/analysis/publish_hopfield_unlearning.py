"""Publica métricas livianas de Hopfield; nunca copia pesos, checkpoints ni datos.

Uso desde TP4: python3 analysis/publish_hopfield_unlearning.py
La lista de archivos admitidos es explícita. Los registros extensos se comprimen.
"""
import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.hopfield_qr import ROOT

RUNS = (
    ('20261022T223236748114Z', 'smoke_completo', 180),
    ('20261022T224138937279Z', 'smoke_reducido', 180),
    ('pilot_completo_20261022', 'exploracion_completa', 6650),
    ('pilot_sin_compartidos_20261022', 'exploracion_reducida', 6650),
    ('confirmacion_sin_compartidos_20261022', 'confirmacion_reducida', 3500),
    ('dano_espacial_confirmado_20261022', 'validacion_espacial', 6600),
    ('comparacion_pseudoinversa_20261022', 'comparacion_pseudoinversa', 1400),
)
FILES = ('COMPLETED', 'config.json', 'resultados.csv', 'entrenamiento.csv',
         'estabilidad.csv', 'resumen.csv', 'resumen_por_qr.csv', 'pares_hebb.csv',
         'candidatos.csv', 'seleccion.json', 'estadisticas_confirmacion.json',
         'comparacion_por_ruido.csv', 'comparacion_por_qr.csv', 'comparacion.csv',
         'comparacion_emparejada.json')
COMPRESSED = {'resultados.csv', 'entrenamiento.csv', 'resumen_por_qr.csv'}


def portable(value):
    """Metadatos legibles en otros clones, sin rutas del equipo de ejecución."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute() and Path(value).is_relative_to(ROOT):
        return Path(value).relative_to(ROOT).as_posix()
    return value


def publish(output):
    source = ROOT / 'results/hopfield_unlearning'
    output = Path(output)
    manifest, inventory = [], []
    for name, phase, new_evaluations in RUNS:
        directory = source / name
        if not (directory / 'COMPLETED').is_file():
            raise ValueError(f'Ejecución incompleta: {directory}')
        config = json.loads((directory / 'config.json').read_text())
        with (directory / 'resultados.csv').open(newline='') as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
        inventory.append({'run': name, 'phase': phase, 'stored_rows': row_count,
                          'new_evaluations': new_evaluations,
                          'patterns_sha256': config.get('patterns_sha256',
                              config.get('reference_config', {}).get('patterns_sha256',
                                  config.get('selection', {}).get('patterns_sha256', '')))})
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            original = directory / filename
            if not original.is_file():
                continue
            raw = original.read_bytes()
            published_name = filename + '.gz' if filename in COMPRESSED else filename
            if filename in COMPRESSED:
                content = gzip.compress(raw, compresslevel=9, mtime=0)
                assert gzip.decompress(content) == raw
            elif filename.endswith('.json'):
                content = (json.dumps(portable(json.loads(raw)), ensure_ascii=False, indent=2) + '\n').encode()
            else:
                content = raw
            if len(content) > 250_000:
                raise ValueError(f'Archivo demasiado grande para esta publicación: {filename}')
            destination = target / published_name
            destination.write_bytes(content)
            manifest.append({'path': str(destination.relative_to(output)), 'bytes': len(content),
                             'source_sha256': sha256(raw).hexdigest(),
                             'published_sha256': sha256(content).hexdigest(),
                             'encoding': 'gzip' if filename in COMPRESSED else 'plain'})
    total = sum(entry['bytes'] for entry in manifest)
    if total > 1_500_000:
        raise ValueError('La evidencia excede el presupuesto de 1.5 MB')
    summary = {'description': 'Derived metrics only; no network weights, checkpoints or original QR images.',
               'new_evaluations_including_smoke': sum(row['new_evaluations'] for row in inventory),
               'new_evaluations_excluding_smoke': sum(row['new_evaluations'] for row in inventory
                                                     if not row['phase'].startswith('smoke')),
               'bytes': total, 'runs': inventory, 'files': manifest}
    (output / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (output / 'inventario.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(inventory)
    print(f'{len(manifest)} archivos de métricas; {total / 1024:.1f} KiB; ningún peso ni checkpoint.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/ejercicio2/desaprendizaje/evidencia')
    args = parser.parse_args()
    publish(args.output)


if __name__ == '__main__':
    main()
