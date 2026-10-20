"""Exporta un inventario liviano de manifiestos, sin datos ni rutas personales."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def export_inventory(root, output):
    snapshot = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []
    for path in sorted(root.rglob('experiment.json')):
        manifest = json.loads(path.read_text())
        runs = manifest.get('runs', 0)
        failed = manifest.get('failed', 0)
        rows.append({
            'snapshot_utc': snapshot,
            'experiment': path.parent.as_posix(),
            'algorithm': manifest['requested'].get('algorithm', 'kohonen'),
            'operation': manifest['requested'].get('operation', 'experiment'),
            'status': manifest.get('status', 'legacy_finished'),
            'planned': manifest.get('planned', runs),
            'finished': runs,
            'failed': failed,
            'pending': manifest.get('pending', 0),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=list(rows[0]) if rows else [
            'snapshot_utc', 'experiment', 'algorithm', 'operation', 'status',
            'planned', 'finished', 'failed', 'pending'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path(
        'reports/ejercicio1/bitacora/experimentos.csv'))
    args = parser.parse_args()
    if args.root.is_absolute():
        parser.error('--root debe ser relativo para no publicar rutas personales')
    rows = export_inventory(args.root, args.output)
    print(f'{len(rows)} experimentos registrados en {args.output}')


if __name__ == '__main__':
    main()
