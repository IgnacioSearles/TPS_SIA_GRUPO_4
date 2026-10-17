"""Entrada para entrenar, experimentar y reanudar ejecuciones."""
import argparse
import json

from experiments.runner import resume, run


def parse_args():
    parser = argparse.ArgumentParser(description='Kohonen y Oja reproducibles')
    parser.add_argument('command', choices=['train', 'experiment', 'resume'])
    parser.add_argument('--config')
    parser.add_argument('--experiment', dest='experiment_path')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.command == 'resume':
        if not args.experiment_path or args.config:
            parser.error('resume requiere --experiment y no admite --config')
    elif not args.config or args.experiment_path or args.retry_failed:
        parser.error('train/experiment requieren --config; --experiment/--retry-failed son de resume')
    return args


def main():
    args = parse_args()
    if args.command == 'resume':
        output = resume(args.experiment_path, retry_failed=args.retry_failed, dry_run=args.dry_run)
    else:
        with open(args.config) as stream:
            config = json.load(stream)
        output = run(config, experiment=args.command == 'experiment', dry_run=args.dry_run)
    if not args.dry_run:
        manifest = json.loads((output / 'experiment.json').read_text())
        if manifest['status'] == 'paused':
            raise SystemExit(130)
        if manifest['failed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
