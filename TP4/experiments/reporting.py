"""Identificación de configuraciones y resúmenes, sin entrenar modelos."""
from copy import deepcopy
import hashlib
import json

import pandas as pd

from src.tp4.export import write_json
from src.tp4.storage import atomic_file
from .results import RunRecord


def group_id(resolved):
    grouping = deepcopy(resolved)
    grouping['model'].pop('seed')
    grouping.pop('output', None)
    serialized = json.dumps(grouping, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:12]


def summarize(table):
    if table.empty:
        return pd.DataFrame(columns=['attempted', 'succeeded']).rename_axis('group')
    groups = table.groupby('group').agg(
        attempted=('status', 'size'),
        succeeded=('status', lambda statuses: (statuses == 'ok').sum()),
    )
    success = table[table.status == 'ok']
    if success.empty:
        return groups
    metrics = success.select_dtypes(include='number').columns.difference(['seed'])
    stats = success.groupby('group')[metrics].agg(['mean', 'std'])
    stats.columns = ['_'.join(column) for column in stats.columns]
    return groups.join(stats)


def write_reports(output, config, records: list[RunRecord], *, status='completed', total=None, interrupted_run=None):
    table = pd.DataFrame([record.to_dict() for record in records],
                         columns=None if records else ['run', 'group', 'seed', 'status'])
    with atomic_file(output / 'runs.csv') as stream:
        table.to_csv(stream, index=False)
    with atomic_file(output / 'summary.csv') as stream:
        summarize(table).to_csv(stream)
    manifest = {
        'requested': config,
        'status': status,
        'planned': len(records) if total is None else total,
        'pending': 0 if total is None else total - len(records),
        'interrupted_run': interrupted_run,
        'runs': len(records),
        'failed': int((table.status == 'failed').sum()),
        'groups': {
            record.group: str(output / record.run / 'config.json')
            for record in records
        },
    }
    write_json(output / 'experiment.json', manifest)
