"""Escrituras por reemplazo atómico: un corte conserva el archivo anterior."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_file(path, *, binary=False):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb' if binary else 'w') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
