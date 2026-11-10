"""Lectura estricta de secciones de configuración.

Cada constructor consume las claves que entiende y luego llama a
`ensure_consumed`, de modo que un error de tipeo en un JSON (o en una ruta de
la grilla de experimentos) falle de inmediato en vez de ignorarse en silencio.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigurationError(ValueError):
    """La configuración es inválida; el mensaje indica la sección y el motivo."""


class Options:
    def __init__(self, values: Mapping[str, Any] | None, context: str) -> None:
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise ConfigurationError(f"'{context}' debe ser un objeto, no {values!r}")
        self._values = dict(values)
        self._context = context
        self._consumed: set[str] = set()

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def take(self, key: str, default: Any) -> Any:
        """Devuelve el valor de `key`, o `default` si la clave se omitió."""
        self._consumed.add(key)
        return self._values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise ConfigurationError(f"Falta la clave obligatoria '{self._context}.{key}'")
        return self.take(key, None)

    def section(self, key: str) -> Options:
        """Subsección; si se omitió se comporta como una sección vacía."""
        return Options(self.take(key, None), f"{self._context}.{key}")

    def optional_section(self, key: str) -> Options | None:
        """Subsección que activa una funcionalidad solo si está presente."""
        if key not in self._values:
            return None
        return self.section(key)

    def keys(self) -> list[str]:
        return list(self._values)

    def ensure_consumed(self) -> None:
        unknown = sorted(set(self._values) - self._consumed)
        if unknown:
            raise ConfigurationError(
                f"Claves desconocidas en '{self._context}': {', '.join(unknown)}"
            )


def choose(registry: Mapping[str, Any], name: Any, kind: str) -> Any:
    """Busca `name` en un registro y lista las alternativas si no existe."""
    if name not in registry:
        raise ConfigurationError(
            f"{kind} desconocido: {name!r}. Opciones: {', '.join(sorted(registry))}"
        )
    return registry[name]
