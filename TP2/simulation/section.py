"""Lectura tipada de objetos JSON de configuración, con errores ubicados."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_MISSING = object()


class ConfigurationError(ValueError):
    """Configuración inválida: clave desconocida, tipo incorrecto o valor fuera de rango."""


class ConfigSection:
    """Vista de solo lectura sobre un objeto JSON que valida tipos y ubica los errores.

    Cada acceso registra la clave leída para que `ensure_no_unknown_keys` pueda
    rechazar las sobrantes: un typo en un archivo de configuración debe fallar
    rápido en vez de pasar inadvertido como un parámetro silenciosamente ignorado.

    Una clave ausente y una con valor `null` son equivalentes: ambas significan
    "usar el valor por defecto", de modo que cualquier parámetro puede omitirse.
    """

    def __init__(self, data: Mapping[str, Any], path: str = "") -> None:
        if not isinstance(data, Mapping):
            raise ConfigurationError(f"{path or 'la raíz'} debe ser un objeto JSON.")
        self._data = data
        self._path = path
        self._known_keys: set[str] = set()
        self._children: list[ConfigSection] = []

    @property
    def path(self) -> str:
        """Ruta de esta sección dentro del archivo, para mensajes de error."""
        return self._path or "la raíz"

    def optional_integer(self, key: str) -> int | None:
        value = self._value(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._type_error(key, "un entero", value)
        return value

    def integer(self, key: str, default: int) -> int:
        value = self.optional_integer(key)
        return default if value is None else value

    def optional_number(self, key: str) -> float | None:
        value = self._value(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._type_error(key, "un número", value)
        return float(value)

    def number(self, key: str, default: float) -> float:
        value = self.optional_number(key)
        return default if value is None else value

    def boolean(self, key: str, default: bool) -> bool:
        value = self._value(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise self._type_error(key, "un booleano", value)
        return value

    def optional_text(self, key: str) -> str | None:
        value = self._value(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise self._type_error(key, "un texto", value)
        return value

    def text(self, key: str, default: str) -> str:
        value = self.optional_text(key)
        return default if value is None else value

    def required_text(self, key: str) -> str:
        value = self.optional_text(key)
        if value is None:
            raise ConfigurationError(f"{self._locate(key)} es obligatorio.")
        return value

    def choice(self, key: str, default: str, options: Sequence[str]) -> str:
        value = self.text(key, default)
        if value not in options:
            raise ConfigurationError(
                f"{self._locate(key)}: '{value}' no es una opción válida. "
                f"Opciones: {', '.join(sorted(options))}."
            )
        return value

    def number_list(self, key: str, default: tuple[float, ...]) -> tuple[float, ...]:
        value = self._value(key)
        if value is None:
            return default
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise self._type_error(key, "una lista de números", value)
        numbers = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise self._type_error(f"{key}[{index}]", "un número", item)
            numbers.append(float(item))
        return tuple(numbers)

    def number_mapping(self, key: str) -> dict[str, float] | None:
        """Lee un objeto `{"nombre": número}`, típico de pesos por componente."""
        value = self._value(key)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise self._type_error(key, "un objeto de nombre a número", value)
        weights = {}
        for name, weight in value.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise self._type_error(f"{key}.{name}", "un número", weight)
            weights[name] = float(weight)
        return weights

    def section(self, key: str) -> ConfigSection:
        """Devuelve la subsección; si falta, una vacía para que apliquen los defaults."""
        return self.optional_section(key) or self._child(key, {})

    def optional_section(self, key: str) -> ConfigSection | None:
        """Devuelve la subsección solo si fue declarada, para funcionalidades opcionales."""
        value = self._value(key)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise self._type_error(key, "un objeto JSON", value)
        return self._child(key, value)

    def ensure_no_unknown_keys(self) -> None:
        """Rechaza las claves que nadie leyó, en esta sección y en las anidadas."""
        for child in self._children:
            child.ensure_no_unknown_keys()
        unknown = sorted(set(self._data) - self._known_keys)
        if unknown:
            raise ConfigurationError(
                f"clave(s) desconocida(s) en {self.path}: {', '.join(unknown)}. "
                f"Claves válidas: {', '.join(sorted(self._known_keys))}."
            )

    def _child(self, key: str, data: Mapping[str, Any]) -> ConfigSection:
        child = ConfigSection(data, self._locate(key))
        self._children.append(child)
        return child

    def _value(self, key: str) -> Any:
        self._known_keys.add(key)
        value = self._data.get(key, _MISSING)
        return None if value is _MISSING else value

    def _locate(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _type_error(self, key: str, expected: str, value: Any) -> ConfigurationError:
        return ConfigurationError(
            f"{self._locate(key)}: se esperaba {expected} y se recibió {value!r}."
        )
