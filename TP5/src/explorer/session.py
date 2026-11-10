"""Lógica del explorador, independiente de HTTP: qué mostrar y cómo decodificar."""

from __future__ import annotations

import base64
import math
from dataclasses import replace
from typing import Any

import numpy as np

from src.data.audio import waveform_to_wav_bytes
from src.data.dataset import Dataset
from src.explorer.latent_map import LatentMap
from src.models.autoencoder import Autoencoder

Array = np.ndarray
# Menos iteraciones que al exportar audio: prioriza una respuesta inmediata.
INTERACTIVE_GRIFFIN_LIM_ITERATIONS = 32
# Cuántas veces el tamaño del mapa puede alejarse una posición pedida.
_MAXIMUM_POSITION_SCALE = 10.0


class InvalidRequest(ValueError):
    """El pedido del navegador no tiene la forma esperada."""


class ExplorerSession:
    def __init__(self, model: Autoencoder, dataset: Dataset, seed: int = 0) -> None:
        self._model = model
        self._dataset = dataset
        self._map = LatentMap(model.encode(dataset.samples))
        self._rng = np.random.default_rng(seed)
        self._codec = (
            None
            if dataset.audio_codec is None
            else replace(
                dataset.audio_codec, griffin_lim_iterations=INTERACTIVE_GRIFFIN_LIM_ITERATIONS
            )
        )
        self._position_limit = _MAXIMUM_POSITION_SCALE * float(np.abs(self._map.points).max())

    def describe(self) -> dict[str, Any]:
        """Todo lo que la página necesita para dibujar el mapa."""
        return {
            "points": [
                {"x": float(x), "y": float(y), "label": label, "group": group}
                for (x, y), label, group in zip(
                    self._map.points, self._dataset.labels, self._dataset.groups
                )
            ],
            "axes": list(self._map.axis_names),
            "image_shape": list(self._dataset.image_shape),
            "is_binary": self._dataset.is_binary,
            "has_audio": self._codec is not None,
        }

    def decode(self, request: dict[str, Any]) -> dict[str, Any]:
        """Decodifica el punto latente que describe `request`.

        Formas aceptadas:
            {"position": [x, y]}                         un punto del mapa
            {"from": [x, y], "to": [x, y], "fraction": t} interpolación latente entre dos puntos
            {"random": true}                              z ~ N(0, I)
        `"audio": false` omite la síntesis de audio (más rápido al arrastrar).
        """
        if not isinstance(request, dict):
            raise InvalidRequest("El pedido debe ser un objeto JSON")
        latent = self._requested_latent(request)
        decoded = np.asarray(self._model.decode(latent[None, :])[0], dtype=float)
        position = self._map.project(latent[None, :])[0]
        response: dict[str, Any] = {
            "position": [float(position[0]), float(position[1])],
            "image": _encode_bytes(np.round(np.clip(decoded, 0.0, 1.0) * 255).astype(np.uint8)),
        }
        if self._codec is not None and request.get("audio", True):
            waveform = self._codec.decode(decoded.reshape(self._dataset.image_shape))
            wav = waveform_to_wav_bytes(waveform, self._codec.sample_rate)
            response["audio"] = _encode_bytes(wav)
        return response

    def _requested_latent(self, request: dict[str, Any]) -> Array:
        if request.get("random") is True:
            return self._rng.standard_normal(self._model.latent_size)
        if "position" in request:
            return self._map.lift(self._position(request["position"]))
        if "from" in request and "to" in request:
            fraction = request.get("fraction")
            if not _is_finite_number(fraction) or not 0.0 <= fraction <= 1.0:
                raise InvalidRequest("'fraction' debe estar entre 0 y 1")
            origin = self._map.lift(self._position(request["from"]))
            destination = self._map.lift(self._position(request["to"]))
            return origin + fraction * (destination - origin)
        raise InvalidRequest("Se esperaba 'position', 'from'/'to' o 'random'")

    def _position(self, value: Any) -> Array:
        if not isinstance(value, list) or len(value) != 2 or not all(map(_is_finite_number, value)):
            raise InvalidRequest("Una posición es una lista [x, y] de números finitos")
        if max(abs(value[0]), abs(value[1])) > self._position_limit:
            raise InvalidRequest("La posición está demasiado lejos del mapa")
        return np.array(value, dtype=float)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _encode_bytes(content: bytes | Array) -> str:
    return base64.b64encode(bytes(content)).decode("ascii")
