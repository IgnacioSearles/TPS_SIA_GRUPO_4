"""Correspondencia entre el espacio latente y un mapa 2D navegable.

Con dos dimensiones latentes el mapa es el latente mismo. Con más, el mapa es
el plano de las dos componentes principales, y al volver del mapa al latente
se completa lo que queda fuera del plano interpolando entre las muestras
vecinas. Así, hacer clic sobre una muestra devuelve exactamente su código y
moverse entre muestras produce mezclas suaves de sus códigos.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray
_COINCIDENCE_DISTANCE = 1e-9


class LatentMap:
    def __init__(self, codes: Array, neighbors: int = 8) -> None:
        if codes.ndim != 2 or codes.shape[1] < 2:
            raise ValueError("El explorador necesita un espacio latente de al menos 2 dimensiones")
        if neighbors < 1:
            raise ValueError("neighbors debe ser positivo")
        codes = np.asarray(codes, dtype=float)
        if codes.shape[1] == 2:
            self._center = np.zeros(2)
            self._basis = np.eye(2)
            self.axis_names = ("z1", "z2")
        else:
            self._center = codes.mean(axis=0)
            _, singular_values, components = np.linalg.svd(
                codes - self._center, full_matrices=False
            )
            self._basis = components[:2]
            explained = singular_values**2 / np.sum(singular_values**2)
            self.axis_names = tuple(f"PC{i + 1} ({explained[i]:.0%})" for i in range(2))
        self.points = self.project(codes)
        self._residuals = codes - self._center - self.points @ self._basis
        self._neighbors = min(neighbors, len(codes))

    def project(self, latents: Array) -> Array:
        """Códigos latentes (N x D) -> posiciones en el mapa (N x 2)."""
        return (np.asarray(latents, dtype=float) - self._center) @ self._basis.T

    def lift(self, position: Array) -> Array:
        """Posición del mapa (2,) -> código latente (D,)."""
        position = np.asarray(position, dtype=float)
        return self._center + position @ self._basis + self._residual_at(position)

    def _residual_at(self, position: Array) -> Array:
        """Parte fuera del plano, promediada entre las muestras vecinas (inverso del cuadrado)."""
        distances = np.linalg.norm(self.points - position, axis=1)
        nearest = np.argsort(distances)[: self._neighbors]
        if distances[nearest[0]] < _COINCIDENCE_DISTANCE:
            return self._residuals[nearest[0]]
        weights = 1.0 / distances[nearest] ** 2
        return weights @ self._residuals[nearest] / weights.sum()
