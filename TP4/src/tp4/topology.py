"""Contrato geométrico e implementaciones, independientes del entrenamiento."""
from abc import ABC, abstractmethod

import numpy as np


class Topology(ABC):
    """Define posiciones, distancias y celdas para índices (fila, columna)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre usado en configuración y gráficos."""

    @abstractmethod
    def positions(self, indices: np.ndarray) -> np.ndarray:
        """Centros de las neuronas; matriz (neuronas, 2)."""

    @abstractmethod
    def pairwise_distances(self, indices: np.ndarray) -> np.ndarray:
        """Distancias entre todas las neuronas; matriz cuadrada."""

    def adjacency(self, pairwise_distances: np.ndarray) -> np.ndarray:
        """Las celdas inmediatas están a una unidad de distancia."""
        return np.isclose(pairwise_distances, 1)

    @abstractmethod
    def cell_vertices(self, center: np.ndarray) -> np.ndarray:
        """Vértices de una celda para dibujar su geometría sin matplotlib."""


class RectangularTopology(Topology):
    name = 'rectangular'

    def positions(self, indices):
        return indices[:, ::-1].astype(float)

    def pairwise_distances(self, indices):
        positions = self.positions(indices)
        differences = positions[:, None] - positions[None, :]
        return np.linalg.norm(differences, axis=2)

    def cell_vertices(self, center):
        offsets = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        return center + offsets


class HexagonalTopology(Topology):
    """Disposición odd-r: las filas impares están desplazadas a la derecha."""

    name = 'hexagonal'

    def positions(self, indices):
        rows, columns = indices.T
        return np.column_stack((columns + .5 * (rows % 2), rows * np.sqrt(3) / 2))

    def pairwise_distances(self, indices):
        rows, columns = indices.T
        axial_columns = columns - (rows - rows % 2) // 2
        cube = np.column_stack((axial_columns, -axial_columns - rows, rows))
        return np.abs(cube[:, None] - cube[None, :]).max(axis=2).astype(float)

    def cell_vertices(self, center):
        angles = np.pi / 2 + np.arange(6) * np.pi / 3
        offsets = np.column_stack((np.cos(angles), np.sin(angles))) / np.sqrt(3)
        return center + offsets


_TOPOLOGIES = {
    RectangularTopology.name: RectangularTopology,
    HexagonalTopology.name: HexagonalTopology,
}


def topology_names():
    return tuple(_TOPOLOGIES)


def create_topology(name: str) -> Topology:
    """Único punto donde un nombre de configuración se convierte en un objeto."""
    try:
        return _TOPOLOGIES[name]()
    except KeyError as error:
        raise ValueError(f'Topología desconocida: {name}') from error
