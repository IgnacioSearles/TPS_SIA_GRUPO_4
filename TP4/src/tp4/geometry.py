"""Grilla y operaciones comunes; la geometría se delega a Topology."""
import numpy as np

from .config import positive_int
from .numerics import distances
from .topology import Topology, create_topology


class Grid:
    def __init__(self, rows: int, cols: int, topology: Topology | str):
        positive_int('rows', rows)
        positive_int('cols', cols)
        if rows * cols < 2:
            raise ValueError('La grilla requiere al menos dos neuronas')

        # Se mantiene la construcción por nombre para compatibilidad con la API.
        self.layout = create_topology(topology) if isinstance(topology, str) else topology
        self.rows = rows
        self.cols = cols
        self.topology = self.layout.name
        row_indices, column_indices = np.indices((rows, cols))
        self.indices = np.column_stack((row_indices.ravel(), column_indices.ravel()))
        self.positions = self.layout.positions(self.indices)
        self.distances = self.layout.pairwise_distances(self.indices)
        self.adjacency = self.layout.adjacency(self.distances)

    def u_matrix(self, weights):
        neighbor_distances = distances(weights, weights) * self.adjacency
        neighbor_counts = self.adjacency.sum(axis=1)
        averages = neighbor_distances.sum(axis=1) / neighbor_counts
        return averages.reshape(self.rows, self.cols)
