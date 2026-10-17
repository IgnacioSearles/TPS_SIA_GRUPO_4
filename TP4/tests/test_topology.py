"""Pruebas del contrato geométrico y la inyección en Kohonen."""
import unittest

import numpy as np

from src.tp4.config import KohonenConfig
from src.tp4.geometry import Grid
from src.tp4.kohonen import Kohonen
from src.tp4.topology import (
    HexagonalTopology,
    RectangularTopology,
    Topology,
    create_topology,
)


class DiagonalTopology(RectangularTopology):
    """Topología de prueba ajena a la fábrica: solo adyacencia diagonal."""

    name = 'diagonal_test'

    def adjacency(self, pairwise_distances):
        return np.isclose(pairwise_distances, np.sqrt(2))


class TopologyTests(unittest.TestCase):
    def test_contract_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            Topology()

    def test_factory(self):
        self.assertIsInstance(create_topology('rectangular'), RectangularTopology)
        self.assertIsInstance(create_topology('hexagonal'), HexagonalTopology)
        with self.assertRaises(ValueError):
            create_topology('unknown')

    def test_known_positions_and_distances(self):
        rectangle = Grid(2, 2, RectangularTopology())
        np.testing.assert_array_equal(rectangle.positions, [[0, 0], [1, 0], [0, 1], [1, 1]])
        self.assertAlmostEqual(rectangle.distances[0, 3], np.sqrt(2))
        self.assertFalse(rectangle.adjacency[0, 3])
        hexagon = Grid(2, 2, HexagonalTopology())
        np.testing.assert_allclose(hexagon.positions[2], [.5, np.sqrt(3) / 2])
        self.assertEqual(hexagon.distances[0, 3], 2)
        self.assertTrue(hexagon.adjacency[1, 2])

    def test_name_and_injection_produce_same_training(self):
        samples = np.random.default_rng(3).normal(size=(10, 3))
        for layout in (RectangularTopology(), HexagonalTopology()):
            config = KohonenConfig(rows=2, cols=3, topology=layout.name, epochs=4)
            selected = Kohonen(config).fit(samples)
            injected = Kohonen(config, topology=layout).fit(samples)
            np.testing.assert_array_equal(selected.weights_, injected.weights_)
            np.testing.assert_array_equal(selected.u_matrix(), injected.u_matrix())

    def test_custom_topology_controls_metrics(self):
        model = Kohonen(KohonenConfig(rows=2, cols=2), topology=DiagonalTopology())
        model.weights_ = np.array([[0.], [10.], [20.], [.1]])
        self.assertEqual(model.grid.topology, 'diagonal_test')
        self.assertEqual(model.metrics([[.01]])['topographic_error'], 0)
        np.testing.assert_allclose(model.u_matrix(), [[.1, 10], [10, .1]])

    def test_cell_geometry(self):
        center = np.array([2., 3.])
        for layout, vertices in ((RectangularTopology(), 4), (HexagonalTopology(), 6)):
            polygon = layout.cell_vertices(center)
            self.assertEqual(polygon.shape, (vertices, 2))
            np.testing.assert_allclose(polygon.mean(axis=0), center)
        with self.assertRaises(ValueError):
            Grid(1, 1, RectangularTopology())


if __name__ == '__main__':
    unittest.main()
