"""Regiones del SOM: Ward con vecindad, estabilidad entre semillas y orden de las etiquetas."""
import unittest

import numpy as np

from analysis.som_campanas import _order_by_median
from analysis.som_regions import co_assignment, contiguous_ward, membership_firmness
from src.tp4.geometry import Grid


class ContiguousWardTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(1, 4, 'rectangular')

    def test_never_joins_similar_but_non_adjacent_neurons(self):
        # Las neuronas 0 y 3 son iguales pero están en los extremos de la fila: no pueden formar una región.
        weights = np.array([[0.0], [10.0], [10.5], [0.0]])
        labels = contiguous_ward(weights, self.grid.adjacency, 3)
        self.assertNotEqual(labels[0], labels[3])
        self.assertEqual(labels[1], labels[2])

    def test_merges_the_closest_adjacent_pair_first(self):
        weights = np.array([[0.0], [.1], [5.0], [9.0]])
        self.assertEqual(list(contiguous_ward(weights, self.grid.adjacency, 3)), [0, 0, 1, 2])

    def test_rejects_impossible_region_counts(self):
        with self.assertRaises(ValueError):
            contiguous_ward(np.zeros((4, 1)), self.grid.adjacency, 5)


class StabilityTests(unittest.TestCase):
    def test_co_assignment_ignores_label_names(self):
        together = co_assignment([[0, 0, 1], [1, 1, 0]])
        np.testing.assert_array_equal(together, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    def test_firmness_drops_for_a_creator_that_changes_group(self):
        reference = [0, 0, 0, 1]
        firmness = membership_firmness(reference, [reference, [0, 0, 1, 1]])
        np.testing.assert_allclose(firmness, [.75, .75, .5, 1.0])


class OrderingTests(unittest.TestCase):
    def test_regions_are_numbered_by_median_followers(self):
        neuron_labels = np.array([0, 1, 2])
        winners = np.array([0, 0, 1, 2])
        followers = np.array([500.0, 700.0, 10.0, 90.0])
        self.assertEqual(list(_order_by_median(neuron_labels, winners, followers)), [2, 0, 1])


if __name__ == '__main__':
    unittest.main()
