"""Regiones del SOM: Ward con vecindad, estabilidad entre semillas y orden de las etiquetas."""
import unittest

import numpy as np
import pandas as pd

from analysis.som_campanas import _order_by_median, region_separation
from analysis.som_pca_comparison import aligned_to_reference
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


class AlignmentTests(unittest.TestCase):
    def test_each_region_takes_the_reference_label_it_overlaps_most(self):
        regions = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        reference = np.array(['R4', 'R4', 'R5', 'R5', 'R5'])
        self.assertEqual(list(aligned_to_reference(regions, reference)), ['R4', 'R4', 'R4', 'R5', 'R5'])


class SeparationTests(unittest.TestCase):
    def setUp(self):
        regions = ['R1', 'R1', 'R2', 'R2', 'R3', 'R3']
        self.frame = pd.DataFrame({
            'region': regions, 'followers': [10, 10, 100, 100, 1000, 1000],
            'argentina_audience_pct': [20, 80, 20, 80, 20, 80],
            'non_follower_reach_pct': [1, 2, 3, 4, 5, 6], 'content_tutorial_pct': [6, 5, 4, 3, 2, 1],
            'sponsored_posts_pct': [1, 1, 2, 2, 3, 4], 'cpm_arg': [10, 40, 10, 40, 10, 40]})

    def test_separates_what_differs_between_regions_and_not_what_differs_within(self):
        separation, _ = region_separation(self.frame)
        self.assertAlmostEqual(separation['log(seguidores)'], 1.0)
        self.assertAlmostEqual(separation['Audiencia argentina'], 0.0)
        self.assertAlmostEqual(separation['log(USD / 1.000 AR)'], 0.0)

    def test_random_baseline_is_the_expected_eta_squared_of_random_groups(self):
        _, baseline = region_separation(self.frame)
        self.assertAlmostEqual(baseline, (3 - 1) / (6 - 1))


if __name__ == '__main__':
    unittest.main()
