"""Métricas del caso de uso para comparar qué separa cada mapa de Kohonen."""
import unittest

import numpy as np

from analysis.representation_summary import eta_squared, representation_label, topic_purity
from src.tp4.preprocessing import DEFAULT_DROP, DEFAULT_LOG


class EtaSquaredTests(unittest.TestCase):
    def test_bounds_for_perfect_and_null_separation(self):
        values = np.array([1., 1., 5., 5.])
        self.assertAlmostEqual(eta_squared([0, 0, 1, 1], values), 1.0)
        self.assertAlmostEqual(eta_squared([0, 1, 0, 1], values), 0.0)

    def test_neuron_labels_do_not_matter(self):
        values = np.array([2., 3., 7., 9., 4.])
        self.assertAlmostEqual(eta_squared([0, 0, 1, 1, 2], values), eta_squared([8, 8, 3, 3, 5], values))

    def test_rejects_constant_values(self):
        with self.assertRaises(ValueError):
            eta_squared([0, 1], [3., 3.])


class TopicPurityTests(unittest.TestCase):
    def test_counts_majority_topic_per_neuron(self):
        topics = ['dogs', 'dogs', 'art', 'art', 'art', 'football']
        self.assertAlmostEqual(topic_purity([0, 0, 0, 1, 1, 1], topics), 4 / 6)
        self.assertAlmostEqual(topic_purity([0, 0, 1, 1, 1, 2], topics), 1.0)


class RepresentationLabelTests(unittest.TestCase):
    def test_labels_each_planned_representation(self):
        def label(exclude=DEFAULT_DROP, log=(), components=None):
            return representation_label({'exclude': list(exclude), 'log_columns': list(log), 'pca_components': components})
        self.assertEqual(label(), '27 variables')
        self.assertEqual(label(exclude=[]), '31 variables')
        self.assertEqual(label(exclude=DEFAULT_DROP + ['followers', 'average_impressions', 'average_video_views']), '24 variables')
        self.assertEqual(label(components=10), 'PCA 10')
        self.assertEqual(label(log=DEFAULT_LOG), 'log')
        self.assertEqual(label(log=DEFAULT_LOG, components=6), 'log + PCA 6')

    def test_rejects_unplanned_selection(self):
        with self.assertRaises(ValueError):
            representation_label({'exclude': ['price_ars'], 'log_columns': [], 'pca_components': None})


if __name__ == '__main__':
    unittest.main()
