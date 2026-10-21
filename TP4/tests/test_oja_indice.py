"""Índice de Oja: orientación respecto de PC1, convergencia y líderes por tema."""
import unittest

import numpy as np
import pandas as pd

from analysis.oja_indice import _first_epoch, comparison_summary, oriented, topic_leaders
from src.tp4.pca import compute_pca


class OrientationTests(unittest.TestCase):
    def test_flips_only_when_opposite_to_reference(self):
        reference = np.array([1.0, 0.0])
        np.testing.assert_array_equal(oriented(np.array([-.9, .1]), reference), [.9, -.1])
        np.testing.assert_array_equal(oriented(np.array([.9, .1]), reference), [.9, .1])

    def test_exact_pc1_reproduces_its_scores_and_variance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 3)) * [3.0, 1.0, .5]
        pca = compute_pca(x)
        summary = comparison_summary(x, -pca.components[0], pca)
        self.assertAlmostEqual(summary['score_correlation'], -1.0)
        self.assertAlmostEqual(summary['explained_variance_ratio'], pca.explained_variance_ratio[0])
        oriented_summary = comparison_summary(x, oriented(-pca.components[0], pca.components[0]), pca)
        self.assertAlmostEqual(oriented_summary['max_score_difference'], 0.0)


class ConvergenceTests(unittest.TestCase):
    def test_first_epoch_is_one_based_and_none_when_never_reached(self):
        alignment = np.array([.5, .99, .9995, .9999])
        self.assertEqual(_first_epoch(alignment, .999), 3)
        self.assertIsNone(_first_epoch(alignment, .99999))


class TopicLeaderTests(unittest.TestCase):
    def test_reports_index_and_cost_leaders_separately(self):
        index = pd.DataFrame({
            'name': ['@a', '@b', '@c'], 'main_topic': ['dogs'] * 3,
            'oja_index': [3.0, 1.0, -2.0], 'cpm_arg': [60.0, 30.0, 15.0],
            'argentina_audience_pct': [30.0, 50.0, 90.0], 'rank': [1, 2, 3],
        })
        leaders = topic_leaders(index).set_index('topic').loc['dogs']
        self.assertEqual(leaders.index_leader, '@a')
        self.assertEqual(leaders.cost_leader, '@c')
        self.assertEqual(leaders.cost_leader_index_rank, 3)


if __name__ == '__main__':
    unittest.main()
