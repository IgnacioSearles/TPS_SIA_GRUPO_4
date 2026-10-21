"""Tablas del EDA orientadas al caso de uso: temas, bloques de variables y relaciones."""
import unittest

import numpy as np
import pandas as pd

from analysis.eda import (
    cost_per_thousand_argentine_reach, follower_relations, main_topic, topic_tables, variable_blocks,
)
from analysis.plots import _coincident_groups


def catalog(topic_rows):
    rows = len(topic_rows)
    topics = pd.DataFrame(topic_rows, columns=['topic_football_pct', 'topic_makeup_pct', 'topic_dogs_pct', 'topic_art_pct'])
    return pd.concat([pd.DataFrame({
        'creator_id': [f'INF{index:03d}' for index in range(rows)],
        'followers': np.linspace(1e4, 1e5, rows),
        'price_usd': np.linspace(100, 1000, rows),
        'average_reach': np.full(rows, 2000.0),
        'estimated_argentina_reach': np.full(rows, 1000.0),
        'argentina_audience_pct': np.full(rows, 50.0),
        'non_follower_reach_pct': np.linspace(20, 80, rows),
        'content_tutorial_pct': np.linspace(60, 0, rows),
        'saves_per_1000': np.linspace(20, 5, rows),
        'profile_visits_per_1000': np.full(rows, 10.0),
        'average_watch_time_seconds': np.full(rows, 20.0),
    }), topics], axis=1)


class TopicTests(unittest.TestCase):
    def test_main_topic_uses_largest_share_and_first_on_ties(self):
        frame = catalog([[70, 10, 10, 10], [10, 10, 10, 70], [40, 40, 10, 10]])
        self.assertEqual(list(main_topic(frame)), ['football', 'art', 'football'])

    def test_specialization_reports_first_and_second_share(self):
        frame = catalog([[70, 20, 5, 5], [10, 15, 60, 15]])
        tables = topic_tables(frame)
        np.testing.assert_array_equal(tables.specialization.main_share, [70, 60])
        np.testing.assert_array_equal(tables.specialization.second_share, [20, 15])
        self.assertEqual(tables.counts.loc['makeup'], 0)
        self.assertEqual(list(tables.counts.index), ['football', 'makeup', 'dogs', 'art'])

    def test_cost_is_usd_per_thousand_argentine_accounts(self):
        frame = catalog([[70, 10, 10, 10]])
        self.assertAlmostEqual(cost_per_thousand_argentine_reach(frame).iloc[0], 100.0)


class VariableBlockTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        size, other = rng.normal(size=200), rng.normal(size=200)
        noise = lambda: rng.normal(scale=.2, size=200)
        self.frame = pd.DataFrame({
            'reach': size + noise(), 'views': size + noise(),
            'saves': -size + noise(), 'captions': -size + noise(),
            'audience': other,
        })

    def test_opposite_variables_land_in_different_blocks(self):
        blocks = variable_blocks(self.frame)
        membership = blocks.membership
        self.assertEqual(membership['reach'], membership['views'])
        self.assertEqual(membership['saves'], membership['captions'])
        self.assertNotEqual(membership['reach'], membership['saves'])
        self.assertEqual((membership == membership['audience']).sum(), 1)

    def test_order_keeps_blocks_contiguous_and_numbered_in_order(self):
        blocks = variable_blocks(self.frame)
        self.assertEqual(set(blocks.ordered_correlations.columns), set(self.frame.columns))
        self.assertEqual(list(blocks.ordered_correlations.index), list(blocks.membership.index))
        labels = blocks.membership.to_numpy()
        self.assertEqual(list(pd.unique(labels)), list(range(1, labels.max() + 1)))
        changes = (np.diff(labels) != 0).sum()
        self.assertEqual(changes, labels.max() - 1)

    def test_summary_mean_correlation_excludes_the_diagonal(self):
        summary = variable_blocks(self.frame).summary()
        singleton = summary[summary.variables == 1]
        self.assertTrue(singleton.correlacion_media.isna().all())
        self.assertTrue((summary.dropna().correlacion_media > .8).all())


class RelationTests(unittest.TestCase):
    def test_spearman_captures_monotonic_but_nonlinear_relation(self):
        frame = catalog([[70, 10, 10, 10]] * 30)
        frame['price_usd'] = np.exp(frame.followers / 1e4)
        relations = follower_relations(frame, ['price_usd'])
        self.assertAlmostEqual(relations.loc['price_usd', 'spearman'], 1.0)
        self.assertLess(relations.loc['price_usd', 'pearson'], .9)


class BiplotLabelTests(unittest.TestCase):
    def test_coincident_arrow_tips_share_one_label(self):
        tips = np.array([[1.0, 0.0], [1.05, 0.02], [-1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(_coincident_groups(tips), [[0, 1], [2], [3]])


if __name__ == '__main__':
    unittest.main()
