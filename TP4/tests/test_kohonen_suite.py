"""Expansión de casos acoplados y comparación de barridos preparados."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.kohonen_experiments import adjusted_rand_index, compare, occupancy_metrics
from experiments.configuration import expand, resolve
from experiments.kohonen_suite import build_suite, confirmation_config, interaction_cases, suite_budget
from experiments.runner import run


class SuiteTests(unittest.TestCase):
    def test_cases_are_coupled_and_combine_with_grid(self):
        config = {'algorithm': 'kohonen', 'seeds': [0, 1],
                  'cases': [{'model.rows': 2, 'model.cols': 6}, {'model.rows': 3, 'model.cols': 4}],
                  'grid': {'model.topology': ['rectangular', 'hexagonal']}}
        runs = expand(config)
        self.assertEqual(len(runs), 8)
        self.assertEqual({(item['model']['rows'], item['model']['cols']) for item in runs}, {(2, 6), (3, 4)})
        broken = deepcopy(config)
        broken['grid']['model.rows'] = [3]
        with self.assertRaises(ValueError):
            expand(broken)
        for cases in ([], ['wrong'], [{'model.seed': 0}]):
            with self.assertRaises(ValueError):
                expand({'algorithm': 'kohonen', 'cases': cases})

    def test_complete_suite_budget_and_unique_configs_per_stage(self):
        suite = build_suite()
        expected = [8, 300, 260, 580, 360, 60, 240, 80, 120, 600]
        self.assertEqual([row[1] for row in suite_budget(suite)], expected)
        for name, config in suite.items():
            runs = [json.dumps(resolve(item), sort_keys=True) for item in expand(config)]
            self.assertEqual(len(set(runs)), len(runs), name)
        self.assertEqual(interaction_cases(), interaction_cases())

    def test_confirmation_preserves_candidates_with_new_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, shape in enumerate(((3, 4), (5, 5))):
                candidate = {'algorithm': 'kohonen', 'model': {
                    'rows': shape[0], 'cols': shape[1], 'topology': 'hexagonal',
                    'epochs': 750, 'seed': 2}, 'preprocessing': {'pca_components': 10}}
                path = Path(temporary) / f'candidate_{index}.json'
                path.write_text(json.dumps(candidate))
                paths.append(path)
            runs = [resolve(item) for item in expand(confirmation_config(paths))]
            self.assertEqual(len(runs), 60)
            self.assertEqual({item['model']['seed'] for item in runs}, set(range(100, 130)))
            self.assertEqual({(item['model']['rows'], item['model']['cols']) for item in runs}, {(3, 4), (5, 5)})
            self.assertTrue(all(item['model']['epochs'] == 750 for item in runs))
            self.assertTrue(all(item['preprocessing']['pca_components'] == 10 for item in runs))


class ComparisonTests(unittest.TestCase):
    def test_ari_is_label_invariant_and_detects_disagreement(self):
        self.assertEqual(adjusted_rand_index([0, 0, 1, 1], [3, 3, 8, 8]), 1)
        self.assertEqual(adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]), 1)
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), -.5)
        self.assertEqual(adjusted_rand_index([0, 1, 2], [3, 4, 5]), 1)
        with self.assertRaises(ValueError):
            adjusted_rand_index([0, 1], [0])

    def test_common_cohesion_and_empty_cells(self):
        x = np.array([[0.], [2.], [10.], [12.]])
        metrics = occupancy_metrics(np.array([0, 0, 2, 2]), 4, x)
        self.assertEqual(metrics['common_centroid_distance'], 1)
        self.assertEqual(metrics['occupied_neurons'], 2)
        self.assertEqual(metrics['empty_fraction'], .5)
        self.assertEqual(metrics['singleton_creator_fraction'], 0)

    def test_comparison_deduplicates_repeated_baselines_and_flags_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data.csv'
            pd.DataFrame(np.random.default_rng(2).normal(size=(12, 4)), columns=list('abcd')).to_csv(data, index=False)
            config = {'algorithm': 'kohonen', 'data': str(data), 'output': str(root / 'runs'),
                      'model': {'rows': 2, 'cols': 2, 'epochs': 2}, 'seeds': [0, 1],
                      'cases': [{'preprocessing.pca_components': None}, {'preprocessing.pca_components': 2}]}
            with patch('src.tp4.export.plot_grid'), patch('src.tp4.export.plot_learning_curves'):
                output = run(config, experiment=True)
            duplicate = output.parent / 'duplicate'
            shutil.copytree(output, duplicate)
            target = compare(output.parent, root / 'comparison')
            summary = pd.read_csv(target / 'summary.csv')
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary.successful_seeds == 2).all())
            self.assertTrue((summary.successful_runs == 4).all())
            self.assertTrue(np.isfinite(summary.common_centroid_distance_mean).all())
            stability = pd.read_csv(target / 'stability.csv')
            self.assertEqual(len(stability), 2)
            (duplicate / 'run_0000' / 'model.npz').unlink()
            compare(output.parent, target)
            self.assertEqual(len(pd.read_csv(target / 'errors.csv')), 1)


if __name__ == '__main__':
    unittest.main()
