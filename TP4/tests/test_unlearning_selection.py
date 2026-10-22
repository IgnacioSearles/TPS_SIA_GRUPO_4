"""El criterio de selección debe excluir candidatos inestables aunque acierten más."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.confirm_hopfield_unlearning_damage import damage_queries
from analysis.select_hopfield_unlearning import freeze_confirmation, rank_candidates
from src.tp4.qr import SIDE


def write_pilot(directory):
    config = {'name': 'synthetic', 'modes': ['dream'], 'learning_rates': [.05, .1],
              'checkpoints': [0, 10], 'training_seeds': [0, 1], 'noise_levels': [0, .2],
              'query_seeds': [10], 'max_iterations': 100, 'remove_shared': True,
              'qr_names': ['a', 'b'], 'patterns_sha256': 'synthetic'}
    (directory / 'config.json').write_text(json.dumps(config))
    (directory / 'COMPLETED').write_text('ok')
    rows, stability = [], []
    for method in ('hebb', 'nearest', 'unlearning'):
        rates = config['learning_rates'] if method == 'unlearning' else [0.]
        for rate in rates:
            seeds = [0, 1] if method == 'unlearning' else [-1]
            for seed in seeds:
                metadata = {'method': method, 'mode': 'dream' if method == 'unlearning' else 'none',
                            'learning_rate': rate, 'training_seed': seed,
                            'attempts': 10 if method == 'unlearning' else 0, 'updates': 5}
                for origin in range(2):
                    for noise in config['noise_levels']:
                        rows.append({**metadata, 'origin': origin, 'noise': noise, 'query_seed': 10,
                                     'correct': noise == 0 or rate == .05 or origin == 0,
                                     'cycle': False, 'capped': False})
                    if method == 'unlearning':
                        stability.append({**metadata, 'state': config['qr_names'][origin],
                                          'stable': not (rate == .05 and seed == 1 and origin == 1)})
    pd.DataFrame(rows).to_csv(directory / 'resultados.csv', index=False)
    pd.DataFrame(stability).to_csv(directory / 'estabilidad.csv', index=False)


class SelectionTests(unittest.TestCase):
    def test_accuracy_does_not_override_stability_requirement(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_pilot(directory)
            _, ranked = rank_candidates(directory)
            self.assertEqual(ranked.iloc[0].learning_rate, .1)
            self.assertEqual(ranked.iloc[0].mean_accuracy, .5)
            self.assertTrue(ranked.iloc[0].eligible)
            self.assertFalse(ranked.iloc[1].eligible)

    def test_freeze_uses_new_seeds_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_pilot(directory)
            destination = directory / 'confirmation.json'
            selection = freeze_confirmation(directory, destination)
            config = json.loads(destination.read_text())
            self.assertEqual(config['learning_rates'], [.1])
            self.assertEqual(config['query_seeds'], list(range(100, 120)))
            self.assertEqual(selection['confirmation_sha256'], sha256(destination.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                freeze_confirmation(directory, destination)

    def test_no_eligible_candidate_does_not_create_a_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_pilot(directory)
            stability = pd.read_csv(directory / 'estabilidad.csv')
            stability['stable'] = False
            stability.to_csv(directory / 'estabilidad.csv', index=False)
            destination = directory / 'confirmation.json'
            result = freeze_confirmation(directory, destination)
            self.assertEqual(result['status'], 'no_viable_candidate')
            self.assertFalse(destination.exists())

    def test_duplicate_query_rejected_even_when_row_count_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_pilot(directory)
            table = pd.read_csv(directory / 'resultados.csv')
            table.iloc[1] = table.iloc[0]
            table.to_csv(directory / 'resultados.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'duplicadas'):
                rank_candidates(directory)


class StructuredDamageTests(unittest.TestCase):
    def test_damage_locations_are_reproducible_and_do_not_modify_originals(self):
        patterns = np.ones((5, SIDE * SIDE))
        plan = {'block_sizes': [3, 29], 'spot_seeds': [100, 101],
                'corner_fill': -1., 'spot_fill': 1.}
        first = damage_queries(patterns, plan)
        second = damage_queries(patterns, plan)
        self.assertEqual(len(first), 5 * 2 * (4 + 2))
        for a, b in zip(first, second):
            self.assertEqual((a.origin, a.noise, a.seed, a.top, a.left),
                             (b.origin, b.noise, b.seed, b.top, b.left))
            np.testing.assert_array_equal(a.pattern, b.pattern)
            self.assertTrue(0 <= a.top <= SIDE - a.noise)
            self.assertTrue(0 <= a.left <= SIDE - a.noise)
            if a.kind == 'corner_white':
                self.assertEqual(np.count_nonzero(a.pattern == -1), a.noise ** 2)
        np.testing.assert_array_equal(patterns, 1.)


if __name__ == '__main__':
    unittest.main()
