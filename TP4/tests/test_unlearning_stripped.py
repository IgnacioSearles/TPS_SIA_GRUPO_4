"""Consultas completas, recuperación reducida y restauración de módulos comunes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.hopfield_qr_unlearning import evaluate, load_config, run
from src.tp4.hopfield import Hopfield, Recall
from src.tp4.queries import Query
from src.tp4.shared_modules import SharedModules
from src.tp4.unlearning import UnlearningTrainer, anti_hebbian_update


class StrippedUnlearningTests(unittest.TestCase):
    def setUp(self):
        self.patterns = np.array([[1, -1, 1, 1, 1, 1],
                                  [1, -1, -1, -1, 1, 1],
                                  [1, -1, 1, 1, -1, -1]], dtype=float)
        self.shared = SharedModules(self.patterns)
        self.active = self.shared.strip(self.patterns)
        self.network = Hopfield(self.active)
        self.mixture = UnlearningTrainer(self.active).mixture

    def evaluate_result(self, query, state, stop_reason='stable'):
        result = Recall(state, 0, (0.,), stop_reason)
        with patch.object(self.network, 'recall', return_value=result) as recall:
            rows = evaluate(self.network, self.patterns, [query], self.mixture,
                            {}, 100, self.shared)
        np.testing.assert_array_equal(recall.call_args.args[0], self.shared.strip(query.pattern))
        return rows[0]

    def test_shared_damage_is_removed_before_recall_and_restored_in_full_output(self):
        damaged = self.patterns[0].copy()
        damaged[:2] *= -1
        row = self.evaluate_result(Query(0, .2, 0, damaged), self.active[0])
        self.assertEqual(row['query_hamming'], 2)
        self.assertEqual(row['network_query_hamming'], 0)
        self.assertEqual(row['hamming'], 0)
        self.assertTrue(row['correct'])
        self.assertEqual(self.network.weights_.shape, (4, 4))

    def test_inverse_means_inverse_in_active_space_not_in_restored_shared_modules(self):
        row = self.evaluate_result(Query(0, 0., 0, self.patterns[0]), -self.active[0])
        self.assertEqual(row['category'], 'inverse')
        self.assertTrue(row['inverse_mixture'])
        self.assertEqual(row['hamming'], 4)
        self.assertFalse(row['correct'])

    def test_correct_bits_at_a_cycle_are_not_counted_as_success(self):
        row = self.evaluate_result(Query(0, 0., 0, self.patterns[0]), self.active[0], 'cycle')
        self.assertEqual(row['hamming'], 0)
        self.assertEqual(row['category'], 'cycle')
        self.assertFalse(row['correct'])

    def test_update_uses_active_neuron_count_for_normalization(self):
        before = self.network.energy(self.mixture)
        anti_hebbian_update(self.network, self.mixture, .2)
        self.assertAlmostEqual(self.network.energy(self.mixture) - before,
                               .2 * (self.active.shape[1] - 1) / 2)

    def test_config_defaults_to_complete_network_and_rejects_nonboolean_flag(self):
        root = Path(__file__).resolve().parents[1]
        full = load_config(root / 'configs/hopfield_unlearning_smoke.json')
        stripped = load_config(root / 'configs/hopfield_unlearning_stripped_smoke.json')
        self.assertFalse(full['remove_shared'])
        self.assertTrue(stripped['remove_shared'])
        for key in set(full) - {'name', 'remove_shared'}:
            self.assertEqual(full[key], stripped[key])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({**full, 'remove_shared': 'true'}))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_identical_patterns_do_not_create_a_zero_neuron_experiment(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / 'configs/hopfield_unlearning_stripped_smoke.json')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch('analysis.hopfield_qr_unlearning.load_patterns', return_value=np.ones((5, 6))):
                with self.assertRaisesRegex(ValueError, 'No hay módulos diferentes'):
                    run(config, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
