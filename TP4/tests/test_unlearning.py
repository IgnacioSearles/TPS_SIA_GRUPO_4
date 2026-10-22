"""Invariantes del desaprendizaje, selección de objetivos y reproducibilidad."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.hopfield_qr_unlearning import load_config, run_specs
from src.tp4.hopfield import Hopfield, Recall
from src.tp4.unlearning import UnlearningTrainer, anti_hebbian_update


def correlated_patterns():
    # La mayoría +1 es un estado espurio estable, distinto de los tres recuerdos.
    patterns = np.ones((3, 12))
    for i in range(3):
        patterns[i, 4 * i:4 * (i + 1)] = -1
    return patterns


class AntiHebbianTests(unittest.TestCase):
    def test_update_raises_target_energy_by_exact_predicted_amount(self):
        patterns = correlated_patterns()
        network = Hopfield(patterns)
        state = np.ones(12)
        before = network.weights_.copy()
        energy = network.energy(state)
        rate = .2
        anti_hebbian_update(network, state, rate)
        expected = before.copy()
        for i in range(12):
            for j in range(12):
                if i != j:
                    expected[i, j] -= rate / 12 * state[i] * state[j]
        np.testing.assert_allclose(network.weights_, expected)
        np.testing.assert_array_equal(network.weights_, network.weights_.T)
        np.testing.assert_array_equal(np.diag(network.weights_), 0)
        self.assertAlmostEqual(network.energy(state) - energy, rate * 11 / 2)
        np.testing.assert_array_equal(network.patterns_, patterns)

    def test_target_and_inverse_produce_same_weight_change(self):
        a, b = Hopfield(correlated_patterns()), Hopfield(correlated_patterns())
        state = np.array([1, -1] * 6)
        anti_hebbian_update(a, state, .1)
        anti_hebbian_update(b, -state, .1)
        np.testing.assert_array_equal(a.weights_, b.weights_)

    def test_bad_arguments_do_not_modify_weights(self):
        network = Hopfield(correlated_patterns())
        before = network.weights_.copy()
        for rate in (0, -1, np.nan, np.inf):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                anti_hebbian_update(network, np.ones(12), rate)
        for state in (np.ones(11), np.zeros(12), np.ones((1, 12))):
            with self.assertRaises(ValueError):
                anti_hebbian_update(network, state, .1)
        np.testing.assert_array_equal(network.weights_, before)


class TrainerTests(unittest.TestCase):
    def test_directed_mode_penalizes_actual_spurious_fixed_point(self):
        trainer = UnlearningTrainer(correlated_patterns(), learning_rate=.2)
        result = trainer.network.recall(trainer.mixture)
        self.assertEqual(result.stop_reason, 'stable')
        self.assertEqual(result.iterations, 0)
        record = trainer.step()
        self.assertEqual(record.action, 'updated')
        self.assertEqual((record.attempt, record.updates), (1, 1))
        self.assertAlmostEqual(record.energy_after - record.energy_before, .2 * 11 / 2)

    def test_stored_patterns_and_inverses_are_never_directly_penalized(self):
        trainer = UnlearningTrainer(correlated_patterns())
        before = trainer.network.weights_.copy()
        for state in (trainer.network.patterns_[0], -trainer.network.patterns_[0]):
            result = Recall(state, 0, (0.,), 'stable')
            with patch.object(trainer.network, 'recall', return_value=result):
                self.assertEqual(trainer.step().action, 'protected')
        self.assertEqual(trainer.updates, 0)
        self.assertEqual(trainer.attempts, 2)
        np.testing.assert_array_equal(trainer.network.weights_, before)

    def test_cycles_and_caps_consume_attempt_but_do_not_update(self):
        trainer = UnlearningTrainer(correlated_patterns())
        before = trainer.network.weights_.copy()
        for reason in ('cycle', 'max_iterations'):
            result = Recall(trainer.mixture, 2, (0., 0., 0.), reason)
            with patch.object(trainer.network, 'recall', return_value=result):
                self.assertEqual(trainer.step().action, reason)
        self.assertEqual((trainer.attempts, trainer.updates), (2, 0))
        np.testing.assert_array_equal(trainer.network.weights_, before)

    def test_dreams_are_reproducible_and_do_not_use_global_rng(self):
        np.random.seed(8)
        expected_global = np.random.random(3)
        np.random.seed(8)
        a = UnlearningTrainer(correlated_patterns(), mode='dream', seed=9)
        b = UnlearningTrainer(correlated_patterns(), mode='dream', seed=9)
        for _ in range(15):
            self.assertEqual(a.step(), b.step())
        np.testing.assert_array_equal(a.network.weights_, b.network.weights_)
        np.testing.assert_array_equal(np.random.random(3), expected_global)

    def test_snapshot_evaluations_do_not_change_training_trajectory(self):
        patterns = correlated_patterns()
        a = UnlearningTrainer(patterns, mode='dream', seed=4)
        b = UnlearningTrainer(patterns, mode='dream', seed=4)
        for _ in range(8):
            a.step()
            for pattern in patterns:
                a.network.recall(pattern)
            b.step()
        np.testing.assert_array_equal(a.network.weights_, b.network.weights_)

    def test_even_majority_ties_are_resolved_without_zero_states(self):
        patterns = np.array([[1., -1., 1.], [-1., 1., -1.]])
        trainer = UnlearningTrainer(patterns)
        np.testing.assert_array_equal(trainer.mixture, patterns[0])

    def test_invalid_trainer_parameters(self):
        for arguments in ({'mode': 'other'}, {'learning_rate': 0},
                          {'max_iterations': 0}, {'max_iterations': True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                UnlearningTrainer(correlated_patterns(), **arguments)


class ExperimentConfigurationTests(unittest.TestCase):
    def test_directed_mode_is_not_repeated_as_independent_random_runs(self):
        config = {'modes': ['directed', 'dream'], 'learning_rates': [.01, .1],
                  'training_seeds': [0, 1, 2]}
        specs = list(run_specs(config))
        self.assertEqual(sum(mode == 'directed' for mode, _, _ in specs), 2)
        self.assertEqual(sum(mode == 'dream' for mode, _, _ in specs), 6)

    def test_configs_and_rejection_of_invalid_grid(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / 'configs/hopfield_unlearning_smoke.json')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            for update in ({'learning_rates': [-.1]}, {'noise_levels': [0, .7]},
                           {'query_seeds': [1, 1]}, {'checkpoints': [5]},
                           {'unexpected': 1}):
                path.write_text(json.dumps({**config, **update}))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
