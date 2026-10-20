import unittest

import numpy as np

from src.tp4.hopfield import Hopfield
from src.tp4.recall_outcome import classify

# Ejemplo del apunte (Unidad 9, "Modelo de Hopfield > Ejemplo").
XI_1 = np.array([1, 1, -1, -1])
XI_2 = np.array([-1, -1, 1, 1])


class ApunteExampleTests(unittest.TestCase):
    def setUp(self):
        self.network = Hopfield(np.vstack([XI_1, XI_2]))

    def test_weights_match_apunte(self):
        expected = np.array([[0, .5, -.5, -.5],
                             [.5, 0, -.5, -.5],
                             [-.5, -.5, 0, .5],
                             [-.5, -.5, .5, 0]])
        np.testing.assert_allclose(self.network.weights_, expected)

    def test_query_reaches_first_pattern_in_one_iteration_and_stays(self):
        result = self.network.recall(np.array([1, -1, -1, -1]))
        np.testing.assert_array_equal(result.state, XI_1)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.stop_reason, 'stable')

    def test_energy_recorded_for_initial_state_and_each_iteration(self):
        # W·ζ = (0,5; 1,5; −0,5; −0,5) → H(ζ) = 0; W·ξ₁ = (1,5; 1,5; −1,5; −1,5) → H(ξ₁) = −3.
        result = self.network.recall(np.array([1, -1, -1, -1]))
        np.testing.assert_allclose(result.energies, [0, -3])

    def test_sequence_of_states_only_when_asked(self):
        query = np.array([1, -1, -1, -1])
        self.assertIsNone(self.network.recall(query).states)
        states = self.network.recall(query, return_states=True).states
        np.testing.assert_array_equal(states, [query, XI_1])

    def test_energy_is_minus_sum_over_pairs_of_weight_times_states(self):
        # H = −½·Σᵢⱼ wᵢⱼ SᵢSⱼ = −Σ_{j>i} wᵢⱼ SᵢSⱼ, con w₁₂ = w₃₄ = 0,5 y el resto −0,5.
        # (1, 1, 1, 1): −(0,5 − 0,5 − 0,5 − 0,5 − 0,5 + 0,5) = 1.
        # ξ₁ = (1, 1, −1, −1): los seis productos valen 0,5 → H = −3, igual que ξ₂.
        self.assertAlmostEqual(self.network.energy(np.ones(4)), 1)
        self.assertAlmostEqual(self.network.energy(XI_1), -3)
        self.assertAlmostEqual(self.network.energy(XI_2), -3)


ORTHOGONAL = np.array([[1, 1, 1, 1, -1, -1, -1, -1],
                       [1, 1, -1, -1, 1, 1, -1, -1]])

# Tres patrones ortogonales de 4 neuronas: w₁₂ = w₁₃ = w₂₄ = w₃₄ = ¼, w₁₄ = w₂₃ = −¼.
# Desde ENTERS_CYCLE = q = sign(ξ₁ + ξ₂ + ξ₃), el modo sincrónico recorre
# q → a = (1, −1, −1, 1) → b = (−1, 1, 1, −1) → a: entra a un ciclo de período 2.
THREE_OF_FOUR = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1]])
ENTERS_CYCLE = np.array([1, 1, 1, -1])


def outcome(network, query, origin, **options):
    result = network.recall(query, **options)
    return classify(result.state, result.stop_reason, network.patterns_, origin)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.network = Hopfield(ORTHOGONAL)

    def test_orthogonal_stored_patterns_are_stable(self):
        for pattern in ORTHOGONAL:
            result = self.network.recall(pattern)
            self.assertEqual((result.stop_reason, result.iterations), ('stable', 0))

    def test_one_error_is_corrected(self):
        query = ORTHOGONAL[0].copy()
        query[2] *= -1
        self.assertEqual(outcome(self.network, query, 0), ('correct', 0))

    def test_inverse_pattern_is_reported_as_inverse(self):
        self.assertEqual(outcome(self.network, -ORTHOGONAL[0], 0), ('inverse', 8))

    def test_landing_on_another_stored_pattern(self):
        # ξ₁ y ξ₂ difieren en 4 de 8 módulos.
        self.assertEqual(outcome(self.network, ORTHOGONAL[1], 0), ('other_pattern', 4))


class SpuriousStateTests(unittest.TestCase):
    def test_mixture_of_three_patterns_is_stable_and_spurious(self):
        # Cada neurona toma una de las 8 combinaciones de signos de (ξ₁, ξ₂, ξ₃):
        # patrones ortogonales cuyo estado mezcla sign(ξ₁ + ξ₂ + ξ₃) es estable.
        patterns = np.array([[1, 1, 1, 1, -1, -1, -1, -1],
                             [1, 1, -1, -1, 1, 1, -1, -1],
                             [1, -1, 1, -1, 1, -1, 1, -1]])
        mixture = np.sign(patterns.sum(axis=0))
        network = Hopfield(patterns)
        result = network.recall(mixture)
        self.assertEqual((result.stop_reason, result.iterations), ('stable', 0))
        self.assertEqual(outcome(network, mixture, 0), ('spurious', 2))


class EnergyRiseTests(unittest.TestCase):
    def test_apunte_example_never_raises_energy(self):
        network = Hopfield(np.vstack([XI_1, XI_2]))
        self.assertEqual(network.recall(np.array([1, -1, -1, -1])).energy_rises, 0)

    def test_synchronous_update_can_raise_energy(self):
        # Desde el estado mezcla de THREE_OF_FOUR, con H = 0, la actualización sincrónica
        # cambia tres neuronas a la vez y llega a (1, −1, −1, 1), con H = 1,5;
        # después oscila con (−1, 1, 1, −1), de igual energía.
        network = Hopfield(THREE_OF_FOUR)
        result = network.recall(ENTERS_CYCLE)
        np.testing.assert_allclose(result.energies, [0, 1.5, 1.5, 1.5])
        self.assertEqual(result.stop_reason, 'cycle')
        self.assertEqual(result.energy_rises, 1)


class ValidationTests(unittest.TestCase):
    def test_rejects_non_bipolar_patterns_and_queries(self):
        with self.assertRaises(ValueError):
            Hopfield(np.array([[1, 0, -1]]))
        network = Hopfield(ORTHOGONAL)
        with self.assertRaises(ValueError):
            network.recall(np.ones(5))


class ZeroFieldTests(unittest.TestCase):
    def test_neuron_with_zero_field_keeps_its_state(self):
        # ξ = (1, 1, 1), w = 1/3 fuera de la diagonal. Desde (−1, 1, −1) las
        # neuronas 1 y 3 tienen campo 0 y conservan −1; la 2 pasa a −1.
        network = Hopfield(np.array([[1, 1, 1]]))
        result = network.recall(np.array([-1, 1, -1]))
        np.testing.assert_array_equal(result.state, [-1, -1, -1])
        self.assertEqual(result.iterations, 1)


class StopReasonTests(unittest.TestCase):
    def setUp(self):
        # Con ξ = (1, −1), w₁₂ = −0,5: desde (1, 1) el modo sincrónico
        # salta a (−1, −1) y vuelve a (1, 1).
        self.network = Hopfield(np.array([[1, -1]]))

    def test_synchronous_oscillation_stops_as_cycle(self):
        result = self.network.recall(np.array([1, 1]))
        self.assertEqual(result.stop_reason, 'cycle')
        self.assertEqual(result.iterations, 2)
        np.testing.assert_array_equal(result.state, [1, 1])

    def test_unstable_endings_are_reported_by_stop_reason(self):
        # (1, 1) está a Hamming 1 de ξ = (1, −1); (−1, −1) también.
        self.assertEqual(outcome(self.network, np.array([1, 1]), 0), ('cycle', 1))
        # La consulta del caso de abajo se mueve a (1, −1, −1, 1), a Hamming 2 de ξ₁.
        network = Hopfield(THREE_OF_FOUR)
        self.assertEqual(outcome(network, ENTERS_CYCLE, 0, max_iterations=1), ('max_iterations', 2))

    def test_fixed_point_is_stable_not_a_cycle(self):
        # Un estado estable también "vuelve" a sí mismo: debe informarse como
        # estado estable, nunca como ciclo de período 1, con cualquier tope.
        network = Hopfield(ORTHOGONAL)
        for limit in (0, 1, 100):
            result = network.recall(ORTHOGONAL[0], max_iterations=limit)
            self.assertEqual((result.stop_reason, result.iterations), ('stable', 0))


class LimitTests(unittest.TestCase):
    def setUp(self):
        self.network = Hopfield(THREE_OF_FOUR)

    def test_query_not_stable_after_the_limit_stops_at_limit(self):
        # Tras 1 iteración la red está en a y el paso siguiente la lleva a b, nuevo.
        result = self.network.recall(ENTERS_CYCLE, max_iterations=1)
        self.assertEqual((result.stop_reason, result.iterations), ('max_iterations', 1))
        np.testing.assert_array_equal(result.state, [1, -1, -1, 1])

    def test_returning_to_a_visited_state_at_the_limit_is_a_cycle(self):
        # Tras 2 iteraciones la red está en b y el paso siguiente vuelve a a: es un
        # ciclo, no un corte por tope. La red no pasa del tope: se queda en b.
        result = self.network.recall(ENTERS_CYCLE, max_iterations=2)
        self.assertEqual((result.stop_reason, result.iterations), ('cycle', 2))
        np.testing.assert_array_equal(result.state, [-1, 1, 1, -1])

    def test_stable_state_at_the_limit_is_stable(self):
        # El ejemplo del apunte llega a ξ₁ en 1 iteración: con tope 1 es estado estable.
        network = Hopfield(np.vstack([XI_1, XI_2]))
        result = network.recall(np.array([1, -1, -1, -1]), max_iterations=1)
        self.assertEqual((result.stop_reason, result.iterations), ('stable', 1))


class PseudoinverseTests(unittest.TestCase):
    def test_equals_hebb_for_orthogonal_patterns(self):
        # Con patrones ortogonales, ΞΞᵀ = N·I y la pseudoinversa se reduce a Hebb.
        hebb = Hopfield(ORTHOGONAL)
        pseudoinverse = Hopfield(ORTHOGONAL, rule='pseudoinverse')
        np.testing.assert_allclose(pseudoinverse.weights_, hebb.weights_)

    def test_weights_are_symmetric_with_zero_diagonal(self):
        network = Hopfield(THREE_OF_FOUR, rule='pseudoinverse')
        np.testing.assert_allclose(network.weights_, network.weights_.T)
        np.testing.assert_allclose(np.diag(network.weights_), 0)

    def test_stores_correlated_patterns_that_hebb_cannot(self):
        # Cinco patrones de 100 neuronas que comparten 70 módulos, como los QR:
        # con Hebb ninguno es estable; con la pseudoinversa, los cinco.
        rng = np.random.default_rng(0)
        patterns = rng.choice([-1., 1.], size=(5, 100))
        patterns[:, :70] = patterns[0, :70]
        hebb = Hopfield(patterns)
        pseudoinverse = Hopfield(patterns, rule='pseudoinverse')
        self.assertFalse(any(hebb.recall(p).iterations == 0 for p in patterns))
        for pattern in patterns:
            result = pseudoinverse.recall(pattern)
            self.assertEqual((result.stop_reason, result.iterations), ('stable', 0))

    def test_rejects_unknown_rule(self):
        with self.assertRaises(ValueError):
            Hopfield(ORTHOGONAL, rule='unknown')


class OrthogonalityTests(unittest.TestCase):
    def test_overlaps_of_apunte_example(self):
        # ξ₂ = −ξ₁: solapamiento −1 entre ellos y 1 consigo mismos.
        network = Hopfield(np.vstack([XI_1, XI_2]))
        np.testing.assert_allclose(network.overlaps(), [[1, -1], [-1, 1]])

    def test_crosstalk_is_zero_with_orthogonal_patterns(self):
        network = Hopfield(ORTHOGONAL)
        np.testing.assert_allclose(network.overlaps(), np.eye(2))
        for index in range(len(ORTHOGONAL)):
            np.testing.assert_allclose(network.crosstalk(index), np.zeros(8))

    def test_crosstalk_of_correlated_patterns(self):
        # ξ₁ = (1, 1, 1, 1) y ξ₂ = (1, 1, 1, −1) coinciden en 3 de 4: m¹² = (3 − 1)/4 = 0,5.
        # C¹ = m²¹·ξ² = (0,5; 0,5; 0,5; −0,5): en la neurona 4 el crosstalk se opone a ξ¹.
        network = Hopfield(np.array([[1, 1, 1, 1], [1, 1, 1, -1]]))
        np.testing.assert_allclose(network.overlaps(), [[1, .5], [.5, 1]])
        np.testing.assert_allclose(network.crosstalk(0), [.5, .5, .5, -.5])
        np.testing.assert_allclose(network.crosstalk(1), [.5, .5, .5, .5])


if __name__ == '__main__':
    unittest.main()
