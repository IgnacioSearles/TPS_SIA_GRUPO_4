"""Una red de Hopfield por zona: partición de los módulos, zonas fijas y coherencia entre zonas."""
import unittest

import numpy as np

from src.tp4.zoned_hopfield import ZonedHopfield


def _patterns(seed, count=2, side=4):
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=(count, side * side))


class ZonePartitionTests(unittest.TestCase):
    def test_zones_cover_every_module_once_even_with_smaller_border_zones(self):
        zoned = ZonedHopfield(_patterns(0, side=5), side=5, zone=2)
        modules = np.concatenate([zone.modules for zone in zoned.zones_])
        self.assertEqual(sorted(modules), list(range(25)))
        self.assertEqual(len(zoned.zones_), 9)

    def test_rejects_patterns_of_another_size(self):
        with self.assertRaises(ValueError):
            ZonedHopfield(_patterns(0, side=4), side=5, zone=2)


class ZoneRecallTests(unittest.TestCase):
    def test_zone_identical_in_every_pattern_has_no_network_and_is_restored(self):
        patterns = _patterns(1)
        patterns[1, :8] = patterns[0, :8]  # las dos filas de arriba son iguales en los dos patrones
        zoned = ZonedHopfield(patterns, side=4, zone=2, only_differing=True)
        fixed = [zone for zone in zoned.zones_ if zone.network is None]
        self.assertEqual(len(fixed), 2)
        query = patterns[0].copy()
        query[:8] *= -1
        np.testing.assert_array_equal(zoned.recall(query).state[:8], patterns[0, :8])

    def test_recovers_a_single_stored_pattern_exactly(self):
        pattern = _patterns(2, count=1)
        zoned = ZonedHopfield(np.vstack([pattern, -pattern]), side=4, zone=2)
        result = zoned.recall(pattern[0])
        self.assertEqual(result.match, 0)
        self.assertTrue(result.consistent)

    def test_zones_that_end_on_different_patterns_are_not_consistent(self):
        first, second = _patterns(3, count=2)
        zoned = ZonedHopfield(np.vstack([first, second]), side=4, zone=2)
        # Arriba a la izquierda se parece al primero; en el resto, al segundo.
        query = second.copy()
        top_left = zoned.zones_[0].modules
        query[top_left] = first[top_left]
        result = zoned.recall(query)
        self.assertIsNone(result.match)
        self.assertFalse(result.consistent)


if __name__ == '__main__':
    unittest.main()
