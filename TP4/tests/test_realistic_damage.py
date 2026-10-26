"""Generadores de daño realista: tipo de cambio, cobertura y reproducibilidad."""
import unittest

import numpy as np

from src.tp4.damage import BLACK, DAMAGES, WHITE, blob_mask, edge_fade, fade, photo, scratches, stain
from src.tp4.qr import SIDE


def qr_like(seed=0):
    return np.random.default_rng(seed).choice([BLACK, WHITE], size=SIDE * SIDE)


class RealisticDamageTests(unittest.TestCase):
    def test_fades_only_turn_black_into_white(self):
        pattern = qr_like()
        for damage in (fade, edge_fade):
            damaged = damage(pattern, .5, np.random.default_rng(1))
            changed = damaged != pattern
            self.assertTrue(changed.any())
            self.assertTrue((pattern[changed] == BLACK).all())
            self.assertTrue((damaged[changed] == WHITE).all())

    def test_fade_rate_matches_probability(self):
        pattern = np.full(SIDE * SIDE, BLACK)
        damaged = fade(pattern, .3, np.random.default_rng(2))
        self.assertAlmostEqual((damaged == WHITE).mean(), .3, delta=.04)

    def test_edge_fade_spares_the_opposite_edge(self):
        pattern = np.full(SIDE * SIDE, BLACK)
        for seed in range(8):
            grid = edge_fade(pattern, 1., np.random.default_rng(seed)).reshape(SIDE, SIDE)
            borders = [grid[0], grid[-1], grid[:, 0], grid[:, -1]]
            lost = [np.mean(b == WHITE) for b in borders]
            self.assertEqual(min(lost), 0)
            self.assertGreater(max(lost), .9)

    def test_blob_covers_at_least_the_target_without_overshooting_much(self):
        for coverage in (.02, .1, .3):
            mask = blob_mask(coverage, np.random.default_rng(3))
            self.assertGreaterEqual(mask.mean(), coverage - 1 / SIDE ** 2)
            self.assertLess(mask.mean(), coverage + .03)

    def test_stain_paints_a_single_color(self):
        pattern = qr_like()
        for color in (BLACK, WHITE):
            damaged = stain(pattern, .1, np.random.default_rng(4), color)
            self.assertTrue((damaged[damaged != pattern] == color).all())

    def test_scratches_only_remove_ink(self):
        pattern = qr_like()
        damaged = scratches(pattern, 4, np.random.default_rng(5))
        changed = damaged != pattern
        self.assertTrue(changed.any())
        self.assertTrue((damaged[changed] == WHITE).all())

    def test_mild_photo_keeps_almost_every_module(self):
        pattern = qr_like()
        damaged = photo(pattern, .1, np.random.default_rng(6))
        self.assertTrue(np.isin(damaged, (BLACK, WHITE)).all())
        self.assertLess((damaged != pattern).mean(), .1)

    def test_every_damage_is_reproducible_and_leaves_the_input_untouched(self):
        pattern = qr_like()
        original = pattern.copy()
        for name, (damage, levels) in DAMAGES.items():
            first = damage(pattern, levels[2], np.random.default_rng(7))
            second = damage(pattern, levels[2], np.random.default_rng(7))
            np.testing.assert_array_equal(first, second, err_msg=name)
            np.testing.assert_array_equal(pattern, original, err_msg=name)

    def test_rejects_patterns_that_are_not_a_qr(self):
        with self.assertRaises(ValueError):
            fade(np.ones(10), .1, np.random.default_rng(0))


if __name__ == '__main__':
    unittest.main()
