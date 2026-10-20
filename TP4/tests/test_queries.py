import unittest

import numpy as np

from src.tp4.queries import Query, make_queries, noisy_query

PATTERNS = np.array([[1, 1, -1, -1, 1, -1],
                     [-1, 1, 1, -1, -1, 1]])


class NoisyQueryTests(unittest.TestCase):
    def test_no_noise_keeps_the_pattern_and_full_noise_inverts_it(self):
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(noisy_query(PATTERNS[0], 0, rng), PATTERNS[0])
        np.testing.assert_array_equal(noisy_query(PATTERNS[0], 1, rng), -PATTERNS[0])


class MakeQueriesTests(unittest.TestCase):
    def test_one_query_per_pattern_noise_level_and_seed(self):
        queries = make_queries(PATTERNS, noise_levels=(0, .5), seeds=range(3))
        self.assertEqual(len(queries), 2 * 2 * 3)
        self.assertIsInstance(queries[0], Query)
        self.assertEqual([(q.origin, q.noise, q.seed) for q in queries[:4]],
                         [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, .5, 0)])
        for query in queries:
            if query.noise == 0:
                np.testing.assert_array_equal(query.pattern, PATTERNS[query.origin])

    def test_queries_are_reproducible(self):
        a = make_queries(PATTERNS, noise_levels=(.3,), seeds=range(5))
        b = make_queries(PATTERNS, noise_levels=(.3,), seeds=range(5))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.pattern, y.pattern)


if __name__ == '__main__':
    unittest.main()
