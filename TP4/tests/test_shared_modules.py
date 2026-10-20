import unittest

import numpy as np

from src.tp4.shared_modules import SharedModules

# Los módulos 0 y 3 son iguales en los tres patrones; 1, 2 y 4 difieren en alguno.
PATTERNS = np.array([[1, 1, -1, -1, 1],
                     [1, -1, -1, -1, -1],
                     [1, 1, 1, -1, -1]])


class SharedModulesTests(unittest.TestCase):
    def test_marks_the_modules_that_differ_in_some_pattern(self):
        shared = SharedModules(PATTERNS)
        np.testing.assert_array_equal(shared.differing, [False, True, True, False, True])
        self.assertEqual(shared.count, 3)

    def test_strip_keeps_only_the_differing_modules(self):
        shared = SharedModules(PATTERNS)
        np.testing.assert_array_equal(shared.strip(PATTERNS), [[1, -1, 1], [-1, -1, -1], [1, 1, -1]])

    def test_restore_undoes_strip_for_the_stored_patterns(self):
        shared = SharedModules(PATTERNS)
        for pattern in PATTERNS:
            np.testing.assert_array_equal(shared.restore(shared.strip(pattern)), pattern)

    def test_restore_puts_back_the_shared_values_of_a_damaged_query(self):
        shared = SharedModules(PATTERNS)
        damaged = -PATTERNS[0]
        np.testing.assert_array_equal(shared.restore(shared.strip(damaged)), [1, -1, 1, -1, -1])


if __name__ == '__main__':
    unittest.main()
