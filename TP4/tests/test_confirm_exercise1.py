"""Controles del protocolo, independientes del desempeño sobre el catálogo."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.confirm_exercise1 import align_regions, common_direction, jobs, run_som
from src.tp4.preprocessing import Preprocessor


class ProtocolTests(unittest.TestCase):
    def test_label_permutations_do_not_change_reference_memberships(self):
        reference = np.repeat(np.arange(5), [2, 3, 4, 5, 6])
        shuffled = np.array([3, 0, 4, 1, 2])[reference]
        np.testing.assert_array_equal(align_regions(shuffled, reference), reference)

    def test_direction_conversion_preserves_score_order(self):
        raw = np.array([[2., 10.], [4., 3.], [5., 17.]])
        fit_scale, ref_scale = np.array([2., 8.]), np.array([4., 3.])
        direction = np.array([.6, .8])
        converted = common_direction(direction, fit_scale, ref_scale)
        original = (raw-raw.mean(0))/fit_scale @ direction
        common = (raw-raw.mean(0))/ref_scale @ converted
        self.assertAlmostEqual(np.corrcoef(original, common)[0, 1], 1.)

    def test_excluded_rows_do_not_fit_subsample_scaler(self):
        frame = pd.DataFrame({'x': [0., 1., 2., 3., 10000.]})
        reference_pre = Preprocessor().fit(frame)
        # Interceptar el fit del SOM permite inspeccionar exactamente su entrada.
        with patch('analysis.confirm_exercise1.Kohonen') as som:
            som.return_value.fit.side_effect = RuntimeError('stop before training')
            with self.assertRaisesRegex(RuntimeError, 'stop before training'):
                run_som(dict(stage='subsamples', train=[0, 1, 2, 3], seed=200),
                        frame, reference_pre, reference_pre.transform(frame), np.zeros(5))
            observed = som.return_value.fit.call_args.args[0]
        expected = Preprocessor().fit_transform(frame.iloc[:4])
        np.testing.assert_allclose(observed, expected)
        np.testing.assert_allclose(observed.mean(0), 0., atol=1e-12)

    def test_planned_counts_and_sampling(self):
        tasks = jobs(pd.DataFrame({'followers': np.arange(50)}))
        self.assertEqual(len(tasks), 460)
        samples = [t for t in tasks if t['stage'] == 'subsamples']
        self.assertEqual(len(samples), 300)
        for sample in range(100):
            group = [t for t in samples if t['sample'] == sample]
            self.assertEqual([t['seed'] for t in group], [200, 201, 202])
            self.assertEqual(len(set(group[0]['train'])), 40)
            self.assertTrue(all(t['train'] == group[0]['train'] for t in group))


if __name__ == '__main__':
    unittest.main()
