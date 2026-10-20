"""La extensión conserva el prefijo y es reanudable con el runner habitual."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.extend_kohonen import extend
from experiments.runner import run
from src.tp4.checkpoint import load_checkpoint
from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen
from src.tp4.preprocessing import Preprocessor


class ExtensionTests(unittest.TestCase):
    def test_extends_without_replaying_and_preserves_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'data.csv'
            frame = pd.DataFrame(np.random.default_rng(4).normal(size=(12, 3)), columns=list('abc'))
            frame.to_csv(path, index=False)
            config = {'algorithm': 'kohonen', 'data': str(path), 'output': str(root / 'initial'),
                      'model': {'rows': 2, 'cols': 3, 'epochs': 4}}
            with patch('src.tp4.export.plot_grid'), patch('src.tp4.export.plot_learning_curves'):
                initial = run(config)
                before = load_checkpoint(initial / 'run_0000' / 'checkpoint.npz')
                extended = extend(initial, 9, root / 'extension')
            checkpoint = load_checkpoint(extended / 'run_0000' / 'checkpoint.npz')
            self.assertEqual(checkpoint.history[:4], before.history)
            self.assertEqual(checkpoint.epoch, 9)
            self.assertEqual(checkpoint.parameters['schedule_epochs'], 4)
            self.assertEqual(checkpoint.history[-1].learning_rate, .01)
            self.assertEqual(checkpoint.history[-1].radius, .5)
            preprocessing = json.loads((initial / 'run_0000' / 'preprocessing.json').read_text())
            samples = Preprocessor.from_state(preprocessing).transform(pd.read_csv(path))
            expected = Kohonen(KohonenConfig(rows=2, cols=3, epochs=9, schedule_epochs=4)).fit(samples)
            np.testing.assert_array_equal(checkpoint.weights, expected.weights_)
            origin = json.loads((extended / 'run_0000' / 'continuation.json').read_text())
            self.assertEqual(origin['source_epoch'], 4)
            with self.assertRaises(ValueError):
                extend(initial, 3, root / 'invalid')


if __name__ == '__main__':
    unittest.main()
