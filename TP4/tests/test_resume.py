"""Continuidad exacta, guardado atómico y recuperación del experimento."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import runner
from experiments.progress import epoch_callback
from experiments.session import ExperimentInterrupted
from src.tp4.checkpoint import load_checkpoint, save_checkpoint
from src.tp4.config import EarlyStopping, KohonenConfig, OjaConfig
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja
from src.tp4.preprocessing import Preprocessor


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(12).normal(size=(16, 4))

    def assert_same_model(self, expected, actual):
        np.testing.assert_array_equal(expected.weights_, actual.weights_)
        self.assertEqual(expected.history_, actual.history_)
        self.assertEqual(expected.epochs_, actual.epochs_)
        self.assertEqual(expected.stop_reason_, actual.stop_reason_)

    def test_exact_resume_with_both_models_topologies_and_stopping(self):
        for enabled in (False, True):
            stop = EarlyStopping(enabled, 2, 2, 5)
            cases = [(Oja, OjaConfig(epochs=9, early_stopping=stop))]
            cases += [(Kohonen, KohonenConfig(rows=2, cols=3, topology=name, epochs=9,
                                              early_stopping=stop)) for name in ('rectangular', 'hexagonal')]
            for model_type, config in cases:
                with self.subTest(model=model_type.__name__, config=config):
                    expected = model_type(config).fit(self.samples)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'checkpoint.npz'
                        def interrupt(state):
                            save_checkpoint(path, state)
                            if state.epoch == 3:
                                raise ExperimentInterrupted()
                        with self.assertRaises(ExperimentInterrupted):
                            model_type(config).fit(self.samples, on_epoch=interrupt)
                        self.assertEqual(len(list(Path(directory).glob('checkpoint*'))), 1)
                        restored = model_type(config).fit(self.samples, checkpoint=load_checkpoint(path))
                        self.assert_same_model(expected, restored)
                        if model_type is Kohonen:
                            self.assertEqual(expected.selected_epoch_, restored.selected_epoch_)

    def test_interruption_inside_epoch_uses_previous_complete_epoch(self):
        config = OjaConfig(epochs=6)
        model = Oja(config)
        update = model._update_sample
        count = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            def interrupt(sample, weights, learning_rate):
                nonlocal count
                count += 1
                if count == 2 * len(self.samples) + 3:
                    raise KeyboardInterrupt()
                update(sample, weights, learning_rate)
            with patch.object(model, '_update_sample', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    model.fit(self.samples, on_epoch=lambda state: save_checkpoint(path, state))
            self.assertEqual(load_checkpoint(path).epoch, 2)
            actual = Oja(config).fit(self.samples, checkpoint=load_checkpoint(path))
            self.assert_same_model(Oja(config).fit(self.samples), actual)

    def test_atomic_replacement_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            states = []
            Oja(OjaConfig(epochs=1)).fit(self.samples, on_epoch=states.append)
            save_checkpoint(path, states[0])
            original = path.read_bytes()
            with patch('src.tp4.checkpoint.np.savez_compressed', side_effect=OSError('disco')):
                with self.assertRaises(OSError):
                    save_checkpoint(path, states[0])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])
            with self.assertRaises(ValueError):
                Oja(OjaConfig(epochs=2)).fit(self.samples, checkpoint=load_checkpoint(path))
            with self.assertRaises(ValueError):
                Oja(OjaConfig(epochs=1)).fit(self.samples + 1, checkpoint=load_checkpoint(path))

    def test_terminal_checkpoint_does_not_train_again(self):
        for config in (OjaConfig(epochs=3), OjaConfig(epochs=9, early_stopping=EarlyStopping(True, 2, 1, 2))):
            states = []
            expected = Oja(config).fit(self.samples, on_epoch=states.append)
            with patch.object(Oja, '_train_epoch', side_effect=AssertionError('no debe entrenar')):
                actual = Oja(config).fit(self.samples, checkpoint=states[-1])
            self.assert_same_model(expected, actual)


class ExperimentResumeTests(unittest.TestCase):
    def test_resume_skips_completed_and_restores_pca_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data.csv'
            frame = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 3)), columns=list('abc'))
            frame.to_csv(data, index=False)
            config = {'algorithm':'oja', 'data':str(data), 'output':str(root/'results'),
                      'model':{'epochs':7}, 'preprocessing':{'pca_components':2}, 'seeds':[0, 1]}
            def controlled_callback(output, control):
                completed = epoch_callback(output, control)
                def completed_epoch(state):
                    if output.name == 'run_0001' and state.epoch == 2:
                        control.requested = True
                    completed(state)
                return completed_epoch
            with patch('src.tp4.export.plot_learning_curves'), patch('experiments.progress.plot_learning_curves'):
                with patch('experiments.runner.epoch_callback', side_effect=controlled_callback):
                    output = runner.run(config, experiment=True)
                self.assertEqual(json.loads((output/'experiment.json').read_text())['status'], 'paused')
                original = (output/'run_0000'/'model.npz').read_bytes()
                checkpoint = load_checkpoint(output/'run_0001'/'checkpoint.npz')
                self.assertEqual(checkpoint.epoch, 2)
                with patch('experiments.runner.execute', wraps=runner.execute) as execute:
                    runner.resume(output)
                    self.assertEqual(execute.call_count, 1)
                self.assertEqual((output/'run_0000'/'model.npz').read_bytes(), original)
                manifest = json.loads((output/'experiment.json').read_text())
                self.assertEqual((manifest['status'], manifest['pending'], manifest['failed']), ('completed', 0, 0))
                state = json.loads((output/'run_0001'/'preprocessing.json').read_text())
                x = Preprocessor.from_state(state).transform(pd.read_csv(data))
                expected = Oja(OjaConfig(epochs=7, seed=1)).fit(x)
                np.testing.assert_array_equal(np.load(output/'run_0001'/'model.npz')['weights'], expected.weights_)
                with patch('experiments.session.source_digest', return_value='changed'):
                    with self.assertRaises(ValueError):
                        runner.resume(output)
                data.write_text(data.read_text() + '\n')
                with self.assertRaises(ValueError):
                    runner.resume(output)

    def test_failed_runs_are_skipped_unless_retry_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'algorithm': 'oja', 'output': directory, 'model': {'epochs': 1}}
            with patch('experiments.runner.execute', side_effect=OSError('fallo temporal')):
                output = runner.run(config)
            with patch('experiments.runner.execute') as execute:
                runner.resume(output)
                execute.assert_not_called()
            with patch('src.tp4.export.plot_learning_curves'):
                runner.resume(output, retry_failed=True)
            manifest = json.loads((output / 'experiment.json').read_text())
            self.assertEqual(manifest['failed'], 0)
            self.assertFalse((output / 'run_0000' / 'error.json').exists())

    def test_pause_before_first_epoch_keeps_valid_empty_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'algorithm':'oja', 'output':directory, 'model':{'epochs':1}}
            with patch('experiments.runner.execute', side_effect=KeyboardInterrupt):
                output = runner.run(config)
            manifest = json.loads((output/'experiment.json').read_text())
            self.assertEqual((manifest['status'], manifest['pending']), ('paused', 1))
            self.assertEqual(len(pd.read_csv(output/'runs.csv')), 0)
            with patch('src.tp4.export.plot_learning_curves'):
                runner.resume(output)
            self.assertEqual(json.loads((output/'experiment.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
