"""Recuperación explícita: reutilizar completas y continuar parciales sin repetir."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.recover_kohonen import prepare_recovery
from experiments.runner import resume, run
from src.tp4.checkpoint import load_checkpoint, save_checkpoint
from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen
from src.tp4.preprocessing import Preprocessor


class RecoveryTests(unittest.TestCase):
    def test_reuses_completed_and_legacy_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data.csv'
            pd.DataFrame(np.random.default_rng(8).normal(size=(14, 3)), columns=list('abc')).to_csv(data, index=False)
            config = {'algorithm': 'kohonen', 'data': str(data), 'output': str(root / 'source'),
                      'model': {'rows': 2, 'cols': 2, 'epochs': 3}, 'seeds': [0, 1, 2]}
            with patch('src.tp4.export.plot_grid'), patch('src.tp4.export.plot_learning_curves'):
                source = run(config, experiment=True)
                from experiments.extend_kohonen import extend
                complete = extend(source, 6, root / 'complete')
            # Se deja solo seed=0 terminada en la fuente de corridas completas.
            for index in (1, 2):
                (complete / f'run_{index:04d}' / 'record.json').unlink()
            # Seed=1 tiene un checkpoint parcial antiguo sin schedule_epochs.
            state_path = source / 'run_0001' / 'preprocessing.json'
            prep = Preprocessor.from_state(json.loads(state_path.read_text()))
            x = prep.transform(pd.read_csv(data))
            states = []
            Kohonen(KohonenConfig(rows=2, cols=2, epochs=3, seed=1)).fit(x, on_epoch=states.append)
            state = states[0]
            parameters = dict(state.parameters)
            parameters.pop('schedule_epochs')
            save_checkpoint(source / 'run_0001' / 'checkpoint.npz', replace(state, parameters=parameters))
            for index in (1, 2):
                (source / f'run_{index:04d}' / 'record.json').unlink()
            (source / 'run_0002' / 'checkpoint.npz').unlink()
            # Simula cambio de firma: resume normal debe rechazarlo.
            plan_path = source / 'plan.json'
            plan = json.loads(plan_path.read_text())
            plan['source_digest'] = 'legacy-code'
            plan_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                resume(source, dry_run=True)
            original = (complete / 'run_0000' / 'model.npz').read_bytes()
            output = prepare_recovery(source, root / 'recovered', 6, [complete])
            manifest = json.loads((output / 'experiment.json').read_text())
            self.assertEqual((manifest['runs'], manifest['pending']), (1, 2))
            self.assertEqual(load_checkpoint(output / 'run_0001' / 'checkpoint.npz').epoch, 1)
            resume(output, dry_run=True)
            from experiments import runner
            with patch('experiments.runner.execute', wraps=runner.execute) as execute:
                with patch('src.tp4.export.plot_grid'), patch('src.tp4.export.plot_learning_curves'):
                    resume(output)
                self.assertEqual(execute.call_count, 2)
            self.assertEqual((output / 'run_0000' / 'model.npz').read_bytes(), original)
            expected = Kohonen(KohonenConfig(rows=2, cols=2, epochs=6, schedule_epochs=3, seed=1)).fit(x)
            np.testing.assert_array_equal(expected.weights_, np.load(output / 'run_0001' / 'model.npz')['weights'])
            self.assertEqual(json.loads(plan_path.read_text())['source_digest'], 'legacy-code')
            data.write_text(data.read_text() + '\n')
            with self.assertRaises(ValueError):
                prepare_recovery(source, root / 'invalid', 6, [complete])
            self.assertFalse((root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
