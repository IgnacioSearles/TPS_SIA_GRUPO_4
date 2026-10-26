"""Contratos de vecindad uniforme y del experimento pareado."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.kohonen_uniform_experiments import config_for, fingerprint, verify_legacy, run_task
from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen
from analysis.som_campanas import MODEL


class UniformExperimentTests(unittest.TestCase):
    def test_radius_one_updates_cardinal_neighbors_equally_and_not_diagonal(self):
        model = Kohonen(KohonenConfig(rows=3, cols=3, neighborhood='uniform'))
        # La central gana; las ocho restantes tienen el mismo prototipo.
        model.weights_ = np.zeros((9, 2))
        model.weights_[4] = [.9, .9]
        before = model.weights_.copy()
        model._update_sample(np.ones(2), learning_rate=.2, radius=1.)
        for index in [1,3,5,7]:
            np.testing.assert_array_equal(model.weights_[index], [.2, .2])
        for index in [0,2,6,8]:
            np.testing.assert_array_equal(model.weights_[index], before[index])
        np.testing.assert_allclose(model.weights_[4], [.92, .92])

    def test_radius_half_moves_only_winner(self):
        model = Kohonen(KohonenConfig(rows=2, cols=2, neighborhood='uniform'))
        model.weights_ = np.array([[.9, .9], [0., 0.], [0., 0.], [0., 0.]])
        before = model.weights_.copy()
        model._update_sample(np.ones(2), learning_rate=.2, radius=.5)
        np.testing.assert_array_equal(model.weights_[1:], before[1:])
        self.assertFalse(np.array_equal(model.weights_[0],before[0]))

    def test_variants_preserve_budget_initialization_and_learning_schedule(self):
        original = asdict(MODEL)
        for name,end in [('uniform_matched',.5),('uniform_neighbors',1.)]:
            configured = asdict(config_for(name))
            self.assertEqual(configured.pop('neighborhood'),'uniform')
            self.assertEqual(configured.pop('radius_end'),end)
            expected = {k:v for k,v in original.items() if k not in ['neighborhood','radius_end']}
            self.assertEqual(configured,expected)
        model = Kohonen(config_for('uniform_neighbors'))
        self.assertEqual(model._step_parameters(1000,49,50),(.01,1.))

    def test_topology_variants_keep_same_seed_and_capacity(self):
        for candidate in ['rect4x4','hex4x4','hex2x8']:
            cfg=config_for('uniform_neighbors',seed=123,candidate=candidate)
            self.assertEqual(cfg.seed,123)
            self.assertEqual(cfg.rows*cfg.cols,16)
            self.assertEqual(cfg.radius_start,2.)

    def test_fingerprint_detects_content_changes_and_new_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'index.html').write_text('antes')
            original=fingerprint(path)
            (path/'index.html').write_text('despues')
            self.assertNotEqual(fingerprint(path),original)
            (path/'notes.md').write_text('nuevo')
            self.assertIn('notes.md',fingerprint(path))

    def test_reuse_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();(root/'data/creadores.csv').write_text('original')
            out=root/'old';out.mkdir()
            data_hash=hashlib.sha256((root/'data/creadores.csv').read_bytes()).hexdigest()
            protocol=dict(smoke=False,data_sha256=data_hash,sources={},tasks=[dict(seed=1)])
            (out/'protocol.json').write_text(json.dumps(protocol))
            (out/'runs.jsonl').write_text(json.dumps(dict(task=dict(seed=1),status='ok'))+'\n')
            verify_legacy(out,root)
            (root/'data/creadores.csv').write_text('cambio')
            with self.assertRaisesRegex(ValueError,'catálogo'):
                verify_legacy(out,root)

    def test_worker_scaler_ignores_excluded_extreme(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.csv'
            pd.DataFrame({'x':[0.,1.,2.,3.,1000000.]}).to_csv(path,index=False)
            task=dict(stage='subsamples',sample=0,seed=200,train=[0,1,2,3])
            with patch('analysis.kohonen_uniform_experiments.Kohonen') as model:
                model.return_value.fit.side_effect=RuntimeError('stop before fitting')
                with self.assertRaisesRegex(RuntimeError,'stop before fitting'):
                    run_task(task,path,{})
                fitted=model.return_value.fit.call_args.args[0]
            np.testing.assert_allclose(fitted.mean(axis=0),0.,atol=1e-12)
            np.testing.assert_allclose(fitted.std(axis=0,ddof=1),1.,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
