"""Contratos de PCA compartido y serialización de resultados tipados."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.representation import analyze_pca
from experiments.reporting import summarize
from experiments.results import ExecutionResult, RunFailure, RunRecord
from src.tp4.config import KohonenConfig, OjaConfig
from src.tp4.export import export_model
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja
from src.tp4.pca import compute_pca
from src.tp4.preprocessing import PreprocessingConfig, Preprocessor
from src.tp4.results import KohonenEpoch, KohonenMetrics, OjaEpoch, OjaMetrics, RunMetrics


class PCATests(unittest.TestCase):
    def test_reconstruction_and_covariance(self):
        samples = np.random.default_rng(4).normal(size=(30, 4)) + [10, 3, -4, 2]
        result = compute_pca(samples)
        np.testing.assert_allclose(result.scores @ result.components + result.mean, samples)
        np.testing.assert_allclose(result.components @ result.components.T, np.eye(4), atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(np.cov(samples, rowvar=False))[::-1]
        np.testing.assert_allclose(result.eigenvalues, eigenvalues)
        self.assertEqual(result.components_for_variance(1), 4)
        dominant = result.components[np.arange(4), np.abs(result.components).argmax(axis=1)]
        self.assertTrue((dominant > 0).all())

    def test_rank_deficiency_and_wide_matrix(self):
        samples = np.array([[1., 2., 3., 6.], [2., 4., 6., 12.], [3., 6., 9., 18.]])
        result = compute_pca(samples)
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.components_for_variance(.9), 1)
        self.assertEqual(result.components.shape, (3, 4))
        np.testing.assert_allclose(result.scores @ result.components + result.mean, samples)
        centered = samples - samples.mean(axis=0)
        np.testing.assert_allclose(compute_pca(centered, center=False).eigenvalues, result.eigenvalues)

    def test_invalid_inputs(self):
        for values in ([[1., 2.]], [[1., 2.], [1., 2.]], [[np.nan], [2.]]):
            with self.assertRaises(ValueError):
                compute_pca(values)
        with self.assertRaises(ValueError):
            compute_pca([[1.], [2.]]).components_for_variance(0)

    def test_analysis_and_preprocessing_share_representation(self):
        frame = pd.DataFrame(np.random.default_rng(3).normal(size=(20, 4)), columns=list('abcd'))
        analysis = analyze_pca(frame, 'test')
        prep = Preprocessor(PreprocessingConfig(exclude=[], pca_components=2))
        transformed = prep.fit_transform(frame)
        np.testing.assert_allclose(prep.components_, analysis.pca.components[:2])
        np.testing.assert_allclose(transformed, analysis.pca.scores[:, :2], atol=1e-14)
        np.testing.assert_allclose(analysis.pca.correlation_loadings[:, :2],
                                   np.corrcoef(analysis.standardized.T, transformed.T)[:4, 4:])


class ResultTests(unittest.TestCase):
    def test_run_record_invariants_and_flat_serialization(self):
        metrics = RunMetrics(KohonenMetrics(2., .1), 4, 40, 'max_epochs', 3, 4, 3)
        execution = ExecutionResult({'algorithm': 'kohonen'}, metrics)
        record = RunRecord('run_0', 'group', 0, metrics=execution.metrics)
        values = record.to_dict()
        self.assertEqual(values['quantization_error'], 2.)
        self.assertEqual(values['selected_epoch'], 4)
        self.assertNotIn('evaluation', values)
        json.dumps(values, allow_nan=False)
        failure = RunFailure('ValueError', 'datos inválidos', 'traceback')
        failed = RunRecord('run_1', 'group', 1, failure=failure)
        self.assertEqual(failed.status, 'failed')
        with self.assertRaises(ValueError):
            RunRecord('run_2', 'group', 2)
        with self.assertRaises(ValueError):
            RunRecord('run_2', 'group', 2, metrics, failure)
        summary = summarize(pd.DataFrame([failed.to_dict()]))
        self.assertEqual(summary.loc['group', 'succeeded'], 0)

    def test_typed_histories_and_exported_schemas(self):
        frame = pd.DataFrame(np.random.default_rng(1).normal(size=(12, 3)), columns=list('abc'))
        prep = Preprocessor(PreprocessingConfig(exclude=[]))
        samples = prep.fit_transform(frame)
        cases = [
            ('kohonen', Kohonen(KohonenConfig(rows=2, cols=3, epochs=3)), KohonenEpoch, KohonenMetrics),
            ('oja', Oja(OjaConfig(epochs=3)), OjaEpoch, OjaMetrics),
        ]
        for algorithm, model, epoch_type, metrics_type in cases:
            model.fit(samples)
            self.assertIsInstance(model.history_[0], epoch_type)
            self.assertIsInstance(model.evaluate(samples), metrics_type)
            self.assertEqual(model.metrics(samples), asdict(model.evaluate(samples)))
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with patch('src.tp4.export.plot_grid'):
                    export_model(output, model, prep, frame, samples, algorithm)
                history = pd.read_csv(output / 'history.csv')
                self.assertEqual(list(history.columns), list(asdict(model.history_[0])))
                np.testing.assert_allclose(history.to_numpy(), pd.DataFrame([asdict(r) for r in model.history_]))
                np.testing.assert_array_equal(np.load(output / 'model.npz')['weights'], model.weights_)
                if algorithm == 'oja':
                    reference = compute_pca(samples - model.mean_, center=False)
                    comparison = json.loads((output / 'comparison.json').read_text())
                    np.testing.assert_array_equal(comparison['pc1_weights'], reference.components[0])


if __name__ == '__main__':
    unittest.main()
