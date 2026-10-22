"""Integridad de la evidencia publicada y comparación reproducible sin reajustes."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.compare_hopfield_unlearning import compare
from analysis.hopfield_qr import ROOT
from analysis.hopfield_qr_unlearning import evaluate
from analysis.publish_hopfield_unlearning import portable
from src.tp4.hopfield import Hopfield
from src.tp4.queries import make_queries
from src.tp4.shared_modules import SharedModules
from src.tp4.unlearning import UnlearningTrainer


class PublishedEvidenceTests(unittest.TestCase):
    def test_shipped_metrics_match_hashes_and_contain_no_network_files(self):
        directory = ROOT / 'reports/ejercicio2/desaprendizaje/evidencia'
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertLess(manifest['bytes'], 1_500_000)
        self.assertEqual(manifest['new_evaluations_excluding_smoke'], 24800)
        for item in manifest['files']:
            path = directory / item['path']
            self.assertTrue(path.resolve().is_relative_to(directory))
            self.assertNotIn(path.suffix, {'.npy', '.npz', '.pkl', '.pt', '.pth'})
            content = path.read_bytes()
            self.assertEqual(len(content), item['bytes'])
            self.assertLess(len(content), 250_001)
            self.assertEqual(sha256(content).hexdigest(), item['published_sha256'])
            if item['encoding'] == 'gzip':
                self.assertEqual(sha256(gzip.decompress(content)).hexdigest(), item['source_sha256'])

    def test_portable_metadata_removes_only_project_prefix(self):
        source = {'confirmation': str(ROOT / 'configs/frozen.json'),
                  'other': ['/another/project.json', {'pilot': str(ROOT / 'results/pilot')}]}
        self.assertEqual(portable(source), {'confirmation': 'configs/frozen.json',
                         'other': ['/another/project.json', {'pilot': 'results/pilot'}]})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.patterns = np.array([[1, -1, 1, 1, 1], [1, -1, -1, 1, 1],
                                  [1, -1, 1, -1, 1], [1, -1, 1, 1, -1],
                                  [1, -1, -1, -1, 1]], dtype=float)

    def write_confirmation(self, directory):
        config = {'modes': ['dream'], 'training_seeds': [0], 'noise_levels': [0, .2],
                  'query_seeds': [100], 'max_iterations': 5, 'active_neurons': 3,
                  'patterns_sha256': sha256(self.patterns.tobytes()).hexdigest()}
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'COMPLETED').write_text('ok')
        shared = SharedModules(self.patterns)
        active = shared.strip(self.patterns)
        queries = make_queries(self.patterns, config['noise_levels'], config['query_seeds'])
        rows = evaluate(Hopfield(active), self.patterns, queries, UnlearningTrainer(active).mixture,
                        {'method': 'unlearning', 'training_seed': 0}, 5, shared)
        pd.DataFrame(rows).to_csv(directory / 'resultados.csv', index=False)

    def test_comparison_reuses_candidate_predictions_and_checks_paired_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'confirmation'
            source.mkdir()
            self.write_confirmation(source)
            before = (source / 'resultados.csv').read_bytes()
            output = directory / 'comparison'
            with patch('analysis.compare_hopfield_unlearning.load_patterns', return_value=self.patterns):
                compare(source, output)
            table = pd.read_csv(output / 'resultados.csv')
            self.assertEqual(len(table), 30)
            self.assertEqual(set(table.method),
                             {'unlearning', 'pseudoinverse_full', 'pseudoinverse_stripped'})
            self.assertTrue((output / 'COMPLETED').exists())
            self.assertEqual(before, (source / 'resultados.csv').read_bytes())

    def test_comparison_rejects_missing_query_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_confirmation(directory)
            table = pd.read_csv(directory / 'resultados.csv').iloc[:-1]
            table.to_csv(directory / 'resultados.csv', index=False)
            output = directory / 'comparison'
            with patch('analysis.compare_hopfield_unlearning.load_patterns', return_value=self.patterns):
                with self.assertRaisesRegex(ValueError, 'consultas del candidato'):
                    compare(directory, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
