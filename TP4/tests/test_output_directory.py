"""Nombres legibles y únicos para los directorios de resultados."""
from datetime import datetime
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from experiments import runner


class OutputDirectoryTests(unittest.TestCase):
    def test_name_combines_experiment_and_local_time(self):
        with tempfile.TemporaryDirectory() as root:
            directory = runner.create_output_directory(root, 'kohonen_hexagonal')
            self.assertTrue(directory.is_dir())
            self.assertRegex(directory.name, r'^kohonen_hexagonal_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')

    def test_same_second_gets_numeric_suffix(self):
        frozen = datetime(2026, 10, 21, 0, 21, 48)
        with tempfile.TemporaryDirectory() as root, patch('experiments.runner.datetime') as clock:
            clock.now.return_value = frozen
            names = [runner.create_output_directory(root, 'oja').name for _ in range(3)]
        self.assertEqual(names, ['oja_2026-10-21_00-21-48', 'oja_2026-10-21_00-21-48_2', 'oja_2026-10-21_00-21-48_3'])

    def test_rejects_names_that_escape_the_root(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('', '../oja', 'a/b', 'a\\b'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.create_output_directory(root, name)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_run_defaults_to_algorithm_name(self):
        with tempfile.TemporaryDirectory() as root:
            config = {'algorithm': 'oja', 'output': root, 'model': {'epochs': 1}}
            with patch('experiments.runner.execute', side_effect=OSError('sin datos')):
                output = runner.run(config)
            self.assertTrue(re.match(r'^oja_\d{4}-', output.name))


if __name__ == '__main__':
    unittest.main()
