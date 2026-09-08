"""Tests de la agregación estadística de resultados de experimentos."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simulation.analysis import GroupSummary, read_results, summarize_by


def _rows(pairs: list[tuple[str, float]]) -> list[dict[str, str]]:
    return [{"fitness.metric": label, "elapsed_seconds": str(value)} for label, value in pairs]


class GroupSummaryTests(unittest.TestCase):
    def test_rejects_a_group_without_measurements(self) -> None:
        with self.assertRaises(ValueError):
            GroupSummary("mse", ())

    def test_reports_mean_deviation_and_error(self) -> None:
        summary = GroupSummary("mse", (10.0, 12.0, 14.0))
        self.assertEqual(summary.runs, 3)
        self.assertAlmostEqual(summary.mean, 12.0)
        self.assertAlmostEqual(summary.standard_deviation, 2.0)
        self.assertAlmostEqual(summary.standard_error, 2.0 / 3 ** 0.5)

    def test_a_single_run_has_no_spread(self) -> None:
        summary = GroupSummary("mse", (7.5,))
        self.assertEqual(summary.standard_deviation, 0.0)
        self.assertEqual(summary.standard_error, 0.0)
        self.assertEqual(summary.confidence_margin(), 0.0)

    def test_confidence_margin_uses_the_student_t_quantile(self) -> None:
        summary = GroupSummary("mse", (10.0, 12.0, 14.0))
        # t(0.975, 2) = 4.302..., no 1.96: con tres corridas la normal subestima.
        self.assertAlmostEqual(summary.confidence_margin(0.95), 4.302653 * summary.standard_error, places=5)

    def test_a_wider_confidence_gives_a_wider_margin(self) -> None:
        summary = GroupSummary("mse", (10.0, 12.0, 14.0))
        self.assertGreater(summary.confidence_margin(0.99), summary.confidence_margin(0.95))

    def test_rejects_a_confidence_outside_the_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            GroupSummary("mse", (1.0, 2.0)).confidence_margin(1.0)


class SummarizeByTests(unittest.TestCase):
    def test_groups_rows_and_orders_them_by_mean(self) -> None:
        rows = _rows([("ssim", 30.0), ("mse", 10.0), ("ssim", 34.0), ("mse", 12.0)])
        summaries = summarize_by(rows, "fitness.metric", "elapsed_seconds")
        self.assertEqual([summary.label for summary in summaries], ["mse", "ssim"])
        self.assertEqual(summaries[0].values, (10.0, 12.0))
        self.assertEqual(summaries[1].values, (30.0, 34.0))

    def test_rejects_an_unknown_group_column(self) -> None:
        with self.assertRaises(ValueError) as error:
            summarize_by(_rows([("mse", 1.0)]), "no.existe", "elapsed_seconds")
        self.assertIn("no.existe", str(error.exception))

    def test_rejects_an_unknown_value_column(self) -> None:
        with self.assertRaises(ValueError):
            summarize_by(_rows([("mse", 1.0)]), "fitness.metric", "no_existe")

    def test_rejects_a_non_numeric_value(self) -> None:
        rows = [{"fitness.metric": "mse", "elapsed_seconds": "n/a"}]
        with self.assertRaises(ValueError):
            summarize_by(rows, "fitness.metric", "elapsed_seconds")

    def test_rejects_an_empty_result_set(self) -> None:
        with self.assertRaises(ValueError):
            summarize_by([], "fitness.metric", "elapsed_seconds")


class ReadResultsTests(unittest.TestCase):
    def test_reads_the_csv_written_by_the_experiment_matrix(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("fitness.metric,elapsed_seconds\nmse,10.5\nssim,30.25\n", encoding="utf-8")
            rows = read_results(path)
        self.assertEqual(rows, [
            {"fitness.metric": "mse", "elapsed_seconds": "10.5"},
            {"fitness.metric": "ssim", "elapsed_seconds": "30.25"},
        ])

    def test_fails_fast_when_the_file_is_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_results(Path("no") / "existe.csv")


if __name__ == "__main__":
    unittest.main()
