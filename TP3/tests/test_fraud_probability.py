import json

import numpy as np
import pandas as pd
import pytest

from experiments.config import deep_merge, load_config
from experiments.fraud.data import REQUIRED_COLUMNS, TEACHER_TARGET
from experiments.fraud.probability import EXPERIMENT_DEFAULTS, predict_probability, run_experiment
from experiments.fraud.validation_analysis import DEFAULTS as ANALYSIS_DEFAULTS, run_analysis


def test_end_to_end_probability_experiment_and_saved_model(tmp_path):
    data = pd.DataFrame({column: np.arange(12, dtype=float) + 1 for column in REQUIRED_COLUMNS})
    data[TEACHER_TARGET] = np.linspace(0.1, 0.9, len(data))
    data["flagged_fraud"] = np.arange(len(data)) % 2
    csv_path = tmp_path / "fraud.csv"
    data.to_csv(csv_path, index=False)
    output = tmp_path / "result"
    config = load_config(experiment_defaults=EXPERIMENT_DEFAULTS)
    config["dataset"] = str(csv_path)
    config["output"] = {"directory": str(output)}
    config["cross_validation"] = {"folds": 3}
    config["training"] = {"epochs": 3, "batch_size": 4, "epsilon": None}

    summary = run_experiment(config)
    predictions = predict_probability(data.iloc[:2], output)

    assert summary["selected_activation"] == "sigmoid"
    assert len(summary["cross_validation"]["results"]) == 3
    assert predictions.shape == (2,)
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert "flagged_fraud" not in json.dumps(summary)
    assert "flagged_fraud" not in (output / "out_of_fold_predictions.csv").read_text()
    assert "flagged_fraud" not in (output / "test_predictions.csv").read_text()
    assert "flagged_fraud" not in (output / "model_metadata.json").read_text()


def test_test_rows_never_reach_cross_validation(tmp_path):
    data = synthetic_dataset(60)
    csv_path = tmp_path / "fraud.csv"
    data.to_csv(csv_path, index=False)
    config = small_config(csv_path, tmp_path / "result")

    summary = run_experiment(config)

    out_of_fold = pd.read_csv(tmp_path / "result" / "out_of_fold_predictions.csv")
    test = pd.read_csv(tmp_path / "result" / "test_predictions.csv")
    assert set(out_of_fold["row"]).isdisjoint(test["row"])
    assert len(out_of_fold) + len(test) == len(data)
    assert summary["split"]["test_samples"] == len(test) == 12


def test_validation_analysis_picks_threshold_on_validation_and_writes_figures(tmp_path):
    data = synthetic_dataset(60)
    csv_path = tmp_path / "fraud.csv"
    data.to_csv(csv_path, index=False)
    run_experiment(small_config(csv_path, tmp_path / "result"))
    analysis_config = deep_merge(ANALYSIS_DEFAULTS, {
        "dataset": str(csv_path),
        "results_directory": str(tmp_path / "result"),
        "output_directory": str(tmp_path / "analysis"),
        "threshold": {"grid_step": 0.01},
    })

    summary = run_analysis(analysis_config)

    assert 0 <= summary["recommended_threshold"] <= 1
    assert summary["test"]["at_threshold"]["recommended"]["threshold"] == summary["recommended_threshold"]
    written = {path.name for path in (tmp_path / "analysis").iterdir()}
    assert {"threshold_summary.json", "threshold_sweep.csv", "threshold_metrics.png",
            "roc_pr_curves.png", "confusion_matrices.png", "predicted_vs_target.png"} <= written


def synthetic_dataset(n_rows: int) -> pd.DataFrame:
    """Features that drive the target, and a label that is the target above 0.85 (like the real data)."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame({column: rng.normal(size=n_rows) for column in REQUIRED_COLUMNS})
    data[TEACHER_TARGET] = 1 / (1 + np.exp(-3 * data["amount_usd"]))
    data["flagged_fraud"] = (data[TEACHER_TARGET] > 0.85).astype(int)
    return data


def small_config(csv_path, output):
    config = load_config(experiment_defaults=EXPERIMENT_DEFAULTS)
    config["dataset"] = str(csv_path)
    config["output"] = {"directory": str(output)}
    config["cross_validation"] = {"folds": 3}
    config["training"] = {"epochs": 3, "batch_size": 8, "epsilon": None}
    return config


def test_generalization_study_can_use_another_activation(tmp_path):
    data = synthetic_dataset(60)
    csv_path = tmp_path / "fraud.csv"
    data.to_csv(csv_path, index=False)
    config = small_config(csv_path, tmp_path / "result")
    config["model"]["activation"] = "relu"
    config["learning_comparison"] = {"activations": ["identity", "sigmoid", "relu"]}

    summary = run_experiment(config)

    assert summary["selected_activation"] == "relu"
    assert set(summary["learning_comparison"]) == {"identity", "sigmoid", "relu"}
    assert json.loads((tmp_path / "result" / "model_metadata.json").read_text())["model"]["activation"] == "relu"
    assert "test_outside_0_1" in summary["final_model"]


def test_learning_comparison_must_include_the_linear_reference_and_the_selected_activation(tmp_path):
    config = small_config(tmp_path / "unused.csv", tmp_path / "result")
    config["model"]["activation"] = "relu"

    with pytest.raises(ValueError, match="must include 'identity' and 'relu'"):
        run_experiment(config)
