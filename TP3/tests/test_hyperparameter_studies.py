import numpy as np
import pandas as pd
import pytest

from experiments.config import load_config
from experiments.fraud_data import REQUIRED_COLUMNS, TEACHER_TARGET
from experiments.study_hyperparameters import STUDY_DEFAULTS, guided_logit_weights, run_studies


def test_guided_logit_weights_recovers_linear_logit_solution():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 6))
    expected_weights = np.array([0.2, -0.1, 0.3, 0.05, -0.4, 0.15])
    logits = X @ expected_weights + 0.25
    probabilities = 1 / (1 + np.exp(-logits))

    weights, bias = guided_logit_weights(X, probabilities, clip=1e-8)

    np.testing.assert_allclose(weights[:, 0], expected_weights, atol=1e-7)
    assert bias == pytest.approx(0.25, abs=1e-7)


def test_hyperparameter_studies_write_kfold_results(tmp_path):
    rng = np.random.default_rng(3)
    data = pd.DataFrame({column: rng.normal(size=20) for column in REQUIRED_COLUMNS})
    features = data[["amount_usd", "quantity_purchased", "session_duration_seconds",
                     "days_since_last_purchase", "account_age_days", "items_viewed_before_purchase"]]
    probabilities = 1 / (1 + np.exp(-features.to_numpy() @ np.array([0.2, -0.1, 0.3, 0.05, -0.4, 0.15])))
    data[TEACHER_TARGET] = probabilities
    dataset = tmp_path / "fraud.csv"
    data.to_csv(dataset, index=False)
    config = load_config(experiment_defaults=STUDY_DEFAULTS)
    config["dataset"] = str(dataset)
    config["training"] = {"epochs": 2, "batch_size": 5, "epsilon": None}
    config["cross_validation"] = {"folds": 2}
    config["studies"] = {"learning_rates": [0.01, 0.05], "guided_noise_std": 0.0, "logit_clip": 1e-6}
    config["output"] = {"directory": str(tmp_path / "reports")}

    results = run_studies(config)

    assert len(results["learning_rate"]) == 2
    assert len(results["initialization"]) == 4
    for filename in (
        "learning_rate_history.csv",
        "learning_rate_summary.csv",
        "learning_rate_curves.png",
        "initialization_by_fold.csv",
        "initialization_history.csv",
        "initialization_comparison.png",
        "initialization_curves.png",
        "report.md",
    ):
        assert (tmp_path / "reports" / filename).is_file()
