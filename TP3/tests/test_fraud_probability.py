import json

import numpy as np
import pandas as pd

from experiments.config import load_config
from experiments.fraud_data import REQUIRED_COLUMNS, TEACHER_TARGET
from experiments.fraud_probability import EXPERIMENT_DEFAULTS, predict_probability, run_experiment


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
    assert "flagged_fraud" not in (output / "model_metadata.json").read_text()
