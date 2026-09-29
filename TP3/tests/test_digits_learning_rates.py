import copy

import pytest

from experiments.digits.learning_rates import STUDY_DEFAULTS, configurations


def test_comparison_changes_only_rate_and_destination_and_does_not_mutate_config():
    original = copy.deepcopy(STUDY_DEFAULTS)
    runs = configurations(original)
    assert original == STUDY_DEFAULTS
    assert [run["optimizer"]["lr"] for run in runs] == [0.001, 0.01, 0.1]
    assert len({run["output"]["directory"] for run in runs}) == 3
    normalized = []
    for run in runs:
        run.pop("output")
        run["optimizer"].pop("lr")
        normalized.append(run)
    assert normalized[0] == normalized[1] == normalized[2]


@pytest.mark.parametrize("rates", [[], [0], [-1], [float("nan")], [float("inf")], [0.1, 0.1]])
def test_invalid_rates_fail_before_training(rates):
    config = copy.deepcopy(STUDY_DEFAULTS)
    config["study"]["learning_rates"] = rates
    with pytest.raises(ValueError):
        configurations(config)
