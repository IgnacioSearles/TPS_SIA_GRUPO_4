import numpy as np
import pytest

from training.metrics import (
    average_precision,
    binary_metrics,
    classification_metrics,
    f_beta,
    precision_recall_curve,
    probability_metrics,
    roc_auc,
    threshold_sweep,
)


def test_binary_classification_metrics_with_bipolar_labels():
    metrics = classification_metrics(
        np.array([[-1.0], [-1.0], [1.0], [1.0]]),
        np.array([[-0.9], [0.2], [0.4], [0.8]]),
    )

    assert metrics["labels"] == [-1.0, 1.0]
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75


def test_multiclass_metrics_with_one_hot_targets():
    metrics = classification_metrics(
        np.eye(3)[[0, 1, 2]],
        np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]]),
    )

    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert metrics["accuracy"] == 2 / 3


def test_probability_metrics_compare_continuous_values_without_a_threshold():
    metrics = probability_metrics(
        np.array([[0.2], [0.8]]),
        np.array([[0.3], [0.6]]),
    )

    assert np.isclose(metrics["mae"], 0.15)
    assert np.isclose(metrics["rmse"], np.sqrt(0.025))
    assert "confusion_matrix" not in metrics

LABELS = np.array([0, 0, 1, 1, 0, 1])
SCORES = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9])


def test_binary_metrics_counts_scores_at_or_above_threshold_as_positive():
    metrics = binary_metrics(LABELS, SCORES, threshold=0.4)

    # Flagged (>= 0.4): 0.4 (neg), 0.8 (pos), 0.7 (neg), 0.9 (pos).
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (2, 2, 1, 1)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert metrics["flagged_rate"] == pytest.approx(4 / 6)


def test_sweep_matches_single_threshold_metrics():
    thresholds = np.array([0.0, 0.35, 0.75, 1.0])
    sweep = threshold_sweep(LABELS, SCORES, thresholds)

    for index, threshold in enumerate(thresholds):
        single = binary_metrics(LABELS, SCORES, threshold)
        assert sweep["tp"][index] == single["tp"]
        assert sweep["fp"][index] == single["fp"]


def test_precision_is_zero_when_nothing_is_flagged():
    metrics = binary_metrics(LABELS, SCORES, threshold=0.95)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_f_beta_weights_recall_more_when_beta_is_above_one():
    precision, recall = np.array([0.5]), np.array([1.0])

    assert f_beta(precision, recall, beta=2.0)[0] > f_beta(precision, recall, beta=1.0)[0]


def test_roc_auc_is_one_for_perfect_ranking_and_zero_for_reversed():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert roc_auc(labels, -scores) == pytest.approx(0.0)


def test_roc_auc_equals_probability_positive_outscores_negative():
    # Pairs (pos, neg): 0.35 beats 0.1 only; 0.8 beats 0.1, 0.4, 0.7; 0.9 beats all three -> 7 of 9.
    assert roc_auc(LABELS, SCORES) == pytest.approx(7 / 9)


def test_average_precision_on_hand_computed_example():
    # Ranking by score: 0.9 (pos), 0.8 (pos), 0.7 (neg), 0.4 (neg), 0.35 (pos), 0.1 (neg).
    # Precision at each positive: 1/1, 2/2, 3/5 -> AP = (1 + 1 + 0.6) / 3.
    assert average_precision(LABELS, SCORES) == pytest.approx((1 + 1 + 0.6) / 3)


def test_precision_recall_curve_groups_tied_scores():
    precision, recall = precision_recall_curve(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.2]))

    np.testing.assert_allclose(precision, [0.5, 2 / 3])
    np.testing.assert_allclose(recall, [0.5, 1.0])


def test_metrics_reject_single_class_labels():
    with pytest.raises(ValueError, match="both classes"):
        roc_auc(np.zeros(3), np.array([0.1, 0.2, 0.3]))
