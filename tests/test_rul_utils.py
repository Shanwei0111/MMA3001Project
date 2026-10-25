"""Tests for training RUL labels and input validation."""

import numpy as np
import pandas as pd
import pytest

from rul_project.rul_utils import (
    add_training_rul,
    split_by_engine,
    evaluate_rul,
)


def test_rul_matches_hand_calculation():
    """Calculate labels correctly even when rows are unsorted."""
    data = pd.DataFrame(
        {
            "engine_id": [2, 1, 2, 1],
            "cycle": [2, 1, 1, 3],
            "sensor_4": [1400.0, 1401.0, 1402.0, 1403.0],
        },
        index=[40, 10, 30, 20],
    )

    result = add_training_rul(data)

    # Engine 1 fails at cycle 3; engine 2 fails at cycle 2.
    assert result["RUL"].tolist() == [0, 2, 1, 0]

    # Preserve row alignment and unrelated sensor measurements.
    assert result.index.tolist() == [40, 10, 30, 20]
    pd.testing.assert_series_equal(
        result["sensor_4"],
        data["sensor_4"],
    )


def test_input_data_is_not_modified():
    """Return a new table without changing the caller's data."""
    data = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 2],
        }
    )
    original = data.copy(deep=True)

    add_training_rul(data)

    pd.testing.assert_frame_equal(data, original)


def test_empty_data_is_rejected():
    data = pd.DataFrame(columns=["engine_id", "cycle"])

    with pytest.raises(ValueError, match="must not be empty"):
        add_training_rul(data)


@pytest.mark.parametrize("column", ["engine_id", "cycle"])
def test_missing_column_is_rejected(column):
    data = pd.DataFrame(
        {
            "engine_id": [1],
            "cycle": [1],
        }
    ).drop(columns=[column])

    with pytest.raises(ValueError, match="Missing required columns"):
        add_training_rul(data)


def test_duplicate_engine_cycle_is_rejected():
    data = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [2, 2],
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        add_training_rul(data)


@pytest.mark.parametrize("column", ["engine_id", "cycle"])
@pytest.mark.parametrize("invalid_value", [0, -1, 1.5, np.nan, np.inf])
def test_invalid_key_values_are_rejected(column, invalid_value):
    data = pd.DataFrame(
        {
            "engine_id": [1],
            "cycle": [1],
        }
    )
    data[column] = invalid_value

    with pytest.raises(ValueError):
        add_training_rul(data)


def test_engine_split_preserves_complete_histories():
    """Keep each engine in one subset and retain every record."""
    data = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "cycle": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )

    train, validation = split_by_engine(
        data,
        n_train=2,
        random_state=42,
    )

    train_ids = set(train["engine_id"])
    validation_ids = set(validation["engine_id"])

    assert len(train_ids) == 2
    assert len(validation_ids) == 2
    assert train_ids.isdisjoint(validation_ids)

    # Recombine both subsets and compare against the original.
    combined = pd.concat([train, validation]).sort_index()
    pd.testing.assert_frame_equal(combined, data)


def test_engine_split_is_reproducible():
    """The same input and seed must produce the same split."""
    data = pd.DataFrame(
        {
            "engine_id": [1, 2, 3, 4],
            "cycle": [1, 1, 1, 1],
        }
    )

    train_a, validation_a = split_by_engine(
        data, n_train=2, random_state=42
    )
    train_b, validation_b = split_by_engine(
        data, n_train=2, random_state=42
    )

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(validation_a, validation_b)


@pytest.mark.parametrize("n_train", [0, -1, 4, 5, 1.5, True])
def test_invalid_split_size_is_rejected(n_train):
    """Reject sizes that are invalid or leave an empty subset."""
    data = pd.DataFrame(
        {
            "engine_id": [1, 2, 3, 4],
            "cycle": [1, 1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="n_train"):
        split_by_engine(data, n_train=n_train)


def test_missing_engine_id_column_is_rejected():
    data = pd.DataFrame({"cycle": [1, 2]})

    with pytest.raises(ValueError, match="Missing required column"):
        split_by_engine(data, n_train=1)


def test_missing_engine_id_value_is_rejected():
    data = pd.DataFrame(
        {
            "engine_id": [1, np.nan, 2],
            "cycle": [1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="must not be missing"):
        split_by_engine(data, n_train=1)

def test_perfect_predictions_have_zero_error():
    scores = evaluate_rul(
        [0, 10, 100],
        [0, 10, 100],
    )

    for value in scores.values():
        assert value == pytest.approx(0.0)


def test_metrics_match_hand_calculation():
    # Signed errors: -3, 0, +4.
    scores = evaluate_rul(
        [10, 20, 30],
        [7, 20, 34],
    )

    assert scores["mae"] == pytest.approx(7 / 3)
    assert scores["rmse"] == pytest.approx((25 / 3) ** 0.5)
    assert scores["bias"] == pytest.approx(1 / 3)
    assert scores["overprediction_pct"] == pytest.approx(100 / 3)


def test_overprediction_receives_larger_penalty():
    # Equal error magnitudes, opposite directions.
    early = evaluate_rul([20], [10])
    late = evaluate_rul([20], [30])

    assert early["nasa_score"] == pytest.approx(
        np.exp(10 / 13) - 1
    )
    assert late["nasa_score"] == pytest.approx(
        np.exp(10 / 10) - 1
    )
    assert late["nasa_score"] > early["nasa_score"]


@pytest.mark.parametrize(
    "actual, predicted",
    [
        ([], []),
        ([10, 20], [10]),
        ([[10]], [[10]]),
        ([np.nan], [10]),
        ([10], [np.inf]),
        ([-1], [10]),
    ],
)
def test_invalid_metric_inputs_are_rejected(actual, predicted):
    with pytest.raises(ValueError):
        evaluate_rul(actual, predicted)


def test_extreme_score_overflow_is_reported():
    # Reject an unrepresentable score instead of silently returning inf.
    with pytest.raises(FloatingPointError):
        evaluate_rul([0], [10000])