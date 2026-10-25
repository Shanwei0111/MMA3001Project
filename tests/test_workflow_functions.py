"""Check endpoint alignment, paired uncertainty and the refactored public API."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from rul_project.analyse_uncertainty import paired_bootstrap
from rul_project.benchmark_performance import operation_estimate
from rul_project.model_configuration import MODEL_NAMES, feature_columns, make_model
from rul_project.prepare_dataset import load_test_endpoints
from rul_project.run_project import main


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_all_models_share_the_documented_feature_order(name):
    """Factories must accept every documented name and preserve the input schema.

    Args:
        name: Documented model name supplied by the parameterized case.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    expected = (
        ["cycle"]
        if name == "Age-only"
        else [
            "cycle",
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
            "sensor_9",
            "sensor_11",
        ]
    )
    assert feature_columns(name) == expected
    assert hasattr(make_model(name), "fit")


def test_misspelled_model_name_is_rejected():
    """A spelling mistake must not silently select a different estimator.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("RandomForest")


def paired_example() -> pd.DataFrame:
    """Provide identical known-error predictions for three models.

    Returns:
        pandas.DataFrame: Two engines per model, with signed errors +2 and -2.
    """
    return pd.DataFrame(
        [
            {
                "model": name,
                "engine_id": engine,
                "RUL": actual,
                "clipped_predicted_RUL": predicted,
            }
            for name in (
                "Multi-sensor linear",
                "Random forest",
                "Histogram gradient boosting",
            )
            for engine, actual, predicted in [(1, 10, 12), (2, 20, 18)]
        ]
    )


def test_paired_bootstrap_matches_known_constant_errors():
    """Constant absolute errors give exact MAE/RMSE bounds and zero paired differences.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    result = paired_bootstrap(paired_example(), repetitions=40)
    for metric in ("mae", "rmse"):
        np.testing.assert_allclose(
            result.iloc[:2][[metric, metric + "_lower", metric + "_upper"]], 2
        )
        np.testing.assert_allclose(
            result.iloc[2:][[metric, metric + "_lower", metric + "_upper"]], 0
        )


@pytest.mark.parametrize("fault", ["duplicate", "engine", "label"])
def test_paired_bootstrap_rejects_misaligned_engines(fault):
    """An invalid pairing must fail rather than produce a convincing interval.

    Args:
        fault: Specified duplicate-ID, engine-alignment or label-alignment defect.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    predictions = paired_example()
    if fault == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif fault == "engine":
        predictions.loc[2, "engine_id"] = 3
    else:
        predictions.loc[2, "RUL"] = 999
    with pytest.raises(ValueError):
        paired_bootstrap(predictions, repetitions=20)


@pytest.mark.parametrize("repetitions", [0, -1, True, 1.5])
def test_invalid_bootstrap_repetition_count_is_rejected(repetitions):
    """Reject ambiguous or empty resampling requests.

    Args:
        repetitions: Invalid bootstrap replicate count supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    with pytest.raises(ValueError, match="positive integer"):
        paired_bootstrap(paired_example(), repetitions=repetitions)


def write_tiny_source(folder: Path) -> None:
    """Write a two-engine fixture with reversed row order and known endpoint labels.

    Args:
        folder: Destination temporary source directory.

    Returns:
        None: Writes three NASA-shaped text files, unrelated to experimental results.
    """
    folder.mkdir(parents=True, exist_ok=True)
    values = np.array(
        [[2, 2] + [1] * 24, [1, 1] + [1] * 24, [2, 1] + [1] * 24, [1, 2] + [1] * 24]
    )
    for name in ("train_FD001.txt", "test_FD001.txt"):
        np.savetxt(folder / name, values, fmt="%d")
    (folder / "RUL_FD001.txt").write_text("10\n20\n", encoding="utf-8")


def test_test_labels_follow_engine_id_after_endpoint_selection(tmp_path):
    """Reversed source rows must not swap official RUL labels between engines.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    write_tiny_source(tmp_path)
    endpoints = load_test_endpoints(tmp_path)
    assert endpoints["engine_id"].tolist() == [1, 2]
    assert endpoints["cycle"].tolist() == [2, 2]
    assert endpoints["RUL"].tolist() == [10, 20]


def test_cli_can_prepare_two_output_directories_in_one_process(tmp_path):
    """Explicit path passing prevents a second invocation reusing cached paths.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    source = tmp_path / "data/raw"
    write_tiny_source(source)
    manifest = {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in source.iterdir()
    }
    (tmp_path / "data/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("first", "second"):
        main(
            [
                "prepare-data",
                "--project-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / name),
            ]
        )
        assert (tmp_path / name / "train_FD001_labeled.csv").is_file()


def test_cli_protects_nested_reference_results(tmp_path):
    """Reject output paths inside the frozen report evidence, including subfolders.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    destination = tmp_path / "results/reference/metrics/accidental"
    with pytest.raises(SystemExit):
        main(
            [
                "train-linear",
                "--project-dir",
                str(tmp_path),
                "--output-dir",
                str(destination),
            ]
        )
    assert not destination.exists()


def test_linear_operation_count_has_documented_units():
    """A one-feature model requires two counted FLOPs and 32 assumed bytes.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    model = LinearRegression().fit([[1], [2]], [2, 1])
    estimate = operation_estimate(model, "Age-only", 1)
    assert estimate["estimated_flops"] == 2
    assert estimate["assumed_transfer_bytes"] == 32
    assert estimate["estimated_flops_per_byte"] == pytest.approx(0.0625)
