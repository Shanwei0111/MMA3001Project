"""Check data integrity, frozen settings, production-test targets and safe imports."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pytest

from rul_project import prepare_dataset
from rul_project.model_configuration import make_hist_gradient_boosting
from rul_project.data_access import prepare_data, verify_data


def fixture_manifest(folder: Path) -> Path:
    """Write a small manifest for isolated tests, unrelated to actual NASA data.

    Args:
        folder: Temporary directory owned by this test.

    Returns:
        Path: JSON fingerprint manifest for the known example file.
    """
    content = b"1 1 0\n"
    path = folder / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "example.txt": {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_split_tests_target_the_production_function():
    """Reject a duplicate test helper that masks production split behavior.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    from test_data_and_metrics import split_by_engine

    assert split_by_engine is prepare_dataset.split_by_engine


def test_hgb_configuration_matches_report():
    """Detect changes to the frozen HGB parameters rather than silently retuning.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    parameters = make_hist_gradient_boosting().get_params()
    expected = {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 42,
    }
    for name, value in expected.items():
        assert parameters[name] == value


def test_missing_raw_source_is_reported(tmp_path):
    """Missing source files must fail before model fitting.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    with pytest.raises(FileNotFoundError, match="Missing"):
        verify_data(tmp_path, fixture_manifest(tmp_path))


def test_corrupt_source_is_rejected_without_overwrite(tmp_path):
    """Reject a corrupt existing file without replacing its bytes.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    manifest = fixture_manifest(tmp_path)
    source = tmp_path / "example.txt"
    source.write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_data(tmp_path, manifest)
    assert source.read_bytes() == b"corrupt\n"


def test_local_data_copy_preserves_bytes(tmp_path):
    """Copy exact verified bytes and preserve the original file.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    manifest = fixture_manifest(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    (original / "example.txt").write_bytes(b"1 1 0\n")
    destination = tmp_path / "raw"
    prepare_data(destination, manifest, source=original)
    assert (destination / "example.txt").read_bytes() == (
        original / "example.txt"
    ).read_bytes()


def test_wrong_local_source_creates_no_raw_payload(tmp_path):
    """Verify acquired bytes before writing a destination payload.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    manifest = fixture_manifest(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    (original / "example.txt").write_bytes(b"wrong\n")
    destination = tmp_path / "raw"
    with pytest.raises(ValueError, match="does not match"):
        prepare_data(destination, manifest, source=original)
    assert not (destination / "example.txt").exists()


def test_importing_all_stages_performs_no_work(tmp_path):
    """Documentation imports must not read data, train or create output files.

    Args:
        tmp_path: Isolated pathlib.Path temporary directory supplied by pytest.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    code = """
import importlib, pkgutil, rul_project
for module in pkgutil.walk_packages(rul_project.__path__, 'rul_project.'):
    importlib.import_module(module.name)
"""
    environment = os.environ.copy()
    environment["RUL_DATA_DIR"] = str(tmp_path / "absent_data")
    environment["RUL_OUTPUT_DIR"] = str(tmp_path / "absent_outputs")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "absent_data").exists()
    assert not (tmp_path / "absent_outputs").exists()


def test_unsupported_memory_model_is_rejected():
    """Reject unsupported models in selected-array memory inspection.

    Returns:
        None: Assertions confirm the expected behavior.

    Raises:
        AssertionError: If observed behavior differs from the stated expectation.

    Usage: Collected and executed by pytest.
    """
    from rul_project.benchmark_performance import numerical_array_bytes

    with pytest.raises(TypeError, match="Unsupported model"):
        numerical_array_bytes(object())
