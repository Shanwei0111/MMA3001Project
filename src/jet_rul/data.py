"""Strict source parsing, dataset integrity, and labels in operational cycles."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = ["s2", "s3", "s4", "s7", "s9", "s11"]
UNITS = {"s2": "degR", "s3": "degR", "s4": "degR", "s7": "psia", "s9": "rpm", "s11": "psia"}


def read_cmapss(path: str | Path) -> pd.DataFrame:
    """Read 26 whitespace-delimited columns and validate complete trajectories.

    Parameters
    ----------
    path : str or Path
        Source training/test TXT file; native sensor units are retained.

    Returns
    -------
    pandas.DataFrame
        Rows sorted by engine_id and cycle, with integer identifiers.

    Raises
    ------
    ValueError
        Malformed numbers, incorrect width, gaps, duplicates or invalid values.
    """
    try:
        values = np.loadtxt(path, ndmin=2)
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot parse source file {path}: {exc}") from exc
    if values.shape[1] != 26:
        raise ValueError(f"Expected 26 columns, received {values.shape[1]}")
    return validate_frame(pd.DataFrame(values, columns=COLUMNS))


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate numeric finite observations and return a sorted copy.

    Parameters
    ----------
    frame : pandas.DataFrame
        Full C-MAPSS schema. A trajectory may begin after cycle 1, but cycles
        must be contiguous; this permits prediction from a trailing history.

    Returns
    -------
    pandas.DataFrame
        Validated copy with integer ID/cycle columns.

    Raises
    ------
    ValueError
        Missing schema, empty input, impossible primary values or bad sequence.
    """
    if frame.empty or list(frame.columns) != COLUMNS:
        raise ValueError("Nonempty input with the exact C-MAPSS column schema is required")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All source fields must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite measurements are unsupported")
    ids = values[:, :2]
    if np.any(ids < 1) or np.any(ids != np.floor(ids)):
        raise ValueError("Engine IDs and cycles must be positive integers")
    result = frame.astype(float).copy()
    result[["engine_id", "cycle"]] = result[["engine_id", "cycle"]].astype(int)
    if (result[SENSORS] <= 0).any().any():
        raise ValueError("Primary absolute temperatures, pressures and speed must be positive")
    if result.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle observations")
    result = result.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    for _, group in result.groupby("engine_id", sort=True):
        if np.any(np.diff(group.cycle.to_numpy()) != 1):
            raise ValueError("Cycle gaps are unsupported; supply contiguous observations")
    return result


def training_labels(frame: pd.DataFrame) -> np.ndarray:
    """Return T_i - t in cycles for verified run-to-failure training rows.

    This function must never be applied to a truncated test trajectory.
    The returned nonnegative array is aligned with frame's row order.
    """
    return (frame.groupby("engine_id").cycle.transform("max") - frame.cycle).to_numpy(dtype=float)


def read_test_labels(path: str | Path, engine_ids: np.ndarray) -> pd.Series:
    """Map one nonnegative integer endpoint label to each sorted engine ID.

    Parameters
    ----------
    path : str or Path
        RUL_FD001.txt file, accessible only to the final evaluator.
    engine_ids : numpy.ndarray
        Unique test engine IDs; the benchmark requires IDs 1 through N.

    Returns
    -------
    pandas.Series
        Endpoint RUL in cycles, indexed by engine ID.

    Raises
    ------
    ValueError
        Labels do not match the expected count, order or integer domain.
    """
    ids = np.sort(np.asarray(engine_ids, dtype=int))
    y = np.loadtxt(path, ndmin=1)
    if (y.ndim != 1 or len(y) != len(ids) or not np.array_equal(ids, np.arange(1, len(ids)+1))
            or not np.isfinite(y).all() or np.any(y < 0) or np.any(y != np.floor(y))):
        raise ValueError("Test labels must be one nonnegative integer per engine, ordered 1..N")
    return pd.Series(y, index=ids, name="true_rul")


def sha256(path: str | Path) -> str:
    """Return the SHA-256 hexadecimal digest of a local file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest(data_dir: str | Path, manifest_path: str | Path) -> dict:
    """Verify all required files against a frozen manifest before reproduction.

    Returns the manifest dictionary. Raises ValueError for a hash mismatch;
    FileNotFoundError identifies missing source or manifest files.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        if sha256(Path(data_dir) / filename) != expected["sha256"]:
            raise ValueError(f"Checksum mismatch for {filename}; do not silently replace benchmark data")
    return manifest
