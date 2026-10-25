"""Read NASA histories, calculate training targets and keep engine groups intact.

Inputs are original headerless 26-column text files. Outputs are labelled CSVs
and a deterministic 80/20 engine split. Censored test histories are handled by
load_test_endpoints; their last observations are never treated as failures.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
SOURCE_COLUMNS += [f"sensor_{number}" for number in range(1, 22)]


def read_history(path: Path) -> pd.DataFrame:
    """Read one unmodified NASA engine-history file and validate its schema.

    Args:
        path: Path to train_FD001.txt or test_FD001.txt, in original source units.

    Returns:
        pandas.DataFrame: Numeric table with named ID, cycle, setting and sensor columns.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the table is empty, not 26 columns or contains nonfinite values.
        OSError: If the file cannot be read.

    Example: read_history(Path("data/raw/train_FD001.txt")).shape == (20631, 26).
    """
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.empty or data.shape[1] != 26:
        raise ValueError("Expected a nonempty history with exactly 26 source columns.")
    data.columns = SOURCE_COLUMNS
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Source histories must contain finite numeric values.")
    return data


def load_test_endpoints(data_dir: Path) -> pd.DataFrame:
    """Align official RUL labels with the final observation of each test engine.

    Args:
        data_dir: Directory containing test_FD001.txt and RUL_FD001.txt.

    Returns:
        pandas.DataFrame: One row per engine, sorted by ID, including RUL in cycles.

    Raises:
        FileNotFoundError: If either source file is absent.
        ValueError: If source schema, label counts, values or ID alignment are invalid.

    Example: load_test_endpoints(Path("data/raw")).iloc[0]["RUL"] == 112.
    """
    history = read_history(data_dir / "test_FD001.txt")
    endpoints = (
        history.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )
    labels = pd.read_csv(data_dir / "RUL_FD001.txt", sep=r"\s+", header=None)
    if labels.shape != (len(endpoints), 1):
        raise ValueError("Expected one official RUL label for every test endpoint.")
    if endpoints["engine_id"].tolist() != list(range(1, len(endpoints) + 1)):
        raise ValueError("Expected consecutive test engine IDs starting at 1.")
    values = labels[0].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Official RUL labels must be finite and nonnegative.")
    # Labels follow engine ID, not the last-row order of an arbitrary input table.
    endpoints["RUL"] = labels[0].to_numpy()
    return endpoints


def prepare_training_data(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Label complete training histories and write the reusable training CSV.

    Args:
        data_dir: Directory containing the original train_FD001.txt.
        output_dir: Destination for train_FD001_labeled.csv; created if necessary.

    Returns:
        pandas.DataFrame: Complete labelled training data, preserving source order.

    Raises:
        FileNotFoundError: If the source history is missing.
        ValueError: If source columns or engine-cycle keys are invalid.
        OSError: If data cannot be read or written.

    Example: prepare_training_data(Path("data/raw"), Path("outputs")) .
    """
    data = add_training_rul(read_history(data_dir / "train_FD001.txt"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "train_FD001_labeled.csv", index=False)
    lifetimes = data.groupby("engine_id")["cycle"].max()
    print(f"Training data: {len(data)} records; {len(lifetimes)} engines")
    print(
        f"Lifetime: {lifetimes.min()} to {lifetimes.max()} cycles; mean {lifetimes.mean():.2f}"
    )
    return data


def save_engine_split(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save the original 80/20 engine-level development split.

    Args:
        output_dir: Directory containing train_FD001_labeled.csv and receiving split CSVs.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Training and validation subsets.

    Raises:
        FileNotFoundError: If training labels have not been prepared.
        ValueError: If engine IDs or the requested split are invalid.
        OSError: If split CSVs cannot be written.

    Example: save_engine_split(Path("outputs")); uses seed 42 and 80 training engines.
    """
    data = pd.read_csv(output_dir / "train_FD001_labeled.csv")
    train, validation = split_by_engine(data, n_train=80, random_state=42)
    train.to_csv(output_dir / "train_split.csv", index=False)
    validation.to_csv(output_dir / "validation_split.csv", index=False)
    print(
        f"Split: {train.engine_id.nunique()} training / {validation.engine_id.nunique()} validation engines"
    )
    print(
        f"Records: {len(train)} training / {len(validation)} validation; shared engines: 0"
    )
    return train, validation


def add_training_rul(data: pd.DataFrame) -> pd.DataFrame:
    """Return uncapped cycle-to-failure labels for complete training histories.

    Args:
        data (pandas.DataFrame): Nonempty table containing engine_id and cycle.
            Keys must be finite positive integers, unique as engine-cycle pairs.
            Other columns can contain arbitrary measurements and are preserved.

    Returns:
        pandas.DataFrame: Copy of data with RUL = max(cycle per engine) - cycle,
        in cycles. Original row order and index are retained; data is unchanged.

    Raises:
        ValueError: If keys/columns are missing, corrupt, invalid or duplicated.

    Examples:
        >>> import pandas as pd
        >>> add_training_rul(pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 3]}))["RUL"].tolist()
        [2, 0]

    The final observed cycle is assumed to represent failure. Censored histories
    are semantically unsupported even when their numeric keys pass validation.
    """
    required_columns = ["engine_id", "cycle"]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Training data must not be empty.")

    result = data.copy()

    for column in required_columns:
        # Convert numeric text when possible; reject corrupt values.
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

        values = result[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite values.")

        if (values <= 0).any():
            raise ValueError(f"{column} must contain positive values.")

        if (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain integer values.")

    if result.duplicated(required_columns).any():
        raise ValueError("Duplicate engine-cycle records found.")

    failure_cycles = result.groupby("engine_id")["cycle"].transform("max")

    result["RUL"] = failure_cycles - result["cycle"]

    return result


def split_by_engine(
    data: pd.DataFrame,
    n_train: int = 80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep complete engine histories together in a deterministic split.

    Args:
        data (pandas.DataFrame): Prepared table with a nonmissing engine_id.
            Upstream label construction is responsible for validating key values.
        n_train (int): Number of training engines; defaults to 80. Must leave at
            least one engine in each subset. Booleans are rejected.
        random_state (int): Seed passed to pandas sampling; defaults to 42.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Independent training and
        validation copies, preserving row order within each subset.

    Raises:
        ValueError: If engine_id is absent/missing or n_train is invalid.

    Examples:
        >>> data = pd.DataFrame({"engine_id": [1, 2, 3]})
        >>> train, valid = split_by_engine(data, n_train=2)
        >>> set(train.engine_id).isdisjoint(valid.engine_id)
        True
    """
    if "engine_id" not in data.columns:
        raise ValueError("Missing required column: engine_id.")

    if data["engine_id"].isna().any():
        raise ValueError("Engine IDs must not be missing.")

    engine_ids = data["engine_id"].drop_duplicates()

    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or not 1 <= n_train < len(engine_ids)
    ):
        raise ValueError(
            "n_train must be an integer that leaves "
            "at least one engine in each subset."
        )

    train_ids = engine_ids.sample(
        n=n_train,
        random_state=random_state,
    )

    training_mask = data["engine_id"].isin(train_ids)

    train_data = data.loc[training_mask].copy()
    validation_data = data.loc[~training_mask].copy()

    return train_data, validation_data
