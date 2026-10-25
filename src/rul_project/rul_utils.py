"""Shared utilities for the FD001 remaining-life project."""

import numpy as np
import pandas as pd


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
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

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
            raise ValueError(
                f"{column} must contain only finite values."
            )

        if (values <= 0).any():
            raise ValueError(
                f"{column} must contain positive values."
            )

        if (values != np.floor(values)).any():
            raise ValueError(
                f"{column} must contain integer values."
            )

    if result.duplicated(required_columns).any():
        raise ValueError("Duplicate engine-cycle records found.")

    failure_cycles = (
        result.groupby("engine_id")["cycle"].transform("max")
    )

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

def evaluate_rul(y_true, y_pred) -> dict[str, float]:
    """Evaluate aligned RUL observations using unrounded supplied predictions.
    
        Args:
            y_true (array-like): Finite nonnegative one-dimensional actual RUL,
                measured in cycles. At least one observation is required.
            y_pred (array-like): Finite one-dimensional predictions in cycles, in
                the same order and with the same length as y_true. Raw negative
                predictions are allowed; this function never clips its inputs.
    
        Returns:
            dict[str, float]: mae, rmse and bias in cycles; overprediction_pct as a
            percentage; nasa_score as the sum of dimensionless asymmetric penalties.
            Error is prediction minus truth; under/overprediction divisors are 13/10.
    
        Raises:
            ValueError: If arrays are empty, nonfinite, not aligned one-dimensional
                arrays or actual RUL contains a negative value.
            FloatingPointError: If squared errors or exponential penalties overflow.
    
        Examples:
            >>> evaluate_rul([10, 20], [10, 20])["rmse"]
            0.0
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Inputs must not be empty.")

    if actual.shape != predicted.shape:
        raise ValueError("Inputs must have equal lengths.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Inputs must contain only finite values.")

    if (actual < 0).any():
        raise ValueError("True RUL must be nonnegative.")

    with np.errstate(over="raise", invalid="raise"):
        errors = predicted - actual

        scaled_errors = np.where(
            errors < 0,
            -errors / 13.0,
            errors / 10.0,
        )

        return {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bias": float(errors.mean()),
            "overprediction_pct": float(
                (errors > 0).mean() * 100
            ),
            "nasa_score": float(np.expm1(scaled_errors).sum()),
        }

def make_hist_gradient_boosting():
    """Create an unfitted HGB candidate with the fixed project configuration.
    
        Returns:
            sklearn.ensemble.HistGradientBoostingRegressor: Squared-error model with
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=20, l2_regularization=1.0, max_bins=255 (library default),
            early_stopping=False and random_state=42.
    
        Examples:
            >>> make_hist_gradient_boosting().max_iter
            200
    
        No arguments are accepted. Early stopping is disabled so grouped validation
        controls evaluation instead of an automatic random row-level holdout.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )