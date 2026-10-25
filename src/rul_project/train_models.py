"""Fit development candidates, compare forest depths and validate by engine.

These functions consume only the 80-engine training subset and 20-engine
validation subset. The official test labels never enter development fitting.
Model factories and feature order come from model_configuration.py.
"""

from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import GroupKFold

from rul_project.model_configuration import (
    feature_columns,
    make_model,
    make_random_forest,
)
from rul_project.rul_metrics import evaluate_rul

VALIDATION_FILES = {
    "Age-only": "baseline_validation_predictions.csv",
    "Multi-sensor linear": "multisensor_validation_predictions.csv",
    "Random forest": "random_forest_validation_predictions.csv",
    "Histogram gradient boosting": "hist_gradient_boosting_validation_predictions.csv",
}


def load_development_split(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared development tables and reject shared engine identities.

    Args:
        output_dir: Directory containing train_split.csv and validation_split.csv.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Nonempty training and validation tables.

    Raises:
        FileNotFoundError: If the preparation/split stages have not been run.
        ValueError: If either subset is empty or an engine appears in both.

    Example: train, validation = load_development_split(Path("results/generated")).
    """
    train = pd.read_csv(output_dir / "train_split.csv")
    validation = pd.read_csv(output_dir / "validation_split.csv")
    if train.empty or validation.empty:
        raise ValueError("Both development subsets must contain records.")
    if set(train.engine_id).intersection(validation.engine_id):
        raise ValueError("Training and validation engines overlap.")
    return train, validation


def fit_validation_model(
    train: pd.DataFrame, validation: pd.DataFrame, model_name: str, output_dir: Path
) -> dict:
    """Fit one candidate and save raw/clipped predictions on unseen engines.

    Args:
        train: Labelled training rows from complete engine groups.
        validation: Labelled held-out-engine rows; must share no training engine.
        model_name: Supported model name from model_configuration.MODEL_NAMES.
        output_dir: Destination for the model-specific prediction CSV.

    Returns:
        dict: Model name, feature count, train/raw/clipped error measures, negative
        prediction count and fit_seconds. Errors are unrounded cycles.

    Raises:
        ValueError: If a model name, feature array or target is invalid or groups overlap.
        OSError: If predictions cannot be written.

    Example: fit_validation_model(train, validation, "Random forest", Path("results/generated")).
    """
    if set(train.engine_id).intersection(validation.engine_id):
        raise ValueError("Training and validation engines overlap.")
    features = feature_columns(model_name)
    model = make_model(model_name)
    start = perf_counter()
    model.fit(train[features], train["RUL"])
    fit_seconds = perf_counter() - start
    raw = model.predict(validation[features])
    delivered = raw.clip(0, None)
    raw_scores = evaluate_rul(validation["RUL"], raw)
    scores = evaluate_rul(validation["RUL"], delivered)
    train_scores = evaluate_rul(
        train["RUL"], model.predict(train[features]).clip(0, None)
    )

    # Preserve row identity so plotted errors remain aligned with their sensor inputs.
    predictions = validation[["engine_id"] + features + ["RUL"]].copy()
    predictions["raw_predicted_RUL"] = raw
    predictions["raw_error"] = raw - validation["RUL"].to_numpy()
    predictions["clipped_predicted_RUL"] = delivered
    predictions["clipped_error"] = delivered - validation["RUL"].to_numpy()
    predictions.to_csv(output_dir / VALIDATION_FILES[model_name], index=False)
    return {
        "model": model_name,
        "n_features": len(features),
        "train_mae": train_scores["mae"],
        "train_rmse": train_scores["rmse"],
        "raw_mae": raw_scores["mae"],
        "raw_rmse": raw_scores["rmse"],
        "clipped_mae": scores["mae"],
        "clipped_rmse": scores["rmse"],
        "negative_raw_predictions": int((raw < 0).sum()),
        "fit_seconds": fit_seconds,
    }


def grouped_cross_validation(train: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Evaluate the fixed candidate on five identical engine-grouped folds.

    Args:
        train: Development training subset, normally 80 complete engine histories.
        model_name: Name of the fixed candidate to fit independently in each fold.

    Returns:
        pandas.DataFrame: Fold, training/validation engine counts, MAE and RMSE.

    Raises:
        ValueError: If there are fewer than five engines or estimator inputs are invalid.

    Example: grouped_cross_validation(train, "Histogram gradient boosting").
    """
    features = feature_columns(model_name)
    inputs, target, groups = train[features], train["RUL"], train["engine_id"]
    rows = []
    for fold, (fit_indices, valid_indices) in enumerate(
        GroupKFold(5).split(inputs, target, groups), start=1
    ):
        # A fresh estimator prevents fitted trees from carrying information across folds.
        model = make_model(model_name)
        model.fit(inputs.iloc[fit_indices], target.iloc[fit_indices])
        predicted = model.predict(inputs.iloc[valid_indices]).clip(0, None)
        scores = evaluate_rul(target.iloc[valid_indices], predicted)
        rows.append(
            {
                "fold": fold,
                "training_engines": groups.iloc[fit_indices].nunique(),
                "validation_engines": groups.iloc[valid_indices].nunique(),
                "mae": scores["mae"],
                "rmse": scores["rmse"],
            }
        )
        print(f"{model_name}: grouped fold {fold}/5 completed")
    return pd.DataFrame(rows)


def compare_forest_depths(
    train: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    """Measure generalization and fit time at depths 4, 8 and 12.

    Args:
        train: Labelled development training table.
        validation: Labelled development validation table with disjoint engines.

    Returns:
        pandas.DataFrame: Depth, train/validation MAE/RMSE and fitting seconds.

    Raises:
        ValueError: If input features/targets are invalid or engine groups overlap.

    Example: compare_forest_depths(train, validation). Training alone is timed.
    """
    if set(train.engine_id).intersection(validation.engine_id):
        raise ValueError("Training and validation engines overlap.")
    features = feature_columns("Random forest")
    rows = []
    for depth in (4, 8, 12):
        model = make_random_forest(max_depth=depth)
        start = perf_counter()
        model.fit(train[features], train["RUL"])
        seconds = perf_counter() - start
        train_scores = evaluate_rul(
            train["RUL"], model.predict(train[features]).clip(0, None)
        )
        valid_scores = evaluate_rul(
            validation["RUL"], model.predict(validation[features]).clip(0, None)
        )
        rows.append(
            {
                "max_depth": depth,
                "train_mae": train_scores["mae"],
                "validation_mae": valid_scores["mae"],
                "train_rmse": train_scores["rmse"],
                "validation_rmse": valid_scores["rmse"],
                "fit_seconds": seconds,
            }
        )
    return pd.DataFrame(rows)


def train_development_models(output_dir: Path, family: str) -> None:
    """Run a documented development stage and write its evidence tables.

    Args:
        output_dir: Prepared-data directory and destination for new results.
        family: "linear", "forest" or "hgb". Linear fits both baseline models;
            forest also runs depth sensitivity; both ensembles run grouped CV.

    Returns:
        None: Writes metrics, predictions and applicable depth/CV CSVs.

    Raises:
        FileNotFoundError: If prepared split files are absent.
        ValueError: If family, grouping or estimator inputs are invalid.
        OSError: If output cannot be written.

    Example: train_development_models(Path("results/generated"), "forest").
    """
    if family not in {"linear", "forest", "hgb"}:
        raise ValueError(f"Unknown model family: {family}")
    train, validation = load_development_split(output_dir)
    names = {
        "linear": ("Age-only", "Multi-sensor linear"),
        "forest": ("Random forest",),
        "hgb": ("Histogram gradient boosting",),
    }[family]
    rows = [fit_validation_model(train, validation, name, output_dir) for name in names]
    metrics = pd.DataFrame(rows)
    filename = {
        "linear": "model_comparison.csv",
        "forest": "random_forest_metrics.csv",
        "hgb": "hist_gradient_boosting_metrics.csv",
    }[family]
    metrics.to_csv(output_dir / filename, index=False)
    print(metrics.round(3).to_string(index=False))
    if family == "forest":
        compare_forest_depths(train, validation).to_csv(
            output_dir / "random_forest_depth_study.csv", index=False
        )
    if family in {"forest", "hgb"}:
        folds = grouped_cross_validation(train, names[0])
        stem = "random_forest" if family == "forest" else "hist_gradient_boosting"
        folds.to_csv(output_dir / f"{stem}_group_cv.csv", index=False)
        print(folds.round(2).to_string(index=False))
