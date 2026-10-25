"""Evaluate the fixed HGB candidate and the same five grouped folds.

Reads the prepared/source files specified in the README under configured
DATA_DIR/OUTPUT_DIR and writes hist_gradient_boosting_metrics.csv, predictions and grouped CV. Importing this module performs
no experiment. CLI usage: ``python -m rul_project histgradientboosting``.
"""
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Evaluate the fixed HGB candidate and the same five grouped folds.

    Returns:
        None: Writes hist_gradient_boosting_metrics.csv, predictions and grouped CV to OUTPUT_DIR and prints summaries.

    Raises:
        FileNotFoundError: If a required source or preceding-stage CSV is absent.
        ValueError: If parsing, schema, alignment or estimator validation fails.
        OSError: If input/output access is unavailable.

    No arguments are accepted. Configure paths through the CLI before import.
    Model settings and output names are preserved from the student's script.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from pathlib import Path
    from time import perf_counter

    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits

    from rul_project.rul_utils import evaluate_rul, make_hist_gradient_boosting


    output_dir = OUTPUT_DIR

    train_data = pd.read_csv(output_dir / "train_split.csv")
    validation_data = pd.read_csv(output_dir / "validation_split.csv")

    features = [
        "cycle",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_9",
        "sensor_11",
    ]

    X_train = train_data[features]
    y_train = train_data["RUL"]

    X_validation = validation_data[features]
    y_validation = validation_data["RUL"]

    shared_ids = set(train_data["engine_id"]).intersection(
        validation_data["engine_id"]
    )
    if shared_ids:
        raise ValueError("Training and validation engines overlap.")

    model = make_hist_gradient_boosting()

    print("Training histogram gradient boosting...")

    with threadpool_limits(limits=1):
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start

        train_predictions = model.predict(X_train).clip(0, None)
        raw_predictions = model.predict(X_validation)

    clipped_predictions = raw_predictions.clip(0, None)

    train_scores = evaluate_rul(y_train, train_predictions)
    raw_scores = evaluate_rul(y_validation, raw_predictions)
    validation_scores = evaluate_rul(
        y_validation,
        clipped_predictions,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Histogram gradient boosting",
                "n_features": len(features),
                "train_mae": train_scores["mae"],
                "train_rmse": train_scores["rmse"],
                "raw_mae": raw_scores["mae"],
                "raw_rmse": raw_scores["rmse"],
                "clipped_mae": validation_scores["mae"],
                "clipped_rmse": validation_scores["rmse"],
                "negative_raw_predictions": int(
                    (raw_predictions < 0).sum()
                ),
                "fit_seconds": fit_seconds,
            }
        ]
    )

    metrics.to_csv(
        output_dir / "hist_gradient_boosting_metrics.csv",
        index=False,
    )

    results = validation_data[
        ["engine_id"] + features + ["RUL"]
    ].copy()

    results["raw_predicted_RUL"] = raw_predictions
    results["raw_error"] = raw_predictions - y_validation.to_numpy()
    results["clipped_predicted_RUL"] = clipped_predictions
    results["clipped_error"] = (
        clipped_predictions - y_validation.to_numpy()
    )

    results.to_csv(
        output_dir / "hist_gradient_boosting_validation_predictions.csv",
        index=False,
    )

    print("\nHistogram gradient boosting: validation performance")
    print("---------------------------------------------------")
    print(metrics.round(3).to_string(index=False))


    # Use the same five engine-grouped folds as the forest experiment.
    groups = train_data["engine_id"]
    group_cv = GroupKFold(n_splits=5)
    cv_rows = []

    with threadpool_limits(limits=1):
        for fold, (fit_indices, valid_indices) in enumerate(
            group_cv.split(X_train, y_train, groups),
            start=1,
        ):
            print(f"\nRunning grouped CV fold {fold}/5...")

            fold_model = make_hist_gradient_boosting()
            fold_model.fit(
                X_train.iloc[fit_indices],
                y_train.iloc[fit_indices],
            )

            fold_predictions = fold_model.predict(
                X_train.iloc[valid_indices]
            ).clip(0, None)

            scores = evaluate_rul(
                y_train.iloc[valid_indices],
                fold_predictions,
            )

            cv_rows.append(
                {
                    "fold": fold,
                    "training_engines": groups.iloc[
                        fit_indices
                    ].nunique(),
                    "validation_engines": groups.iloc[
                        valid_indices
                    ].nunique(),
                    "mae": scores["mae"],
                    "rmse": scores["rmse"],
                }
            )

    cv_results = pd.DataFrame(cv_rows)

    cv_results.to_csv(
        output_dir / "hist_gradient_boosting_group_cv.csv",
        index=False,
    )

    print("\nHistogram gradient boosting: engine-grouped CV")
    print("---------------------------------------------")
    print(cv_results.round(2).to_string(index=False))
    print(f"\nMean fold MAE: {cv_results['mae'].mean():.2f} cycles")
    print(f"Mean fold RMSE: {cv_results['rmse'].mean():.2f} cycles")
    print(
        "Fold RMSE standard deviation: "
        f"{cv_results['rmse'].std():.2f} cycles"
    )


if __name__ == "__main__":
    main()
