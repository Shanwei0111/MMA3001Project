"""Evaluate the fixed forest, depth sensitivity and five grouped folds.

Reads the prepared/source files specified in the README under configured
DATA_DIR/OUTPUT_DIR and writes random_forest_metrics.csv, predictions, depth study and grouped CV. Importing this module performs
no experiment. CLI usage: ``python -m rul_project randomforest``.
"""
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Evaluate the fixed forest, depth sensitivity and five grouped folds.

    Returns:
        None: Writes random_forest_metrics.csv, predictions, depth study and grouped CV to OUTPUT_DIR and prints summaries.

    Raises:
        FileNotFoundError: If a required source or preceding-stage CSV is absent.
        ValueError: If parsing, schema, alignment or estimator validation fails.
        OSError: If input/output access is unavailable.

    No arguments are accepted. Configure paths through the CLI before import.
    Model settings and output names are preserved from the student's script.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.model_selection import GroupKFold
    from time import perf_counter
    from pathlib import Path

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error


    # Load the same training and validation datasets.
    output_dir = OUTPUT_DIR

    train_data = pd.read_csv(output_dir / "train_split.csv")
    validation_data = pd.read_csv(output_dir / "validation_split.csv")


    # Use the same inputs as the multi-sensor linear model.
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


    # Create the random forest with fixed initial settings.
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=1,
    )

    print("Training random forest...")

    # Fit using training records only.
    model.fit(X_train, y_train)

    # Generate raw and nonnegative predictions.
    predictions = model.predict(X_validation)
    clipped_predictions = predictions.clip(0, None)


    # Evaluate raw predictions.
    raw_mae = mean_absolute_error(y_validation, predictions)
    raw_rmse = mean_squared_error(y_validation, predictions) ** 0.5

    # Evaluate nonnegative predictions.
    clipped_mae = mean_absolute_error(y_validation, clipped_predictions)
    clipped_rmse = (
        mean_squared_error(y_validation, clipped_predictions) ** 0.5
    )

    negative_count = int((predictions < 0).sum())


    # Display the validation results.
    print("\nRandom forest: raw predictions")
    print("------------------------------")
    print(f"Input features: {len(features)}")
    print(f"Validation records: {len(y_validation)}")
    print(f"MAE: {raw_mae:.2f} cycles")
    print(f"RMSE: {raw_rmse:.2f} cycles")
    print(f"Negative predictions: {negative_count}")

    print("\nRandom forest: nonnegative predictions")
    print("--------------------------------------")
    print(f"MAE: {clipped_mae:.2f} cycles")
    print(f"RMSE: {clipped_rmse:.2f} cycles")
    print(f"Negative predictions: {(clipped_predictions < 0).sum()}")


    # Save predictions and errors for each validation record.
    results = validation_data[
        ["engine_id"] + features + ["RUL"]
    ].copy()

    results["raw_predicted_RUL"] = predictions
    results["raw_error"] = predictions - y_validation

    results["clipped_predicted_RUL"] = clipped_predictions
    results["clipped_error"] = clipped_predictions - y_validation

    prediction_path = (
        output_dir / "random_forest_validation_predictions.csv"
    )

    results.to_csv(prediction_path, index=False)


    # Save the evaluation metrics without rounding.
    metrics = pd.DataFrame(
        [
            {
                "model": "Random forest",
                "n_features": len(features),
                "raw_mae": raw_mae,
                "raw_rmse": raw_rmse,
                "clipped_mae": clipped_mae,
                "clipped_rmse": clipped_rmse,
                "negative_raw_predictions": negative_count,
            }
        ]
    )

    metrics_path = output_dir / "random_forest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("\nPredictions saved to:", prediction_path)
    print("Metrics saved to:", metrics_path)

    # Evaluate the fitted model on its training records.
    train_predictions = model.predict(X_train).clip(0, None)

    train_mae = mean_absolute_error(y_train, train_predictions)
    train_rmse = (
        mean_squared_error(y_train, train_predictions) ** 0.5
    )

    print("\nTraining versus validation performance")
    print("--------------------------------------")
    print(f"Training records: {len(y_train)}")
    print(f"Validation records: {len(y_validation)}")
    print(f"Training MAE: {train_mae:.2f} cycles")
    print(f"Validation MAE: {clipped_mae:.2f} cycles")
    print(f"Training RMSE: {train_rmse:.2f} cycles")
    print(f"Validation RMSE: {clipped_rmse:.2f} cycles")

    # Compare tree depths while keeping other settings fixed.
    depth_results = []

    for depth in [4, 8, 12]:

        print(f"\nEvaluating max_depth={depth}...")

        trial_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=1,
        )

        # Measure fitting time only.
        start_time = perf_counter()
        trial_model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start_time

        # Use the same nonnegative prediction rule.
        trial_train_predictions = trial_model.predict(
            X_train
        ).clip(0, None)

        trial_validation_predictions = trial_model.predict(
            X_validation
        ).clip(0, None)

        # Record training and validation performance.
        depth_results.append(
            {
                "max_depth": depth,
                "train_mae": mean_absolute_error(
                    y_train,
                    trial_train_predictions,
                ),
                "validation_mae": mean_absolute_error(
                    y_validation,
                    trial_validation_predictions,
                ),
                "train_rmse": mean_squared_error(
                    y_train,
                    trial_train_predictions,
                ) ** 0.5,
                "validation_rmse": mean_squared_error(
                    y_validation,
                    trial_validation_predictions,
                ) ** 0.5,
                "fit_seconds": fit_seconds,
            }
        )

    depth_comparison = pd.DataFrame(depth_results)

    depth_comparison.to_csv(
        output_dir / "random_forest_depth_study.csv",
        index=False,
    )

    print("\nRandom forest depth comparison")
    print("------------------------------")
    print(depth_comparison.round(2).to_string(index=False))

    # Evaluate the fixed candidate using engine-grouped cross-validation.
    group_cv = GroupKFold(n_splits=5)
    groups = train_data["engine_id"]

    cv_results = []

    for fold, (fit_indices, validation_indices) in enumerate(
        group_cv.split(X_train, y_train, groups=groups),
        start=1,
    ):

        print(f"\nRunning grouped CV fold {fold}/5...")

        # Select the records assigned to this fold.
        fold_X_train = X_train.iloc[fit_indices]
        fold_y_train = y_train.iloc[fit_indices]

        fold_X_validation = X_train.iloc[validation_indices]
        fold_y_validation = y_train.iloc[validation_indices]

        # Create a fresh model for every fold.
        cv_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=1,
        )

        cv_model.fit(fold_X_train, fold_y_train)

        fold_predictions = cv_model.predict(
            fold_X_validation
        ).clip(0, None)

        fold_mae = mean_absolute_error(
            fold_y_validation,
            fold_predictions,
        )
        fold_rmse = mean_squared_error(
            fold_y_validation,
            fold_predictions,
        ) ** 0.5

        cv_results.append(
            {
                "fold": fold,
                "training_engines": groups.iloc[
                    fit_indices
                ].nunique(),
                "validation_engines": groups.iloc[
                    validation_indices
                ].nunique(),
                "mae": fold_mae,
                "rmse": fold_rmse,
            }
        )

    cv_table = pd.DataFrame(cv_results)

    cv_table.to_csv(
        output_dir / "random_forest_group_cv.csv",
        index=False,
    )

    print("\nEngine-grouped cross-validation")
    print("-------------------------------")
    print(cv_table.round(2).to_string(index=False))

    print(f"\nMean fold MAE: {cv_table['mae'].mean():.2f} cycles")
    print(f"Mean fold RMSE: {cv_table['rmse'].mean():.2f} cycles")
    print(f"Fold RMSE standard deviation: {cv_table['rmse'].std():.2f} cycles")

if __name__ == "__main__":
    main()
