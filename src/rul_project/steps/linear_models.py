"""Fit age-only and seven-feature linear regression baselines.

    Reads the prepared/source files specified in the README under configured
    DATA_DIR/OUTPUT_DIR and writes model_comparison.csv and per-model validation predictions. Importing this module performs
    no experiment. CLI usage: ``python -m rul_project linear_models``.
    """
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Fit age-only and seven-feature linear regression baselines.

        Returns:
            None: Writes model_comparison.csv and per-model validation predictions to OUTPUT_DIR and prints summaries.

        Raises:
            FileNotFoundError: If a required source or preceding-stage CSV is absent.
            ValueError: If parsing, schema, alignment or estimator validation fails.
            OSError: If input/output access is unavailable.

        No arguments are accepted. Configure paths through the CLI before import.
        Model settings and output names are preserved from the student's script.
        """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from pathlib import Path

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error


    # Load the same training and validation datasets.
    output_dir = OUTPUT_DIR

    train_data = pd.read_csv(output_dir / "train_split.csv")
    validation_data = pd.read_csv(output_dir / "validation_split.csv")


    # Define the features for the multi-sensor model.
    multisensor_features = [
        "cycle",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_9",
        "sensor_11",
    ]

    # Each configuration contains:
    # model name, input features, and prediction output filename.
    model_settings = [
        (
            "Age-only",
            ["cycle"],
            "baseline_validation_predictions.csv",
        ),
        (
            "Multi-sensor linear",
            multisensor_features,
            "multisensor_validation_predictions.csv",
        ),
    ]

    # Collect evaluation metrics for all models.
    comparison_rows = []


    # Repeat the same training and evaluation procedure for each feature set.
    for model_name, features, file_name in model_settings:

        # Select inputs and targets.
        X_train = train_data[features]
        y_train = train_data["RUL"]

        X_validation = validation_data[features]
        y_validation = validation_data["RUL"]

        # Fit the model using training records only.
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate raw and nonnegative predictions.
        predictions = model.predict(X_validation)
        clipped_predictions = predictions.clip(0, None)

        # Evaluate raw predictions.
        raw_mae = mean_absolute_error(y_validation, predictions)
        raw_rmse = mean_squared_error(y_validation, predictions) ** 0.5

        # Evaluate nonnegative predictions.
        clipped_mae = mean_absolute_error(
            y_validation,
            clipped_predictions,
        )
        clipped_rmse = mean_squared_error(
            y_validation,
            clipped_predictions,
        ) ** 0.5

        # Store the metrics for this model.
        comparison_rows.append(
            {
                "model": model_name,
                "n_features": len(features),
                "raw_mae": raw_mae,
                "raw_rmse": raw_rmse,
                "clipped_mae": clipped_mae,
                "clipped_rmse": clipped_rmse,
                "negative_raw_predictions": int(
                    (predictions < 0).sum()
                ),
            }
        )

        # Preserve input features, predictions, and errors.
        result_columns = ["engine_id"] + features + ["RUL"]
        results = validation_data[result_columns].copy()

        results["raw_predicted_RUL"] = predictions
        results["raw_error"] = predictions - y_validation

        results["clipped_predicted_RUL"] = clipped_predictions
        results["clipped_error"] = clipped_predictions - y_validation

        results.to_csv(
            output_dir / file_name,
            index=False,
        )

        print(f"Completed model: {model_name}")


    # Create and save the model comparison table.
    comparison = pd.DataFrame(comparison_rows)

    comparison_path = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nLinear regression model comparison")
    print("----------------------------------")
    print(comparison.round(2).to_string(index=False))

    print("\nValidation records:", len(validation_data))
    print("Comparison saved to:", comparison_path)

if __name__ == "__main__":
    main()
