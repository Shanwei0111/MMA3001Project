"""Refit four fixed models and score the 100 official test endpoints.

    Reads the prepared/source files specified in the README under configured
    DATA_DIR/OUTPUT_DIR and writes official metrics, predictions, error analysis and paired bootstrap. Importing this module performs
    no experiment. CLI usage: ``python -m rul_project testevaluation``.
    """
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Refit four fixed models and score the 100 official test endpoints.

        Returns:
            None: Writes official metrics, predictions, error analysis and paired bootstrap to OUTPUT_DIR and prints summaries.

        Raises:
            FileNotFoundError: If a required source or preceding-stage CSV is absent.
            ValueError: If parsing, schema, alignment or estimator validation fails.
            OSError: If input/output access is unavailable.

        No arguments are accepted. Configure paths through the CLI before import.
        Model settings and output names are preserved from the student's script.
        """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from rul_project.rul_utils import evaluate_rul
    from rul_project.rul_utils import make_hist_gradient_boosting
    import pandas as pd
    import numpy as np

    # Locate the source data and output folders.
    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR

    output_dir.mkdir(exist_ok=True)


    # Define the source column names.
    column_names = [
        "engine_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ]

    sensor_names = [f"sensor_{i}" for i in range(1, 22)]


    # Load the truncated test histories.
    test_data = pd.read_csv(
        data_dir / "test_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    if test_data.shape[1] != 26:
        raise ValueError("Expected 26 columns in the test data.")

    test_data.columns = column_names + sensor_names


    # Select the final observed record for each test engine.
    test_endpoints = (
        test_data
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )


    # Load the official RUL at each engine's final observation.
    rul_data = pd.read_csv(
        data_dir / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
    )

    if rul_data.shape[1] != 1:
        raise ValueError("Expected one column in the RUL label file.")

    if len(test_endpoints) != len(rul_data):
        raise ValueError("Test engine count does not match RUL label count.")

    # Official labels correspond to engine IDs 1 through N.
    expected_ids = list(range(1, len(test_endpoints) + 1))

    if test_endpoints["engine_id"].tolist() != expected_ids:
        raise ValueError("Expected consecutive test engine IDs starting at 1.")

    test_endpoints["RUL"] = rul_data[0].to_numpy()


    # Display the aligned endpoints and labels.
    print("Official test dataset summary")
    print("-----------------------------")
    print("Test observations:", len(test_data))
    print("Test engines:", test_data["engine_id"].nunique())
    print("Endpoint labels:", len(rul_data))

    print("\nFirst five test endpoints:")
    print(
        test_endpoints[["engine_id", "cycle", "RUL"]]
        .head()
        .to_string(index=False)
    )


    # Save one labeled endpoint per test engine.
    output_path = output_dir / "test_endpoints_labeled.csv"
    test_endpoints.to_csv(output_path, index=False)

    print("\nTest endpoints saved to:", output_path)

    # Load all complete training histories with their RUL labels.
    full_train_data = pd.read_csv(
        output_dir / "train_FD001_labeled.csv"
    )

    print("\nFinal model fitting")
    print("-------------------")
    print("Training engines:", full_train_data["engine_id"].nunique())
    print("Training records:", len(full_train_data))
    print("Test endpoints:", len(test_endpoints))


    # Use the previously defined multi-sensor feature set.
    multisensor_features = [
        "cycle",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_9",
        "sensor_11",
    ]

    # Keep the model configurations fixed for official testing.
    final_models = [
        (
            "Age-only",
            ["cycle"],
            LinearRegression(),
        ),
        (
            "Multi-sensor linear",
            multisensor_features,
            LinearRegression(),
        ),
        (
            "Random forest",
            multisensor_features,
            RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=1,
            ),
        ),
        (
            "Histogram gradient boosting",
            multisensor_features,
            make_hist_gradient_boosting(),
        ),
    ]

    test_metrics = []
    prediction_tables = []


    # Refit each model and evaluate one endpoint per test engine.
    for model_name, features, final_model in final_models:

        X_full_train = full_train_data[features]
        y_full_train = full_train_data["RUL"]

        X_test = test_endpoints[features]
        y_test = test_endpoints["RUL"]

        # Official test labels are not passed to fit.
        final_model.fit(X_full_train, y_full_train)

        raw_predictions = final_model.predict(X_test)
        clipped_predictions = raw_predictions.clip(0, None)

        # Calculate metrics using the same output rules as validation.
        raw_mae = mean_absolute_error(y_test, raw_predictions)
        raw_rmse = mean_squared_error(y_test, raw_predictions) ** 0.5

        clipped_mae = mean_absolute_error(
            y_test,
            clipped_predictions,
        )
        clipped_rmse = mean_squared_error(
            y_test,
            clipped_predictions,
        ) ** 0.5

        test_metrics.append(
            {
                "model": model_name,
                "n_features": len(features),
                "raw_mae": raw_mae,
                "raw_rmse": raw_rmse,
                "clipped_mae": clipped_mae,
                "clipped_rmse": clipped_rmse,
                "negative_raw_predictions": int(
                    (raw_predictions < 0).sum()
                ),
            }
        )

        # Preserve every engine's prediction for error analysis.
        model_results = test_endpoints[
            ["engine_id", "cycle", "RUL"]
        ].copy()

        model_results["model"] = model_name
        model_results["raw_predicted_RUL"] = raw_predictions
        model_results["raw_error"] = raw_predictions - y_test

        model_results["clipped_predicted_RUL"] = clipped_predictions
        model_results["clipped_error"] = clipped_predictions - y_test

        prediction_tables.append(model_results)

        print(f"Completed official evaluation: {model_name}")


    # Save the official test metrics.
    test_comparison = pd.DataFrame(test_metrics)

    test_comparison.to_csv(
        output_dir / "official_test_metrics.csv",
        index=False,
    )

    # Combine the prediction tables from all four models.
    all_test_predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    all_test_predictions.to_csv(
        output_dir / "official_test_predictions.csv",
        index=False,
    )

    print("\nOfficial test endpoint performance")
    print("----------------------------------")
    print(test_comparison.round(2).to_string(index=False))

    print("\nTest engines evaluated per model:", len(test_endpoints))
    print("Official test metrics and predictions saved successfully.")

    # Select the random forest's official test predictions.
    forest_results = all_test_predictions[
        all_test_predictions["model"] == "Random forest"
    ].copy()

    # Verify that each engine contributes one endpoint.
    if forest_results["engine_id"].duplicated().any():
        raise ValueError("Expected one test endpoint per engine.")

    # Calculate error magnitude and prediction direction.
    errors = forest_results["clipped_error"]
    forest_results["absolute_error"] = errors.abs()

    bias = errors.mean()
    overprediction_count = int((errors > 0).sum())
    underprediction_count = int((errors < 0).sum())
    overprediction_percentage = (errors > 0).mean() * 100


    # Display the error summary.
    print("\nRandom forest: official test error analysis")
    print("-------------------------------------------")
    print("Test engines:", len(forest_results))
    print(f"Mean signed error: {bias:.2f} cycles")
    print("Overpredicted engines:", overprediction_count)
    print("Underpredicted engines:", underprediction_count)
    print(
        f"Overprediction percentage: "
        f"{overprediction_percentage:.1f}%"
    )


    # Identify the five largest absolute errors.
    worst_cases = forest_results.nlargest(
        5,
        "absolute_error",
    )

    print("\nFive largest absolute errors:")
    print(
        worst_cases[
            [
                "engine_id",
                "cycle",
                "RUL",
                "clipped_predicted_RUL",
                "clipped_error",
                "absolute_error",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )


    # Save the detailed error analysis.
    forest_results.to_csv(
        output_dir / "random_forest_test_error_analysis.csv",
        index=False,
    )

    # Supplement the existing test evaluation with shared metrics.
    extended_rows = []

    for model_name, model_results in all_test_predictions.groupby(
        "model",
        sort=False,
    ):
        scores = evaluate_rul(
            model_results["RUL"],
            model_results["clipped_predicted_RUL"],
        )

        extended_rows.append(
            {
                "model": model_name,
                "test_engines": len(model_results),
                **scores,
            }
        )

    extended_metrics = pd.DataFrame(extended_rows)

    extended_metrics.to_csv(
        output_dir / "official_test_extended_metrics.csv",
        index=False,
    )

    print("\nOfficial test: supplementary error metrics")
    print("------------------------------------------")
    print(extended_metrics.round(2).to_string(index=False))

    # Align model predictions by engine ID.
    prediction_matrix = all_test_predictions.pivot(
        index="engine_id",
        columns="model",
        values="clipped_predicted_RUL",
    ).sort_index()

    if prediction_matrix.isna().any().any():
        raise ValueError("Missing model predictions for some test engines.")

    actual_rul = (
        test_endpoints
        .set_index("engine_id")
        .loc[prediction_matrix.index, "RUL"]
        .to_numpy(dtype=float)
    )

    forest_errors = (
        prediction_matrix["Random forest"].to_numpy() - actual_rul
    )

    linear_errors = (
        prediction_matrix["Multi-sensor linear"].to_numpy() - actual_rul
    )


    # Generate paired engine-level bootstrap samples.
    n_engines = len(actual_rul)
    n_bootstrap = 2000
    rng = np.random.default_rng(42)

    sample_indices = rng.integers(
        0,
        n_engines,
        size=(n_bootstrap, n_engines),
    )

    # Each row represents one bootstrap sample of engines.
    forest_samples = forest_errors[sample_indices]
    linear_samples = linear_errors[sample_indices]

    forest_mae_samples = np.abs(forest_samples).mean(axis=1)
    forest_rmse_samples = np.sqrt(
        (forest_samples ** 2).mean(axis=1)
    )

    linear_mae_samples = np.abs(linear_samples).mean(axis=1)
    linear_rmse_samples = np.sqrt(
        (linear_samples ** 2).mean(axis=1)
    )


    # Calculate point estimates from the original test sample.
    forest_mae = np.abs(forest_errors).mean()
    forest_rmse = np.sqrt((forest_errors ** 2).mean())

    linear_mae = np.abs(linear_errors).mean()
    linear_rmse = np.sqrt((linear_errors ** 2).mean())

    bootstrap_metrics = [
        (
            "Forest MAE",
            forest_mae,
            forest_mae_samples,
        ),
        (
            "Forest RMSE",
            forest_rmse,
            forest_rmse_samples,
        ),
        (
            "MAE difference: forest - linear",
            forest_mae - linear_mae,
            forest_mae_samples - linear_mae_samples,
        ),
        (
            "RMSE difference: forest - linear",
            forest_rmse - linear_rmse,
            forest_rmse_samples - linear_rmse_samples,
        ),
    ]


    # Report percentile 95% bootstrap intervals.
    summary_rows = []

    for metric_name, estimate, samples in bootstrap_metrics:

        lower, upper = np.percentile(samples, [2.5, 97.5])

        summary_rows.append(
            {
                "metric": metric_name,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    bootstrap_summary = pd.DataFrame(summary_rows)

    bootstrap_summary.to_csv(
        output_dir / "bootstrap_summary.csv",
        index=False,
    )

    print("\nPaired engine-bootstrap results")
    print("-------------------------------")
    print(bootstrap_summary.round(2).to_string(index=False))

if __name__ == "__main__":
    main()
