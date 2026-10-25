"""Plot validation baseline and official four-model endpoint diagnostics.

Reads the prepared/source files specified in the README under configured
DATA_DIR/OUTPUT_DIR and writes PNG figures; source CSVs are left unchanged. Importing this module performs
no experiment. CLI usage: ``python -m rul_project plotbaseline``.
"""
from rul_project.settings import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Plot validation baseline and official four-model endpoint diagnostics.

    Returns:
        None: Writes PNG figures; source CSVs are left unchanged to OUTPUT_DIR and prints summaries.

    Raises:
        FileNotFoundError: If a required source or preceding-stage CSV is absent.
        ValueError: If parsing, schema, alignment or estimator validation fails.
        OSError: If input/output access is unavailable.

    No arguments are accepted. Configure paths through the CLI before import.
    Model settings and output names are preserved from the student's script.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd


    # Load the saved validation predictions.
    output_dir = OUTPUT_DIR
    data_path = output_dir / "baseline_validation_predictions.csv"

    results = pd.read_csv(data_path)

    # Sort by cycle so the prediction lines are drawn in order.
    line_data = results.sort_values("cycle")

    # Create the figure.
    plt.figure(figsize=(10, 6))

    # Show the actual RUL of every validation record.
    plt.scatter(
        results["cycle"],
        results["RUL"],
        color="gray",
        alpha=0.2,
        s=10,
        label="Actual RUL",
    )

    # Show the original linear predictions.
    plt.plot(
        line_data["cycle"],
        line_data["raw_predicted_RUL"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Raw age-only predictions",
    )

    # Show the predictions after applying the nonnegative constraint.
    plt.plot(
        line_data["cycle"],
        line_data["clipped_predicted_RUL"],
        color="red",
        linewidth=2,
        label="Nonnegative age-only predictions",
    )

    # Add labels and a reference line at zero RUL.
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.xlabel("Cycle")
    plt.ylabel("RUL (cycles)")
    plt.title("Age-only Baseline on Validation Engines")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # Save the figure before displaying it.
    figure_path = output_dir / "age_only_baseline.png"
    plt.savefig(figure_path, dpi=200)

    # Load the saved official test predictions.
    test_predictions = pd.read_csv(
        output_dir / "official_test_predictions.csv"
    )

    forest_results = test_predictions.loc[
        test_predictions["model"] == "Random forest"
    ].copy()

    if forest_results.empty:
        raise ValueError("No random forest predictions were found.")

    if forest_results["engine_id"].duplicated().any():
        raise ValueError("Expected one prediction per test engine.")

    actual = forest_results["RUL"].to_numpy(dtype=float)
    predicted = forest_results[
        "clipped_predicted_RUL"
    ].to_numpy(dtype=float)

    errors = predicted - actual
    bias = errors.mean()

    fig, axes = plt.subplots(figsize=(12, 5), ncols=2)

    # Left panel: predicted versus actual RUL.
    axes[0].scatter(
        actual,
        predicted,
        color="steelblue",
        alpha=0.75,
        s=35,
        edgecolors="white",
        linewidths=0.4,
    )

    axis_limit = max(actual.max(), predicted.max()) * 1.05

    axes[0].plot(
        [0, axis_limit],
        [0, axis_limit],
        color="black",
        linestyle="--",
        label="Perfect prediction",
    )

    axes[0].set_xlim(0, axis_limit)
    axes[0].set_ylim(0, axis_limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Actual RUL (cycles)")
    axes[0].set_ylabel("Predicted RUL (cycles)")
    axes[0].set_title("Random Forest: Official Test Endpoints")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Right panel: signed errors across test engines.
    axes[1].hist(
        errors,
        bins=15,
        color="steelblue",
        edgecolor="white",
    )

    axes[1].axvline(
        0,
        color="black",
        linestyle="--",
        label="Zero error",
    )

    axes[1].axvline(
        bias,
        color="darkorange",
        linewidth=2,
        label=f"Mean error: {bias:+.2f} cycles",
    )

    axes[1].set_xlabel("Prediction - actual RUL (cycles)")
    axes[1].set_ylabel("Number of engines")
    axes[1].set_title("Random Forest: Signed Error Distribution")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()

    figure_path = output_dir / "random_forest_test_diagnostics.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")

    # Generate diagnostics for the additional models.
    model_names = [
        "Age-only",
        "Multi-sensor linear",
        "Random forest",
        "Histogram gradient boosting",
    ]

    test_predictions = pd.read_csv(
        output_dir / "official_test_predictions.csv"
    )

    available_models = set(test_predictions["model"])
    missing_models = set(model_names) - available_models

    if missing_models:
        raise ValueError(
            f"Missing model predictions: {sorted(missing_models)}. "
            "Run the updated official test evaluation first."
        )

    # Verify that every compared model uses the same engines and labels.
    reference_labels = None

    for model_name in model_names:
        model_data = test_predictions.loc[
            test_predictions["model"] == model_name
        ]

        if model_data["engine_id"].duplicated().any():
            raise ValueError(f"Duplicate test engines for {model_name}.")

        labels = model_data.set_index("engine_id")["RUL"].sort_index()

        if reference_labels is None:
            reference_labels = labels
        else:
            pd.testing.assert_series_equal(labels, reference_labels)

    selected = test_predictions.loc[
        test_predictions["model"].isin(model_names)
    ]

    # Use common axis limits for the new prediction scatter plots.
    common_limit = 1.05 * max(
        selected["RUL"].max(),
        selected["clipped_predicted_RUL"].max(),
    )

    diagnostic_models = [
        (
        "Age-only",
        "Age-only Linear Regression",
        "age_only_test_diagnostics.png",
        ),
        (
            "Multi-sensor linear",
            "Multi-sensor Linear Regression",
            "multisensor_linear_test_diagnostics.png",
        ),
        (
            "Histogram gradient boosting",
            "Histogram Gradient Boosting",
            "hist_gradient_boosting_test_diagnostics.png",
        ),
    ]

    for model_name, title, filename in diagnostic_models:
        model_data = test_predictions.loc[
            test_predictions["model"] == model_name
        ]

        actual = model_data["RUL"].to_numpy(dtype=float)
        predicted = model_data[
            "clipped_predicted_RUL"
        ].to_numpy(dtype=float)
        errors = predicted - actual

        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

        axes[0].scatter(
            actual,
            predicted,
            color="steelblue",
            alpha=0.75,
            s=35,
        )
        axes[0].plot(
            [0, common_limit],
            [0, common_limit],
            "k--",
            label="Perfect prediction",
        )
        axes[0].set_xlim(0, common_limit)
        axes[0].set_ylim(0, common_limit)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("Actual RUL (cycles)")
        axes[0].set_ylabel("Predicted RUL (cycles)")
        axes[0].set_title(f"{title}\nOfficial Test Endpoints")
        axes[0].legend()
        axes[0].grid(alpha=0.2)

        axes[1].hist(
            errors,
            bins=15,
            color="steelblue",
            edgecolor="white",
        )
        axes[1].axvline(
            0,
            color="black",
            linestyle="--",
            label="Zero error",
        )
        axes[1].axvline(
            errors.mean(),
            color="darkorange",
            linewidth=2,
            label=f"Mean error: {errors.mean():+.2f} cycles",
        )
        axes[1].set_xlabel("Prediction - actual RUL (cycles)")
        axes[1].set_ylabel("Number of engines")
        axes[1].set_title("Signed Error Distribution")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.2)

        fig.tight_layout()
        figure_path = output_dir / filename
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        print("Figure saved to:", figure_path)


    # Calculate comparable metrics directly from the plotted predictions.
    from rul_project.rul_utils import evaluate_rul

    comparison_rows = []

    for model_name in model_names:
        model_data = test_predictions.loc[
            test_predictions["model"] == model_name
        ]

        scores = evaluate_rul(
            model_data["RUL"],
            model_data["clipped_predicted_RUL"],
        )

        comparison_rows.append(
            {
                "model": model_name,
                "mae": scores["mae"],
                "rmse": scores["rmse"],
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    display_labels = [
        "Age-only\nlinear",
        "Multi-sensor\nlinear",
        "Random\nforest",
        "Histogram\ngradient boosting",
    ]

    colors = ["gray", "steelblue", "seagreen", "darkorange"]

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

    for ax, metric in zip(axes, ["mae", "rmse"]):
        values = comparison[metric].to_numpy()

        bars = ax.bar(
            display_labels,
            values,
            color=colors,
            width=0.65,
        )

        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylabel(f"{metric.upper()} (cycles)")
        ax.set_title(f"Official Test {metric.upper()} — Lower Is Better")
        ax.set_ylim(0, max(values.max() * 1.2, 1))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Model Comparison: {len(reference_labels)} FD001 Test Endpoints"
    )
    fig.tight_layout()

    figure_path = output_dir / "official_test_model_comparison.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")

    print("Figure saved to:", figure_path)
    print("Figure saved to:", figure_path)
    plt.close("all")

if __name__ == "__main__":
    main()
