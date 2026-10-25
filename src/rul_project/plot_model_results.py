"""Draw validation and test diagnostics using saved predictions, without refitting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul_project.model_configuration import MODEL_NAMES


def plot_age_baseline(results: pd.DataFrame, destination: Path) -> None:
    """Draw actual training-history RUL against raw and clipped age predictions.

    Args:
        results: Age-only validation table with cycle, RUL and prediction columns.
        destination: PNG path whose parent already exists.

    Returns:
        None: Saves the figure and closes its Matplotlib resources.

    Raises:
        KeyError: If required prediction columns are missing.
        OSError: If the figure cannot be written.

    Example: plot_age_baseline(results, Path("results/generated/age_only_baseline.png")).
    """
    line = results.sort_values("cycle")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        results["cycle"],
        results["RUL"],
        color="gray",
        alpha=0.2,
        s=10,
        label="Actual RUL",
    )
    # Age is the sole predictor, so equal-cycle predictions lie on the same line.
    ax.plot(
        line["cycle"],
        line["raw_predicted_RUL"],
        "b--",
        label="Raw age-only predictions",
    )
    ax.plot(
        line["cycle"],
        line["clipped_predicted_RUL"],
        color="red",
        label="Nonnegative age-only predictions",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Cycle",
        ylabel="RUL (cycles)",
        title="Age-only Baseline on Validation Engines",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=250)
    plt.close(fig)


def plot_endpoint_diagnostics(
    results: pd.DataFrame, destination: Path, model_names: tuple[str, ...]
) -> None:
    """Plot predictions and signed errors with common axes across compared models.

    Args:
        results: Combined official endpoint table with IDs, model, RUL and predictions.
        destination: PNG path for the complete diagnostic grid.
        model_names: Ordered model names; one row of two panels is drawn per model.

    Returns:
        None: Saves and closes the diagnostic figure.

    Raises:
        ValueError: If a requested model has no rows or duplicate engine IDs.
        OSError: If the figure cannot be written.

    Example: plot_endpoint_diagnostics(data, path, ("Random forest", "Histogram gradient boosting")).
    """
    fig, axes = plt.subplots(
        len(model_names),
        2,
        figsize=(10, 2.8 * len(model_names)),
        squeeze=False,
        layout="constrained",
    )
    colours = {"Random forest": "#386e9f", "Histogram gradient boosting": "#a95221"}
    for row, name in zip(axes, model_names):
        data = results.loc[results["model"] == name]
        if data.empty or data.engine_id.duplicated().any():
            plt.close(fig)
            raise ValueError(f"Expected one endpoint per engine for {name}.")
        actual, predicted = (
            data["RUL"].to_numpy(),
            data["clipped_predicted_RUL"].to_numpy(),
        )
        errors = predicted - actual
        colour = colours.get(name, "#386e9f")
        title = "HGB" if name == "Histogram gradient boosting" else name
        row[0].scatter(actual, predicted, color=colour, alpha=0.75, s=16)
        row[0].plot([0, 200], [0, 200], "k--", lw=1, label="Perfect prediction")
        row[0].set(
            xlim=(0, 200),
            ylim=(0, 200),
            xlabel="Actual RUL (cycles)",
            ylabel="Predicted RUL (cycles)",
            title=f"{title}: Test endpoints",
        )
        # Identical bins and limits make optimistic error tails directly comparable.
        row[1].hist(
            errors, bins=np.linspace(-50, 90, 15), color=colour, edgecolor="white"
        )
        row[1].axvline(0, color="black", ls="--", lw=1, label="Zero error")
        row[1].axvline(
            errors.mean(),
            color="#b01d2c",
            lw=1.2,
            label=f"Mean: {errors.mean():+.2f} cycles",
        )
        row[1].set(
            xlim=(-50, 90),
            ylim=(0, 25),
            xlabel="Prediction - actual RUL (cycles)",
            ylabel="Number of engines",
            title=f"{title}: Signed errors",
        )
        for ax in row:
            ax.legend(fontsize=9)
            ax.grid(alpha=0.18)
            ax.set_axisbelow(True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, destination: Path) -> None:
    """Compare official test MAE and RMSE using the fixed four-model order.

    Args:
        metrics: Table containing model, mae and rmse in cycles.
        destination: PNG output path.

    Returns:
        None: Saves a labelled two-panel bar chart and closes it.

    Raises:
        KeyError: If a required model or metric is absent.
        OSError: If the figure cannot be written.

    Example: plot_metric_comparison(metrics, Path("results/generated/official_test_model_comparison.png")).
    """
    ordered = metrics.set_index("model").loc[list(MODEL_NAMES)]
    labels = [
        "Age-only\nlinear",
        "Multi-sensor\nlinear",
        "Random\nforest",
        "Histogram gradient\nboosting",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric in zip(axes, ("mae", "rmse")):
        bars = ax.bar(
            labels, ordered[metric], color=["#888888", "#386e9f", "#367f53", "#a95221"]
        )
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(
            title=f"Official test {metric.upper()} - lower is better",
            ylabel=f"{metric.upper()} (cycles)",
            ylim=(0, ordered[metric].max() * 1.18),
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def generate_result_figures(output_dir: Path) -> None:
    """Create the report and per-model diagnostics from current run CSVs.

    Args:
        output_dir: Directory containing official metrics/predictions and age baseline.

    Returns:
        None: Writes age baseline, four model diagnostics, combined RF/HGB diagnostics
        and the four-model MAE/RMSE chart.

    Raises:
        FileNotFoundError: If prerequisite CSVs are absent.
        ValueError: If a required model has missing/duplicated engine predictions.
        OSError: If any figure cannot be saved.

    Example: generate_result_figures(Path("results/generated")). Uses Times New Roman when available.
    """
    predictions = pd.read_csv(output_dir / "official_test_predictions.csv")
    metrics = pd.read_csv(output_dir / "official_test_extended_metrics.csv")
    # A local style context avoids changing a caller's global Matplotlib settings.
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 11}):
        plot_age_baseline(
            pd.read_csv(output_dir / "baseline_validation_predictions.csv"),
            output_dir / "age_only_baseline.png",
        )
        for name, stem in [
            ("Age-only", "age_only"),
            ("Multi-sensor linear", "multisensor_linear"),
            ("Random forest", "random_forest"),
            ("Histogram gradient boosting", "hist_gradient_boosting"),
        ]:
            plot_endpoint_diagnostics(
                predictions, output_dir / f"{stem}_test_diagnostics.png", (name,)
            )
        plot_endpoint_diagnostics(
            predictions,
            output_dir / "rf_hgb_test_diagnostics.png",
            ("Random forest", "Histogram gradient boosting"),
        )
        plot_metric_comparison(
            metrics, output_dir / "official_test_model_comparison.png"
        )
