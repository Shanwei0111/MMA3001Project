"""Estimate paired engine-sampling uncertainty from saved endpoint predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

COMPARISONS = (
    ("RF", "Random forest", None),
    ("HGB", "Histogram gradient boosting", None),
    ("RF - multi-sensor LR", "Random forest", "Multi-sensor linear"),
    ("HGB - multi-sensor LR", "Histogram gradient boosting", "Multi-sensor linear"),
    ("HGB - RF", "Histogram gradient boosting", "Random forest"),
)


def paired_bootstrap(
    predictions: pd.DataFrame, repetitions: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Calculate point estimates and percentile intervals using paired engine IDs.

    Args:
        predictions: Table with model, engine_id, RUL and clipped_predicted_RUL.
            Every compared model must have one row for each identical test engine.
        repetitions: Number of bootstrap replicates; positive integer, default 2000.
        seed: Seed for NumPy's generator, default 42.

    Returns:
        pandas.DataFrame: Five comparisons with MAE/RMSE estimates and 95% bounds,
        all measured in cycles. Results condition on the already fitted models.

    Raises:
        ValueError: If groups are empty/duplicated/misaligned, values are invalid
            or repetitions is not a positive integer.

    Example: paired_bootstrap(pd.read_csv("results/generated/official_test_predictions.csv")).
    """
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, (int, np.integer))
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer.")
    names = ("Multi-sensor linear", "Random forest", "Histogram gradient boosting")
    groups = {
        name: predictions.loc[predictions["model"] == name].sort_values("engine_id")
        for name in names
    }
    reference = groups[names[0]]
    if reference.empty:
        raise ValueError("No aligned model endpoints were supplied.")
    for frame in groups.values():
        if (
            frame.engine_id.duplicated().any()
            or frame.engine_id.tolist() != reference.engine_id.tolist()
        ):
            raise ValueError("Expected identical unique test engines for each model.")
        if not np.array_equal(frame["RUL"].to_numpy(), reference["RUL"].to_numpy()):
            raise ValueError("Actual RUL labels disagree between models.")
        values = frame[["RUL", "clipped_predicted_RUL"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, 0] < 0).any():
            raise ValueError(
                "Bootstrap values must be finite with nonnegative true RUL."
            )

    # One index matrix is reused for every model: differences stay engine-paired.
    indices = np.random.default_rng(seed).integers(
        0, len(reference), size=(repetitions, len(reference))
    )
    estimates, samples = {}, {}
    for name, frame in groups.items():
        errors = frame["clipped_predicted_RUL"].to_numpy() - frame["RUL"].to_numpy()
        estimates[name] = (np.abs(errors).mean(), np.sqrt((errors**2).mean()))
        samples[name] = (
            np.abs(errors[indices]).mean(axis=1),
            np.sqrt((errors[indices] ** 2).mean(axis=1)),
        )
    rows = []
    for label, name, baseline in COMPARISONS:
        row = {"comparison": label}
        for index, metric in enumerate(("mae", "rmse")):
            estimate = estimates[name][index] - (
                estimates[baseline][index] if baseline else 0
            )
            distribution = samples[name][index] - (
                samples[baseline][index] if baseline else 0
            )
            lower, upper = np.percentile(distribution, [2.5, 97.5])
            row.update(
                {metric: estimate, metric + "_lower": lower, metric + "_upper": upper}
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_uncertainty_analysis(output_dir: Path) -> pd.DataFrame:
    """Write both original and extended bootstrap tables from saved predictions.

    Args:
        output_dir: Directory containing official_test_predictions.csv and
            receiving bootstrap_summary.csv and extended_bootstrap_summary.csv.

    Returns:
        pandas.DataFrame: Extended five-comparison bootstrap summary.

    Raises:
        FileNotFoundError: If official evaluation has not been run.
        ValueError: If engine predictions are invalid or misaligned.
        OSError: If output tables cannot be written.

    Example: save_uncertainty_analysis(Path("results/generated")). No model is refitted.
    """
    summary = paired_bootstrap(
        pd.read_csv(output_dir / "official_test_predictions.csv")
    )
    summary.to_csv(output_dir / "extended_bootstrap_summary.csv", index=False)
    original_rows = []
    for label, row_index, metric in [
        ("Forest MAE", 0, "mae"),
        ("Forest RMSE", 0, "rmse"),
        ("MAE difference: forest - linear", 2, "mae"),
        ("RMSE difference: forest - linear", 2, "rmse"),
    ]:
        row = summary.iloc[row_index]
        original_rows.append(
            {
                "metric": label,
                "estimate": row[metric],
                "ci_lower": row[metric + "_lower"],
                "ci_upper": row[metric + "_upper"],
            }
        )
    pd.DataFrame(original_rows).to_csv(
        output_dir / "bootstrap_summary.csv", index=False
    )
    print(summary.round(3).to_string(index=False))
    return summary
