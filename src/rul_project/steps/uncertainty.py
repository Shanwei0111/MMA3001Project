"""Reproduce RF/HGB paired uncertainty and the report comparison figures.

Reads official_test_predictions.csv and bootstrap_summary.csv after the official
evaluation stage. Uses the same 2,000 paired endpoint samples and seed 42 as the
report; model fitting and test-label-based bias corrections are not performed.
"""
from rul_project.settings import OUTPUT_DIR


def main() -> None:
    """Extend uncertainty analysis to HGB using existing predictions.

    Returns:
        None: Writes extended_bootstrap_summary.csv, rf_hgb_test_diagnostics.png
        and official_test_model_comparison.png in OUTPUT_DIR.

    Raises:
        FileNotFoundError: If official predictions/original bootstrap are missing.
        AssertionError: If model endpoints are misaligned or original intervals
            are not reproduced by the paired resampling calculation.
        ValueError: If CSV values or plotting data are invalid.
        OSError: If figures or CSV results cannot be written.

    No arguments are accepted. Run ``python -m rul_project uncertainty`` after
    testevaluation. Bounds describe endpoint sampling for fixed fitted models.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import findfont, FontProperties
    OUT = OUTPUT_DIR
    SOURCE = OUTPUT_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE / "official_test_predictions.csv")
    names = ["Multi-sensor linear", "Random forest", "Histogram gradient boosting"]
    groups = {name: data.loc[data.model == name].sort_values("engine_id") for name in names}
    reference = groups[names[0]]
    assert len(reference) == 100 and reference.engine_id.is_unique
    for frame in groups.values():
        assert len(frame) == 100 and frame.engine_id.is_unique
        np.testing.assert_array_equal(frame.engine_id, reference.engine_id)
        np.testing.assert_array_equal(frame.RUL, reference.RUL)
    indices = np.random.default_rng(42).integers(0, 100, size=(2000, 100))
    samples, estimates = {}, {}
    for name, frame in groups.items():
        error = frame.clipped_predicted_RUL.to_numpy() - frame.RUL.to_numpy()
        samples[name] = (np.abs(error[indices]).mean(axis=1), np.sqrt((error[indices]**2).mean(axis=1)))
        estimates[name] = (np.abs(error).mean(), np.sqrt((error**2).mean()))

    rows = []
    comparisons = [
        ("RF", "Random forest", None),
        ("HGB", "Histogram gradient boosting", None),
        ("RF - multi-sensor LR", "Random forest", "Multi-sensor linear"),
        ("HGB - multi-sensor LR", "Histogram gradient boosting", "Multi-sensor linear"),
        ("HGB - RF", "Histogram gradient boosting", "Random forest"),
    ]
    for label, name, baseline in comparisons:
        row = {"comparison": label}
        for k, metric in enumerate(["mae", "rmse"]):
            estimate = estimates[name][k] - (estimates[baseline][k] if baseline else 0)
            dist = samples[name][k] - (samples[baseline][k] if baseline else 0)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            row.update({metric: estimate, metric+"_lower": lo, metric+"_upper": hi})
        rows.append(row)
    summary = pd.DataFrame(rows)
    # The extension must reproduce the four original RF bootstrap quantities exactly.
    old = pd.read_csv(SOURCE / "bootstrap_summary.csv")
    for old_row, new_row, metric in [(0, 0, "mae"), (1, 0, "rmse"), (2, 2, "mae"), (3, 2, "rmse")]:
        np.testing.assert_allclose(old.iloc[old_row][["estimate", "ci_lower", "ci_upper"]].to_numpy(dtype=float),
            summary.iloc[new_row][[metric, metric+"_lower", metric+"_upper"]].to_numpy(dtype=float), atol=1e-10)
    summary.to_csv(OUT / "extended_bootstrap_summary.csv", index=False)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    print("Figure font:", findfont(FontProperties(family="Times New Roman")))
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.4), layout="constrained")
    bins = np.linspace(-50, 90, 15)
    for axrow, name, short, colour in zip(axes, names[1:], ["Random forest", "HGB"], ["#386e9f", "#a95221"]):
        frame = groups[name]
        truth, pred = frame.RUL.to_numpy(), frame.clipped_predicted_RUL.to_numpy()
        error = pred - truth
        axrow[0].scatter(truth, pred, s=14, alpha=.72, color=colour)
        axrow[0].plot([0, 200], [0, 200], "k--", lw=1, label="Perfect prediction")
        axrow[0].set(xlim=(0, 200), ylim=(0, 200), xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)", title=short + ": Test endpoints")
        axrow[0].legend(loc="upper left", fontsize=10)
        axrow[1].hist(error, bins=bins, color=colour, alpha=.85, edgecolor="white")
        axrow[1].axvline(0, color="black", ls="--", lw=1, label="Zero error")
        axrow[1].axvline(error.mean(), color="#b01d2c", lw=1.2, label=f"Mean: +{error.mean():.2f} cycles")
        axrow[1].set(xlim=(-50, 90), ylim=(0, 25), xlabel="Prediction - actual RUL (cycles)", ylabel="Number of engines", title=short + ": Signed errors")
        axrow[1].legend(loc="upper right", fontsize=10)
        for ax in axrow:
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
    fig.savefig(OUT / "rf_hgb_test_diagnostics.png", dpi=300)
    plt.close(fig)

    metrics = pd.read_csv(SOURCE / "official_test_extended_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = ["Age-only\nlinear", "Multi-sensor\nlinear", "Random\nforest", "Histogram gradient\nboosting"]
    for ax, metric in zip(axes, ["mae", "rmse"]):
        bars = ax.bar(labels, metrics[metric], color=["#888888", "#386e9f", "#367f53", "#a95221"])
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(title=f"Official test {metric.upper()} - lower is better", ylabel=f"{metric.upper()} (cycles)", ylim=(0, metrics[metric].max()*1.18))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "official_test_model_comparison.png", dpi=300)
    plt.close(fig)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
