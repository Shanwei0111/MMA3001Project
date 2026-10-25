"""Refit the four fixed candidates and evaluate one official endpoint per engine."""

from pathlib import Path

import pandas as pd

from rul_project.model_configuration import MODEL_NAMES, feature_columns, make_model
from rul_project.prepare_dataset import load_test_endpoints
from rul_project.rul_metrics import evaluate_rul


def fit_and_predict_endpoints(
    train: pd.DataFrame, endpoints: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Fit a fresh candidate on complete training data and predict test endpoints.

    Args:
        train: Complete labelled run-to-failure training table.
        endpoints: One labelled official test endpoint per engine, in ID order.
        model_name: Supported case-sensitive model name.

    Returns:
        pandas.DataFrame: ID, cycle, actual RUL, model name, raw/clipped predictions
        and signed errors. Test labels are used only after prediction.

    Raises:
        ValueError: If endpoints are duplicated/empty or estimator data is invalid.

    Example: fit_and_predict_endpoints(train, endpoints, "Random forest").
    """
    if train.empty or endpoints.empty or endpoints["engine_id"].duplicated().any():
        raise ValueError("Expected training data and one endpoint per test engine.")
    features = feature_columns(model_name)
    model = make_model(model_name)
    # Fit receives training labels only; official RUL never influences the estimator.
    model.fit(train[features], train["RUL"])
    raw = model.predict(endpoints[features])
    result = endpoints[["engine_id", "cycle", "RUL"]].copy()
    result["model"] = model_name
    result["raw_predicted_RUL"] = raw
    result["raw_error"] = raw - result["RUL"].to_numpy()
    result["clipped_predicted_RUL"] = raw.clip(0, None)
    result["clipped_error"] = result["clipped_predicted_RUL"] - result["RUL"]
    return result


def summarize_test_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate raw/clipped errors and the supplementary asymmetric metrics.

    Args:
        predictions: Combined model/engine endpoint predictions from final fitting.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Original metric table and the
        extended MAE/RMSE/bias/overprediction/NASA-score table.

    Raises:
        ValueError: If metric inputs are empty, nonfinite, misaligned or invalid.
        FloatingPointError: If an asymmetric penalty cannot be represented.

    Example: metrics, extended = summarize_test_predictions(predictions).
    """
    basic, extended = [], []
    for name, result in predictions.groupby("model", sort=False):
        raw = evaluate_rul(result["RUL"], result["raw_predicted_RUL"])
        delivered = evaluate_rul(result["RUL"], result["clipped_predicted_RUL"])
        basic.append(
            {
                "model": name,
                "n_features": len(feature_columns(name)),
                "raw_mae": raw["mae"],
                "raw_rmse": raw["rmse"],
                "clipped_mae": delivered["mae"],
                "clipped_rmse": delivered["rmse"],
                "negative_raw_predictions": int(
                    (result["raw_predicted_RUL"] < 0).sum()
                ),
            }
        )
        extended.append({"model": name, "test_engines": len(result), **delivered})
    return pd.DataFrame(basic), pd.DataFrame(extended)


def evaluate_official_test(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Refit all fixed models and save official endpoint metrics and failure cases.

    Args:
        data_dir: Raw NASA directory containing censored test histories and labels.
        output_dir: Directory containing full training labels and receiving test results.

    Returns:
        pandas.DataFrame: All 400 predictions for the standard 100-engine FD001 test.

    Raises:
        FileNotFoundError: If raw files or prepared training labels are absent.
        ValueError: If label alignment or estimator inputs are invalid.
        OSError: If results cannot be written.

    Example: evaluate_official_test(Path("data/raw"), Path("results/generated")).
    """
    train = pd.read_csv(output_dir / "train_FD001_labeled.csv")
    endpoints = load_test_endpoints(data_dir)
    endpoints.to_csv(output_dir / "test_endpoints_labeled.csv", index=False)
    tables = []
    for name in MODEL_NAMES:
        tables.append(fit_and_predict_endpoints(train, endpoints, name))
        print(f"Completed official evaluation: {name}")
    predictions = pd.concat(tables, ignore_index=True)
    metrics, extended = summarize_test_predictions(predictions)
    predictions.to_csv(output_dir / "official_test_predictions.csv", index=False)
    metrics.to_csv(output_dir / "official_test_metrics.csv", index=False)
    extended.to_csv(output_dir / "official_test_extended_metrics.csv", index=False)
    # Keep the largest errors visible; do not discard engines based on test performance.
    for name, stem in [
        ("Random forest", "random_forest"),
        ("Histogram gradient boosting", "hist_gradient_boosting"),
    ]:
        cases = predictions.loc[predictions["model"] == name].copy()
        cases["absolute_error"] = cases["clipped_error"].abs()
        cases.to_csv(output_dir / f"{stem}_test_error_analysis.csv", index=False)
    print(extended.round(2).to_string(index=False))
    return predictions
