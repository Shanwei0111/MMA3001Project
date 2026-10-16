"""Endpoint metrics and paired engine-level bootstrap uncertainty."""
import numpy as np


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    """Return error metrics for finite, equally shaped nonempty RUL arrays.

    Parameters
    ----------
    y_true, prediction : array-like, shape (n_engines,)
        Actual and predicted remaining cycles. Predictions may be raw or clipped;
        the caller must identify which convention it is reporting.

    Returns
    -------
    dict
        MAE/RMSE/bias in cycles; summed and mean NASA score, overprediction rate.

    Raises
    ------
    ValueError
        Inputs are empty, nonfinite, mismatched, or targets are negative.
    OverflowError
        The asymmetric exponential score cannot be represented in float64.
    """
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or pred.shape != truth.shape:
        raise ValueError("Metrics require nonempty aligned one-dimensional arrays")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all() or np.any(truth < 0):
        raise ValueError("Metrics require finite inputs and nonnegative true RUL")
    error = pred - truth
    exponent = np.where(error < 0, -error/13.0, error/10.0)
    with np.errstate(over="raise", invalid="raise"):
        try:
            penalties = np.expm1(exponent)
            score = float(penalties.sum())
            rmse = float(np.sqrt(np.mean(error**2)))
        except FloatingPointError as exc:
            raise OverflowError("Unrepresentable error/score; inspect predictions") from exc
    return {"n": len(truth), "mae": float(np.abs(error).mean()), "rmse": rmse,
            "bias": float(error.mean()), "nasa_score": score, "nasa_mean": score/len(truth),
            "overprediction_fraction": float((error > 0).mean())}


def bootstrap_metrics(y: np.ndarray, a: np.ndarray, b: np.ndarray, *,
                      repetitions: int = 2000, seed: int = 3001) -> dict:
    """Bootstrap paired engine endpoints; differences are model A minus B.

    Returns percentile 95% intervals for each model's MAE/RMSE and their paired
    differences. This estimates sampling uncertainty conditional on fitted models,
    not an individual-engine prediction interval or retraining uncertainty.
    """
    y, a, b = map(lambda v: np.asarray(v, dtype=float), (y, a, b))
    evaluate(y, a)
    evaluate(y, b)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    index = np.random.default_rng(seed).integers(0, len(y), size=(repetitions, len(y)))
    ea, eb = (a-y)[index], (b-y)[index]
    values = {"a_mae": np.abs(ea).mean(axis=1), "b_mae": np.abs(eb).mean(axis=1),
              "a_rmse": np.sqrt((ea**2).mean(axis=1)), "b_rmse": np.sqrt((eb**2).mean(axis=1))}
    values["mae_difference"] = values["a_mae"]-values["b_mae"]
    values["rmse_difference"] = values["a_rmse"]-values["b_rmse"]
    return {name: np.quantile(v, [0.025, 0.975]).tolist() for name, v in values.items()}
