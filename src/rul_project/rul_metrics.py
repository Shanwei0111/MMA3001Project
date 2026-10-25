"""Calculate regression errors and the asymmetric 13/10 RUL penalty.

All errors use prediction minus truth. Metric evaluation does not clip inputs;
the caller explicitly chooses raw or delivered nonnegative predictions.
"""

import numpy as np


def evaluate_rul(y_true, y_pred) -> dict[str, float]:
    """Evaluate aligned RUL observations using unrounded supplied predictions.

    Args:
        y_true (array-like): Finite nonnegative one-dimensional actual RUL,
            measured in cycles. At least one observation is required.
        y_pred (array-like): Finite one-dimensional predictions in cycles, in
            the same order and with the same length as y_true. Raw negative
            predictions are allowed; this function never clips its inputs.

    Returns:
        dict[str, float]: mae, rmse and bias in cycles; overprediction_pct as a
        percentage; nasa_score as the sum of dimensionless asymmetric penalties.
        Error is prediction minus truth; under/overprediction divisors are 13/10.

    Raises:
        ValueError: If arrays are empty, nonfinite, not aligned one-dimensional
            arrays or actual RUL contains a negative value.
        FloatingPointError: If squared errors or exponential penalties overflow.

    Examples:
        >>> evaluate_rul([10, 20], [10, 20])["rmse"]
        0.0
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Inputs must not be empty.")

    if actual.shape != predicted.shape:
        raise ValueError("Inputs must have equal lengths.")

    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Inputs must contain only finite values.")

    if (actual < 0).any():
        raise ValueError("True RUL must be nonnegative.")

    with np.errstate(over="raise", invalid="raise"):
        errors = predicted - actual

        scaled_errors = np.where(
            errors < 0,
            -errors / 13.0,
            errors / 10.0,
        )

        return {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(errors.mean()),
            "overprediction_pct": float((errors > 0).mean() * 100),
            "nasa_score": float(np.expm1(scaled_errors).sum()),
        }
