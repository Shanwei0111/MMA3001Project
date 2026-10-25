"""Define the ordered inputs and fixed model settings used throughout the project.

This is the only place where the final estimators are configured. Development,
official testing and performance measurements therefore cannot drift apart when
a setting is edited. The RF depth study changes depth explicitly and separately.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

MODEL_NAMES = (
    "Age-only",
    "Multi-sensor linear",
    "Random forest",
    "Histogram gradient boosting",
)
MULTISENSOR_FEATURES = (
    "cycle",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_11",
)
RANDOM_SEED = 42


def feature_columns(model_name: str) -> list[str]:
    """Return the predictor names in the exact fitted order.

    Args:
        model_name: One of MODEL_NAMES; IDs and target labels are never features.

    Returns:
        list[str]: Cycle alone for Age-only; otherwise the seven ordered inputs.

    Raises:
        ValueError: If model_name is unsupported.

    Example: feature_columns("Age-only") returns ["cycle"].
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    return ["cycle"] if model_name == "Age-only" else list(MULTISENSOR_FEATURES)


def make_random_forest(max_depth: int = 8) -> RandomForestRegressor:
    """Construct a fresh forest for a fixed fit or the depth sensitivity study.

    Args:
        max_depth: Maximum tree depth. Final fits use 8; the study tests 4, 8, 12.

    Returns:
        RandomForestRegressor: Unfitted 100-tree forest with minimum leaf size 10,
        bootstrap enabled, all features eligible, seed 42 and one worker.

    Raises:
        ValueError: During fitting if scikit-learn rejects a supplied depth.

    Example: make_random_forest().n_estimators == 100.
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def make_model(model_name: str):
    """Construct an unfitted estimator from the shared final configuration.

    Args:
        model_name: One of the four case-sensitive names in MODEL_NAMES.

    Returns:
        sklearn.base.RegressorMixin: Fresh LinearRegression, forest or HGB model.

    Raises:
        ValueError: If model_name is unsupported.

    Example: make_model("Multi-sensor linear").fit(X_train, y_train).
    """
    feature_columns(model_name)  # Reject misspelled names before creating a model.
    if model_name in MODEL_NAMES[:2]:
        return LinearRegression()
    if model_name == "Random forest":
        return make_random_forest()
    return make_hist_gradient_boosting()


def make_hist_gradient_boosting():
    """Create an unfitted HGB candidate with the fixed project configuration.

    Returns:
        sklearn.ensemble.HistGradientBoostingRegressor: Squared-error model with
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=20, l2_regularization=1.0, max_bins=255 (library default),
        early_stopping=False and random_state=42.

    Examples:
        >>> make_hist_gradient_boosting().max_iter
        200

    No arguments are accepted. Early stopping is disabled so grouped validation
    controls evaluation instead of an automatic random row-level holdout.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
