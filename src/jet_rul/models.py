"""Comparable regressors, model persistence and supported prediction contracts."""
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import SENSORS, validate_frame
from .features import build_features, feature_names


def engine_weights(frame: pd.DataFrame) -> np.ndarray:
    """Return mean-one sample weights with equal total weight per engine."""
    weights = 1.0/frame.groupby("engine_id").engine_id.transform("size").to_numpy()
    return weights/weights.mean()


def fit_regressor(frame: pd.DataFrame, spec: dict, *, seed: int = 3001,
                  trends: bool = True):
    """Fit ridge/age-only ridge or a forest using engine-balanced row weights.

    Parameters
    ----------
    frame : pandas.DataFrame
        Feature rows plus target in cycles. Contains only fit-partition engines.
    spec : dict
        kind ('ridge', 'age', 'forest') and relevant hyperparameters.
    seed : int
        Forest bootstrap/feature randomization seed.
    trends : bool
        Select 19 full features or seven current-only features.

    Returns
    -------
    estimator, list[str]
        Fitted scikit-learn estimator and ordered predictor names.
    """
    names = ["age"] if spec["kind"] == "age" else feature_names(trends)
    x, y = frame[names].to_numpy(float), frame.target.to_numpy(float)
    weights = engine_weights(frame)
    if spec["kind"] in ("ridge", "age"):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"], solver="svd"))
        # Scaling is deliberately unweighted and fitted only on this partition.
        estimator.fit(x, y, ridge__sample_weight=weights)
    elif spec["kind"] == "forest":
        estimator = RandomForestRegressor(n_estimators=spec["trees"], max_depth=spec["depth"],
                                          min_samples_leaf=spec["leaf"], max_features=spec["features"],
                                          random_state=seed, n_jobs=1, bootstrap=True)
        estimator.fit(x, y, sample_weight=weights)
    else:
        raise ValueError(f"Unknown estimator kind: {spec['kind']}")
    return estimator, names


@dataclass
class RULModel:
    """Persisted estimator and its input contract.

    Attributes
    ----------
    estimator : object
        Fitted scikit-learn estimator; never load untrusted joblib artifacts.
    names : list[str]
        Ordered predictor names.
    window : int
        Trailing history length in cycles.
    bounds : dict
        Training-only observed primary-channel minima and maxima.
    version : str
        Configuration fingerprint linking to the run record.
    """
    estimator: object
    names: list[str]
    window: int
    bounds: dict
    version: str

    def predict(self, observations: pd.DataFrame, *, subset: str = "FD001") -> dict:
        """Predict one engine endpoint and return RUL, traceability and flags.

        Reject unsupported subsets, multiple engines, fewer than 30 contiguous
        samples or invalid observations. Slight empirical-envelope excursions
        produce warnings rather than false physical-invalidity claims.
        """
        if subset != "FD001":
            raise ValueError("This model supports only the declared FD001 regime")
        frame = validate_frame(observations)
        if frame.engine_id.nunique() != 1 or len(frame) < 30:
            raise ValueError("One engine with at least 30 contiguous observations is required")
        feat = build_features(frame, self.window)
        if feat.empty:
            raise ValueError("No supported prediction endpoint")
        raw = float(self.estimator.predict(feat[self.names].tail(1).to_numpy(float))[0])
        if not np.isfinite(raw):
            raise ValueError("Model returned a nonfinite prediction")
        flags = []
        for sensor in SENSORS:
            low, high = self.bounds[sensor]
            if ((frame[sensor].tail(self.window) < low) | (frame[sensor].tail(self.window) > high)).any():
                flags.append(f"{sensor}_outside_training_envelope")
        if raw < 0:
            flags.append("negative_prediction_clipped")
        return {"engine_id": int(frame.engine_id.iloc[-1]), "cycle": int(frame.cycle.iloc[-1]),
                "raw_rul_prediction": raw, "predicted_rul_cycles": max(0., raw),
                "input_status": "warning" if flags else "ok", "flags": flags,
                "model_version": self.version}

    def save(self, path: str | Path) -> None:
        """Save the fitted bundle to a local joblib path, creating its parent."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)


def load_model(path: str | Path) -> RULModel:
    """Load a trusted local model, rejecting objects with an incorrect type.

    Joblib uses pickle and is suitable only for artifacts whose origin is trusted.
    """
    model = joblib.load(path)
    if not isinstance(model, RULModel):
        raise ValueError("Unexpected model artifact type")
    return model
