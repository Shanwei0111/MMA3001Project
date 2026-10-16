"""Causal sensor summaries with a direct reference and vectorized implementation."""
import numpy as np
import pandas as pd
from .data import SENSORS


def feature_names(trends: bool = True) -> list[str]:
    """Return the fixed predictor order: age, then current/mean/slope channels.

    Slopes have native sensor units per cycle. With trends=True there are 19
    predictors; otherwise seven. Engine identity is never a predictor.
    """
    return ["age"] + [f"{s}_{stat}" for stat in (["current", "mean", "slope"] if trends else ["current"]) for s in SENSORS]


def build_features(frame: pd.DataFrame, window: int = 30, *, reference: bool = False,
                   trends: bool = True) -> pd.DataFrame:
    """Compute within-engine trailing features for endpoints at cycle >=30.

    Parameters
    ----------
    frame : pandas.DataFrame
        Validated, contiguous observations in native units.
    window : int
        History length, between 2 and 30 inclusive.
    reference : bool
        Use transparent Python window iteration for independent optimization
        verification. The default uses strided NumPy views and einsum.
    trends : bool
        Include trailing means/slopes; False implements the current-only ablation.

    Returns
    -------
    pandas.DataFrame
        engine_id, cycle and ordered predictors. Empty if no supported endpoints.

    Raises
    ------
    ValueError
        Unsupported window or nonfinite/incorrectly ordered input.
    """
    if not isinstance(window, (int, np.integer)) or not 2 <= window <= 30:
        raise ValueError("window must be an integer from 2 to 30")
    parts = []
    positions = np.arange(window, dtype=float) - (window-1)/2
    denominator = float(positions @ positions)
    for engine_id, group in frame.groupby("engine_id", sort=True):
        values = group[SENSORS].to_numpy(dtype=float)
        cycles = group.cycle.to_numpy()
        if not np.isfinite(values).all() or np.any(np.diff(cycles) != 1):
            raise ValueError("Features require finite, ordered, contiguous observations")
        if len(values) < window:
            continue
        if reference:
            windows = [values[t-window+1:t+1] for t in range(window-1, len(values))]
            means = np.array([w.mean(axis=0) for w in windows])
            slopes = np.array([positions @ w / denominator for w in windows])
        else:
            # A view avoids copying every overlapping history. Centered positions
            # sum to zero, so no sensor centering is needed for the slope.
            windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
            means = windows.mean(axis=-1)
            slopes = np.einsum("npw,w->np", windows, positions, optimize=False) / denominator
        current = values[window-1:]
        ages = cycles[window-1:]
        x = np.column_stack([ages, current, means, slopes] if trends else [ages, current])
        part = pd.DataFrame(x, columns=feature_names(trends))
        part.insert(0, "cycle", ages.astype(int))
        part.insert(0, "engine_id", int(engine_id))
        # Common endpoint support for every candidate W; a history may start
        # after cycle 1, but cannot include an incomplete selected window.
        part = part.loc[part.cycle >= 30]
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["engine_id", "cycle"] + feature_names(trends))
    return pd.concat(parts, ignore_index=True)


def attach_targets(features: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Attach run-to-failure labels to feature rows without exposing them to X.

    Returns a copy with target in cycles. Only full training trajectories may
    provide the failure-cycle mapping.
    """
    result = features.copy()
    ends = training.groupby("engine_id").cycle.max()
    result["target"] = result.engine_id.map(ends) - result.cycle
    return result


def training_sample(features: pd.DataFrame, stride: int = 5) -> pd.DataFrame:
    """Take every stride-th supported window and each engine's final window.

    This computational optimization reduces redundant correlated training
    windows. Validation endpoints are never subsampled. Returns a sorted copy.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    keep = (features.cycle - 30) % stride == 0
    keep |= features.cycle == features.groupby("engine_id").cycle.transform("max")
    return features.loc[keep].copy().reset_index(drop=True)
