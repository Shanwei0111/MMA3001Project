import numpy as np
import pytest
from jet_rul.metrics import evaluate, bootstrap_metrics
from jet_rul.features import build_features, attach_targets
from jet_rul.models import fit_regressor, engine_weights, RULModel, load_model
from jet_rul.data import SENSORS
from jet_rul.experiment import make_partitions


def test_metric_known_values_and_penalty_direction():
    result = evaluate([10, 20], [13, 16])
    assert result["mae"] == 3.5
    assert result["rmse"] == pytest.approx(np.sqrt(12.5))
    assert result["bias"] == -0.5
    assert evaluate([20], [30])["nasa_score"] > evaluate([20], [10])["nasa_score"]
    assert evaluate([20], [20])["nasa_score"] == 0


@pytest.mark.parametrize("truth,pred", [([], []), ([1], [1, 2]), ([1], [np.nan]), ([-1], [1])])
def test_bad_metrics(truth, pred):
    with pytest.raises(ValueError):
        evaluate(truth, pred)


def test_score_overflow_is_explicit():
    with pytest.raises(OverflowError):
        evaluate([1], [1e9])


def test_bootstrap_paired_identity():
    result = bootstrap_metrics(np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([1, 3, 5]), repetitions=100)
    assert result["mae_difference"] == [0, 0]
    assert result["rmse_difference"] == [0, 0]


def test_balanced_engine_weights(trajectories):
    frame = trajectories.loc[(trajectories.engine_id == 1) | (trajectories.cycle <= 20)].copy()
    weights = engine_weights(frame)
    assert weights[frame.engine_id == 1].sum() == pytest.approx(weights[frame.engine_id == 2].sum())


def test_ridge_collinearity_persistence_and_history(trajectories, tmp_path):
    features = attach_targets(build_features(trajectories), trajectories)
    estimator, names = fit_regressor(features, {"kind": "ridge", "alpha": 1.})
    bounds = {s: [0., 10000.] for s in SENSORS}
    bundle = RULModel(estimator, names, 30, bounds, "test")
    history = trajectories.loc[trajectories.engine_id == 1].iloc[:30]
    prediction = bundle.predict(history)
    assert np.isfinite(prediction["raw_rul_prediction"])
    with pytest.raises(ValueError, match="30"):
        bundle.predict(history.iloc[:29])
    with pytest.raises(ValueError, match="FD001"):
        bundle.predict(history, subset="FD002")
    path = tmp_path/"model.joblib"
    bundle.save(path)
    assert load_model(path).predict(history) == prediction


def test_scaler_uses_fit_rows_only(trajectories):
    features = attach_targets(build_features(trajectories), trajectories)
    train = features.loc[features.engine_id == 1].copy()
    model, names = fit_regressor(train, {"kind": "ridge", "alpha": 1.})
    before = model.named_steps["standardscaler"].mean_.copy()
    held = features.loc[features.engine_id == 2].copy()
    held[names] += 1e6
    model.predict(held[names].to_numpy(float))
    np.testing.assert_array_equal(model.named_steps["standardscaler"].mean_, before)


def test_partitions_are_reproducible_and_disjoint(trajectories):
    import pandas as pd
    frames = []
    for i in range(1, 13):
        group = trajectories.loc[trajectories.engine_id == 1].copy()
        group.engine_id = i
        frames.append(group)
    frame = pd.concat(frames)
    config = {"seed": 3001, "holdout_engines": 3, "cv_folds": 3}
    splits = make_partitions(frame, config)
    assert splits == make_partitions(frame, config)
    assert set(splits["development"]).isdisjoint(splits["holdout"])
    for fold in splits["folds"]:
        assert set(fold["train"]).isdisjoint(fold["validation"])
        assert set(fold["train"] + fold["validation"]) == set(splits["development"])


def test_negative_raw_prediction_is_retained(trajectories):
    class Negative:
        def predict(self, x):
            return np.full(len(x), -7.)
    model = RULModel(Negative(), ["age"], 30, {s: [0., 1e4] for s in SENSORS}, "negative")
    result = model.predict(trajectories.loc[trajectories.engine_id == 1])
    assert result["raw_rul_prediction"] == -7.
    assert result["predicted_rul_cycles"] == 0.
    assert "negative_prediction_clipped" in result["flags"]
