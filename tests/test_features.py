import numpy as np
import pytest
from jet_rul.features import build_features, attach_targets, training_sample
from jet_rul.data import SENSORS


@pytest.mark.parametrize("window", [10, 20, 30])
def test_exact_linear_slopes_and_means(trajectories, window):
    result = build_features(trajectories, window)
    first = result.iloc[0]
    assert first.cycle == 30
    for j, sensor in enumerate(SENSORS):
        assert first[f"{sensor}_slope"] == pytest.approx(j+1)
        assert first[f"{sensor}_mean"] == pytest.approx(200+(j+1)*(30-(window-1)/2))


def test_constant_sensor_has_zero_slope(trajectories):
    trajectories["s3"] = 400.
    result = build_features(trajectories)
    np.testing.assert_allclose(result.s3_slope, 0, atol=1e-12)


def test_vectorization_matches_reference(trajectories):
    a = build_features(trajectories, reference=True)
    b = build_features(trajectories)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_future_mutation_cannot_change_past(trajectories):
    before = build_features(trajectories)
    trajectories.loc[trajectories.cycle > 35, SENSORS] += 99999
    after = build_features(trajectories)
    np.testing.assert_allclose(before.loc[before.cycle <= 35], after.loc[after.cycle <= 35])


def test_engine_isolation(trajectories):
    combined = build_features(trajectories)
    separate = build_features(trajectories.loc[trajectories.engine_id == 2])
    np.testing.assert_allclose(combined.loc[combined.engine_id == 2], separate)


def test_sampling_retains_failure_and_labels(trajectories):
    sample = training_sample(attach_targets(build_features(trajectories), trajectories), stride=7)
    assert sample.groupby("engine_id").cycle.max().tolist() == [50, 50]
    assert sample.loc[sample.cycle == 50, "target"].tolist() == [0, 0]


@pytest.mark.parametrize("window", [1, 31, 2.5])
def test_invalid_window(trajectories, window):
    with pytest.raises(ValueError):
        build_features(trajectories, window)
