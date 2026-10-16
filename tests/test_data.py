import numpy as np
import pytest
from jet_rul.data import read_cmapss, validate_frame, training_labels, read_test_labels, check_manifest, sha256


def test_whitespace_parser(trajectories, tmp_path):
    path = tmp_path/"source.txt"
    np.savetxt(path, trajectories.to_numpy(), delimiter="   ")
    result = read_cmapss(path)
    assert result.shape == (100, 26)
    assert result.engine_id.dtype.kind == "i"


def test_wrong_width(tmp_path):
    path = tmp_path/"bad.txt"
    np.savetxt(path, np.ones((3, 25)))
    with pytest.raises(ValueError, match="26"):
        read_cmapss(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(trajectories, bad_value):
    trajectories = trajectories.astype(float)
    trajectories.loc[0, "s2"] = bad_value
    with pytest.raises(ValueError, match="nonfinite"):
        validate_frame(trajectories)


@pytest.mark.parametrize("field,value", [("engine_id", 1.5), ("cycle", 0), ("s2", -5), ("s7", 0)])
def test_invalid_domains(trajectories, field, value):
    trajectories = trajectories.astype(float)
    trajectories.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_frame(trajectories)


def test_gaps_duplicates_and_sorting(trajectories):
    with pytest.raises(ValueError, match="gaps"):
        validate_frame(trajectories.drop(index=10))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_frame(trajectories.loc[list(range(100))+[0]])
    result = validate_frame(trajectories.iloc[::-1])
    assert result.cycle.iloc[0] == 1


def test_label_zero_only_at_failure(trajectories):
    short = trajectories.loc[(trajectories.engine_id == 1) & (trajectories.cycle <= 4)]
    np.testing.assert_array_equal(training_labels(short), [3, 2, 1, 0])


def test_test_endpoint_labels_are_external(tmp_path):
    path = tmp_path/"rul.txt"
    path.write_text("17\n42\n")
    labels = read_test_labels(path, np.array([2, 1]))
    assert labels.loc[1] == 17
    assert labels.loc[2] == 42
    with pytest.raises(ValueError):
        read_test_labels(path, np.array([1, 2, 3]))


def test_manifest_detects_changed_data(tmp_path):
    import json
    source = tmp_path/"train.txt"
    source.write_text("original")
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"files": {"train.txt": {"sha256": sha256(source)}}}))
    check_manifest(tmp_path, manifest)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Checksum"):
        check_manifest(tmp_path, manifest)
