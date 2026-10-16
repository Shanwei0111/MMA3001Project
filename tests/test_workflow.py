"""Small synthetic integration test of the selection/evaluation boundary."""
import json
import numpy as np
import pandas as pd
import pytest
from jet_rul import experiment
from jet_rul.data import sha256


def test_development_does_not_parse_official_targets(trajectories, tmp_path, monkeypatch):
    groups = []
    for engine in range(1, 13):
        group = trajectories.loc[trajectories.engine_id == 1].copy()
        group.engine_id = engine
        groups.append(group)
    np.savetxt(tmp_path/'train_FD001.txt', pd.concat(groups).to_numpy())
    # Invalid numeric contents are deliberate: development may checksum these
    # files for integrity, but must never parse their observations or targets.
    for name in ['test_FD001.txt', 'RUL_FD001.txt']:
        (tmp_path/name).write_text('forbidden during selection')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'files': {p.name: {'sha256': sha256(p)} for p in tmp_path.glob('*.txt')}}))
    config = dict(seed=3001, holdout_engines=3, cv_folds=3, windows=[30], ridge_alpha=[1.],
                  forest_depth=[2], forest_leaf=[2], forest_features=[1.], forest_trees=3,
                  train_stride=5, forest_improvement_required=.05, bootstrap_repetitions=30)
    monkeypatch.setattr(experiment, '_development_studies', lambda *args: None)
    out = tmp_path/'results'
    result = experiment.develop(tmp_path, config, out, manifest)
    assert result['official_test_used_for_selection'] is False
    assert result['selected'] in ['ridge', 'forest']
    assert len(pd.read_csv(out/'holdout_predictions.csv')) == 3
    with pytest.raises(ValueError, match='Configuration changed'):
        experiment.evaluate_official(tmp_path, {**config, 'seed': 1}, out, manifest)
