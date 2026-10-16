"""Persisted engine partitions, bounded model selection and final evaluation.

Development reads training measurements only; official target values enter
only evaluate_official. All model selection precedes that function.
"""
from pathlib import Path
import hashlib
import itertools
import json
import platform
import time
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from .data import SENSORS, read_cmapss, read_test_labels, check_manifest
from .features import build_features, attach_targets, training_sample
from .metrics import evaluate, bootstrap_metrics
from .models import fit_regressor, RULModel


def write_json(path: str | Path, value: dict) -> None:
    """Write readable strict JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(config: dict) -> str:
    """Return a deterministic SHA-256 of a configuration's canonical JSON."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def make_partitions(training: pd.DataFrame, config: dict) -> dict:
    """Create fixed engine partitions and one pseudo-endpoint per engine.

    The truncation is uniform on [30,T_i-1]. It is a retrospective evaluation
    scenario; its horizon distribution may differ from official truncation.
    Failure time defines labels/sampling eligibility, never predictors.
    """
    ids = np.sort(training.engine_id.unique())
    n_holdout = config["holdout_engines"]
    if len(ids) <= n_holdout + config["cv_folds"]:
        raise ValueError("Insufficient engines for requested holdout and CV")
    shuffled = np.random.default_rng(config["seed"]).permutation(ids)
    dev, hold = sorted(shuffled[n_holdout:].tolist()), sorted(shuffled[:n_holdout].tolist())
    ends = training.groupby("engine_id").cycle.max()
    rng = np.random.default_rng(config["seed"]+1)
    endpoints = {str(int(i)): int(rng.integers(30, int(ends.loc[i]))) for i in ids}
    folds = []
    dev_array = np.array(dev)
    for fit, validation in KFold(config["cv_folds"], shuffle=True, random_state=config["seed"]).split(dev_array):
        folds.append({"train": dev_array[fit].tolist(), "validation": dev_array[validation].tolist()})
    return {"development": dev, "holdout": hold, "folds": folds, "endpoints": endpoints,
            "seed": config["seed"], "sampling": "one uniform integer cycle in [30,T_i-1] per engine"}


def endpoint_rows(features: pd.DataFrame, endpoints: dict, ids: list[int]) -> pd.DataFrame:
    """Select and order the exact saved validation endpoint for every engine.

    Raises ValueError if any requested endpoint is absent or duplicated.
    """
    request = pd.DataFrame({"engine_id": ids, "cycle": [endpoints[str(i)] for i in ids]})
    merged = request.merge(features, on=["engine_id", "cycle"], how="left", validate="one_to_one")
    if merged.isna().any().any() or len(merged) != len(ids):
        raise ValueError("Every evaluation engine must have its exact supported endpoint")
    return merged


def _pred(estimator, names, frame):
    return estimator.predict(frame[names].to_numpy(float))


def _cross_validate(spec, cache, splits, config):
    results, fit_times = [], []
    features = cache[spec["window"]]
    for fold in splits["folds"]:
        fit_frame = training_sample(features.loc[features.engine_id.isin(fold["train"])], config["train_stride"])
        validation = endpoint_rows(features, splits["endpoints"], fold["validation"])
        start = time.perf_counter()
        model, names = fit_regressor(fit_frame, spec, seed=config["seed"])
        fit_times.append(time.perf_counter()-start)
        raw = _pred(model, names, validation)
        results.append(pd.DataFrame({"engine_id": validation.engine_id, "true_rul": validation.target,
                                     "raw": raw, "prediction": np.maximum(0., raw)}))
    combined = pd.concat(results).sort_values("engine_id")
    metrics = evaluate(combined.true_rul, combined.prediction)
    return {**spec, **metrics, "fit_seconds_mean": float(np.mean(fit_times))}


def develop(data_dir: str | Path, config: dict, output_dir: str | Path, manifest: str | Path) -> dict:
    """Tune on development engines and freeze selection on an internal holdout.

    Returns and writes a selection record with parameters, split/configuration
    fingerprints and holdout evidence. Does not parse official test data/labels.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    check_manifest(data_dir, manifest)
    train = read_cmapss(Path(data_dir)/"train_FD001.txt")
    splits = make_partitions(train, config)
    write_json(output/"splits.json", splits)
    write_json(output/"run_config.json", config)
    cache = {w: attach_targets(build_features(train, w), train) for w in config["windows"]}
    specs = []
    for window, alpha in itertools.product(config["windows"], config["ridge_alpha"]):
        specs.append({"kind": "ridge", "window": window, "alpha": alpha})
    for window, depth, leaf, features in itertools.product(
            config["windows"], config["forest_depth"], config["forest_leaf"], config["forest_features"]):
        specs.append({"kind": "forest", "window": window, "trees": config["forest_trees"],
                      "depth": depth, "leaf": leaf, "features": features})
    records = []
    with threadpool_limits(limits=1):
        for index, spec in enumerate(specs, 1):
            result = _cross_validate(spec, cache, splits, config)
            records.append(result)
            pd.DataFrame(records).to_csv(output/"cv_results.csv", index=False)
            print(f"CV {index}/{len(specs)} {spec} RMSE={result['rmse']:.3f}", flush=True)
        winners = {}
        for kind in ["ridge", "forest"]:
            candidates = [r for r in records if r["kind"] == kind]
            best = min(candidates, key=lambda r: (r["rmse"], r["mae"]))
            winners[kind] = {k: best[k] for k in ["kind", "window"] +
                             (["alpha"] if kind == "ridge" else ["trees", "depth", "leaf", "features"])}
        age_candidates = [{"kind": "age", "window": 30, "alpha": a} for a in config["ridge_alpha"]]
        age_results = [_cross_validate(spec, cache, splits, config) for spec in age_candidates]
        best_age = min(age_results, key=lambda r: r["rmse"])
        winners["age"] = {k: best_age[k] for k in ["kind", "window", "alpha"]}
        pd.DataFrame(records + age_results).to_csv(output/"cv_results.csv", index=False)
        hold_metrics, held_predictions = {}, {}
        for kind, spec in winners.items():
            features = cache[spec["window"]]
            fit_frame = training_sample(features.loc[features.engine_id.isin(splits["development"])], config["train_stride"])
            held = endpoint_rows(features, splits["endpoints"], splits["holdout"])
            model, names = fit_regressor(fit_frame, spec, seed=config["seed"])
            raw = _pred(model, names, held)
            held_predictions[kind] = np.maximum(0., raw)
            hold_metrics[kind] = evaluate(held.target, held_predictions[kind])
        improvement = 1-hold_metrics["forest"]["rmse"]/hold_metrics["ridge"]["rmse"]
        selected = ("forest" if improvement >= config["forest_improvement_required"] and
                    hold_metrics["forest"]["nasa_score"] <= hold_metrics["ridge"]["nasa_score"] else "ridge")
        selection = {"config_sha256": fingerprint(config), "specs": winners, "selected": selected,
                     "holdout_metrics": hold_metrics, "holdout_relative_rmse_improvement": improvement,
                     "rule": "forest requires >=5% lower holdout RMSE and no worse NASA score; otherwise ridge",
                     "target": "uncapped operational cycles", "official_test_used_for_selection": False,
                     "training_stride": config["train_stride"],
                     "training_rows_full": len(train), "training_windows_sampled": len(training_sample(cache[30], config["train_stride"])),
                     "holdout_bootstrap_forest_minus_ridge": bootstrap_metrics(
                         held.target.to_numpy(), held_predictions["forest"], held_predictions["ridge"],
                         repetitions=config["bootstrap_repetitions"], seed=config["seed"])}
        write_json(output/"selection.json", selection)
        hold_table = held[["engine_id", "cycle", "target"]].rename(columns={"target": "true_rul"}).copy()
        for kind, values in held_predictions.items():
            hold_table[kind] = values
        hold_table.to_csv(output/"holdout_predictions.csv", index=False)
        # Selection is now frozen. These analyses characterize fixed models;
        # neither the held-out nor final-test results change hyperparameters.
        _development_studies(train, cache, splits, winners, config, output)
    return selection


def _development_studies(train, cache, splits, winners, config, output):
    spec = winners["forest"]
    fold = splits["folds"][0]
    feature = cache[spec["window"]]
    fit = training_sample(feature.loc[feature.engine_id.isin(fold["train"])], config["train_stride"])
    val = endpoint_rows(feature, splits["endpoints"], fold["validation"])
    sweep = []
    for trees in config["tree_sweep"]:
        start = time.perf_counter()
        model, names = fit_regressor(fit, {**spec, "trees": trees}, seed=config["seed"])
        duration = time.perf_counter()-start
        sweep.append({"trees": trees, "fit_seconds": duration,
                      **evaluate(val.target, np.maximum(0, _pred(model, names, val)))})
    pd.DataFrame(sweep).to_csv(output/"tree_sweep.csv", index=False)
    ablations = []
    for stride in [1, 5]:
        for trends in [False, True]:
            features = attach_targets(build_features(train, spec["window"], trends=trends), train)
            fit_frame = training_sample(features.loc[features.engine_id.isin(fold["train"])], stride)
            validation = endpoint_rows(features, splits["endpoints"], fold["validation"])
            start = time.perf_counter()
            model, names = fit_regressor(fit_frame, spec, seed=config["seed"], trends=trends)
            duration = time.perf_counter()-start
            ablations.append({"stride": stride, "trends": trends, "training_windows": len(fit_frame),
                              "fit_seconds": duration,
                              **evaluate(validation.target, np.maximum(0, _pred(model, names, validation)))})
    pd.DataFrame(ablations).to_csv(output/"ablation.csv", index=False)
    seeds = []
    for seed in [3001, 3002, 3003]:
        model, names = fit_regressor(fit, spec, seed=seed)
        seeds.append({"seed": seed, **evaluate(val.target, np.maximum(0, _pred(model, names, val)))})
    pd.DataFrame(seeds).to_csv(output/"seed_sensitivity.csv", index=False)
    sensitivity = []
    fit_ids = splits["development"]
    hold_ids = splits["holdout"]
    reference_train = train.loc[train.engine_id.isin(fit_ids)]
    held_obs = train.loc[train.engine_id.isin(hold_ids)].copy()
    std = reference_train[SENSORS].std(ddof=0)
    perturbations = [("clean", 0.), ("noise_1pct_std", .01), ("noise_5pct_std", .05), ("T30_bias_1std", 1.)]
    for kind in ["ridge", "forest"]:
        spec = winners[kind]
        features = cache[spec["window"]]
        fitted, names = fit_regressor(training_sample(features.loc[features.engine_id.isin(fit_ids)], config["train_stride"]),
                                     spec, seed=config["seed"])
        for name, strength in perturbations:
            changed = held_obs.copy()
            rng = np.random.default_rng(config["seed"])
            if name.startswith("noise"):
                changed[SENSORS] = changed[SENSORS] + rng.normal(size=(len(changed), len(SENSORS)))*std.to_numpy()*strength
            elif name.startswith("T30"):
                changed["s3"] += float(std["s3"])*strength
            changed_features = attach_targets(build_features(changed, spec["window"]), changed)
            endpoint = endpoint_rows(changed_features, splits["endpoints"], hold_ids)
            sensitivity.append({"model": kind, "scenario": name,
                                **evaluate(endpoint.target, np.maximum(0, _pred(fitted, names, endpoint)))})
    pd.DataFrame(sensitivity).to_csv(output/"robustness.csv", index=False)


def evaluate_official(data_dir: str | Path, config: dict, output_dir: str | Path,
                      manifest: str | Path) -> dict:
    """Refit frozen configurations on all training engines and evaluate endpoints.

    Writes all declared models' predictions, raw/clipped metrics, error strata,
    bootstrap intervals, trusted local model bundles and example predictions.
    Test metrics do not influence the frozen selected model.
    """
    output = Path(output_dir)
    selection = json.loads((output/"selection.json").read_text(encoding="utf-8"))
    if selection["config_sha256"] != fingerprint(config):
        raise ValueError("Configuration changed after selection; official evaluation refused")
    check_manifest(data_dir, manifest)
    training = read_cmapss(Path(data_dir)/"train_FD001.txt")
    testing = read_cmapss(Path(data_dir)/"test_FD001.txt")
    labels = read_test_labels(Path(data_dir)/"RUL_FD001.txt", testing.engine_id.unique())
    output.joinpath("models").mkdir(exist_ok=True)
    table = pd.DataFrame({"engine_id": labels.index, "true_rul": labels.values})
    metrics, strata, fitted = [], [], {}
    bounds = {s: [float(training[s].min()), float(training[s].max())] for s in SENSORS}
    with threadpool_limits(limits=1):
        for kind, spec in selection["specs"].items():
            features = attach_targets(build_features(training, spec["window"]), training)
            fit_frame = training_sample(features, config["train_stride"])
            start = time.perf_counter()
            estimator, names = fit_regressor(fit_frame, spec, seed=config["seed"])
            fit_seconds = time.perf_counter()-start
            test_features = build_features(testing, spec["window"]).groupby("engine_id", sort=True).tail(1).sort_values("engine_id")
            if not np.array_equal(test_features.engine_id.to_numpy(), labels.index.to_numpy()):
                raise ValueError("Test endpoint order/count does not match labels")
            raw = _pred(estimator, names, test_features)
            pred = np.maximum(0., raw)
            table[kind] = pred
            table[f"{kind}_raw"] = raw
            table["last_cycle"] = test_features.cycle.to_numpy()
            metrics.append({"model": kind, "selected": kind == selection["selected"], "fit_seconds": fit_seconds,
                            "clipped_count": int((raw < 0).sum()), **evaluate(labels.values, pred)})
            metrics.append({"model": kind+"_raw", "selected": False, "fit_seconds": fit_seconds,
                            "clipped_count": 0, **evaluate(labels.values, raw)})
            for name, mask in [("<=30", labels.values <= 30), ("31-80", (labels.values > 30) & (labels.values <= 80)),
                               (">80", labels.values > 80)]:
                if mask.any():
                    strata.append({"model": kind, "rul_band": name, **evaluate(labels.values[mask], pred[mask])})
            bundle = RULModel(estimator, names, spec["window"], bounds, selection["config_sha256"][:12])
            bundle.save(output/"models"/f"{kind}.joblib")
            fitted[kind] = bundle
            print(f"OFFICIAL {kind}: RMSE={metrics[-2]['rmse']:.3f}, MAE={metrics[-2]['mae']:.3f}", flush=True)
    table.to_csv(output/"test_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output/"test_metrics.csv", index=False)
    pd.DataFrame(strata).to_csv(output/"error_by_rul.csv", index=False)
    intervals = bootstrap_metrics(table.true_rul, table.forest, table.ridge,
                                  repetitions=config["bootstrap_repetitions"], seed=config["seed"])
    write_json(output/"bootstrap_intervals.json", intervals)
    chosen = fitted[selection["selected"]]
    examples = [chosen.predict(testing.loc[testing.engine_id == i]) for i in [1, 2, 3]]
    write_json(output/"demo_predictions.json", {"selected_model": selection["selected"], "examples": examples})
    # All test trajectories are used; no engine is dropped for poor performance.
    warnings = [chosen.predict(g) for _, g in testing.groupby("engine_id")]
    write_json(output/"support_audit.json", {"engines": len(warnings),
               "engines_with_warnings": sum(bool(v["flags"]) for v in warnings),
               "predictions": warnings})
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "psutil", "pytest", "pdoc"]}
    write_json(output/"environment.json", {"python": platform.python_version(), "platform": platform.platform(),
               "processor": platform.processor(), "versions": versions, "seed": config["seed"], "threads": 1,
               "config_sha256": fingerprint(config), "model_target": "uncapped_RUL_cycles"})
    return {"selection": selection, "metrics": metrics, "intervals": intervals}
