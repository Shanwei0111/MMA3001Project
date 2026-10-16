"""Measured runtime, memory and explicitly scoped arithmetic estimates."""
from pathlib import Path
import cProfile
import io
import json
import pstats
import time
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from .data import read_cmapss
from .features import build_features, attach_targets, training_sample
from .models import load_model, fit_regressor
from .experiment import write_json


def benchmark_call(function, repetitions: int = 30, inner: int = 1) -> dict:
    """Warm a callable and return median/IQR seconds per call over repetitions.

    inner amortizes clock overhead; all observations include Python call overhead.
    """
    function()
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        values.append((time.perf_counter()-start)/inner)
    return {"median_seconds": float(np.median(values)), "q25_seconds": float(np.quantile(values, .25)),
            "q75_seconds": float(np.quantile(values, .75)), "repetitions": repetitions, "inner": inner}


def profile_project(data_dir: str | Path, config: dict, output_dir: str | Path) -> dict:
    """Profile frozen models without tuning against official test outcomes.

    Records actual runtime distributions, three fit repetitions, current/peak
    process memory, serialized model bytes and tree decision-path comparisons.
    Native working-set measurements include interpreter/libraries/data.
    """
    out = Path(output_dir)
    train = read_cmapss(Path(data_dir)/"train_FD001.txt")
    test = read_cmapss(Path(data_dir)/"test_FD001.txt")
    selection = json.loads((out/"selection.json").read_text())
    repetitions = config["timing_repetitions"]
    profile = {"timing_scope": "single thread; warm in-memory models; file/model loading excluded from prediction",
               "models": {}}
    with threadpool_limits(limits=1):
        profile["feature_reference"] = benchmark_call(lambda: build_features(train, 30, reference=True), 10)
        profile["feature_vectorized"] = benchmark_call(lambda: build_features(train, 30), 10)
        a, b = build_features(train, 30, reference=True), build_features(train, 30)
        profile["feature_max_absolute_difference"] = float(np.max(np.abs(a.to_numpy(float)-b.to_numpy(float))))
        profile["feature_speedup"] = profile["feature_reference"]["median_seconds"]/profile["feature_vectorized"]["median_seconds"]
        profiler = cProfile.Profile()
        profiler.runcall(build_features, train, 30)
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
        (out/"profiles").mkdir(exist_ok=True)
        (out/"profiles"/"features.txt").write_text(stream.getvalue(), encoding="utf-8")
        for kind, spec in selection["specs"].items():
            bundle = load_model(out/"models"/f"{kind}.joblib")
            endpoint = build_features(test, spec["window"]).groupby("engine_id", sort=True).tail(1)
            x = endpoint[bundle.names].to_numpy(float)
            history = test.loc[test.engine_id == 1].tail(30)
            fit_frame = training_sample(attach_targets(build_features(train, spec["window"]), train), config["train_stride"])
            fit_seconds = []
            for _ in range(3):
                start = time.perf_counter()
                fit_regressor(fit_frame, spec, seed=config["seed"])
                fit_seconds.append(time.perf_counter()-start)
            info = {"model_one": benchmark_call(lambda: bundle.estimator.predict(x[:1]), repetitions, 10),
                    "model_batch100": benchmark_call(lambda: bundle.estimator.predict(x), repetitions, 5),
                    "history_to_prediction": benchmark_call(lambda: bundle.predict(history), repetitions, 1),
                    "fit_seconds": fit_seconds, "fit_median_seconds": float(np.median(fit_seconds)),
                    "serialized_bytes": (out/"models"/f"{kind}.joblib").stat().st_size,
                    "features": len(bundle.names)}
            if kind == "forest":
                estimator = bundle.estimator
                # Each visited leaf adds one node but no threshold comparison.
                paths = np.column_stack([t.decision_path(x).getnnz(axis=1)-1 for t in estimator.estimators_])
                state = [t.tree_.__getstate__() for t in estimator.estimators_]
                bytes_per_node = int(state[0]["nodes"].dtype.itemsize)
                info.update({"trees": len(state), "mean_comparisons": float(paths.sum(axis=1).mean()),
                             "max_comparisons": int(paths.sum(axis=1).max()),
                             "total_nodes": int(sum(t.tree_.node_count for t in estimator.estimators_)),
                             "node_storage_bytes": int(sum(s["nodes"].nbytes+s["values"].nbytes for s in state)),
                             "node_record_bytes": bytes_per_node,
                             "estimated_traversal_bytes": float((paths.sum(axis=1).mean()+len(state))*bytes_per_node),
                             "averaging_flops": len(state)})
            else:
                d = len(bundle.names)
                info.update({"dot_and_intercept_flops": 2*d, "scaling_flops": 2*d,
                             "fused_cold_bytes": 8*(2*d+2), "fused_flops_per_byte": (2*d)/(8*(2*d+2))})
            profile["models"][kind] = info
    mem = psutil.Process().memory_info()
    profile["process_current_rss_mb"] = mem.rss/1e6
    profile["process_peak_rss_mb"] = getattr(mem, "peak_wset", mem.rss)/1e6
    profile["peak_memory_scope"] = ("Windows process lifetime peak working set" if hasattr(mem, "peak_wset")
                                   else "RSS at profiling completion; not a sampled lifetime peak")
    write_json(out/"performance.json", profile)
    return profile
