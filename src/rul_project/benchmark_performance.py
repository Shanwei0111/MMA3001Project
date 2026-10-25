"""Measure runtime, fitted storage, operation counts and training hotspots.

File loading and feature selection are outside the timed prediction calls.
Profiles are separate instrumented fits. Selected arrays and serialized bytes
are reported separately because neither is a measurement of peak process RAM.
"""

from pathlib import Path
from time import perf_counter
import cProfile
import io
import pickle
import platform
import pstats

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from rul_project.model_configuration import MODEL_NAMES, feature_columns, make_model

FIT_REPETITIONS = 3
PREDICTION_REPETITIONS = 30


def save_training_profile(
    model_name: str,
    template,
    inputs: pd.DataFrame,
    targets: pd.Series,
    output_dir: Path,
) -> None:
    """Profile a separate cloned fit after ordinary runtime measurements.

    Args:
        model_name: Readable name used to construct output filenames.
        template: Unfitted scikit-learn estimator to clone.
        inputs: Training feature table in fitted column order.
        targets: Aligned training RUL in cycles.
        output_dir: Destination for binary .prof and readable top-15 text profiles.

    Returns:
        None: Writes both profile formats without changing the estimator template.

    Raises:
        ValueError: If cloning or fitting rejects the model/data.
        OSError: If profile files cannot be written.

    Example: save_training_profile("RF", make_model("Random forest"), X, y, Path("results/generated")).
    """
    model = clone(template)
    profiler = cProfile.Profile()
    profiler.enable()
    model.fit(inputs, targets)
    profiler.disable()
    stem = model_name.lower().replace(" ", "_").replace("-", "_")
    profiler.dump_stats(str(output_dir / f"{stem}_training.prof"))
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats(
        "cumulative"
    ).print_stats(15)
    (output_dir / f"{stem}_training_profile.txt").write_text(
        buffer.getvalue(), encoding="utf-8"
    )


def operation_estimate(model, model_name: str, n_features: int) -> dict:
    """Describe per-prediction linear arithmetic or a tree-comparison upper bound.

    Args:
        model: Fitted supported scalar-output regressor.
        model_name: Human-readable model label for the output table.
        n_features: Number of fitted predictor columns.

    Returns:
        dict: Linear FLOPs/assumed bytes/intensity, or tree count, node count,
        maximum depth, comparison upper bound and output combination arithmetic.

    Raises:
        TypeError: If the estimator type is unsupported.
        AttributeError: If the estimator is unfitted or HGB internals changed.

    Example: operation_estimate(fitted_model, "Age-only", 1). Comparisons are not FLOPs.
    """
    if isinstance(model, LinearRegression):
        # Read d inputs, d coefficients and an intercept; write one float64 result.
        flops, byte_count = 2 * n_features, 8 * (2 * n_features + 2)
        return {
            "model": model_name,
            "n_features": n_features,
            "estimated_flops": flops,
            "assumed_transfer_bytes": byte_count,
            "estimated_flops_per_byte": flops / byte_count,
        }
    if isinstance(model, RandomForestRegressor):
        depths = [tree.tree_.max_depth for tree in model.estimators_]
        node_count = sum(tree.tree_.node_count for tree in model.estimators_)
    elif isinstance(model, HistGradientBoostingRegressor):
        trees = hgb_trees(model)
        depths = [tree.get_max_depth() for tree in trees]
        node_count = sum(len(tree.nodes) for tree in trees)
    else:
        raise TypeError(f"Unsupported model: {type(model).__name__}")
    # RF uses B-1 additions and a division; HGB adds B already-scaled leaf outputs.
    return {
        "model": model_name,
        "n_trees": len(depths),
        "total_nodes": node_count,
        "maximum_actual_depth": max(depths),
        "comparison_upper_bound": sum(depths),
        "output_combination_flops": len(depths),
    }


def forest_path_comparisons(
    model: RandomForestRegressor, inputs: pd.DataFrame
) -> np.ndarray:
    """Count threshold comparisons actually visited by each forest prediction.

    Args:
        model: Fitted random forest.
        inputs: Nonempty predictor table with matching fitted feature order.

    Returns:
        numpy.ndarray: Integer comparison counts per record, summed over all trees.

    Raises:
        ValueError: If the estimator rejects the input dimensions or values.

    Example: forest_path_comparisons(forest, endpoints[features]).mean().
    """
    comparisons = np.zeros(len(inputs), dtype=np.int64)
    values = inputs.to_numpy(dtype=np.float32)
    for tree in model.estimators_:
        # Each visited path contains one final leaf, which performs no threshold test.
        path = tree.decision_path(values)
        comparisons += np.diff(path.indptr) - 1
    return comparisons


def run_performance_benchmark(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Measure all four models with identical prepared data and timing boundaries.

    Args:
        output_dir: Directory containing training labels and official test endpoints;
            receives runtime, memory, operation, profile and environment outputs.

    Returns:
        dict[str, pandas.DataFrame]: Result tables keyed by output CSV filename.

    Raises:
        FileNotFoundError: If prepared data is absent.
        ValueError: If tables are empty/duplicated or estimator inputs are invalid.
        OSError: If evidence cannot be saved.

    Example: run_performance_benchmark(Path("results/generated")). Uses one thread even outside the CLI.
    """
    # Enforce the advertised timing budget for direct API callers as well as the CLI.
    with threadpool_limits(limits=1):
        return _benchmark_tables(output_dir)


def _benchmark_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Collect benchmark tables while the public caller holds the thread limit.

    Args:
        output_dir: Directory containing prepared training and endpoint CSVs.

    Returns:
        dict[str, pandas.DataFrame]: Saved runtime, memory and operation tables.

    Raises:
        FileNotFoundError: If prepared inputs are absent.
        ValueError: If tables or estimator inputs are invalid.
        OSError: If benchmark evidence cannot be written.

    Internal usage: called only by run_performance_benchmark inside its thread context.
    """
    train = pd.read_csv(output_dir / "train_FD001_labeled.csv")
    endpoints = pd.read_csv(output_dir / "test_endpoints_labeled.csv")
    if train.empty or endpoints.empty or endpoints.engine_id.duplicated().any():
        raise ValueError("Expected nonempty training data and unique test endpoints.")
    runtime, sizes, memory, linear_ops, tree_ops, fitted = [], [], [], [], [], {}
    for name in MODEL_NAMES:
        features = feature_columns(name)
        inputs, target = train[features], train["RUL"]
        batch = endpoints[features]
        single = batch.iloc[[0]]
        durations = []
        for _ in range(FIT_REPETITIONS):
            model = make_model(name)
            start = perf_counter()
            model.fit(inputs, target)
            durations.append(perf_counter() - start)
        fitted[name] = model
        one_time = benchmark_prediction(model, single)
        batch_time = benchmark_prediction(model, batch)
        runtime.append(
            {
                "model": name,
                "n_features": len(features),
                "fit_s": np.median(durations),
                "fit_q25_s": np.percentile(durations, 25),
                "fit_q75_s": np.percentile(durations, 75),
                "one_ms": one_time["median_ms"],
                "one_q25_ms": one_time["q25_ms"],
                "one_q75_ms": one_time["q75_ms"],
                "batch_ms": batch_time["median_ms"],
                "batch_q25_ms": batch_time["q25_ms"],
                "batch_q75_ms": batch_time["q75_ms"],
                "batch_ms_per_engine": batch_time["median_ms"] / len(batch),
            }
        )
        serialized = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        array_bytes = numerical_array_bytes(model)
        sizes.append(
            {
                "model": name,
                "serialized_bytes": serialized,
                "serialized_kib": serialized / 1024,
            }
        )
        memory.append(
            {
                "model": name,
                "numerical_array_bytes": array_bytes,
                "numerical_array_kib": array_bytes / 1024,
            }
        )
        estimate = operation_estimate(model, name, len(features))
        (linear_ops if isinstance(model, LinearRegression) else tree_ops).append(
            estimate
        )
        print(f"Benchmarked: {name}")

    forest = fitted["Random forest"]
    paths = forest_path_comparisons(forest, endpoints[feature_columns("Random forest")])
    forest_structure = next(row for row in tree_ops if row["model"] == "Random forest")
    measured_paths = [
        {
            "test_engines": len(endpoints),
            "n_trees": forest.n_estimators,
            "total_nodes": forest_structure["total_nodes"],
            "maximum_actual_depth": forest_structure["maximum_actual_depth"],
            "mean_threshold_comparisons": paths.mean(),
            "min_threshold_comparisons": paths.min(),
            "max_threshold_comparisons": paths.max(),
            "output_averaging_flops": forest.n_estimators,
        }
    ]
    environment = [
        {
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "training_engines": train.engine_id.nunique(),
            "training_records": len(train),
            "prediction_batch_size": len(endpoints),
            "fit_repetitions": FIT_REPETITIONS,
            "prediction_repetitions": PREDICTION_REPETITIONS,
            "requested_computation_threads": 1,
        }
    ]
    tables = {
        "model_runtime_comparison.csv": pd.DataFrame(runtime),
        "model_size_comparison.csv": pd.DataFrame(sizes),
        "model_numerical_array_memory.csv": pd.DataFrame(memory),
        "linear_arithmetic_intensity.csv": pd.DataFrame(linear_ops),
        "tree_model_operation_summary.csv": pd.DataFrame(tree_ops),
        "random_forest_operation_summary.csv": pd.DataFrame(measured_paths),
        "benchmark_environment.csv": pd.DataFrame(environment),
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    # Keep profiler overhead out of every runtime value written above.
    for name in MODEL_NAMES:
        save_training_profile(
            name,
            make_model(name),
            train[feature_columns(name)],
            train["RUL"],
            output_dir,
        )
    print(tables["model_runtime_comparison.csv"].round(3).to_string(index=False))
    return tables


def benchmark_prediction(model, inputs):
    """Measure warmed estimator prediction calls without loading files.

    Args:
        model: Fitted scikit-learn regressor accepting inputs in predict().
        inputs (pandas.DataFrame): Nonempty table of features in fitted order.

    Returns:
        dict[str, float]: median_ms, q25_ms and q75_ms for 30 calls, in ms.

    Raises:
        ValueError: If predict() rejects inputs or their feature schema.

    One warm-up call is excluded from timing. Clipping and feature selection
    belong outside this helper. Example: benchmark_prediction(model, X.iloc[[0]]).
    """
    model.predict(inputs)
    durations = []

    for _ in range(PREDICTION_REPETITIONS):
        start = perf_counter()
        model.predict(inputs)
        durations.append((perf_counter() - start) * 1000)

    return {
        "median_ms": float(np.median(durations)),
        "q25_ms": float(np.percentile(durations, 25)),
        "q75_ms": float(np.percentile(durations, 75)),
    }


def hgb_trees(model):
    """Flatten fitted HGB predictors using version-dependent internals.

    Args:
        model (HistGradientBoostingRegressor): A fitted scalar-output estimator.

    Returns:
        list: Internal TreePredictor objects in boosting iteration order.

    Raises:
        AttributeError: If model is unfitted or its private layout is unsupported.

    Usage: hgb_trees(fitted_hgb). Intended for scikit-learn 1.9.1 inspection.
    """
    return [tree for iteration in model._predictors for tree in iteration]


def numerical_array_bytes(model):
    """Count selected fitted numerical arrays rather than process RAM.

    Args:
        model: Fitted LinearRegression, RandomForestRegressor or
            HistGradientBoostingRegressor. Scalar-output regression is assumed.

    Returns:
        int: Coefficient/intercept bytes for linear models, node/value bytes
        for RF, or baseline/bin thresholds/nodes/category bitsets for HGB.
        Python objects, temporary workspace and process/peak RAM are excluded.

    Raises:
        TypeError: If the estimator type is unsupported.
        AttributeError: If unfitted or private HGB internals are incompatible.

    Usage: numerical_array_bytes(fitted_model). HGB internals are version-dependent.
    """
    if isinstance(model, LinearRegression):
        arrays = [
            np.asarray(model.coef_),
            np.asarray(model.intercept_),
        ]

    elif isinstance(model, RandomForestRegressor):
        arrays = []
        for estimator in model.estimators_:
            state = estimator.tree_.__getstate__()
            arrays.extend([state["nodes"], state["values"]])

    elif isinstance(model, HistGradientBoostingRegressor):
        arrays = [np.asarray(model._baseline_prediction)]
        arrays.extend(model._bin_mapper.bin_thresholds_)

        for tree in hgb_trees(model):
            arrays.extend(
                [
                    tree.nodes,
                    tree.raw_left_cat_bitsets,
                    tree.binned_left_cat_bitsets,
                ]
            )

    else:
        raise TypeError(f"Unsupported model: {type(model).__name__}")

    return int(sum(array.nbytes for array in arrays))
