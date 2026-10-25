"""Compare the computational cost of four fixed RUL models."""

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
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from rul_project.rul_utils import make_hist_gradient_boosting


from rul_project.settings import OUTPUT_DIR
FIT_REPETITIONS = 3
PREDICTION_REPETITIONS = 30


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
    return [
        tree
        for iteration in model._predictors
        for tree in iteration
    ]


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


def save_table(rows, filename, title):
    """Save an unrounded CSV and print a three-decimal summary.
    
        Args:
            rows (list[dict]): Tabular records compatible with pandas.DataFrame.
            filename (str): CSV filename relative to the configured OUTPUT_DIR.
            title (str): Heading printed above the displayed summary.
    
        Returns:
            pandas.DataFrame: The unrounded table written to disk.
    
        Raises:
            OSError: If the output cannot be written.
    
        Usage: save_table(records, "timing.csv", "Timing results").
    """
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / filename, index=False)

    print(f"\n{title}")
    print("-" * len(title))
    print(table.round(3).to_string(index=False))
    return table


def save_training_profile(model_name, template, inputs, targets):
    """Profile a separate cloned fit after runtime benchmarking.
    
        Args:
            model_name (str): Human-readable name used to construct output filenames.
            template: Unfitted scikit-learn-compatible estimator to clone.
            inputs (pandas.DataFrame): Training feature table in fixed feature order.
            targets (array-like): Aligned training RUL in cycles.
    
        Returns:
            None: Writes binary .prof and readable top-15 cumulative-time text files.
    
        Raises:
            ValueError: If clone or fit rejects the configuration/data.
            OSError: If profile files cannot be written.
    
        Usage: save_training_profile("RF", template, X_train, y_train).
        Profiled times include instrumentation overhead; use benchmark timing for comparisons.
    """
    profile_model = clone(template)
    profiler = cProfile.Profile()

    profiler.enable()
    profile_model.fit(inputs, targets)
    profiler.disable()

    stem = model_name.lower().replace(" ", "_").replace("-", "_")
    profiler.dump_stats(str(OUTPUT_DIR / f"{stem}_training.prof"))

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.strip_dirs().sort_stats("cumulative").print_stats(15)

    report_path = OUTPUT_DIR / f"{stem}_training_profile.txt"
    report_path.write_text(buffer.getvalue(), encoding="utf-8")
    print("Profile saved to:", report_path)


def main():
    """Benchmark the four fixed models on the complete FD001 training set.
    
        Returns:
            None: Writes runtime quartiles, serialized sizes, selected-array memory,
            linear arithmetic intensity, tree operations, profiles and environment CSV.
    
        Raises:
            FileNotFoundError: If prepared training or endpoint CSVs are missing.
            ValueError: If inputs are empty, duplicated or invalid for fitting.
            OSError: If results cannot be written.
    
        No arguments are accepted. Run ``python -m rul_project performance`` after
        testevaluation. Uses three fits and 30 warm prediction calls with one thread.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(
        OUTPUT_DIR / "train_FD001_labeled.csv"
    )
    test_data = pd.read_csv(
        OUTPUT_DIR / "test_endpoints_labeled.csv"
    )

    if test_data["engine_id"].duplicated().any():
        raise ValueError("Expected one test endpoint per engine.")

    if train_data.empty or test_data.empty:
        raise ValueError("Benchmark datasets must not be empty.")

    multi_features = [
        "cycle",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_9",
        "sensor_11",
    ]

    # Keep configurations identical to the evaluated models.
    model_settings = [
        (
            "Age-only",
            ["cycle"],
            LinearRegression(),
        ),
        (
            "Multi-sensor linear",
            multi_features,
            LinearRegression(),
        ),
        (
            "Random forest",
            multi_features,
            RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=1,
            ),
        ),
        (
            "Histogram gradient boosting",
            multi_features,
            make_hist_gradient_boosting(),
        ),
    ]

    runtime_rows = []
    size_rows = []
    memory_rows = []
    intensity_rows = []
    tree_rows = []
    fitted_models = {}

    # File loading is outside all measured regions.
    with threadpool_limits(limits=1):
        for model_name, features, template in model_settings:
            print(f"\nBenchmarking model: {model_name}")

            # Feature selection is also outside the timers.
            X_train = train_data[features]
            y_train = train_data["RUL"]
            X_test = test_data[features]
            X_single = X_test.iloc[[0]]

            fit_times = []

            for _ in range(FIT_REPETITIONS):
                model = clone(template)

                start = perf_counter()
                model.fit(X_train, y_train)
                fit_times.append(perf_counter() - start)

            fitted_models[model_name] = model

            single = benchmark_prediction(model, X_single)
            batch = benchmark_prediction(model, X_test)

            runtime_rows.append(
                {
                    "model": model_name,
                    "n_features": len(features),
                    "fit_s": float(np.median(fit_times)),
                    "fit_q25_s": float(np.percentile(fit_times, 25)),
                    "fit_q75_s": float(np.percentile(fit_times, 75)),
                    "one_ms": single["median_ms"],
                    "one_q25_ms": single["q25_ms"],
                    "one_q75_ms": single["q75_ms"],
                    "batch_ms": batch["median_ms"],
                    "batch_q25_ms": batch["q25_ms"],
                    "batch_q75_ms": batch["q75_ms"],
                    "batch_ms_per_engine": (
                        batch["median_ms"] / len(X_test)
                    ),
                }
            )

            serialized_bytes = len(
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            )
            size_rows.append(
                {
                    "model": model_name,
                    "serialized_bytes": serialized_bytes,
                    "serialized_kib": serialized_bytes / 1024,
                }
            )

            array_bytes = numerical_array_bytes(model)
            memory_rows.append(
                {
                    "model": model_name,
                    "numerical_array_bytes": array_bytes,
                    "numerical_array_kib": array_bytes / 1024,
                }
            )

            if isinstance(model, LinearRegression):
                d = len(features)
                flops = 2 * d

                # Read d inputs, d coefficients and one intercept;
                # write one output. Assume float64 and no cache reuse.
                transfer_bytes = (2 * d + 2) * 8

                intensity_rows.append(
                    {
                        "model": model_name,
                        "n_features": d,
                        "estimated_flops": flops,
                        "assumed_transfer_bytes": transfer_bytes,
                        "estimated_flops_per_byte": (
                            flops / transfer_bytes
                        ),
                    }
                )

            elif isinstance(model, RandomForestRegressor):
                depths = [
                    estimator.tree_.max_depth
                    for estimator in model.estimators_
                ]
                tree_rows.append(
                    {
                        "model": model_name,
                        "n_trees": len(depths),
                        "total_nodes": sum(
                            estimator.tree_.node_count
                            for estimator in model.estimators_
                        ),
                        "maximum_actual_depth": max(depths),
                        "comparison_upper_bound": sum(depths),
                        "output_combination_flops": len(depths),
                    }
                )

            elif isinstance(model, HistGradientBoostingRegressor):
                trees = hgb_trees(model)
                depths = [tree.get_max_depth() for tree in trees]

                tree_rows.append(
                    {
                        "model": model_name,
                        "n_trees": len(trees),
                        "total_nodes": sum(
                            len(tree.nodes) for tree in trees
                        ),
                        "maximum_actual_depth": max(depths),
                        "comparison_upper_bound": sum(depths),
                        # Add each tree output to the baseline.
                        # Learning-rate scaling is stored in leaf values.
                        "output_combination_flops": len(trees),
                    }
                )

        # Preserve the actual forest path-count analysis.
        forest = fitted_models["Random forest"]
        X_tree = test_data[multi_features].to_numpy(dtype=np.float32)
        comparisons = np.zeros(len(X_tree), dtype=np.int64)

        for estimator in forest.estimators_:
            path = estimator.decision_path(X_tree)
            comparisons += np.diff(path.indptr) - 1

        forest_structure = next(
            row for row in tree_rows
            if row["model"] == "Random forest"
        )

        save_table(
            [
                {
                    "test_engines": len(X_tree),
                    "n_trees": forest_structure["n_trees"],
                    "total_nodes": forest_structure["total_nodes"],
                    "maximum_actual_depth": (
                        forest_structure["maximum_actual_depth"]
                    ),
                    "mean_threshold_comparisons": comparisons.mean(),
                    "min_threshold_comparisons": comparisons.min(),
                    "max_threshold_comparisons": comparisons.max(),
                    "output_averaging_flops": forest.n_estimators,
                }
            ],
            "random_forest_operation_summary.csv",
            "Random forest: measured prediction-path comparisons",
        )

        # Profile separate fits only after all timing benchmarks.
        for model_name, features, template in model_settings:
            save_training_profile(
                model_name,
                template,
                train_data[features],
                train_data["RUL"],
            )

    save_table(
        runtime_rows,
        "model_runtime_comparison.csv",
        "Runtime comparison",
    )
    save_table(
        size_rows,
        "model_size_comparison.csv",
        "Serialized model size comparison",
    )
    save_table(
        memory_rows,
        "model_numerical_array_memory.csv",
        "Selected fitted-array memory",
    )
    save_table(
        intensity_rows,
        "linear_arithmetic_intensity.csv",
        "Linear models: analytical arithmetic intensity",
    )
    save_table(
        tree_rows,
        "tree_model_operation_summary.csv",
        "Tree models: structure and per-prediction operation bounds",
    )

    information = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "training_engines": train_data["engine_id"].nunique(),
        "training_records": len(train_data),
        "prediction_batch_size": len(test_data),
        "fit_repetitions": FIT_REPETITIONS,
        "prediction_repetitions": PREDICTION_REPETITIONS,
        "requested_computation_threads": 1,
    }

    pd.DataFrame([information]).to_csv(
        OUTPUT_DIR / "benchmark_environment.csv",
        index=False,
    )

    print("\nBenchmark information")
    print("---------------------")
    for key, value in information.items():
        print(f"{key}: {value}")

    print("\nMeasurement scope")
    print("-----------------")
    print("Prediction timing measures warm raw model.predict calls.")
    print("File loading, feature selection and clipping are excluded.")
    print("Serialized size is not process RAM usage.")
    print("Array counts exclude Python overhead and training workspace.")
    print("HGB array inspection uses version-dependent private attributes.")
    print("Tree comparison bounds are not measured FLOPs.")
    print("Linear arithmetic intensity assumes no cache reuse.")
    print("Profiling adds overhead; use benchmark times for comparisons.")


if __name__ == "__main__":
    main()