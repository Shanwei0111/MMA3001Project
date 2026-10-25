"""Run named project stages with explicit paths and a consistent thread budget.

Use ``python -m rul_project --help``. The all command follows the dependency
order shown in README.md. Paths are passed directly to reusable stage functions.
"""

from pathlib import Path
import argparse
import os

STAGES = (
    "prepare-data",
    "split-engines",
    "train-linear",
    "train-forest",
    "train-hgb",
    "evaluate-test",
    "analyse-uncertainty",
    "plot-results",
    "benchmark",
)
# Retain old spellings for students following earlier terminal notes.
LEGACY_COMMANDS = {
    "preparedata": "prepare-data",
    "splitdata": "split-engines",
    "linear_models": "train-linear",
    "randomforest": "train-forest",
    "histgradientboosting": "train-hgb",
    "testevaluation": "evaluate-test",
    "uncertainty": "analyse-uncertainty",
    "plotbaseline": "plot-results",
    "performance": "benchmark",
}


def run_stage(stage: str, data_dir: Path, output_dir: Path) -> None:
    """Execute one named stage using explicit data and output directories.

    Args:
        stage: Canonical stage name in STAGES.
        data_dir: Directory containing verified original FD001 source files.
        output_dir: Prepared/intermediate result directory for the current run.

    Returns:
        None: Writes the selected stage's results and prints its summary.

    Raises:
        ValueError: If stage or input data is unsupported.
        FileNotFoundError: If raw data or an earlier-stage CSV is missing.
        OSError: If files cannot be read/written.

    Example: run_stage("prepare-data", Path("data/raw"), Path("results/generated")).
    """
    from rul_project.prepare_dataset import prepare_training_data, save_engine_split
    from rul_project.train_models import train_development_models
    from rul_project.evaluate_test_set import evaluate_official_test
    from rul_project.analyse_uncertainty import save_uncertainty_analysis
    from rul_project.plot_model_results import generate_result_figures
    from rul_project.benchmark_performance import run_performance_benchmark

    actions = {
        "prepare-data": lambda: prepare_training_data(data_dir, output_dir),
        "split-engines": lambda: save_engine_split(output_dir),
        "train-linear": lambda: train_development_models(output_dir, "linear"),
        "train-forest": lambda: train_development_models(output_dir, "forest"),
        "train-hgb": lambda: train_development_models(output_dir, "hgb"),
        "evaluate-test": lambda: evaluate_official_test(data_dir, output_dir),
        "analyse-uncertainty": lambda: save_uncertainty_analysis(output_dir),
        "plot-results": lambda: generate_result_figures(output_dir),
        "benchmark": lambda: run_performance_benchmark(output_dir),
    }
    if stage not in actions:
        raise ValueError(f"Unknown workflow stage: {stage}")
    print(f"\n=== {stage} ===")
    actions[stage]()


def main(argv: list[str] | None = None) -> None:
    """Parse paths, verify source data where needed and run requested stages.

    Args:
        argv: Optional argument list for programmatic use; None reads the CLI.

    Returns:
        None: Creates the requested run outputs or confirms raw-data fingerprints.

    Raises:
        SystemExit: For --help, invalid arguments or a protected output directory.
        ValueError: If fingerprints, labels or model inputs are invalid.
        FileNotFoundError: If source/prerequisite files are absent.
        OSError: If filesystem or explicitly requested network access fails.

    Example: main(["all", "--output-dir", "results/generated/trial"]).
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the NASA FD001 four-model RUL project"
    )
    parser.add_argument(
        "stage", choices=("data", "all") + STAGES + tuple(LEGACY_COMMANDS)
    )
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--data-source",
        type=Path,
        help="Local folder containing the three original FD001 files",
    )
    parser.add_argument(
        "--download-data", action="store_true", help="Fetch the official NASA archive"
    )
    args = parser.parse_args(argv)
    stage = LEGACY_COMMANDS.get(args.stage, args.stage)
    root = args.project_dir.resolve()
    data_dir = (args.data_dir or root / "data/raw").resolve()
    output_dir = (args.output_dir or root / "results/generated").resolve()
    protected = (
        root / "results/reference",
        root / "results/verification",
        root / "tests/reports",
    )
    if output_dir == root or any(
        output_dir.is_relative_to(path.resolve()) for path in protected
    ):
        parser.error(
            "Choose a generated-results directory outside protected reference/test evidence."
        )
    if output_dir == data_dir:
        parser.error("Raw data and generated results require separate directories.")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from rul_project.data_access import prepare_data

    if stage in {"data", "all", "prepare-data", "evaluate-test"}:
        prepare_data(
            data_dir, root / "data/manifest.json", args.data_source, args.download_data
        )
    elif args.data_source is not None or args.download_data:
        parser.error(
            "Data acquisition options apply to data, all, prepare-data or evaluate-test."
        )
    if stage == "data":
        print(f"Verified the three FD001 source files in {data_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits

    # Equal thread limits keep the ensemble/linear timing comparison consistent.
    with threadpool_limits(limits=1):
        for selected in STAGES if stage == "all" else (stage,):
            run_stage(selected, data_dir, output_dir)
