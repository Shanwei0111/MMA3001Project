"""Command-line entry points for the student's reproducible learning workflow.

Use ``python -m rul_project --help``. Run from the repository root, or pass
--project-dir, --data-dir and --output-dir explicitly. Generated results are
written separately from the immutable evidence/reported snapshot.
"""
from pathlib import Path
import argparse
import importlib
import os

STAGES = (
    "preparedata", "splitdata", "linear_models", "randomforest",
    "histgradientboosting", "testevaluation", "plotbaseline", "performance", "uncertainty",
)


def main(argv: list[str] | None = None) -> None:
    """Parse workflow options and explicitly execute selected stages.

    Args:
        argv: Optional argument list for programmatic invocation; None reads
            the process command line. Do not reuse different path configurations
            in one Python process because stage settings are cached on import.

    Returns:
        None: Creates requested output files or reports verified raw-data hashes.

    Raises:
        SystemExit: For --help or invalid command-line syntax.
        FileNotFoundError: For missing source/prerequisite files.
        ValueError: For invalid raw fingerprints or stage data.
        OSError: For filesystem or explicitly requested download failures.

    Example: main(["preparedata", "--output-dir", "outputs/trial"]).
    """
    parser = argparse.ArgumentParser(description="NASA FD001 four-model RUL workflow")
    parser.add_argument("stage", choices=("data", "all") + STAGES)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-source", type=Path, help="Folder containing the original three FD001 files")
    parser.add_argument("--download-data", action="store_true", help="Explicitly request the official NASA archive")
    args = parser.parse_args(argv)
    root = args.project_dir.resolve()
    data_dir = (args.data_dir or root / "data/raw").resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    if output_dir == (root / "evidence/reported").resolve():
        parser.error("Use a separate output directory; evidence/reported is the frozen report snapshot.")
    os.environ["RUL_DATA_DIR"] = str(data_dir)
    os.environ["RUL_OUTPUT_DIR"] = str(output_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    from rul_project.data_access import prepare_data
    if args.stage in {"data", "all", "preparedata", "testevaluation"}:
        prepare_data(data_dir, root / "data/manifest.json", args.data_source, args.download_data)
    elif args.data_source is not None or args.download_data:
        parser.error("Data acquisition options apply to data, all, preparedata or testevaluation.")
    if args.stage == "data":
        print(f"Verified three FD001 source files in {data_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    # Use the same thread budget for all models; each stage keeps original seeds.
    with threadpool_limits(limits=1):
        for stage in STAGES if args.stage == "all" else (args.stage,):
            print(f"\n=== {stage} ===")
            importlib.import_module(f"rul_project.steps.{stage}").main()
