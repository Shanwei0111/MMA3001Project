"""Command-line workflows for development, frozen evaluation and prediction."""
import argparse
import json
from pathlib import Path
from .data import read_cmapss
from .experiment import develop, evaluate_official
from .models import load_model
from .profiling import profile_project


def main() -> None:
    """Parse arguments and execute a reproducible local workflow.

    Run python -m jet_rul --help for available commands. Source data default to
    data/raw, with a fallback to the supplied sibling CMAPSSData folder.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["develop", "evaluate", "profile", "figures", "reproduce", "predict"])
    parser.add_argument("--config", type=Path, default=Path("configs/fd001.json"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--engine", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_dir = args.data_dir or (Path("data/raw") if Path("data/raw/train_FD001.txt").exists() else Path("../CMAPSSData"))
    if args.command in ("develop", "reproduce"):
        develop(data_dir, config, args.output_dir, args.manifest)
    if args.command in ("evaluate", "reproduce"):
        evaluate_official(data_dir, config, args.output_dir, args.manifest)
    if args.command in ("profile", "reproduce"):
        profile_project(data_dir, config, args.output_dir)
    if args.command in ("figures", "reproduce"):
        from .plotting import generate_figures
        generate_figures(data_dir, args.output_dir)
    if args.command == "predict":
        selection = json.loads((args.output_dir/"selection.json").read_text())
        bundle = load_model(args.output_dir/"models"/f"{selection['selected']}.joblib")
        observed = read_cmapss(data_dir/"test_FD001.txt")
        result = bundle.predict(observed.loc[observed.engine_id == args.engine])
        print(json.dumps(result, indent=2))
