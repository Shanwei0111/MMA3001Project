"""Create a Moodle zipped clone from remote main and verified local FD001 data."""
from pathlib import Path
import argparse
import json
import subprocess
import tempfile
import zipfile

REMOTE = "https://github.com/Shanwei0111/MMA3001Project.git"


def main() -> None:
    """Clone remote main, optionally add verified NASA data, and create a ZIP.

    Returns:
        None: Writes the requested ZIP; prints its commit, path and size.

    Raises:
        SystemExit: If the destination exists, preventing silent overwrite.
        subprocess.CalledProcessError: If Git cannot clone/read remote main.
        ValueError: If supplied data differs from the committed fingerprints.
        OSError: If local or network access fails.

    Usage: python scripts/package_submission.py --destination ../submission.zip
    --data-source data/raw. Git metadata is retained as development evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--data-source", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}; choose a new name.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    from rul_project.data_access import prepare_data
    with tempfile.TemporaryDirectory(prefix="mma3001-clone-") as folder:
        clone = Path(folder) / "MMA3001Project"
        subprocess.run(["git", "clone", "--branch", "main", REMOTE, str(clone)], check=True)
        revision = subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip()
        if args.data_source is not None:
            prepare_data(clone / "data/raw", clone / "data/manifest.json", source=args.data_source.resolve())
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
            for path in sorted(clone.rglob("*")):
                if path.is_file():
                    archive.write(path, "MMA3001Project/" + path.relative_to(clone).as_posix())
        print(json.dumps({"clone_commit": revision, "zip": str(destination), "bytes": destination.stat().st_size,
                          "raw_data_included": args.data_source is not None, "git_history_included": True}, indent=2))


if __name__ == "__main__":
    main()
