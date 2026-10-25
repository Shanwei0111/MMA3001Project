"""Generate API pages and verify their source and HTML fingerprints.

Run ``python scripts/build_docs.py`` after edits and commit docs/html together
with docs/source_manifest.json. CI runs --check and regenerates the pages.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
__all__ = ["source_hashes", "html_hashes", "main"]


def source_hashes() -> dict[str, str]:
    """Fingerprint all package modules and repository support scripts.

    Returns:
        dict[str, str]: Repository-relative POSIX paths and their SHA-256 hashes.

    Raises:
        OSError: If a source file cannot be read.

    Example: source_hashes()["src/rul_project/rul_utils.py"].
    """
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}


def html_hashes() -> dict[str, str]:
    """Fingerprint generated files with normalized text line endings.

    Returns:
        dict[str, str]: Documentation-relative paths and SHA-256 hashes.

    Raises:
        OSError: If a generated file cannot be read.

    Example: html_hashes(). Normalization handles platform checkout differences.
    """
    directory = ROOT / "docs/html"
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def main() -> None:
    """Build pdoc pages or reject stale/incomplete committed documentation.

    Returns:
        None: Writes HTML and source_manifest.json, or prints freshness status.

    Raises:
        SystemExit: If --check finds changed sources/HTML or missing pages.
        subprocess.CalledProcessError: If pdoc cannot import/render the API.
        OSError: If documentation files cannot be accessed.

    Usage: python scripts/build_docs.py [--check]. Requires pdoc 16.0.0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "docs/source_manifest.json"
    if args.check:
        if not manifest_path.is_file():
            raise SystemExit("Missing documentation manifest; run scripts/build_docs.py.")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["sources"] != source_hashes() or saved["html"] != html_hashes():
            raise SystemExit("Source or HTML changed; regenerate and commit the API documentation.")
        expected = ["index.html", "rul_project/rul_utils.html", "rul_project/cli.html", "rul_project/data_access.html"]
        if not all((ROOT / "docs/html" / name).is_file() for name in expected):
            raise SystemExit("API documentation is incomplete.")
        print(f"Documentation synchronized: {len(saved['sources'])} sources, {len(saved['html'])} generated files.")
        return
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([
        sys.executable, "-m", "pdoc", "--docformat", "google", "--no-search",
        "--footer-text", "MMA3001 | Chua Shanwei 35393343",
        "--output-directory", "docs/html", "rul_project",
        "scripts/build_docs.py", "scripts/verify_reproduction.py", "scripts/package_submission.py",
    ], cwd=ROOT, env=env, check=True)
    manifest = {"generator": "pdoc", "version": importlib.metadata.version("pdoc"),
                "docformat": "google", "sources": source_hashes(), "html": html_hashes()}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Generated API pages for {len(manifest['sources'])} source files.")


if __name__ == "__main__":
    main()
