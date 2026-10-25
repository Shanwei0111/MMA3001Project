"""Generate API pages and verify their source and HTML fingerprints.

Run ``python tools/build_api_documentation.py`` after edits and commit docs/api together
with docs/source_manifest.json. CI runs --check and regenerates the pages.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
__all__ = ["source_hashes", "html_hashes", "build_landing_pages", "main"]


def build_landing_pages() -> None:
    """Create the offline entry page and a rendered version of the README.

    Returns:
        None: Writes START_HERE.html and docs/project_guide.html from current content.

    Raises:
        OSError: If the README cannot be read or pages cannot be written.

    Usage: Called by this script after pdoc; no separate manual page editing is needed.
    """
    import markdown2

    style = """
    body{font:16px/1.6 system-ui,Arial,sans-serif;color:#173047;background:#f7f9fb;margin:0}
    main{max-width:1100px;margin:36px auto;padding:0 26px 40px}h1,h2,h3{color:#102f4d;line-height:1.25}
    h1{font-size:34px}a{color:#175d8f}p.lead{font-size:18px;color:#526577}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
    .card{display:block;text-decoration:none;background:white;border:1px solid #d7e0e8;border-radius:9px;padding:22px;color:inherit}
    .card:hover{border-color:#a26739;box-shadow:0 3px 10px #17304716}.card h2{font-size:21px;margin:0 0 8px}.card p{margin:0;color:#526577}
    .rule{height:4px;width:100px;background:#a26739;margin:16px 0 25px}footer{border-top:1px solid #d7e0e8;margin-top:30px;padding-top:14px;color:#526577}
    table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{border:1px solid #d7e0e8;padding:9px;text-align:left;vertical-align:top}th{background:#eaf0f5}
    pre{background:#eaf0f5;padding:18px;overflow:auto;border-radius:6px}code{font-family:Consolas,monospace;font-size:.9em}img{max-width:100%}
    @media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:28px}main{padding:0 16px 25px}}
    """
    prefix = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'
        + style
        + "</style>"
    )
    cards = [
        (
            "README and workflow",
            "Purpose, folder map, environment and complete reproduction commands.",
            "docs/project_guide.html",
        ),
        (
            "HTML source-code documentation",
            "Generated API pages with parameters, returns, exceptions and View Source.",
            "docs/api/index.html",
        ),
        (
            "Licensing agreement",
            "MIT license for original code. NASA data attribution is in data/README.md.",
            "LICENSE",
        ),
        (
            "Source code",
            "Purpose-named Python modules for data, models, evaluation, plots and performance.",
            "docs/api/rul_project.html",
        ),
        (
            "Test functions and report",
            "Open the recorded readable pytest report. Test functions are in tests/.",
            "tests/reports/pytest_report.html",
        ),
        (
            "Final GitHub repository",
            "Accessible main branch, meaningful commits and preserved development history.",
            "https://github.com/Shanwei0111/MMA3001Project",
        ),
    ]
    body = '<h1>MMA3001 project repository</h1><div class="rule"></div><p class="lead">Chua Shanwei · 35393343<br>Turbofan Remaining Useful Life Prediction</p><div class="grid">'
    body += "".join(
        f'<a class="card" href="{url}"><h2>{title}</h2><p>{description}</p></a>'
        for title, description, url in cards
    )
    body += '</div><p><a href="reports/MMA3001_Chua_Shanwei_35393343_Written_Project_Documentation.pdf">Open the written report PDF</a> · <a href="results/reference/README.md">Browse the frozen result folders</a></p><footer>Start with the workflow guide. Source, tests, data, results and reports each have a dedicated folder. The hidden .git folder preserves version history.</footer>'
    (ROOT / "START_HERE.html").write_text(
        prefix
        + "<title>MMA3001 - Start here</title></head><body><main>"
        + body
        + "</main></body></html>\n",
        encoding="utf-8",
    )
    guide = str(
        markdown2.markdown(
            (ROOT / "README.md").read_text(encoding="utf-8"),
            extras=["tables", "fenced-code-blocks", "header-ids"],
        )
    )
    # The README is rendered one folder lower; preserve working local links.
    guide = re.sub(
        r'href="([^\"]+)"',
        lambda match: 'href="'
        + (
            match.group(1)
            if match.group(1).startswith(("http:", "https:", "#", "mailto:"))
            else "../" + match.group(1)
        )
        + '"',
        guide,
    )
    (ROOT / "docs/project_guide.html").write_text(
        prefix
        + '<title>MMA3001 - Project guide</title></head><body><main><p><a href="../START_HERE.html">Back to project entry</a></p>'
        + guide
        + "</main></body></html>\n",
        encoding="utf-8",
    )


def source_hashes() -> dict[str, str]:
    """Fingerprint all package modules and repository support scripts.

    Returns:
        dict[str, str]: Repository-relative POSIX paths and their SHA-256 hashes.

    Raises:
        OSError: If a source file cannot be read.

    Example: source_hashes()["src/rul_project/prepare_dataset.py"].
    """
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    return {
        p.relative_to(ROOT)
        .as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest()
        for p in files
    }


def html_hashes() -> dict[str, str]:
    """Fingerprint generated files with normalized text line endings.

    Returns:
        dict[str, str]: Documentation-relative paths and SHA-256 hashes.

    Raises:
        OSError: If a generated file cannot be read.

    Example: html_hashes(). Normalization handles platform checkout differences.
    """
    directory = ROOT / "docs/api"
    return {
        p.relative_to(directory)
        .as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def main() -> None:
    """Build pdoc pages or reject stale/incomplete committed documentation.

    Returns:
        None: Writes HTML and source_manifest.json, or prints freshness status.

    Raises:
        SystemExit: If --check finds changed sources/HTML or missing pages.
        subprocess.CalledProcessError: If pdoc cannot import/render the API.
        OSError: If documentation files cannot be accessed.

    Usage: python tools/build_api_documentation.py [--check]. Requires pdoc 16.0.0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "docs/source_manifest.json"
    if args.check:
        if not manifest_path.is_file():
            raise SystemExit(
                "Missing documentation manifest; run tools/build_api_documentation.py."
            )
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["sources"] != source_hashes() or saved["html"] != html_hashes():
            raise SystemExit(
                "Source or HTML changed; regenerate and commit the API documentation."
            )
        for name, fingerprint in saved["guide_sources"].items():
            if (
                hashlib.sha256(
                    (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                != fingerprint
            ):
                raise SystemExit(
                    "The README guide changed; regenerate the entry pages."
                )
        for name, fingerprint in saved["entry_pages"].items():
            if (
                hashlib.sha256(
                    (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                != fingerprint
            ):
                raise SystemExit(
                    "Generated entry pages changed; rebuild the documentation."
                )
        expected = [
            "index.html",
            "rul_project/rul_metrics.html",
            "rul_project/run_project.html",
            "rul_project/data_access.html",
        ]
        if not all((ROOT / "docs/api" / name).is_file() for name in expected):
            raise SystemExit("API documentation is incomplete.")
        print(
            f"Documentation synchronized: {len(saved['sources'])} sources, {len(saved['html'])} generated files."
        )
        return
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pdoc",
            "--docformat",
            "google",
            "--no-search",
            "--footer-text",
            "MMA3001 | Chua Shanwei 35393343",
            "--output-directory",
            "docs/api",
            "rul_project",
            "tools/build_api_documentation.py",
            "tools/verify_reproducibility.py",
            "tools/create_submission_zip.py",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    build_landing_pages()
    manifest = {
        "generator": "pdoc",
        "version": importlib.metadata.version("pdoc"),
        "docformat": "google",
        "sources": source_hashes(),
        "html": html_hashes(),
        "guide_sources": {
            "README.md": hashlib.sha256(
                (ROOT / "README.md").read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
        },
        "entry_pages": {
            name: hashlib.sha256(
                (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for name in ["START_HERE.html", "docs/project_guide.html"]
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated API pages for {len(manifest['sources'])} source files.")


if __name__ == "__main__":
    main()
