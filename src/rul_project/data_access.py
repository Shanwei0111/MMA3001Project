"""Acquire and verify the three FD001 files without silently changing source data.

NASA data is attributed in data/README.md and is not covered by this project's
MIT license. Only the three named archive members are copied; paths in remote
archives are never extracted directly onto the filesystem.
"""

from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

ARCHIVE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


def verify_data(data_dir: Path, manifest_path: Path) -> dict:
    """Validate raw FD001 file bytes against the committed SHA-256 manifest.

    Args:
        data_dir: Directory containing the three original NASA text files.
        manifest_path: JSON mapping filenames to sha256 and byte-count records.

    Returns:
        dict: Verified manifest records, keyed by source filename.

    Raises:
        FileNotFoundError: If the manifest or a required raw file is absent.
        ValueError: If a fingerprint differs from the recorded source bytes.
        OSError: If a file cannot be read.

    Example: verify_data(Path("data/raw"), Path("data/manifest.json")).
    No source file is modified by this function.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest.items():
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run the data stage with --data-source or --download-data."
            )
        content = path.read_bytes()
        if (
            len(content) != record["bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError(
                f"Source fingerprint mismatch: {name}; obtain the original unmodified FD001 file."
            )
    return manifest


def prepare_data(
    data_dir: Path,
    manifest_path: Path,
    source: Path | None = None,
    download: bool = False,
) -> dict:
    """Populate raw data from a local NASA folder or the official NASA archive.

    Args:
        data_dir: Destination directory for raw FD001 files.
        manifest_path: Committed JSON file containing expected fingerprints.
        source: Optional local directory containing the three raw text files.
        download: Whether to retrieve NASA's public archive when source is absent.

    Returns:
        dict: Validated source manifest from verify_data().

    Raises:
        ValueError: If both acquisition options are requested, existing files are
            corrupt, archive members are ambiguous or downloaded bytes mismatch.
        FileNotFoundError: If data/source/manifest files are absent.
        OSError: If local or network access fails.
        zipfile.BadZipFile: If the remote payload is not a valid ZIP.

    Existing files are verified, never silently overwritten. Example:
    prepare_data(Path("data/raw"), Path("data/manifest.json"), source=Path("../CMAPSSData")).
    """
    if source is not None and download:
        raise ValueError("Choose either a local source or a NASA download.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, record in manifest.items():
        existing = data_dir / name
        if (
            existing.exists()
            and hashlib.sha256(existing.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(
                f"Refusing to overwrite a mismatched existing file: {existing}"
            )
    payloads = {}
    if source is not None:
        payloads = {name: (source / name).read_bytes() for name in manifest}
    elif download:
        # Read a bounded response and copy only explicitly named source members.
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
            archive_bytes = response.read(100_000_001)
        if len(archive_bytes) > 100_000_000:
            raise ValueError("NASA archive exceeds the 100 MB acquisition limit.")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for name in manifest:
                matches = [
                    item for item in archive.namelist() if Path(item).name == name
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one archive member named {name}.")
                payloads[name] = archive.read(matches[0])
    for name, content in payloads.items():
        if hashlib.sha256(content).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(
                f"Acquired data does not match the recorded source: {name}"
            )
    for name, content in payloads.items():
        if not (data_dir / name).exists():
            (data_dir / name).write_bytes(content)
    return verify_data(data_dir, manifest_path)
