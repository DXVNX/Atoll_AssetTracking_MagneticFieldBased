"""Safely acquire and freeze the UJIIndoorLoc-Mag raw trajectory files.

Trajectory files remain opaque bytes here. Parsing sensor rows belongs to the
next project phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable


DATASET_NAME = "UJIIndoorLoc-Mag"
UCI_DATASET_ID = 343
DOI = "10.24432/C5D311"
LICENSE = "CC BY 4.0"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/343/ujiindoorloc+mag.zip"
ARCHIVE_FILENAME = "ujiindoorloc+mag.zip"
EXPECTED_TRAIN_COUNT = 270
EXPECTED_TEST_COUNT = 11
MANIFEST_COLUMNS = (
    "trajectory_id",
    "relative_path",
    "source_split",
    "trajectory_type",
    "filename",
    "size_bytes",
    "sha256",
)

LINE_RE = re.compile(r"^l\d+[nr]?_\d+\.txt$", re.IGNORECASE)
CURVE_RE = re.compile(r"^c\d+[nr]?\d+[nr]?_\d+\.txt$", re.IGNORECASE)
TEST_RE = re.compile(r"^tt\d+\.txt$", re.IGNORECASE)


class AcquisitionError(RuntimeError):
    """Raised when an archive or frozen raw dataset fails validation."""


@dataclass(frozen=True)
class ManifestRecord:
    trajectory_id: str
    relative_path: str
    source_split: str
    trajectory_type: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DiscoveryReport:
    records: tuple[ManifestRecord, ...]
    warnings: tuple[str, ...]
    unexpected_files: tuple[str, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_trajectory(relative_path: str | Path) -> tuple[str, str]:
    """Return ``(source_split, trajectory_type)`` without opening the file."""
    path = PurePosixPath(str(relative_path).replace("\\", "/"))
    filename = path.name
    folder_parts = {part.lower() for part in path.parts[:-1]}

    if TEST_RE.fullmatch(filename):
        return "test_public", "test"
    if LINE_RE.fullmatch(filename):
        return "train_public", "line"
    if CURVE_RE.fullmatch(filename):
        return "train_public", "curve"
    if filename.lower().endswith(".txt"):
        if folder_parts & {"test", "tests", "testing"}:
            return "test_public", "unknown"
        if folder_parts & {"line", "lines", "curve", "curves", "train", "training"}:
            return "train_public", "unknown"
    return "unknown", "unknown"


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\x00" in handle.read(8192)


def discover_trajectories(raw_dir: Path) -> DiscoveryReport:
    """Discover trajectory files recursively and validate byte-level integrity."""
    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        raise AcquisitionError(f"Raw dataset directory does not exist: {raw_dir}")

    records: list[ManifestRecord] = []
    unexpected: list[str] = []
    zero_byte: list[str] = []
    binary: list[str] = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(raw_dir).as_posix()
        source_split, trajectory_type = classify_trajectory(relative)
        if source_split == "unknown":
            unexpected.append(relative)
            continue
        size = path.stat().st_size
        if size == 0:
            zero_byte.append(relative)
        if _looks_binary(path):
            binary.append(relative)
        records.append(
            ManifestRecord(
                trajectory_id=path.stem,
                relative_path=relative,
                source_split=source_split,
                trajectory_type=trajectory_type,
                filename=path.name,
                size_bytes=size,
                sha256=sha256_file(path),
            )
        )

    if not records:
        raise AcquisitionError(f"No UJI trajectory files discovered under {raw_dir}")
    if zero_byte:
        raise AcquisitionError(f"Zero-byte trajectory files: {', '.join(zero_byte)}")
    if binary:
        raise AcquisitionError(f"Unexpected binary trajectory files: {', '.join(binary)}")

    ids: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        ids[record.trajectory_id].append(record.relative_path)
        hashes[record.sha256].append(record.relative_path)
    duplicate_ids = {key: paths for key, paths in ids.items() if len(paths) > 1}
    duplicate_hashes = {key: paths for key, paths in hashes.items() if len(paths) > 1}
    if duplicate_ids:
        detail = "; ".join(f"{key}: {paths}" for key, paths in duplicate_ids.items())
        raise AcquisitionError(f"Duplicate trajectory IDs detected: {detail}")
    if duplicate_hashes:
        detail = "; ".join(f"{key}: {paths}" for key, paths in duplicate_hashes.items())
        raise AcquisitionError(f"Duplicate trajectory file hashes detected: {detail}")

    train_count = sum(r.source_split == "train_public" for r in records)
    test_count = sum(r.source_split == "test_public" for r in records)
    if test_count == 0:
        raise AcquisitionError("No public test trajectory files were discovered")

    warnings: list[str] = []
    if train_count != EXPECTED_TRAIN_COUNT:
        warnings.append(f"Expected approximately {EXPECTED_TRAIN_COUNT} training trajectories; found {train_count}.")
    if test_count != EXPECTED_TEST_COUNT:
        warnings.append(f"Expected approximately {EXPECTED_TEST_COUNT} test trajectories; found {test_count}.")
    if unexpected:
        warnings.append(f"Ignored {len(unexpected)} unexpected non-trajectory file(s).")
    unknown_types = sum(r.trajectory_type == "unknown" for r in records)
    if unknown_types:
        warnings.append(f"Classified {unknown_types} trajectory file(s) with unknown type.")
    return DiscoveryReport(tuple(records), tuple(warnings), tuple(unexpected))


def validate_zip_members(archive_path: Path) -> None:
    """Reject absolute paths, traversal, links, and encrypted archive members."""
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts or (member.parts and ":" in member.parts[0]):
                raise AcquisitionError(f"Unsafe archive path: {info.filename}")
            if info.flag_bits & 0x1:
                raise AcquisitionError(f"Encrypted archive member is unsupported: {info.filename}")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise AcquisitionError(f"Symbolic links are not allowed in the archive: {info.filename}")


def extract_archive_safely(archive_path: Path, destination: Path) -> None:
    validate_zip_members(archive_path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)


def write_manifest(records: Iterable[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.relative_path):
            writer.writerow(asdict(record))


def read_manifest(path: Path) -> tuple[ManifestRecord, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
                raise AcquisitionError(f"Manifest schema mismatch in {path}")
            return tuple(
                ManifestRecord(
                    trajectory_id=row["trajectory_id"],
                    relative_path=row["relative_path"],
                    source_split=row["source_split"],
                    trajectory_type=row["trajectory_type"],
                    filename=row["filename"],
                    size_bytes=int(row["size_bytes"]),
                    sha256=row["sha256"],
                )
                for row in reader
            )
    except (OSError, ValueError, KeyError) as exc:
        raise AcquisitionError(f"Could not read manifest {path}: {exc}") from exc


def verify_frozen_dataset(raw_dir: Path, manifest_path: Path) -> DiscoveryReport:
    if not manifest_path.is_file():
        raise AcquisitionError(f"Existing raw data has no manifest: {manifest_path}")
    stored = sorted(read_manifest(manifest_path), key=lambda item: item.relative_path)
    discovered = discover_trajectories(raw_dir)
    current = sorted(discovered.records, key=lambda item: item.relative_path)
    if current != stored:
        raise AcquisitionError(
            "Existing raw data does not match raw_manifest.csv. Refusing to overwrite it without --force."
        )
    return discovered


def download_archive(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mag-indoor-tracking/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as exc:
        raise AcquisitionError(f"Download failed from {url}: {exc}") from exc


def _source_metadata(archive_path: Path, report: DiscoveryReport) -> dict[str, object]:
    return {
        "dataset_name": DATASET_NAME,
        "uci_dataset_id": UCI_DATASET_ID,
        "doi": DOI,
        "license": LICENSE,
        "source_url": SOURCE_URL,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_filename": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "trajectory_count": len(report.records),
        "train_public_count": sum(r.source_split == "train_public" for r in report.records),
        "test_public_count": sum(r.source_split == "test_public" for r in report.records),
        "warnings": list(report.warnings),
        "unexpected_files": list(report.unexpected_files),
    }


def _write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def acquire_dataset(
    dataset_dir: Path,
    *,
    force: bool = False,
    local_archive: Path | None = None,
    downloader: Callable[[str, Path], None] = download_archive,
) -> tuple[DiscoveryReport, bool]:
    """Acquire or verify UJI data, returning ``(report, changed)``."""
    dataset_dir = dataset_dir.resolve()
    raw_dir = dataset_dir / "raw"
    manifests_dir = dataset_dir / "manifests"
    manifest_path = manifests_dir / "raw_manifest.csv"
    source_path = manifests_dir / "source.json"
    (dataset_dir / "interim").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "processed").mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    if raw_dir.exists() and any(raw_dir.rglob("*")) and not force:
        if not source_path.is_file():
            raise AcquisitionError(f"Existing raw data has no source metadata: {source_path}")
        return verify_frozen_dataset(raw_dir, manifest_path), False

    with tempfile.TemporaryDirectory(prefix="uji-mag-acquire-") as temporary:
        temp_root = Path(temporary)
        archive_path = temp_root / (local_archive.name if local_archive else ARCHIVE_FILENAME)
        if local_archive is not None:
            if not local_archive.is_file():
                raise AcquisitionError(f"Local archive does not exist: {local_archive}")
            shutil.copy2(local_archive, archive_path)
        else:
            downloader(SOURCE_URL, archive_path)

        staging = dataset_dir / ".raw-staging"
        backup = dataset_dir / ".raw-backup"
        manifest_backup = manifests_dir / ".raw_manifest.csv.backup"
        source_backup = manifests_dir / ".source.json.backup"
        stale = [path for path in (staging, backup, manifest_backup, source_backup) if path.exists()]
        if stale:
            raise AcquisitionError(f"Stale acquisition paths require manual inspection: {stale}")

        extract_archive_safely(archive_path, staging)
        try:
            report = discover_trajectories(staging)
            staged_manifest = temp_root / "raw_manifest.csv"
            staged_source = temp_root / "source.json"
            write_manifest(report.records, staged_manifest)
            _write_json(_source_metadata(archive_path, report), staged_source)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        raw_swapped = False
        manifest_installed = False
        source_installed = False
        try:
            if raw_dir.exists():
                raw_dir.rename(backup)
            if manifest_path.exists():
                manifest_path.rename(manifest_backup)
            if source_path.exists():
                source_path.rename(source_backup)
            staging.rename(raw_dir)
            raw_swapped = True
            shutil.move(staged_manifest, manifest_path)
            manifest_installed = True
            shutil.move(staged_source, source_path)
            source_installed = True
            verify_frozen_dataset(raw_dir, manifest_path)
        except Exception:
            if raw_swapped and raw_dir.exists():
                shutil.rmtree(raw_dir)
            if backup.exists():
                backup.rename(raw_dir)
            if manifest_installed and manifest_path.exists():
                manifest_path.unlink()
            if source_installed and source_path.exists():
                source_path.unlink()
            if manifest_backup.exists():
                manifest_backup.rename(manifest_path)
            if source_backup.exists():
                source_backup.rename(source_path)
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        for old_path in (backup, manifest_backup, source_backup):
            if old_path.is_dir():
                shutil.rmtree(old_path)
            elif old_path.exists():
                old_path.unlink()
        return report, True


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "external" / "ujiindoorloc_mag"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace raw data after staging and validation.")
    parser.add_argument("--archive", type=Path, help="Use a local ZIP instead of downloading.")
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset_dir(), help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report, changed = acquire_dataset(args.dataset_dir, force=args.force, local_archive=args.archive)
    except AcquisitionError as exc:
        print(f"ERROR: {exc}")
        return 1
    train_count = sum(r.source_split == "train_public" for r in report.records)
    test_count = sum(r.source_split == "test_public" for r in report.records)
    action = "acquired" if changed else "verified; no files rewritten"
    print(f"UJIIndoorLoc-Mag {action}.")
    print(f"Training trajectories: {train_count}")
    print(f"Test trajectories: {test_count}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
