import csv
import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_uji_mag import (
    AcquisitionError,
    MANIFEST_COLUMNS,
    acquire_dataset,
    classify_trajectory,
    discover_trajectories,
    read_manifest,
    write_manifest,
)


def _write_trajectory(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("UJIIndoorLoc-Mag/train/lines/l3r_03.txt", b"training-line\n")
        archive.writestr("UJIIndoorLoc-Mag/train/curves/c5n1r_05.txt", b"training-curve\n")
        archive.writestr("UJIIndoorLoc-Mag/tests/tt03.txt", b"public-test\n")
        archive.writestr("UJIIndoorLoc-Mag/README.md", b"metadata only\n")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("lines/l3r_03.txt", ("train_public", "line")),
        ("curves/c5n1r_05.txt", ("train_public", "curve")),
        ("tests/tt03.txt", ("test_public", "test")),
        ("tests/future_name.txt", ("test_public", "unknown")),
        ("notes/readme.md", ("unknown", "unknown")),
    ],
)
def test_filename_and_split_classification(path, expected):
    assert classify_trajectory(path) == expected


def test_manifest_generation_has_required_schema_and_hashes(tmp_path):
    raw = tmp_path / "raw"
    _write_trajectory(raw / "lines" / "l1n_01.txt", b"line-one\n")
    _write_trajectory(raw / "tests" / "tt01.txt", b"test-one\n")
    report = discover_trajectories(raw)
    manifest = tmp_path / "raw_manifest.csv"
    write_manifest(report.records, manifest)
    rows = read_manifest(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        assert tuple(csv.DictReader(handle).fieldnames) == MANIFEST_COLUMNS
    assert len(rows) == 2
    assert all(len(row.sha256) == 64 for row in rows)
    assert {row.source_split for row in rows} == {"train_public", "test_public"}


def test_duplicate_trajectory_id_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    _write_trajectory(raw / "a" / "l1n_01.txt", b"first\n")
    _write_trajectory(raw / "b" / "l1n_01.txt", b"second\n")
    _write_trajectory(raw / "tests" / "tt01.txt", b"test\n")
    with pytest.raises(AcquisitionError, match="Duplicate trajectory IDs"):
        discover_trajectories(raw)


def test_duplicate_hash_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    _write_trajectory(raw / "lines" / "l1n_01.txt", b"same\n")
    _write_trajectory(raw / "lines" / "l2n_01.txt", b"same\n")
    _write_trajectory(raw / "tests" / "tt01.txt", b"test\n")
    with pytest.raises(AcquisitionError, match="Duplicate trajectory file hashes"):
        discover_trajectories(raw)


def test_local_acquisition_is_idempotent(tmp_path):
    archive = tmp_path / "fixture.zip"
    _make_archive(archive)
    dataset_dir = tmp_path / "dataset"
    first_report, first_changed = acquire_dataset(dataset_dir, local_archive=archive)
    manifest = dataset_dir / "manifests" / "raw_manifest.csv"
    source = dataset_dir / "manifests" / "source.json"
    first_manifest_bytes = manifest.read_bytes()
    first_source_bytes = source.read_bytes()
    raw_mtimes = {p.relative_to(dataset_dir): p.stat().st_mtime_ns for p in (dataset_dir / "raw").rglob("*") if p.is_file()}
    second_report, second_changed = acquire_dataset(dataset_dir, local_archive=archive)
    assert first_changed is True
    assert second_changed is False
    assert second_report.records == first_report.records
    assert manifest.read_bytes() == first_manifest_bytes
    assert source.read_bytes() == first_source_bytes
    assert {p.relative_to(dataset_dir): p.stat().st_mtime_ns for p in (dataset_dir / "raw").rglob("*") if p.is_file()} == raw_mtimes
    metadata = json.loads(source.read_text(encoding="utf-8"))
    assert metadata["uci_dataset_id"] == 343
    assert metadata["test_public_count"] == 1


def test_existing_raw_data_must_match_manifest(tmp_path):
    archive = tmp_path / "fixture.zip"
    _make_archive(archive)
    dataset_dir = tmp_path / "dataset"
    acquire_dataset(dataset_dir, local_archive=archive)
    next((dataset_dir / "raw").rglob("l3r_03.txt")).write_bytes(b"changed\n")
    with pytest.raises(AcquisitionError, match="does not match"):
        acquire_dataset(dataset_dir, local_archive=archive)


def test_force_replaces_dataset_only_after_validating_staged_archive(tmp_path):
    first_archive = tmp_path / "first.zip"
    second_archive = tmp_path / "second.zip"
    _make_archive(first_archive)
    _make_archive(second_archive)
    with zipfile.ZipFile(second_archive, "a") as archive:
        archive.writestr("UJIIndoorLoc-Mag/train/lines/l4n_01.txt", b"another-line\n")
    dataset_dir = tmp_path / "dataset"
    first_report, _ = acquire_dataset(dataset_dir, local_archive=first_archive)
    second_report, changed = acquire_dataset(dataset_dir, force=True, local_archive=second_archive)
    assert changed is True
    assert len(second_report.records) == len(first_report.records) + 1
    assert len(read_manifest(dataset_dir / "manifests" / "raw_manifest.csv")) == 4


def test_unsafe_zip_member_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", b"nope")
    with pytest.raises(AcquisitionError, match="Unsafe archive path"):
        acquire_dataset(tmp_path / "dataset", local_archive=archive)
