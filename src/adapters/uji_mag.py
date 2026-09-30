"""Parser for the raw UJIIndoorLoc-Mag trajectory text format.

This adapter only decodes published samples and segment annotations.  It does
not rotate vectors, create windows, extract features, or evaluate models.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd

SAMPLE_COLUMNS = (
    "timestamp", "mx", "my", "mz", "ax", "ay", "az",
    "orientation_x", "orientation_y", "orientation_z",
)
OUTPUT_COLUMNS = (
    "sample_idx", "timestamp", "elapsed_s", "mx", "my", "mz", "ax", "ay", "az",
    "orientation_x", "orientation_y", "orientation_z", "trajectory_id", "source_split",
    "trajectory_type", "segment_index", "segment_id", "directional_segment_id",
)
COORDINATE_DECIMALS = 7
_SEPARATOR = re.compile(r"^(?P<prefix>.*?)<(?P<count>\d+)>\s*$")


class UJIParseError(ValueError):
    """A structured error for a malformed raw trajectory."""

    def __init__(self, path: Path, trajectory_id: str, reason: str, line_number: int | None = None):
        self.path = Path(path)
        self.trajectory_id = trajectory_id
        self.reason = reason
        self.line_number = line_number
        location = f" line {line_number}" if line_number is not None else ""
        super().__init__(f"{self.path} [{trajectory_id}]{location}: {reason}")


@dataclass(frozen=True)
class UJISegment:
    segment_index: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    first_sample: int
    last_sample: int
    segment_id: str
    directional_segment_id: str


@dataclass
class UJITrajectory:
    trajectory_id: str
    source_split: str
    trajectory_type: str
    path: Path
    samples: pd.DataFrame
    segments: tuple[UJISegment, ...]
    warnings: tuple[str, ...]
    median_dt_ms: float | None
    sampling_rate_est_hz: float | None
    max_gap_ms: float | None
    nonpositive_timestamp_count: int


def canonical_endpoint(lat: float, lon: float) -> tuple[float, float]:
    return round(float(lat), COORDINATE_DECIMALS), round(float(lon), COORDINATE_DECIMALS)


def _endpoint_token(point: tuple[float, float]) -> str:
    return f"{point[0]:.{COORDINATE_DECIMALS}f},{point[1]:.{COORDINATE_DECIMALS}f}"


def segment_identifiers(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> tuple[str, str]:
    start = canonical_endpoint(start_lat, start_lon)
    end = canonical_endpoint(end_lat, end_lon)
    unordered = "|".join(_endpoint_token(p) for p in sorted((start, end)))
    directed = f"{_endpoint_token(start)}->{_endpoint_token(end)}"
    return (
        "seg_" + hashlib.sha1(unordered.encode("ascii")).hexdigest()[:12],
        "dir_" + hashlib.sha1(directed.encode("ascii")).hexdigest()[:12],
    )


def _error(path: Path, trajectory_id: str, reason: str, line_number: int | None = None) -> UJIParseError:
    return UJIParseError(path, trajectory_id, reason, line_number)


def parse_uji_trajectory(
    path: str | Path,
    trajectory_id: str,
    source_split: str,
    trajectory_type: str,
) -> UJITrajectory:
    """Parse one UJI trajectory using zero-based, inclusive FS/LS annotations."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _error(path, trajectory_id, f"cannot read file: {exc}") from exc
    if not lines:
        raise _error(path, trajectory_id, "empty file")

    separator_index = None
    segment_count = None
    warnings: list[str] = []
    for index, raw in enumerate(lines):
        match = _SEPARATOR.match(raw.strip())
        if match:
            separator_index = index
            segment_count = int(match.group("count"))
            if match.group("prefix"):
                warnings.append(f"MALFORMED_SEPARATOR_PREFIX:{match.group('prefix')}")
            break
    if separator_index is None or segment_count is None:
        raise _error(path, trajectory_id, "missing <segment-count> separator")

    sensor_rows: list[list[float]] = []
    for index, raw in enumerate(lines[:separator_index]):
        fields = raw.split()
        if len(fields) != 10:
            raise _error(path, trajectory_id, f"sensor row has {len(fields)} fields; expected 10", index + 1)
        try:
            values = [float(value) for value in fields]
        except ValueError as exc:
            raise _error(path, trajectory_id, "sensor row contains a non-numeric value", index + 1) from exc
        sensor_rows.append(values)
    if not sensor_rows:
        raise _error(path, trajectory_id, "trajectory has no sensor rows")

    config_lines = [(index, raw) for index, raw in enumerate(lines[separator_index + 1 :], separator_index + 1) if raw.strip()]
    if len(config_lines) != segment_count:
        raise _error(path, trajectory_id, f"declares {segment_count} segments but has {len(config_lines)} configuration rows", separator_index + 1)

    segments: list[UJISegment] = []
    expected_first = 0
    for segment_index, (line_index, raw) in enumerate(config_lines):
        fields = raw.split()
        if len(fields) != 6:
            raise _error(path, trajectory_id, f"configuration row has {len(fields)} fields; expected 6", line_index + 1)
        try:
            start_lat, start_lon, end_lat, end_lon = map(float, fields[:4])
            first_sample, declared_last = map(int, fields[4:])
        except ValueError as exc:
            raise _error(path, trajectory_id, "configuration row contains an invalid number", line_index + 1) from exc
        if first_sample != expected_first:
            raise _error(path, trajectory_id, f"non-contiguous FS {first_sample}; expected {expected_first}", line_index + 1)
        last_sample = declared_last
        if last_sample == len(sensor_rows) and segment_index == segment_count - 1:
            last_sample -= 1
            warnings.append(f"FINAL_LS_ONE_PAST_END:{declared_last}->{last_sample}")
        if first_sample < 0 or last_sample < first_sample or last_sample >= len(sensor_rows):
            raise _error(path, trajectory_id, f"invalid inclusive sample range {first_sample}..{declared_last} for {len(sensor_rows)} samples", line_index + 1)
        segment_id, directional_id = segment_identifiers(start_lat, start_lon, end_lat, end_lon)
        segments.append(UJISegment(segment_index, start_lat, start_lon, end_lat, end_lon, first_sample, last_sample, segment_id, directional_id))
        expected_first = last_sample + 1
    if expected_first != len(sensor_rows):
        raise _error(path, trajectory_id, f"segment ranges cover {expected_first} of {len(sensor_rows)} samples")

    frame = pd.DataFrame(sensor_rows, columns=SAMPLE_COLUMNS)
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame.insert(0, "sample_idx", range(len(frame)))
    frame.insert(2, "elapsed_s", (frame["timestamp"] - frame["timestamp"].iloc[0]) / 1000.0)
    frame["trajectory_id"] = trajectory_id
    frame["source_split"] = source_split
    frame["trajectory_type"] = trajectory_type
    frame["segment_index"] = -1
    frame["segment_id"] = ""
    frame["directional_segment_id"] = ""
    for segment in segments:
        mask = frame["sample_idx"].between(segment.first_sample, segment.last_sample)
        frame.loc[mask, ["segment_index", "segment_id", "directional_segment_id"]] = (
            segment.segment_index, segment.segment_id, segment.directional_segment_id
        )
    frame["segment_index"] = frame["segment_index"].astype("int64")
    frame = frame.loc[:, OUTPUT_COLUMNS]

    deltas = frame["timestamp"].diff().dropna().astype(float).tolist()
    duplicate_count = sum(delta == 0 for delta in deltas)
    backward_count = sum(delta < 0 for delta in deltas)
    nonpositive = duplicate_count + backward_count
    if nonpositive:
        warnings.append(f"NONPOSITIVE_TIMESTAMP_DELTAS:{nonpositive}")
    if duplicate_count:
        warnings.append(f"DUPLICATE_TIMESTAMPS:{duplicate_count}")
    if backward_count:
        warnings.append(f"BACKWARD_TIMESTAMPS:{backward_count}")
    positive = [delta for delta in deltas if delta > 0]
    median_dt = float(median(positive)) if positive else None
    max_gap = float(max(positive)) if positive else None
    if median_dt and max_gap > median_dt * 5:
        warnings.append(f"TIMESTAMP_GAP_MS:{max_gap:g}")
    rate = 1000.0 / median_dt if median_dt else None
    return UJITrajectory(trajectory_id, source_split, trajectory_type, path, frame, tuple(segments), tuple(warnings), median_dt, rate, max_gap, nonpositive)
