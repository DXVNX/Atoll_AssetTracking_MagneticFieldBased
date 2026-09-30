from pathlib import Path

import pandas as pd
import pytest

from src.adapters.uji_mag import OUTPUT_COLUMNS, UJIParseError, parse_uji_trajectory, segment_identifiers


def _write(path: Path, rows: list[str], configs: list[str], separator: str | None = None) -> Path:
    marker = separator if separator is not None else f"<{len(configs)}>"
    path.write_text("\n".join(rows + [marker] + configs) + "\n", encoding="utf-8")
    return path


def _samples(n: int = 4) -> list[str]:
    return [f"{1000 + i * 100} {i}.1 {i}.2 {i}.3 {i}.4 {i}.5 {i}.6 {i}.7 {i}.8 {i}.9" for i in range(n)]


def test_line_file_schema_elapsed_and_assignment(tmp_path):
    path = _write(tmp_path / "l1n_01.txt", _samples(), ["39.1 -0.1 39.2 -0.2 0 3"])
    result = parse_uji_trajectory(path, "l1n_01", "train_public", "line")
    assert tuple(result.samples.columns) == OUTPUT_COLUMNS
    assert result.samples["elapsed_s"].tolist() == pytest.approx([0, .1, .2, .3])
    assert result.samples["segment_index"].tolist() == [0, 0, 0, 0]
    assert result.sampling_rate_est_hz == pytest.approx(10)


def test_curve_boundaries_are_zero_based_inclusive(tmp_path):
    path = _write(tmp_path / "curve.txt", _samples(5), [
        "39.1 -0.1 39.2 -0.2 0 1",
        "39.2 -0.2 39.3 -0.3 2 4",
    ])
    result = parse_uji_trajectory(path, "curve", "train_public", "curve")
    assert result.samples["segment_index"].tolist() == [0, 0, 1, 1, 1]
    assert [s.first_sample for s in result.segments] == [0, 2]
    assert [s.last_sample for s in result.segments] == [1, 4]


def test_test_file_is_parsed_without_special_label_logic(tmp_path):
    path = _write(tmp_path / "tt01.txt", _samples(3), ["39.1 -0.1 39.2 -0.2 0 2"])
    result = parse_uji_trajectory(path, "tt01", "test_public", "test")
    assert set(result.samples["source_split"]) == {"test_public"}
    assert set(result.samples["trajectory_type"]) == {"test"}


def test_malformed_row_has_structured_error(tmp_path):
    path = _write(tmp_path / "bad.txt", ["1000 1 2"], ["39.1 -0.1 39.2 -0.2 0 0"])
    with pytest.raises(UJIParseError) as caught:
        parse_uji_trajectory(path, "bad", "train_public", "line")
    assert caught.value.trajectory_id == "bad"
    assert caught.value.path == path
    assert caught.value.line_number == 1


def test_out_of_bounds_segment_is_rejected(tmp_path):
    path = _write(tmp_path / "bad_bounds.txt", _samples(3), ["39.1 -0.1 39.2 -0.2 0 4"])
    with pytest.raises(UJIParseError, match="invalid inclusive sample range"):
        parse_uji_trajectory(path, "bad_bounds", "train_public", "line")


def test_direction_invariant_and_directional_ids():
    forward = segment_identifiers(39.1, -0.1, 39.2, -0.2)
    reverse = segment_identifiers(39.2, -0.2, 39.1, -0.1)
    assert forward[0] == reverse[0]
    assert forward[1] != reverse[1]


def test_empirical_archive_anomalies_are_normalized_with_warnings(tmp_path):
    path = _write(tmp_path / "anomaly.txt", _samples(3), ["39.1 -0.1 39.2 -0.2 0 3"], "49<1>")
    result = parse_uji_trajectory(path, "anomaly", "train_public", "curve")
    assert result.segments[0].last_sample == 2
    assert "MALFORMED_SEPARATOR_PREFIX:49" in result.warnings
    assert "FINAL_LS_ONE_PAST_END:3->2" in result.warnings


def test_timestamp_warnings_do_not_drop_samples(tmp_path):
    rows = _samples(4)
    rows[2] = rows[2].replace("1200", "1100", 1)
    path = _write(tmp_path / "timestamps.txt", rows, ["39.1 -0.1 39.2 -0.2 0 3"])
    result = parse_uji_trajectory(path, "timestamps", "train_public", "line")
    assert len(result.samples) == 4
    assert result.nonpositive_timestamp_count == 1
    assert "NONPOSITIVE_TIMESTAMP_DELTAS:1" in result.warnings
    assert "DUPLICATE_TIMESTAMPS:1" in result.warnings
    assert pd.api.types.is_integer_dtype(result.samples["timestamp"])


def test_backward_timestamp_warning(tmp_path):
    rows = _samples(3)
    rows[2] = rows[2].replace("1200", "1050", 1)
    path = _write(tmp_path / "backward.txt", rows, ["39.1 -0.1 39.2 -0.2 0 2"])
    result = parse_uji_trajectory(path, "backward", "train_public", "line")
    assert "BACKWARD_TIMESTAMPS:1" in result.warnings
