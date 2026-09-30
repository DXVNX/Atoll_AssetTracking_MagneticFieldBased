import numpy as np
import pandas as pd
import pytest

from src.feature_sets import FEATURE_SETS, extract_window_features
from src.features import FEATURE_COLS_V1, FEATURE_COLS_V2, extract_features, extract_features_v2


def _canonical_window():
    values = np.arange(1, 6, dtype=float)
    return pd.DataFrame({
        "mx": values, "my": values + 1, "mz": values + 2,
        "ax": values / 10, "ay": values / 20, "az": values / 30,
        "orientation_azimuth": [0, 10, 20, 30, 40],
        "orientation_pitch": [0, 1, 2, 3, 4], "orientation_roll": [0, -1, -2, -3, -4],
        "trajectory_id": "secret", "source_split": "train_public", "label": "target",
    })


def test_feature_name_order_and_dimensions_are_stable():
    expected = {
        "mag_mean": 1, "magnitude_stats": 9, "body_xyz_magnitude": 36,
        "legacy_v1_mag5": 5, "legacy_v2_world13": 13,
        "uji_legacy_v2_world13": 13, "uji_expanded_world": 63,
        "accelerometer_stats": 27, "body_magnetic_accelerometer": 63,
        "magnitude_accelerometer": 36, "uji_world_accelerometer": 90,
    }
    assert {name: len(spec.feature_names) for name, spec in FEATURE_SETS.items()} == expected
    assert FEATURE_SETS["legacy_v1_mag5"].feature_names == tuple(FEATURE_COLS_V1)
    assert FEATURE_SETS["legacy_v2_world13"].feature_names == tuple(FEATURE_COLS_V2)


@pytest.mark.parametrize("name", ["mag_mean", "magnitude_stats", "body_xyz_magnitude", "uji_legacy_v2_world13", "uji_expanded_world", "accelerometer_stats", "body_magnetic_accelerometer", "magnitude_accelerometer", "uji_world_accelerometer"])
def test_canonical_features_are_finite_and_exclude_metadata(name):
    result = extract_window_features(_canonical_window(), name)
    assert np.isfinite(result).all()
    forbidden = {"trajectory_id", "source_split", "label", "filename"}
    assert forbidden.isdisjoint(result.index)


def test_ablation_combinations_preserve_component_order():
    frame = _canonical_window()
    mag = extract_window_features(frame, "magnitude_stats")
    accel = extract_window_features(frame, "accelerometer_stats")
    combined = extract_window_features(frame, "magnitude_accelerometer")
    assert combined.index.tolist() == mag.index.tolist() + accel.index.tolist()
    assert combined.to_numpy() == pytest.approx(np.concatenate([mag, accel]))


def test_legacy_v1_wrapper_exactly_matches_historical_function():
    frame = pd.DataFrame({"magnitude_smooth": [1.0, 2.0, 4.0, 8.0]})
    expected = extract_features(frame)
    actual = extract_window_features(frame, "legacy_v1_mag5")
    assert actual.index.tolist() == FEATURE_COLS_V1
    assert actual.to_numpy() == pytest.approx([expected[name] for name in FEATURE_COLS_V1])


def test_legacy_v2_wrapper_exactly_matches_historical_function():
    frame = pd.DataFrame({
        "magnitude_smooth": [10.0, 11.0, 12.0], "mh": [8.0, 9.0, 10.0],
        "mv": [2.0, 3.0, 4.0], "Bx_w": [1.0, 2.0, 3.0],
        "By_w": [4.0, 5.0, 6.0], "inclination": [10.0, 20.0, 30.0],
    })
    expected = extract_features_v2(frame)
    actual = extract_window_features(frame, "legacy_v2_world13")
    assert actual.index.tolist() == FEATURE_COLS_V2
    assert actual.to_numpy() == pytest.approx([expected[name] for name in FEATURE_COLS_V2])


def test_saved_historical_first_row_matches_legacy_extractors():
    frame = pd.read_csv("data/processed/static_P1_flat_facingnorth_worldframe.csv")
    saved_v1 = pd.read_csv("fingerprint_maps/features_all.csv").iloc[0]
    saved_v2 = pd.read_csv("fingerprint_maps/features_all_v2.csv").iloc[0]
    assert extract_window_features(frame, "legacy_v1_mag5").to_numpy() == pytest.approx(saved_v1[FEATURE_COLS_V1].to_numpy(dtype=float))
    assert extract_window_features(frame, "legacy_v2_world13").to_numpy() == pytest.approx(saved_v2[FEATURE_COLS_V2].to_numpy(dtype=float))

