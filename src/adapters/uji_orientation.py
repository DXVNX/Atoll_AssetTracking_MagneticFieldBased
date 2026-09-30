"""Decode UJI Android TYPE_ORIENTATION angles into device-to-world ENU rotations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def android_orientation_matrix(azimuth_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Reconstruct Android SensorManager's device-to-world ENU matrix.

    TYPE_ORIENTATION uses degrees and the historical z/x/y convention.  This
    matrix is the inverse of Android SensorManager.getOrientation's documented
    extraction formulas: atan2(R01,R11), asin(-R21), atan2(-R20,R22).
    """
    azimuth, pitch, roll = np.radians([azimuth_deg, pitch_deg, roll_deg])
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [ca * cr - sa * sp * sr, sa * cp, ca * sr + sa * sp * cr],
        [-sa * cr - ca * sp * sr, ca * cp, -sa * sr + ca * sp * cr],
        [-cp * sr, -sp, cp * cr],
    ])


def uji_body_to_world(vector: np.ndarray, azimuth_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    return android_orientation_matrix(azimuth_deg, pitch_deg, roll_deg) @ np.asarray(vector)


def add_uji_world_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"mx", "my", "mz", "orientation_azimuth", "orientation_pitch", "orientation_roll"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"UJI world transform missing columns: {missing}")
    vectors = frame[["mx", "my", "mz"]].to_numpy(dtype=float)
    angles = frame[["orientation_azimuth", "orientation_pitch", "orientation_roll"]].to_numpy(dtype=float)
    world = np.vstack([uji_body_to_world(vector, *angle) for vector, angle in zip(vectors, angles)])
    result = frame.copy()
    result[["mw_east", "mw_north", "mw_up"]] = world
    result["mw_horizontal"] = np.hypot(result["mw_east"], result["mw_north"])
    result["mw_vertical"] = result["mw_up"]
    result["mw_inclination"] = np.degrees(np.arctan2(result["mw_vertical"], result["mw_horizontal"]))
    return result

