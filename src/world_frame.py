"""
World-frame transformation module for magnetic field tracking.

Transforms calibrated magnetometer readings from the phone body frame into the
NED (North-East-Down) navigation frame using the phone's Azimuth, Pitch, Roll.

Orientation source: Android sensor-fusion Euler angles (Azimuth, Pitch, Roll)
provided directly in the raw sensor export CSV. These are platform-fused
(accelerometer + gyro + magnetometer) and already available — no separate
complementary/Madgwick filter is needed.

Rotation convention: ZYX (aerospace / Tait-Bryan)
  R = Rz(ψ) · Ry(θ) · Rx(φ)   maps NED world frame → phone body frame
  R.T                           maps phone body frame → NED world frame
  (R is orthogonal, so R^{-1} = R^T)

NED sign convention for vertical component:
  mv = Bz_w  (positive downward, per NED convention: North-East-Down)
  In most indoor locations at mid-latitudes, the geomagnetic field dips
  downward, so mv is typically negative (field points into the ground
  from the surface perspective, but NED Z points down so it's negative
  relative to the field direction). The sign is preserved as-is from
  the rotation output — no negation applied.

Literature basis:
  - Survey (Ouyang & Abed-Meraim, Electronics 2022, Eq. 13): mb = A·R_bn·mn + b
  - Analysis paper (Sensors 2022, Section 6.1, Eq. 15): xi = {mh, mv, magnitude}
  - Survey Section 2.1, Eq. 3-4: inclination I = arctan(mz/H), declination D = arctan(my/mx)
"""

import numpy as np
import pandas as pd


def get_rotation_matrix(azimuth_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build the 3x3 ZYX rotation matrix R that maps NED world frame → phone body frame.

    Parameters
    ----------
    azimuth_deg : float
        Compass heading (ψ) in degrees. 0° = North, 90° = East, clockwise.
    pitch_deg : float
        Nose up/down (θ) in degrees.
    roll_deg : float
        Tilt left/right (φ) in degrees.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix.  R maps world→body.  Use R.T for body→world.
    """
    psi   = np.radians(azimuth_deg)
    theta = np.radians(pitch_deg)
    phi   = np.radians(roll_deg)

    Rz = np.array([[ np.cos(psi), -np.sin(psi), 0],
                   [ np.sin(psi),  np.cos(psi), 0],
                   [ 0,            0,           1]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0,             1, 0            ],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(phi), -np.sin(phi)  ],
                   [0, np.sin(phi),  np.cos(phi)  ]])
    return Rz @ Ry @ Rx


def body_to_world(mb: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate a magnetometer vector from phone body frame to NED world frame.

    Since R maps world→body (R = Rz·Ry·Rx), the inverse is R.T:
        mn = R^T @ mb

    This is verified empirically by the sanity check: |mn| must equal |mb|
    (rotation preserves magnitude). We use R.T (not R) because R was
    constructed as the world-to-body rotation.

    Parameters
    ----------
    mb : np.ndarray, shape (3,) or (N, 3)
        Calibrated body-frame magnetic vector(s).
    R : np.ndarray, shape (3, 3)
        Rotation matrix (world → body).

    Returns
    -------
    mn : np.ndarray, same shape as mb
        World-frame (NED) magnetic vector(s).
    """
    if mb.ndim == 1:
        return R.T @ mb
    else:
        # Vectorised for (N, 3): each row is a sample
        return (R.T @ mb.T).T


def compute_horizontal_vertical(mn: np.ndarray) -> tuple[float, float]:
    """
    Decompose a world-frame magnetic vector into horizontal and vertical components.

    NED convention:
      - mh = sqrt(mn_x² + mn_y²)   horizontal field strength (always >= 0)
      - mv = mn_z                   vertical component (NED: positive = downward)

    The sign of mv is kept as-is from the NED frame. In the northern hemisphere,
    the geomagnetic field dips downward, so mv is typically negative (the field
    vector points partly upward in NED coordinates at indoor locations distorted
    by building materials).

    Parameters
    ----------
    mn : np.ndarray, shape (3,)
        World-frame magnetic vector [Bx_w, By_w, Bz_w].

    Returns
    -------
    mh : float
        Horizontal component magnitude.
    mv : float
        Vertical component (signed, NED convention).
    """
    mh = float(np.sqrt(mn[0]**2 + mn[1]**2))
    mv = float(mn[2])
    return mh, mv


def compute_inclination(mh: float, mv: float) -> float:
    """
    Compute the magnetic field inclination (dip angle).

    Defined per Survey paper Section 2.1, Eq. 3:
        I = arctan2(mv, mh)

    Uses arctan2 to handle all quadrants correctly.

    Parameters
    ----------
    mh : float
        Horizontal component magnitude.
    mv : float
        Vertical component (signed).

    Returns
    -------
    inclination_deg : float
        Inclination in degrees. Negative = field dips upward from horizontal,
        Positive = field dips downward.
    """
    return float(np.degrees(np.arctan2(mv, mh)))


def add_world_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NED-frame columns to a cleaned DataFrame.

    Requires columns: Bx_cal, By_cal, Bz_cal, Azimuth, Pitch, Roll.
    Adds columns:     Bx_w, By_w, Bz_w, mh, mv, inclination, declination.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned recording with calibrated body-frame magnetometer and
        orientation angles.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of input with world-frame columns appended.
    """
    B_phone = df[['Bx_cal', 'By_cal', 'Bz_cal']].values  # (N, 3)
    angles  = df[['Azimuth', 'Pitch', 'Roll']].values

    B_world = np.zeros_like(B_phone)
    for i in range(len(B_phone)):
        R = get_rotation_matrix(angles[i, 0], angles[i, 1], angles[i, 2])
        B_world[i] = body_to_world(B_phone[i], R)

    df = df.copy()
    df['Bx_w'] = B_world[:, 0]
    df['By_w'] = B_world[:, 1]
    df['Bz_w'] = B_world[:, 2]

    # Horizontal and vertical components (Analysis paper Eq. 15 decomposition)
    df['mh'] = np.sqrt(df['Bx_w']**2 + df['By_w']**2)   # horizontal magnitude
    df['mv'] = df['Bz_w']                                  # vertical component (NED: positive down)

    # Inclination and declination (Survey paper Section 2.1, Eq. 3-4)
    df['inclination'] = np.degrees(np.arctan2(df['mv'], df['mh']))
    df['declination'] = np.degrees(np.arctan2(df['By_w'], df['Bx_w']))

    return df
