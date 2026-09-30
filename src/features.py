"""
Feature extraction module for magnetic field tracking.

V1 (5 scalars): statistics of smoothed magnitude — rotation-dependent baseline.
V2 (13 scalars): orientation-independent features using world-frame decomposition
    from Ouyang & Abed-Meraim (2022), Analysis paper Section 6.1, Eq. 15.

Both extractors operate on the full recording (no windowing). The same
interface is used for static reference recordings (fed to fingerprint_db)
and for sliding-window trajectory evaluation (the caller slices the window).
"""

import numpy as np
import pandas as pd


# ─── Feature column name lists (canonical ordering) ─────────────────────

FEATURE_COLS_V1 = ['mean', 'std', 'median', 'p25', 'p75']

FEATURE_COLS_V2 = [
    # Magnitude statistics (v1 baseline — kept for fair comparison)
    'mag_mean', 'mag_std', 'mag_median', 'mag_p25', 'mag_p75',
    # Horizontal component: strength of field projected onto horizontal plane
    'mh_mean', 'mh_std',
    # Vertical component: most stable under rotation (Analysis paper Fig. 6a)
    'mv_mean', 'mv_std',
    # World-frame direction of horizontal component
    'Bx_w_mean', 'By_w_mean',
    # Inclination: rotation-invariant dip angle (Survey Eq. 3)
    'incl_mean', 'incl_std',
]


def _get_window_params(df: pd.DataFrame, window_sec: float, step_sec: float):
    # Estimate sample rate dt
    if 'time' in df.columns and len(df) > 1:
        dt = df['time'].diff().median()
    else:
        dt = 0.1 # default to 10Hz if no time column
    
    window_samples = max(1, int(window_sec / dt))
    step_samples = max(1, int(step_sec / dt))
    return window_samples, step_samples


def extract_windowed_features(df: pd.DataFrame, window_sec: float = 1.0, step_sec: float = 0.5) -> list[dict]:
    """
    Extract V1 features over sliding windows for a single recording.
    """
    window_samples, step_samples = _get_window_params(df, window_sec, step_sec)
    
    features_list = []
    for start in range(0, len(df) - window_samples + 1, step_samples):
        window = df.iloc[start : start + window_samples]
        features_list.append(extract_features(window))
        
    # If the recording is shorter than one window, just extract the whole thing
    if not features_list and len(df) > 0:
        features_list.append(extract_features(df))
        
    return features_list


def extract_windowed_features_v2(df: pd.DataFrame, window_sec: float = 1.0, step_sec: float = 0.5) -> list[dict]:
    """
    Extract V2 features over sliding windows for a single recording.
    """
    window_samples, step_samples = _get_window_params(df, window_sec, step_sec)
    
    features_list = []
    for start in range(0, len(df) - window_samples + 1, step_samples):
        window = df.iloc[start : start + window_samples]
        features_list.append(extract_features_v2(window))
        
    if not features_list and len(df) > 0:
        features_list.append(extract_features_v2(df))
        
    return features_list


def extract_features(df: pd.DataFrame) -> dict:
    """
    V1 feature extraction: 5 statistics of smoothed magnitude.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed recording with a 'magnitude_smooth' column
        (output of preprocessing.preprocess_recording).

    Returns
    -------
    dict
        Keys: 'mean', 'std', 'median', 'p25', 'p75'.
    """
    s = df['magnitude_smooth']
    return {
        'mean':   s.mean(),
        'std':    s.std(),
        'median': s.median(),
        'p25':    s.quantile(0.25),
        'p75':    s.quantile(0.75),
    }


def extract_features_v2(df: pd.DataFrame) -> dict:
    """
    V2 feature extraction: 13-scalar orientation-independent feature vector.

    Requires world-frame columns (Bx_w, By_w, Bz_w, mh, mv, inclination),
    which are added by world_frame.add_world_frame(). If these columns are
    not present, the caller must run add_world_frame() first.

    Parameters
    ----------
    df : pd.DataFrame
        World-frame recording with columns: magnitude_smooth, mh, mv,
        Bx_w, By_w, inclination.

    Returns
    -------
    dict
        Keys matching FEATURE_COLS_V2 (13 scalars).
    """
    mag = df['magnitude_smooth']
    return {
        # Magnitude (rotation-invariant scalar, v1 baseline)
        'mag_mean':   mag.mean(),
        'mag_std':    mag.std(),
        'mag_median': mag.median(),
        'mag_p25':    mag.quantile(0.25),
        'mag_p75':    mag.quantile(0.75),
        # Horizontal component
        'mh_mean':    df['mh'].mean(),
        'mh_std':     df['mh'].std(),
        # Vertical component
        'mv_mean':    df['mv'].mean(),
        'mv_std':     df['mv'].std(),
        # World-frame horizontal direction
        'Bx_w_mean':  df['Bx_w'].mean(),
        'By_w_mean':  df['By_w'].mean(),
        # Inclination
        'incl_mean':  df['inclination'].mean(),
        'incl_std':   df['inclination'].std(),
    }


def features_dict_to_array(features: dict, version: str = 'v1') -> np.ndarray:
    """
    Convert a feature dict to a numpy array in canonical column order.

    Parameters
    ----------
    features : dict
        Output of extract_features() or extract_features_v2().
    version : str
        'v1' or 'v2'.

    Returns
    -------
    np.ndarray, shape (5,) for v1 or (13,) for v2.
    """
    cols = FEATURE_COLS_V1 if version == 'v1' else FEATURE_COLS_V2
    return np.array([features[c] for c in cols])
