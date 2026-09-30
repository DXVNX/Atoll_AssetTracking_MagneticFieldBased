import pandas as pd
import numpy as np
from typing import List, Dict

def build_fingerprint_map(
    feature_vectors_by_zone: Dict[str, List[np.ndarray]], 
    feature_cols: List[str] = None
) -> pd.DataFrame:
    """
    Aggregates raw per-sample feature vectors into one row per zone.
    Returns DataFrame with zone label, mean vector, per-feature variance, and sample count.
    
    If feature_cols is provided, the output columns will be named according to it
    (e.g., 'mag_mean', 'var_mag_mean', etc.). Otherwise, defaults to V1 naming.
    """
    # Default to V1 columns if not provided
    if feature_cols is None:
        feature_cols = ['mean', 'std', 'median', 'p25', 'p75']
        
    rows = []
    for zone, vectors in feature_vectors_by_zone.items():
        arr = np.array(vectors)
        mean_vec = arr.mean(axis=0)
        var_vec = arr.var(axis=0) if len(arr) > 1 else np.zeros_like(mean_vec)
        count = len(vectors)
        
        row = {'point': zone}
        for i, col in enumerate(feature_cols):
            row[col] = mean_vec[i]
            row[f'var_{col}'] = var_vec[i]
            
        row['sample_count'] = count
        rows.append(row)
        
    return pd.DataFrame(rows)

def save_fingerprint_map(df: pd.DataFrame, path: str):
    """Save the fingerprint map to a CSV file."""
    df.to_csv(path, index=False)

def load_fingerprint_map(path: str) -> pd.DataFrame:
    """Load the fingerprint map from a CSV file."""
    return pd.read_csv(path)
