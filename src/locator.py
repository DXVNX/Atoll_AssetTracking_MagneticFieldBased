import numpy as np
import pandas as pd
from collections import Counter
from typing import Literal, Tuple, List
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Locator:
    def __init__(self, fingerprint_map: pd.DataFrame, method: Literal["knn", "svm", "lda"] = "knn", feature_cols: List[str] = None, k: int = 1, raw_features_df: pd.DataFrame = None):
        self.method = method
        self.k = k
        self.fingerprint_map = fingerprint_map
        self.raw_features_df = raw_features_df
        
        # If feature_cols not provided, infer from dataframe (V1 vs V2)
        if feature_cols is None:
            if 'mag_mean' in fingerprint_map.columns:
                self.feature_cols = [
                    'mag_mean', 'mag_std', 'mag_median', 'mag_p25', 'mag_p75',
                    'mh_mean', 'mh_std', 'mv_mean', 'mv_std',
                    'Bx_w_mean', 'By_w_mean', 'incl_mean', 'incl_std'
                ]
            else:
                self.feature_cols = ['mean', 'std', 'median', 'p25', 'p75']
        else:
            self.feature_cols = feature_cols
            
        self.db_features = fingerprint_map[self.feature_cols].values
        self.db_points = fingerprint_map['point'].values
        
        # Train classifiers if needed
        if self.method in ["svm", "lda"]:
            if self.raw_features_df is not None:
                self.X_train = self.raw_features_df[self.feature_cols].values
                self.y_train = self.raw_features_df['point'].values
            else:
                self.X_train = self.db_features
                self.y_train = self.db_points
                
            if self.method == "svm":
                self.clf = SVC(kernel='rbf', probability=True)
                self.clf.fit(self.X_train, self.y_train)
            elif self.method == "lda":
                self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                self.clf.fit(self.X_train, self.y_train)

    def estimate(self, query_vector: np.ndarray) -> Tuple[str, float]:
        if self.method == "knn":
            dists = np.linalg.norm(self.db_features - query_vector, axis=1)
            if self.k == 1:
                idx = np.argmin(dists)
                return self.db_points[idx], float(dists[idx])
            else:
                top_k_indices = np.argsort(dists)[:self.k]
                top_k_points = self.db_points[top_k_indices]
                top_k_dists = dists[top_k_indices]
                counts = Counter(top_k_points)
                majority_label = counts.most_common(1)[0][0]
                return majority_label, float(top_k_dists[0])
                
        elif self.method == "svm":
            pred = self.clf.predict([query_vector])[0]
            try:
                probs = self.clf.predict_proba([query_vector])[0]
                conf = 1.0 - np.max(probs) # 1 - prob as a distance-like proxy (smaller is better)
            except:
                conf = 0.0
            return pred, float(conf)
            
        elif self.method == "lda":
            pred = self.clf.predict([query_vector])[0]
            try:
                probs = self.clf.predict_proba([query_vector])[0]
                conf = 1.0 - np.max(probs)
            except:
                conf = 0.0
            return pred, float(conf)

def estimate_location(query_vector: np.ndarray, fingerprint_map: pd.DataFrame, k: int = 1, method: Literal["knn", "svm", "lda"] = "knn", feature_cols: List[str] = None, raw_features_df: pd.DataFrame = None) -> Tuple[str, float]:
    """
    Backwards compatible standalone function. 
    Instantiates a Locator on the fly and estimates location.
    """
    locator = Locator(fingerprint_map, method=method, feature_cols=feature_cols, k=k, raw_features_df=raw_features_df)
    return locator.estimate(query_vector)
