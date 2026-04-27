import pandas as pd
import numpy as np
from typing import List

class FeatureExtractor:
    """Extracts statistical features from data windows."""
    
    @staticmethod
    def extract_from_windows(windows: List[pd.DataFrame]) -> np.ndarray:
        print(f"Extracting features from {len(windows)} windows...")
        feature_matrix = []
        for window in windows:
            numeric_window = window.select_dtypes(include=[np.number])
            features = []
            for col in numeric_window.columns:
                data = numeric_window[col].values
                features.extend([
                    np.mean(data),
                    np.std(data),
                    np.min(data),
                    np.max(data),
                    np.median(data),
                    np.percentile(data, 75) - np.percentile(data, 25) # IQR
                ])
            feature_matrix.append(features)
        return np.array(feature_matrix)
