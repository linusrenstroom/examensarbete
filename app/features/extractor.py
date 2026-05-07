import pandas as pd
import numpy as np
from typing import List

class FeatureExtractor:
    """Extracts statistical features from data windows (numpy optimized)."""
    
    @staticmethod
    def extract_from_windows(windows: np.ndarray) -> np.ndarray:
        """
        Input: windows of shape (n_windows, window_size, n_sensors)
        Output: feature_matrix of shape (n_windows, n_sensors * 6)
        """
        if windows.size == 0:
            return np.array([])
            
        n_windows, win_size, n_sensors = windows.shape
        print(f"Extracting features from {n_windows} windows (Vectorized)...")
        
        # Compute features using vectorized numpy operations across the window axis (axis=1)
        means = np.mean(windows, axis=1)          # (n_windows, n_sensors)
        stds = np.std(windows, axis=1)            # (n_windows, n_sensors)
        mins = np.min(windows, axis=1)            # (n_windows, n_sensors)
        maxs = np.max(windows, axis=1)            # (n_windows, n_sensors)
        medians = np.median(windows, axis=1)      # (n_windows, n_sensors)
        
        # IQR: 75th percentile - 25th percentile
        q75 = np.percentile(windows, 75, axis=1)
        q25 = np.percentile(windows, 25, axis=1)
        iqrs = q75 - q25
        
        # Concatenate all features: (n_windows, n_sensors * 6)
        # We want: [sensor1_mean, sensor1_std..., sensor2_mean, sensor2_std...]
        # So we stack along a new axis and then reshape
        all_features = np.stack([means, stds, mins, maxs, medians, iqrs], axis=2)
        # Shape is now (n_windows, n_sensors, 6)
        
        return all_features.reshape(n_windows, -1)
