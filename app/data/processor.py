import pandas as pd
import numpy as np
from typing import List

class DataProcessor:
    """Handles data loading and windowing operations."""
    def __init__(self, file_path: str, window_size: int, step_size: int):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size

    def load_raw_data(self) -> pd.DataFrame:
        print(f"Loading data from {self.file_path}...")
        # Specific loading for the sensor data format
        return pd.read_csv(self.file_path, sep=';', decimal=',', skiprows=[0, 2])

    def create_windows(self, df: pd.DataFrame) -> np.ndarray:
        print(f"Creating overlapping windows (W={self.window_size}, S={self.step_size})...")
        data = df.select_dtypes(include=[np.number]).values
        n_samples = data.shape[0]
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.step_size + 1
        
        # Use numpy stride tricks for ultra-fast windowing without copying data
        from numpy.lib.stride_tricks import as_strided
        
        window_shape = (n_windows, self.window_size, data.shape[1])
        window_strides = (data.strides[0] * self.step_size, data.strides[0], data.strides[1])
        
        windows = as_strided(data, shape=window_shape, strides=window_strides)
        return windows
