import pandas as pd
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

    def create_windows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        print("Creating overlapping windows...")
        windows = []
        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            windows.append(df.iloc[i : i + self.window_size])
        return windows
