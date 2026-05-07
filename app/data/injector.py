import numpy as np
import pandas as pd
from typing import Tuple

class AnomalyInjector:
    """
    Injects anomalies as described in the reference paper:
    1. Outliers: 1.3x and 0.7x of the mean value (locked to 20% proportion).
    """
    def __init__(self, outlier_fraction: float = 0.2, seed: int = 42):
        self.outlier_fraction = outlier_fraction
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def inject(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        print(f"Injecting anomalies (reproducible seed={self.seed}): {self.outlier_fraction*100}% outliers...")
        n_samples = len(df)
        labels = np.ones(n_samples) # 1 for normal
        corrupted_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate means for all columns to use as base for anomalies
        means = df[numeric_cols].mean()

        segment_len = 50
        
        # 1. Inject Outliers
        n_outlier_segments = int((n_samples * self.outlier_fraction) / segment_len)
        for _ in range(n_outlier_segments):
            start = self.rng.randint(0, n_samples - segment_len)
            end = start + segment_len
            labels[start:end] = -1
            
            factor = self.rng.choice([1.3, 0.7])
            for col in numeric_cols:
                corrupted_df.iloc[start:end, corrupted_df.columns.get_loc(col)] = means[col] * factor
                
        return corrupted_df, labels
