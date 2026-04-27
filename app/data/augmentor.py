import pandas as pd
import numpy as np

class DataAugmentor:
    """Augments sensor data by adding noise to the existing dataset."""
    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.noise_level <= 0:
            return df
            
        print(f"Augmenting data: Adding noise (level={self.noise_level})...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        noisy_df = df.copy()
        noise = np.random.normal(0, self.noise_level, size=noisy_df[numeric_cols].shape)
        noisy_df[numeric_cols] += noise
            
        return noisy_df
