import pandas as pd
import numpy as np

class DataAugmentor:
    """Augments sensor data by creating copies with Gaussian noise."""
    def __init__(self, n_copies: int = 1, noise_level: float = 0.01):
        self.n_copies = n_copies
        self.noise_level = noise_level

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates noisy copies of the input DataFrame and returns a combined result."""
        if self.n_copies == 0:
            return df
            
        print(f"Augmenting training data: Creating {self.n_copies} noisy copies (noise_level={self.noise_level})...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        augmented_dfs = [df] # Start with original
        
        for _ in range(self.n_copies):
            noisy_df = df.copy()
            # Generate Gaussian noise
            noise = np.random.normal(0, self.noise_level, size=noisy_df[numeric_cols].shape)
            noisy_df[numeric_cols] += noise
            augmented_dfs.append(noisy_df)
            
        return pd.concat(augmented_dfs, ignore_index=True)
