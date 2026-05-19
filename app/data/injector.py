import numpy as np                                      # Import NumPy for numerical operations
import pandas as pd                                     # Import Pandas for data manipulation
from typing import Tuple                                # Import Tuple for type hinting returns

class AnomalyInjector:                                  # Define the class responsible for injecting anomalies
    def __init__(self, fraction: float = 0.2, seed: int = 42): # Initialize with the share of anomalies and a random seed
        self.fraction = fraction                        # Store the fraction of data to be corrupted
        self.seed = seed                                # Store the seed for reproducible results
        self.rng = np.random.RandomState(seed)          # Create a private random number generator instance
        self.segment_len = 50                           # Define the length of each anomaly block (25 seconds @ 2Hz)

    def inject(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]: # Method to inject anomalies into a DataFrame
        n_samples = len(df)                             # Get the total number of rows in the input data
        labels = np.ones(n_samples)                     # Initialize an array of ones representing 'normal' (1)
        corrupted_df = df.copy()                        # Create a deep copy of the data to avoid original mutation
        cols = df.select_dtypes(include=[np.number]).columns # Identify all numeric columns for manipulation
        
        means = df[cols].mean()                         # Calculate the global mean for every numeric sensor
        n_segments = int((n_samples * self.fraction) / self.segment_len) # Calculate how many 50-point blocks to inject

        for _ in range(n_segments):                     # Loop through the required number of anomaly segments
            start = self.rng.randint(0, n_samples - self.segment_len) # Pick a random starting index for the block
            end = start + self.segment_len              # Define the end index based on the fixed segment length
            labels[start:end] = -1                      # Mark the samples in this range as anomalies (-1)
            
            mult = self.rng.choice([1.3, 0.7])          # Randomly choose between a 30% increase or decrease
            for col in cols:                            # Iterate through each sensor column
                loc = corrupted_df.columns.get_loc(col) # Get the integer position of the current column
                corrupted_df.iloc[start:end, loc] = means[col] * mult # Replace raw values with the fixed global outlier

        return corrupted_df, labels                     # Return the modified data and the ground truth labels
