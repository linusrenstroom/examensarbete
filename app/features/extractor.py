import numpy as np                                      # Import NumPy for vectorized math

class FeatureExtractor:                                 # Define the class for calculating window statistics
    @staticmethod                                       # Declare as static because it doesn't need class state
    def extract_from_windows(windows: np.ndarray) -> np.ndarray: # Method to turn raw windows into feature vectors
        if windows.size == 0: return np.array([])       # Return empty array if input is empty to avoid errors
        
        # We calculate 6 stats along axis 1 (the time axis inside each window)
        f_mean = np.mean(windows, axis=1)                # Calculate the average level of the signal
        f_std = np.std(windows, axis=1)                  # Calculate the volatility/noise level
        f_min = np.min(windows, axis=1)                  # Capture the lowest point in the window
        f_max = np.max(windows, axis=1)                  # Capture the highest point in the window
        f_med = np.median(windows, axis=1)               # Get the middle value (robust to single spikes)
        
        q75, q25 = np.percentile(windows, [75, 25], axis=1) # Get the 75th and 25th percentiles
        f_iqr = q75 - q25                                # Calculate the range where the middle 50% of data lives
        
        features = np.stack([f_mean, f_std, f_min, f_max, f_med, f_iqr], axis=2) # Combine stats into a 3D block
        n_win, n_sens, n_stats = features.shape          # Get the dimensions of the feature block
        
        return features.reshape(n_win, -1)              # Flatten sensors and stats into a single row per window
