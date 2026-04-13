import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def augment_data(X, number_of_copies, noise_level):
    """
    Augments the data by creating copies with added Gaussian noise.
    """
    augmented = []
    for df in X:
        augmented.append(df)  # Keep original
        for _ in range(number_of_copies):
            noisy_df = df.copy()
            numeric_cols = noisy_df.select_dtypes(include=[np.number]).columns
            noise = np.random.normal(0, noise_level, size=noisy_df[numeric_cols].shape)
            noisy_df[numeric_cols] = noisy_df[numeric_cols] + noise
            augmented.append(noisy_df)
    return augmented

def create_windows(df, window_size, step_size):
    """
    Slices the DataFrame into overlapping windows.
    """
    windows = []
    for i in range(0, len(df) - window_size + 1, step_size):
        windows.append(df.iloc[i : i + window_size])
    return windows

def build_feature_matrix(windows):
    """
    Extracts statistical features (mean, std, min, max) from each window.
    """
    feature_matrix = []
    for window in windows:
        numeric_window = window.select_dtypes(include=[np.number])
        features = []
        for col in numeric_window.columns:
            features.extend([
                numeric_window[col].mean(),
                numeric_window[col].std(),
                numeric_window[col].min(),
                numeric_window[col].max()
            ])
        feature_matrix.append(features)
    return feature_matrix

def train_isolation_forest(feature_matrix):
    """
    Trains an Isolation Forest model on the feature matrix.
    """
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(feature_matrix)
    return model

# Parameters
number_of_copies = 2
noise = 0.01
window_size = 100
step_size = 50

# Load data (skipping ID and Unit rows)
def load_data(file_path):
    return [pd.read_csv(file_path, sep=';', decimal=',', skiprows=[0, 2])]

X = load_data("input.txt")

# 1. Augment
augmented_cycles = augment_data(X, number_of_copies, noise)

# 2. Flatten all cycles into one combined DataFrame
all_data = pd.concat(augmented_cycles, ignore_index=True)

# 3. Slide windows over combined data
windows = create_windows(all_data, window_size, step_size)

# 4. Extract features from every window
feature_matrix = np.array(build_feature_matrix(windows))

# 5. Train
model = train_isolation_forest(feature_matrix)

print(f"Trained on {len(feature_matrix)} windows.")