import pandas as pd                                     # Import Pandas for CSV handling
import numpy as np                                      # Import NumPy for array manipulations
from numpy.lib.stride_tricks import as_strided          # Import specialized tool for zero-copy windowing

class DataProcessor:                                    # Define the class for data loading and segmentation
    def __init__(self, path: str, window: int, step: int): # Initialize with file path, window size, and step size
        self.path = path                                # Store the input file path
        self.window = window                            # Store the number of samples per window
        self.step = step                                # Store the number of samples to move the window

    def load(self) -> pd.DataFrame:                     # Method to read the specific industrial CSV format
        return pd.read_csv(self.path, sep=';', decimal=',', skiprows=[0, 2]) # Read CSV with custom separators and skip headers

    def create_windows(self, df: pd.DataFrame) -> np.ndarray: # Method to transform time-series into overlapping windows
        data = df.select_dtypes(include=[np.number]).values # Convert numeric columns into a raw NumPy matrix
        n_rows, n_cols = data.shape                     # Extract the dimensions (samples and sensors)
        n_win = (n_rows - self.window) // self.step + 1 # Calculate how many windows fit into the sequence
        
        s0, s1 = data.strides                           # Get the memory strides (bytes to skip to next row/col)
        new_shape = (n_win, self.window, n_cols)        # Define the 3D shape of the windowed result
        new_strides = (s0 * self.step, s0, s1)          # Define how to jump through memory to create windows
        
        return as_strided(data, shape=new_shape, strides=new_strides) # Create the windows view without copying data
