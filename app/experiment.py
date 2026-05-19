import os                                               # Import OS for directory management
import numpy as np                                      # Import NumPy for array operations
import time                                             # Import time for execution profiling
import pandas as pd
from app.data.processor import DataProcessor            # Link to our data handling class
from app.data.injector import AnomalyInjector            # Link to our anomaly generation class
from app.features.extractor import FeatureExtractor      # Link to our statistics extraction class
from app.models.detector import AnomalyDetector          # Link to our ML model class

class AnomalyExperiment:                                # Define the main controller for the study
    def __init__(self, config: dict):                   # Initialize with window and step configuration
        self.config = config                            # Store configuration dictionary
        input_path = os.path.join("datasets", config['file_name']) # Build full path to dataset
        self.processor = DataProcessor(input_path, config['window_size'], config['step_size']) # Create processor
        self.injector = AnomalyInjector(fraction=0.2)   # Create injector with 20% corruption target
        self.detector = AnomalyDetector(contamination=0.2, n_estimators=300) # Create detector with matching contamination

    def run(self, output_dir: str) -> dict:             # Main workflow method returns metrics and timings
        raw = self.processor.load()                     # Load the raw sensor data from disk
        split = int(len(raw) * 0.7)                     # Calculate the 70% mark for the training split
        
        train_raw = raw.iloc[:split]                    # Take the first 70% as clean training data
        test_raw = raw.iloc[split:]                     # Take the remaining 30% for testing
        
        # 1. Training Set Processing (Timing)
        print("--- Feature Extraction (Train) ---")
        start_proc = time.time()                        # Record start time for feature extraction
        train_wins = self.processor.create_windows(train_raw) # Segment training data into windows
        x_train = FeatureExtractor.extract_from_windows(train_wins) # Extract features from training windows
        end_proc = time.time()                          # Record end time for feature extraction
        dur_proc = end_proc - start_proc
        print(f"Completed in {dur_proc:.2f}s")
        
        # 2. Test Set Anomaly Injection
        corrupted_test, y_raw = self.injector.inject(test_raw) # Inject anomalies into the raw test slice
        
        # 3. Test Set Processing
        print("--- Feature Extraction (Test) ---")
        test_wins = self.processor.create_windows(corrupted_test) # Segment corrupted test data
        x_test = FeatureExtractor.extract_from_windows(test_wins) # Extract features from test windows
        y_test = self._label_windows(y_raw)             # Propagate raw labels to the window level
        
        # 4. Train (Timing)
        print("--- Model Training ---")
        start_train = time.time()                       # Record start time for model training
        self.detector.train(x_train)                    # Train the Isolation Forest on clean data
        end_train = time.time()                         # Record end time for model training
        dur_train = end_train - start_train
        print(f"Completed in {dur_train:.2f}s")

        # 5. Evaluate (Timing)
        print("--- Evaluation ---")
        start_eval = time.time()                        # Record start time for inference/evaluation
        report = self.detector.evaluate(x_test, y_test) # Evaluate the model and get the report
        end_eval = time.time()                          # Record end time for inference/evaluation
        dur_eval = end_eval - start_eval
        print(f"Completed in {dur_eval:.2f}s")

        # Create a dictionary for all metrics
        metrics = {
            'f1_score': report['Anomaly']['f1-score'],
            'time_feature_extraction': dur_proc,
            'time_training': dur_train,
            'time_inference': dur_eval,
            'total_time': dur_proc + dur_train + dur_eval
        }

        # Save metrics locally to the specific result folder
        os.makedirs(output_dir, exist_ok=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
        
        # Save classification report locally
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(pd.DataFrame(report).transpose().to_string())

        return metrics

    def _label_windows(self, y_raw: np.ndarray) -> np.ndarray: # Helper to map samples to windows
        y_win = []                                      # Initialize empty list for window labels
        for i in range(0, len(y_raw) - self.config['window_size'] + 1, self.config['step_size']): # Loop with defined step
            chunk = y_raw[i : i + self.config['window_size']] # Get all sample labels inside current window
            y_win.append(-1 if -1 in chunk else 1)      # Label window as -1 if it contains ANY anomaly
        return np.array(y_win)                          # Return the window labels as a NumPy array
