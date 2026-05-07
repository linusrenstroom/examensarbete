import numpy as np
import pandas as pd
import os
from app.data.processor import DataProcessor
from app.data.injector import AnomalyInjector
from app.features.extractor import FeatureExtractor
from app.models.detector import AnomalyDetector

class AnomalyExperiment:
    """Orchestrates the anomaly detection experiment workflow focused on window and step size evaluation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = "datasets"
        
        # Unique output directory for this experiment run
        self.output_dir = os.path.join(
            "results", 
            f"win{config['window_size']}_step{config['step_size']}"
        )
        
        # Locked Parameters for the study
        self.contamination = 0.2
        self.n_estimators = 300
        self.seed = 42
        self.outlier_fraction = 0.2 
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        input_path = os.path.join(self.data_dir, config['file_name'])
        
        self.processor = DataProcessor(
            input_path, 
            config['window_size'], 
            config['step_size']
        )
        self.injector = AnomalyInjector(
            outlier_fraction=self.outlier_fraction,
            seed=self.seed
        )
        self.detector = AnomalyDetector(
            contamination=self.contamination, 
            n_estimators=self.n_estimators,
            random_state=self.seed
        )

    def run(self) -> float:
        # 1. Load Data
        raw_data = self.processor.load_raw_data()
        
        # 2. Time-based Split (70% Train, 30% Test)
        split_idx = int(len(raw_data) * 0.7)
        train_raw = raw_data.iloc[:split_idx].copy()
        test_raw = raw_data.iloc[split_idx:].copy()
        
        print(f"Data split: Train={len(train_raw)} samples, Test={len(test_raw)} samples")

        # 3. Training Set Processing
        print(f"\n--- Training Set Processing (Window: {self.config['window_size']}, Step: {self.config['step_size']}) ---")
        train_windows = self.processor.create_windows(train_raw)
        X_train = FeatureExtractor.extract_from_windows(train_windows)

        # 4. Inject Anomalies into Test Data
        print("\n--- Test Set Anomaly Injection ---")
        test_raw_corrupted, y_samples = self.injector.inject(test_raw)
        
        test_data_path = os.path.join(self.output_dir, "test_data_with_bugs.csv")
        test_raw_corrupted.to_csv(test_data_path, index=False)

        # 5. Test Set Processing
        print("\n--- Test Set Processing ---")
        test_windows = self.processor.create_windows(test_raw_corrupted)
        X_test = FeatureExtractor.extract_from_windows(test_windows)
        
        y_test = self._label_windows(y_samples)

        # 6. Train & Evaluate
        self.detector.train(X_train)
        report = self.detector.evaluate(X_test, y_test)

        # 7. Save Evaluation Text (Classification Report)
        report_path = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("--- ANOMALY DETECTION REPORT ---\n\n")
            f.write(pd.DataFrame(report).transpose().to_string())
        
        # 8. Save Results CSV
        results_path = os.path.join(self.output_dir, "experiment_results.csv")
        self.detector.results.to_csv(results_path, index=False)

        print(f"\nExperiment complete. All outputs saved to '{self.output_dir}/'.")
        
        # Return F1-score for summary purposes
        return report['Anomaly']['f1-score']

    def _label_windows(self, y_samples: np.ndarray) -> np.ndarray:
        y_test = []
        for i in range(0, len(y_samples) - self.config['window_size'] + 1, self.config['step_size']):
            window_labels = y_samples[i : i + self.config['window_size']]
            # Label as anomaly if ANY sample in the window is an anomaly
            y_test.append(-1 if -1 in window_labels else 1)
        return np.array(y_test)
