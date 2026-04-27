import numpy as np
import pandas as pd
import os
from app.data.processor import DataProcessor
from app.data.injector import AnomalyInjector
from app.data.augmentor import DataAugmentor
from app.features.extractor import FeatureExtractor
from app.models.detector import AnomalyDetector
from app.visualization.plotters import AnomalyHeatmapPlotter, ScorePlotter

class AnomalyExperiment:
    """Orchestrates the anomaly detection experiment workflow with updated paths."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = "datasets"
        self.output_dir = "results"
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        input_path = os.path.join(self.data_dir, config['file_name'])
        
        self.processor = DataProcessor(
            input_path, 
            config['window_size'], 
            config['step_size']
        )
        self.augmentor = DataAugmentor(config['noise_level'])
        self.injector = AnomalyInjector(
            outlier_fraction=config['outlier_fraction'], 
            noise_fraction=config['noise_fraction']
        )
        self.detector = AnomalyDetector(
            contamination=config['contamination'], 
            n_estimators=config['n_estimators']
        )

    def run(self):
        # 1. Load Data
        raw_data = self.processor.load_raw_data()

        # 2. Augment Training Data
        train_raw = raw_data.copy()
        train_raw_augmented = self.augmentor.augment(train_raw)

        # 3. Training Set Processing
        print("\n--- Training Set Processing ---")
        train_windows = self.processor.create_windows(train_raw_augmented)
        X_train = FeatureExtractor.extract_from_windows(train_windows)

        # 4. Inject Anomalies into Test Data
        test_raw = raw_data.copy()
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
        
        # 9. Visualization
        self._visualize(results_path, test_data_path)

        print(f"\nExperiment complete. All outputs saved to '{self.output_dir}/'.")

    def _visualize(self, results_path, test_data_path):
        heatmap_plotter = AnomalyHeatmapPlotter(self.output_dir)
        heatmap_plotter.plot(
            results_path, 
            test_data_path, 
            self.config['window_size'], 
            self.config['step_size']
        )
        
        score_plotter = ScorePlotter(self.output_dir)
        score_plotter.plot(results_path)

    def _label_windows(self, y_samples: np.ndarray) -> np.ndarray:
        y_test = []
        for i in range(0, len(y_samples) - self.config['window_size'] + 1, self.config['step_size']):
            window_labels = y_samples[i : i + self.config['window_size']]
            y_test.append(-1 if -1 in window_labels else 1)
        return np.array(y_test)
