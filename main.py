import numpy as np
import pandas as pd
from data_processing import DataProcessor
from features import FeatureExtractor
from anomalies import AnomalyInjector, AnomalyDetector
from augmentation import DataAugmentor

def run_experiment(config):
    # 1. Pipeline Setup
    processor = DataProcessor(config['file_path'], config['window_size'], config['step_size'])
    augmentor = DataAugmentor(config['n_copies'], config['noise_level'])
    injector = AnomalyInjector(config['anomaly_fraction'], config['intensity'])
    detector = AnomalyDetector(contamination=config['contamination'])

    # 2. Initial Data Load
    raw_data = processor.load_raw_data()
    
    # 3. Split Raw Data (Time-based split: First 70% for Training)
    split_idx = int(len(raw_data) * config['train_split'])
    train_raw = raw_data.iloc[:split_idx]
    test_raw = raw_data.iloc[split_idx:]

    # 4. Augment Training Data ONLY (Make it more robust)
    train_raw_augmented = augmentor.augment(train_raw)

    # 5. Windowing & Feature Extraction
    print("\n--- Training Set Processing ---")
    train_windows = processor.create_windows(train_raw_augmented)
    X_train = FeatureExtractor.extract_from_windows(train_windows)

    print("\n--- Test Set Processing ---")
    test_windows = processor.create_windows(test_raw)
    X_test_clean = FeatureExtractor.extract_from_windows(test_windows)

    # 6. Inject Anomalies into Test Set
    X_test_corrupted, y_test = injector.inject(X_test_clean)

    # 7. Train & Evaluate
    detector.train(X_train)
    detector.evaluate(X_test_corrupted, y_test)

    # 8. Save
    detector.results.to_csv("experiment_results.csv", index=False)
    print("\nExperiment complete. Results saved to 'experiment_results.csv'.")

if __name__ == "__main__":
    config = {
        'file_path': 'input.txt',
        'window_size': 100,
        'step_size': 50,
        'train_split': 0.7,
        'n_copies': 1,           # Reverted to 1
        'noise_level': 0.05,     # Level of Gaussian noise for augmentation
        'anomaly_fraction': 0.1,
        'intensity': 10.0,
        'contamination': 0.1,    # Match injected anomaly fraction
        'seed': 42
    }

    run_experiment(config)
