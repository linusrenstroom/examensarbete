import numpy as np
from data_processing import DataProcessor
from features import FeatureExtractor
from anomalies import AnomalyInjector, AnomalyDetector

def run_experiment(config):
    # 1. Pipeline Setup
    processor = DataProcessor(config['file_path'], config['window_size'], config['step_size'])
    injector = AnomalyInjector(config['anomaly_fraction'], config['intensity'])
    detector = AnomalyDetector()

    # 2. Process & Feature Extraction
    raw_data = processor.load_raw_data()
    windows = processor.create_windows(raw_data)
    features = FeatureExtractor.extract_from_windows(windows)

    # 3. Shuffle and Split
    print("Shuffling and splitting data...")
    indices = np.arange(len(features))
    np.random.seed(config.get('seed', 42))
    np.random.shuffle(indices)
    
    features_shuffled = features[indices]
    split_idx = int(len(features_shuffled) * config['train_split'])
    
    X_train = features_shuffled[:split_idx]
    X_test_clean = features_shuffled[split_idx:]

    # 4. Inject Anomalies into Test Set
    X_test_corrupted, y_test = injector.inject(X_test_clean)

    # 5. Train and Evaluate
    detector.train(X_train)
    detector.evaluate(X_test_corrupted, y_test)

    # 6. Save results
    detector.results.to_csv("experiment_results.csv", index=False)
    print("\nExperiment complete. Results saved to 'experiment_results.csv'.")

if __name__ == "__main__":
    # Configuration Dictionary
    config = {
        'file_path': 'input.txt',
        'window_size': 100,
        'step_size': 50,
        'train_split': 0.7,
        'anomaly_fraction': 0.1,
        'intensity': 10.0,
        'seed': 42
    }

    run_experiment(config)
