import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Tuple, Dict

class AnomalyInjector:
    """Injects synthetic anomalies into a feature matrix."""
    def __init__(self, fraction: float = 0.1, intensity: float = 10.0):
        self.fraction = fraction
        self.intensity = intensity

    def inject(self, feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Injecting synthetic anomalies (fraction={self.fraction}, intensity={self.intensity})...")
        n_samples = feature_matrix.shape[0]
        n_anomalies = int(n_samples * self.fraction)
        
        labels = np.ones(n_samples) # 1 for normal
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        labels[anomaly_indices] = -1 # -1 for anomaly
        
        corrupted_matrix = feature_matrix.copy()
        for idx in anomaly_indices:
            # Corrupt a larger portion of features (up to half) to ensure detectability
            n_to_corrupt = np.random.randint(1, corrupted_matrix.shape[1] // 2 + 1)
            f_indices = np.random.choice(corrupted_matrix.shape[1], n_to_corrupt, replace=False)
            
            for f_idx in f_indices:
                std_dev = np.std(feature_matrix[:, f_idx])
                shift = np.random.choice([-1, 1]) * self.intensity * (std_dev + 1e-9)
                corrupted_matrix[idx, f_idx] += shift
                
        return corrupted_matrix, labels

class AnomalyDetector:
    """Enhanced Isolation Forest with Scaling and Tuning."""
    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        # Increased estimators for stability
        self.model = IsolationForest(
            n_estimators=200, 
            contamination=contamination, 
            random_state=random_state,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.results: pd.DataFrame = None

    def train(self, X_train: np.ndarray):
        print("Scaling and training Isolation Forest...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_true: np.ndarray) -> Dict:
        print("Predicting and evaluating...")
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        self.results = pd.DataFrame({
            'ground_truth': y_true,
            'prediction': y_pred,
            'score': scores
        })
        
        # Calculate F1-score specifically for the anomaly class (-1)
        # This is the most crucial metric for evaluating detection performance
        f1 = f1_score(y_true, y_pred, pos_label=-1)
        
        print("\n--- Optimized Evaluation Results ---")
        print(f"Anomaly Class F-score (F1): {f1:.4f}")
        print("\nFull Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Anomaly', 'Normal']))
        
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_dict['anomaly_f1'] = f1
        return report_dict
