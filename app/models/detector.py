import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from typing import Dict

class AnomalyDetector:
    """Isolation Forest model with integrated scaling."""
    def __init__(self, contamination: float = 0.10, n_estimators: int = 200, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
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
        
        f1 = f1_score(y_true, y_pred, pos_label=-1)
        print(f"\nAnomaly Class F1-score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Anomaly', 'Normal']))
        
        return classification_report(y_true, y_pred, output_dict=True)
