import pandas as pd                                     # Import Pandas for result storage
import numpy as np                                      # Import NumPy for score handling
from sklearn.ensemble import IsolationForest           # Import the anomaly detection algorithm
from sklearn.preprocessing import StandardScaler        # Import scaler to normalize sensor ranges
from sklearn.metrics import f1_score, classification_report # Import evaluation metrics

class AnomalyDetector:                                  # Define the class for the machine learning model
    def __init__(self, contamination: float = 0.10, n_estimators: int = 200, random_state: int = 42): # Initialize with parameters
        self.scaler = StandardScaler()                  # Create a scaler to ensure sensors have equal weight
        self.model = IsolationForest(                   # Configure the Isolation Forest algorithm
            n_estimators=n_estimators,                  # Set the number of trees in the forest
            contamination=contamination,                # Set the expected percentage of anomalies
            random_state=random_state                   # Set seed for reproducible tree splits
        )
        self.results = None                             # Placeholder for test predictions and scores

    def train(self, X_train: np.ndarray):               # Method to fit the model to normal data
        x_scaled = self.scaler.fit_transform(X_train)   # Learn mean/std and scale the training features
        self.model.fit(x_scaled)                        # Build the isolation trees using the scaled data

    def evaluate(self, X_test: np.ndarray, y_true: np.ndarray): # Method to predict and evaluate on test data
        x_scaled = self.scaler.transform(X_test)        # Scale test data using training parameters
        preds = self.model.predict(x_scaled)            # Get predictions (1 for normal, -1 for anomaly)
        scores = self.model.decision_function(x_scaled) # Get raw anomaly scores (lower means more anomalous)
        
        self.results = pd.DataFrame({'ground_truth': y_true, 'prediction': preds, 'score': scores}) # Store results
        
        print("\nClassification Report:")               # Print header for report
        print(classification_report(y_true, preds, target_names=['Anomaly', 'Normal'])) # Print sklearn report
        
        return classification_report(y_true, preds, output_dict=True) # Return report as dictionary
