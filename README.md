# Anomaly Detection for Sensor Data (Examensarbete)

A modular Python framework for detecting anomalies in industrial sensor data using the **Isolation Forest** algorithm. This project was developed as part of a thesis to evaluate unsupervised learning performance on time-series data through synthetic anomaly injection.

## Project Overview

The system processes raw sensor data, transforms it into statistical feature windows, and trains an Isolation Forest model to distinguish between normal operation and anomalous behavior. To evaluate the model's effectiveness, the system automatically injects "synthetic bugs" (anomalies) into a test set and measures how many the model can successfully identify.

## Project Structure

- **`main.py`**: The central orchestrator. Defines the experiment configuration and runs the end-to-end pipeline.
- **`data_processing.py`**: Handles raw CSV loading (from `input.txt`) and slices the data into overlapping sliding windows.
- **`features.py`**: Extracts advanced statistical features from each window (Mean, Std, Min, Max, Median, and Interquartile Range).
- **`anomalies.py`**: 
    - `AnomalyInjector`: Injects synthetic spikes/shifts into the data for testing.
    - `AnomalyDetector`: Wraps the Isolation Forest model, including automatic feature scaling using `StandardScaler`.
- **`input.txt`**: The raw sensor dataset (semicolon-separated).
- **`experiment_results.csv`**: The output file containing ground truth labels, model predictions, and anomaly scores for every test window.

## 🛠️ Installation & Setup

Ensure you have Python installed, then install the required libraries:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

Execute the main experiment script:

```bash
python main.py
```

## Methodology

1.  **Windowing**: The raw time-series is divided into windows (default: 100 samples) with a step size (default: 50) to capture local temporal patterns.
2.  **Feature Engineering**: Instead of raw data, the model sees 6 statistical descriptors per sensor, making it robust to minor noise while sensitive to structural changes.
3.  **Shuffle & Split**: Data is randomized before splitting into training (70%) and testing (30%) sets to ensure the model learns a generalized "Normal" state.
4.  **Synthetic Injection**: 10% of the test windows are corrupted with spikes equal to 10 standard deviations of that specific feature.
5.  **Unsupervised Training**: The Isolation Forest is trained **only** on the clean training set.
6.  **Evaluation**: The model is tested on the corrupted set, and performance is measured using the **F1-score**, balancing Precision and Recall.

## Interpreting Results

After running the experiment, the console will output a classification report:

- **Precision**: How many of the flagged anomalies were actually the synthetic bugs we injected?
- **Recall**: What percentage of the injected synthetic bugs did the model successfully catch?
- **F1-Score**: The harmonic mean of Precision and Recall. This is your primary metric for evaluating the model's overall success.
- **Anomaly Score**: Found in `experiment_results.csv`. Negative scores indicate a high likelihood of an anomaly; positive scores indicate normal behavior.
