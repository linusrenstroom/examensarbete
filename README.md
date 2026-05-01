# Anomaly Detection for Sensor Data (Examensarbete)

A modular Python framework for detecting anomalies in industrial sensor data using the **Isolation Forest** algorithm. This project was developed as part of a thesis to evaluate unsupervised learning performance on time-series data through synthetic anomaly injection.

## Project Overview

The system processes raw sensor data, transforms it into statistical feature windows, and trains an Isolation Forest model to distinguish between normal operation and anomalous behavior. To evaluate the model's effectiveness, the system automatically injects "synthetic bugs" (anomalies) into a test set and measures how many the model can successfully identify.

## Project Structure

- **`main.py`**: The central orchestrator. Defines the experiment configuration and runs the end-to-end pipeline.
- **`app/data/processor.py`**: Handles raw data loading and slices the data into overlapping sliding windows.
- **`app/features/extractor.py`**: Extracts advanced statistical features from each window (Mean, Std, Min, Max, Median, and Interquartile Range).
- **`app/data/injector.py`**: Injects synthetic anomalies (outliers and noise) into the data for testing.
- **`app/models/detector.py`**: Wraps the Isolation Forest model, including automatic feature scaling.
- **`datasets/`**: Directory for raw sensor datasets.
- **`results/`**: Output directory for evaluation reports, results CSV, and visualizations.

## 🛠️ Installation & Setup

Ensure you have Python installed, then install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

Execute the main experiment script:

```bash
python main.py
```

## Methodology

1.  **Windowing**: The raw time-series is divided into windows (e.g., 200 samples) with a step size (e.g., 100) to capture local temporal patterns.
2.  **Feature Engineering**: Instead of raw data, the model sees statistical descriptors per sensor, making it robust to minor noise while sensitive to structural changes.
3.  **Training**: The Isolation Forest is trained on the raw sensor data to learn the "Normal" state.
4.  **Synthetic Injection**: Test data is created by injecting synthetic anomalies (spikes and noise) into the raw dataset to evaluate detection performance.
5.  **Evaluation**: The model is tested on the corrupted set, and performance is measured using metrics like Precision, Recall, and F1-score.

## Interpreting Results

After running the experiment, the outputs are saved in the `results/` directory:

- **`classification_report.txt`**: Detailed performance metrics (Precision, Recall, F1-Score).
- **`experiment_results.csv`**: Contains ground truth labels, model predictions, and anomaly scores for every test window.
- **Visualizations**: Heatmaps and score plots showing where anomalies were detected.
