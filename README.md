# Anomaly Detection for Sensor Data (Examensarbete)

A modular Python framework for detecting anomalies in industrial sensor data using the **Isolation Forest** algorithm. This project evaluates the impact of **window size** and **step size** on unsupervised anomaly detection performance through a grid search approach.

## Project Overview

The system processes raw sensor data, transforms it into statistical feature windows, and trains an Isolation Forest model. Anomalies are injected at the **raw data level** before windowing to ensure realistic feature-level effects. The study focuses on how different windowing strategies (size and overlap) affect the detection accuracy (F1-score).

## Project Structure

- **`main.py`**: The central orchestrator. Runs a grid search across various window and step sizes.
- **`app/experiment.py`**: Controls the experiment workflow (data splitting, injection, training, evaluation).
- **`app/data/processor.py`**: Handles raw data loading and efficient zero-copy windowing using NumPy strides.
- **`app/data/injector.py`**: Injects synthetic outliers (block-wise) into the raw sensor data.
- **`app/features/extractor.py`**: Extracts 6 statistical features (Mean, Std, Min, Max, Median, IQR) from each window.
- **`app/models/detector.py`**: Wraps the Isolation Forest model with standard scaling and evaluation reporting.
- **`datasets/`**: Directory for raw sensor datasets (expects CSV with `;` separator).
- **`results/`**: Output directory. Contains a `grid_search_summary.csv` and detailed results for each parameter combination.

## Study Parameters (Locked)

To ensure a controlled evaluation, the following parameters are locked:

- **Contamination**: 0.2 (20% expected anomalies).
- **Outlier Fraction**: 0.2 (Locked to match contamination).
- **N Estimators**: 300.
- **Seed**: 42.
- **Anomaly Segment Length**: 50 samples.

## Installation & Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to Run

1. Place your dataset in the `datasets/` folder.
2. Configure the `window_sizes` and `step_sizes` grids in `main.py`.
3. Execute the grid search:

```bash
python main.py
```

## Methodology

1.  **Data Splitting**: The raw data is split into a clean training set (first 70%) and a test set (remaining 30%).
2.  **Raw Injection**: 20% of the raw test data is corrupted with synthetic outliers (1.3x and 0.7x of the mean value) in segments of 50 samples.
3.  **Windowing**: Both training and corrupted test data are divided into windows based on the current `window_size` and `step_size`.
4.  **Feature Engineering**: Statistical descriptors (Mean, Std, Min, Max, Median, IQR) are extracted from each window for all sensor columns.
5.  **Training**: The Isolation Forest is trained on the clean features from the training set.
6.  **Evaluation**: The model is tested on the features from the corrupted test set. A window is labeled as an anomaly if it contains at least one corrupted raw sample.
7.  **Results**: F1-scores are saved for each experiment run to facilitate comparative analysis.
