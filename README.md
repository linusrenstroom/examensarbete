# Anomaly Detection for Sensor Data (Examensarbete)

A modular Python framework for detecting anomalies in industrial sensor data using the **Isolation Forest** algorithm. This project evaluates the impact of **window size** and **step size** on unsupervised anomaly detection performance.

## Project Overview

The system processes raw sensor data, transforms it into statistical feature windows, and trains an Isolation Forest model. Anomalies are injected at the **raw data level** before windowing to ensure realistic feature-level effects.

## Study Parameters (Locked)

To ensure a controlled evaluation, the following parameters are locked:

- **Contamination**: 0.2 (20% expected anomalies).
- **Outlier Fraction**: 0.2 (Locked to match contamination).
- **N Estimators**: 300.
- **Seed**: 42.

## Project Structure

- **`main.py`**: The central orchestrator.
- **`app/data/processor.py`**: Handles raw data loading and windowing.
- **`app/features/extractor.py`**: Extracts statistical features from each window.
- **`app/data/injector.py`**: Injects synthetic outliers into the **raw sensor data**.
- **`app/models/detector.py`**: Wraps the Isolation Forest model.
- **`datasets/`**: Directory for raw sensor datasets.
- **`results/`**: Output directory. Each experiment run is saved in a unique subdirectory based on window and step size (e.g., `results/win200_step100/`).

## Installation & Setup

```bash
pip install pandas numpy scikit-learn
```

## How to Run

1. Configure the `window_size` and `step_size` in `main.py`.
2. Execute the experiment:

```bash
python main.py
```

## Methodology

1.  **Raw Injection**: 20% of the raw data is corrupted with synthetic outliers (1.3x and 0.7x of the mean value).
2.  **Windowing**: The corrupted raw data is then divided into windows.
3.  **Feature Engineering**: Statistical descriptors are extracted from these windows.
4.  **Training**: The Isolation Forest is trained on the raw (un-corrupted) data.
5.  **Evaluation**: The model is tested on the corrupted dataset, and results are saved for future visualization and analysis.
