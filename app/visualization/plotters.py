import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class BasePlotter:
    """Base class for shared visualization logic."""
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

class AnomalyHeatmapPlotter(BasePlotter):
    """Refactored Heatmap visualization into a class."""
    
    def plot(self, results_path: str, test_data_path: str, window_size: int, step_size: int, limit: int = 20000):
        print("Generating anomaly heatmap...")
        try:
            results = pd.read_csv(results_path)
            test_data = pd.read_csv(test_data_path)
        except FileNotFoundError as e:
            print(f"Error: Could not find files for plotting: {e}")
            return

        signals = [
            'Sidstyrning Inlopp - Position',
            'Sidstyrning Inlopp - Kraft',
            'Sidstyrning Inlopp - Centrum (OS-DS)/2'
        ]
        signals = [s for s in signals if s in test_data.columns]
        
        test_data_sub = test_data.iloc[:limit]
        results_sub = results.iloc[:limit // step_size]

        fig, axes = plt.subplots(len(signals), 1, figsize=(18, 4 * len(signals)), sharex=True)
        if len(signals) == 1: axes = [axes]

        for ax, sig in zip(axes, signals):
            ax.plot(test_data_sub.index, test_data_sub[sig], color='black', alpha=0.5, label=f'Sensor: {sig}', linewidth=1)
            
            added_labels = set()
            for i, row in results_sub.iterrows():
                start = i * step_size
                end = start + window_size
                
                if row['ground_truth'] == -1:
                    label = 'Actual Bug' if 'GT' not in added_labels else ""
                    ax.axvspan(start, end, color='blue', alpha=0.1, label=label)
                    added_labels.add('GT')

                if row['prediction'] == -1:
                    label = 'Model Alarm' if 'Larm' not in added_labels else ""
                    ax.axvline(x=start, color='red', linestyle='--', alpha=0.6, label=label)
                    added_labels.add('Larm')

            ax.set_ylabel("Value")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.2)

        plt.suptitle("Multi-Sensor Anomaly View", fontsize=16)
        plt.xlabel("Samples")
        plt.xlim(0, limit)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_file = os.path.join(self.output_dir, 'anomaly_heatmap.png')
        plt.savefig(output_file, dpi=200)
        print(f"Heatmap saved to {output_file}")

class ScorePlotter(BasePlotter):
    """Refactored Score visualization into a class."""

    def plot(self, results_path: str):
        print("Generating score plots...")
        try:
            results = pd.read_csv(results_path)
        except FileNotFoundError:
            print(f"Error: {results_path} not found.")
            return

        plt.figure(figsize=(15, 6))
        plt.plot(results.index, results['score'], label='Anomaly Score', color='blue', linewidth=1.5)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Threshold')

        gt_anomalies = results[results['ground_truth'] == -1]
        if not gt_anomalies.empty:
            plt.scatter(gt_anomalies.index, gt_anomalies['score'], 
                        color='orange', marker='x', s=100, label='Injected Bugs', zorder=5)

        plt.title("Isolation Forest Decision Scores")
        plt.xlabel("Window Index")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        output_file = os.path.join(self.output_dir, 'anomaly_scores.png')
        plt.savefig(output_file)
        print(f"Score plot saved to {output_file}")
