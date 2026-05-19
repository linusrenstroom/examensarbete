import pandas as pd
import os
from app.experiment import AnomalyExperiment

if __name__ == "__main__":
    print("\n" + "#"*50)
    print("ANOMALY DETECTION EXPERIMENT STARTING...")
    print("Files will be saved to: " + os.path.abspath("results"))
    print("#"*50 + "\n")
    """
    Maskininlärningsbaserad Anomalidetektion: 
    En studie inom anomalidetektion på sidstyrning i ett Steckelvalsverk.
    Högskolan i Gävle - Linus Renström.

    Grid Search: Evaluating Window Size and Step Size with Performance Metrics.
    """
    
    # Define the parameter grid to investigate
    window_sizes = [20, 50, 100, 200]
    step_sizes = [1, 5, 10, 25]
    
    summary_results = []
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    for w in window_sizes:
        for s in step_sizes:
            print("\n" + "="*50)
            print(f"STARTING EXPERIMENT: Window={w}, Step={s}")
            print("="*50 + "\n")
            
            config = {
                'file_name': 'sidstyrning-februari.txt',
                'window_size': w,
                'step_size': s
            }
            
            try:
                # Build the specific output directory path
                output_dir = os.path.join("results", f"win{w}_step{s}")
                
                # Initialize and run the experiment
                experiment = AnomalyExperiment(config)
                metrics = experiment.run(output_dir)
                
                # Combine parameters with results and timings for the global summary
                result_entry = {
                    'window_size': w,
                    'step_size': s,
                    'anomaly_f1_score': metrics['f1_score'],
                    'time_extraction_sec': metrics['time_feature_extraction'],
                    'time_training_sec': metrics['time_training'],
                    'time_inference_sec': metrics['time_inference'],
                    'total_time_sec': metrics['total_time']
                }
                
                summary_results.append(result_entry)
                
                # Save global summary
                summary_df = pd.DataFrame(summary_results)
                summary_path = "results/grid_search_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                
                print(f"\n>>> EXPERIMENT SUCCESS <<<")
                print(f"Results saved in: {os.path.abspath(output_dir)}")
                print(f"Global summary updated: {os.path.abspath(summary_path)}")
                print("="*50)
                
            except Exception as e:
                print(f"FAILED experiment Window={w}, Step={s}: {e}")

    print("\n" + "#"*50)
    print("GRID SEARCH COMPLETE")
    print(f"Final summary with performance metrics saved to 'results/grid_search_summary.csv'")
    print("#"*50)
