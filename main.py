import pandas as pd
import os
from app.experiment import AnomalyExperiment

if __name__ == "__main__":
    """
    Maskininlärningsbaserad Anomalidetektion: 
    En studie inom anomalidetektion på sidstyrning i ett Steckelvalsverk.
    Högskolan i Gävle - Linus Renström.

    Grid Search: Evaluating Window Size and Step Size.
    """
    
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
                experiment = AnomalyExperiment(config)
                f1_score = experiment.run()
                
                summary_results.append({
                    'window_size': w,
                    'step_size': s,
                    'anomaly_f1_score': f1_score
                })
                
                # Save intermediate summary after each run to prevent data loss
                summary_df = pd.DataFrame(summary_results)
                summary_df.to_csv("results/grid_search_summary.csv", index=False)
                
            except Exception as e:
                print(f"FAILED experiment Window={w}, Step={s}: {e}")

    print("\n" + "#"*50)
    print("GRID SEARCH COMPLETE")
    print(f"Final summary saved to 'results/grid_search_summary.csv'")
    print("#"*50)
