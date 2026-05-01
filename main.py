from app.experiment import AnomalyExperiment

if __name__ == "__main__":
    # Updated config with file_name and folder organization
    config = {
        'file_name': 'sidstyrning-februari.txt',
        'window_size': 200,
        'step_size': 100,
        'outlier_fraction': 0.20,
        'noise_fraction': 0.20,
        'contamination': 0.2,
        'n_estimators': 300,
        'seed': 42
    }

    experiment = AnomalyExperiment(config)
    experiment.run()
