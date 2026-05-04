from app.experiment import AnomalyExperiment

if __name__ == "__main__":
    # The study evaluates window_size and step_size.
    # Other parameters are locked in AnomalyExperiment (contamination=0.2, n_estimators=300, seed=42).
    config = {
        'file_name': 'sidstyrning-februari.txt',
        'window_size': 200,
        'step_size': 50
    }

    experiment = AnomalyExperiment(config)
    experiment.run()
