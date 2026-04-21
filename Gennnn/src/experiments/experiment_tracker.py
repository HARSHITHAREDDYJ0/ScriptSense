# src/experiments/experiment_tracker.py

import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, save_dir="experiments/logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log_experiment(self, model_name, params, metrics):
        experiment = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "parameters": params,
            "metrics": metrics
        }

        file_path = os.path.join(self.save_dir, f"{model_name}.json")

        # Append mode
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(experiment)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Logged experiment for {model_name}")