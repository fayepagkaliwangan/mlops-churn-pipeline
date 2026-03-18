import mlflow
import os

def setup_mlflow_experiment(experiment_name="Telco_Churn_Prediction"):
    """
    Sets up the MLflow tracking URI and creates an experiment.
    """
    # Create a local logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Tell MLflow to save data locally in the logs folder
    tracking_uri = f"sqlite:///logs/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set up the experiment
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking is set up. Experiment: {experiment_name}")
    print(f"Tracking URI: {tracking_uri}")

if __name__ == "__main__":
    setup_mlflow_experiment()