from pipelines.training_pipelines import train_pipeline
import logging

from zenml.client import Client


if __name__ == "__main__":
    tracking_uri = experiment_tracker = Client().active_stack.experiment_tracker.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}") # pour voir le chemin de là où se trouve le uri

    train_pipeline("data")
    logging.info("La pipeline est exécutée avec succès !")

    
