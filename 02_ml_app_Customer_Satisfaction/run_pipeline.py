from pipelines.training_pipelines import train_pipeline
import logging


if __name__ == "__main__":
    train_pipeline("data")
    logging.info("La pipeline est exécutée avec succès !")