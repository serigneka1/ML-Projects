import logging

from zenml import step
import mlflow
import pandas as pd
from zenml.client import Client
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin

from src.model_dev import TrainRegressionModel
from configuration import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker = experiment_tracker.name)
def train_model(
         X_train: pd.DataFrame,
         y_train: pd.DataFrame,
        config : ModelNameConfig
    ) -> ClassifierMixin:

    # Création d'une configuration pour le modèle
    # config = ModelNameConfig(model_name="LogisticRegression")
    model = None


    try:
        if config.model_name == "LogisticRegression":
            model = TrainRegressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train_model(X_train, y_train)
            return trained_model
        else:
            logging.info("Ne trouve pas le modèle {}.".format(config.model_name))
        
    except Exception as e:
        logging.error("Erreur pendant l'entrainement du modèle: {}.".format(e))
        raise e
