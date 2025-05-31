from zenml import step
import pandas as pd
from src.model_dev import TrainRegressionModel
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from configuration import ModelNameConfig
import logging


@step
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
            trained_model = model.train_model(X_train, y_train)
            return trained_model
        else:
            logging.info("Ne trouve pas le modèle {}.".format(config.model_name))
        
    except Exception as e:
        logging.error("Erreur pendant l'entrainement du modèle: {}.".format(e))
        raise e
