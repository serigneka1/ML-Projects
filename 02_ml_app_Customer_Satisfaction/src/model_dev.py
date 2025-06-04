import logging
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin



class Model(ABC):

    @abstractmethod
    def train_model(self, X_train, y_train, **kwargs )-> ClassifierMixin:
        pass

class TrainRegressionModel(Model):

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs)-> ClassifierMixin:
        """
        Entrainement du modèle de classification de regression logistique
        Args:
            X_train (pd.DataFrame): Les features d'entrainement
            y_train (pd.DataFrame): Les targets des données d'entrainement

        Raises:
            e: soulève une erreur si l'entrainement du modèle échoue

        Returns:
            ClassifierMixin: retourn un modèle de type classification
        """
        reg = LogisticRegression(**kwargs)
        try:
            reg.fit(X_train, y_train)
            logging.info("Entrainement du modéle terminé !!!")
            return reg
        except Exception as e:
            logging.error("L'entrainement du modèle a échoué !!!! ")
            raise e




