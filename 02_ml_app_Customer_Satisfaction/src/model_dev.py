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
       # Entrainement du modèle
       reg = LogisticRegression(**kwargs)

       try:
        reg.fit(X_train, y_train)
        logging.info("Entrainement du modéle terminé !!!")
        return reg
       except Exception as e:
          logging.error("L'entrainement du modèle a échoué !!!! ")
          raise e 




