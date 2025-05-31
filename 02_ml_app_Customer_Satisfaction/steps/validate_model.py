from zenml import step
import pandas as pd

import logging

from sklearn.base import ClassifierMixin
from src.evaluate_model import Accuracy



@step
def validate_model(model: ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series) -> float:
    try:
        predictions = model.predict(X_test)


        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_score(y_test, predictions)
        logging.info("Evaluation terminée ...")

        return accuracy
    except Exception as e:
        logging.error("Erreur lors de l'évaluation: {}".format(e))
        raise e




    

