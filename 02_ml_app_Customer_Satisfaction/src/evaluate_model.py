from abc import abstractmethod, ABC
import pandas as pd
from sklearn.metrics import accuracy_score




class EvaluationStrategy(ABC):
    """
    Le blueprint du cacule des scores
    """
    @abstractmethod
    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        blueprint de la fonction qui calcule le score 
        Args:
            y_true (pd.Series): les données réelles pour les tests
            y_pred (pd.Series): Les données prédites 
        Returns: retourne le score 
        """
        pass


class Accuracy(EvaluationStrategy):

    def calculate_score(self, y_true: pd.Series, y_pred: pd.Series) -> float:

        accuracy = accuracy_score(y_true, y_pred)

        return accuracy
    
