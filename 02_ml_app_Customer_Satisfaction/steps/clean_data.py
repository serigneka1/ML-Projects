from zenml import step
import pandas as pd
import numpy as np
from steps.ingest_data import get_data
from src.clean_data import DataCleansingStrategy, PreprocessingStrategy, DivideDataStrategy

from typing import Tuple
from typing_extensions import Annotated
import logging


@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Réalise les opérations de nettoyage du dataset.

    Args:
        data: DataFrame brut en entrée.

    Returns:
        Tuple contenant X_train, X_test, y_train, y_test.
    """
    try:
        # Étape 1 : Nettoyage et prétraitement
        preprocess_strategy = PreprocessingStrategy()
        data_cleaning = DataCleansingStrategy(preprocess_strategy)
        processed_data = data_cleaning.handle_data(data)

        # Étape 2 : Division en train/test
        divide_data_strategy = DivideDataStrategy()
        data_divider = DataCleansingStrategy(divide_data_strategy)
        X_train, X_test, y_train, y_test = data_divider.handle_data(processed_data)

        logging.info("La transformation des données est terminée.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"La transformation des données a échoué: {e}")
        raise