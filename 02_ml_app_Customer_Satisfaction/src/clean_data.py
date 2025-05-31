from abc import ABC, abstractmethod
from typing import Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Union




class CleanDataStrategy(ABC):
    """
    Appliquer la strategy abstract
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame):
        """ 
            Applique un certain nombre d'étape de preprocessing sur le dataframe fournie
        Args:
            data (pd.DataFrame): prend un dataframe en entrée

        Returns:
            Any: retourn un autre dataframe ou d'autre types
        """
        pass


class PreprocessingStrategy(CleanDataStrategy):
    """
        Applique un ensemble de méthode de nétoyage (valeurs manquantes, renommage colonnes
        encodage des données catégorielles)
    Args:
        CleanDataStrategy (class strategy): recoit en entrée la stratégie abstract
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et encode le DataFrame fourni.

        Args:
            data (pd.DataFrame): DataFrame original.

        Returns:
            pd.DataFrame: DataFrame nettoyé et encodé.
        """
        df = data.copy()

        # Nettoyage des valeurs manquantes
        try:
            df = df[df["Arrival Delay in Minutes"].notna()]
            logging.info("Valeurs manquantes supprimées dans la colonne: 'Arrival Delay in Minutes'.")
        except Exception as e:
            logging.error(f"Erreur pendant la gestion des valeurs manquantes : {e}")
            raise
        df.drop(columns=["Gender"], inplace=True, errors="ignore")

        # Renommer les colonnes en français
        rename_dict = {
        'satisfaction': 'satisfaction',
        'Customer Type': 'type_client',
        'Age': 'age',
        'Type of Travel': 'type_voyage',
        'Class': 'classe_vol',
        'Flight Distance': 'distance_vol',
        'Seat comfort': 'confort_siege',
        'Departure/Arrival time convenient': 'ponctualite',
        'Food and drink': 'repas_boisson',
        'Gate location': 'prox_porte',
        'Inflight wifi service': 'wifi_bord',
        'Inflight entertainment': 'divertissement',
        'Online support': 'support_enligne',
        'Ease of Online booking': 'resa_facile',
        'On-board service': 'service_bord',
        'Leg room service': 'espace_jambes',
        'Baggage handling': 'gestion_bagages',
        'Checkin service': 'enregistrement',
        'Cleanliness': 'proprete',
        'Online boarding': 'embarquement',
        'Departure Delay in Minutes': 'retard_dep',
        'Arrival Delay in Minutes': 'retard_arr'
        }

        try:
            df.rename(columns=rename_dict, inplace=True)
            logging.info("Renommage des colonnes réussi.")
        except Exception as e:
            logging.error(f"Erreur pendant le renommage des colonnes : {e}")
            raise
        #df = df.drop(columns=['genre'], axis=1)
        # Encodage
        try:
            # Exemple : encode le genre
            # df["genre"] = df["genre"].map({"Male": 1, "Female": 0})

            df["satisfaction"] = df["satisfaction"].apply(lambda x: 1 if x == 'satisfied' else 0)
            df["type_client"] = df["type_client"].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
            df["type_voyage"] = df["type_voyage"].apply(lambda x: 1 if x == 'Personal Travel' else 0)
            df["classe_vol"] = df["classe_vol"].map({
                'Eco': 1,
                'Business': 2,
                'Eco Plus': 3
            })

            logging.info("Encodage des colonnes terminé.")
        except Exception as e:
            logging.error(f"Erreur pendant l'encodage des colonnes : {e}")
            raise

        return df

class DivideDataStrategy(CleanDataStrategy):
    """Applique la stratégie de division des données

    Args:
        CleanDataStrategy (_type_): s'applique sur les données propres
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """ applique les trannsformations nécessaires pour faire la division des données

        Args:
            data (_type_): recoit en entrée un dataset

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: retourne des échantillons de données d'entraienement et de tests
        """
        df = data.copy()
        X = df.drop("satisfaction", axis=1)
        y = df["satisfaction"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Le dataframe est bien divisé en en entrainement (80%) et test(20%)")

        return X_train, X_test, y_train, y_test
        

class DataCleansingStrategy:
    """Applique la stratégie de nettoyage choie
    """
    def __init__(self, strategy: CleanDataStrategy):
        """
        Initialise avec une stratégie spécifique.
        Args:
            strategy (CleanDataStrategy): La stratégie de nettoyage à appliquer.
        """
        self.strategy = strategy

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Exécute la stratégie de nettoyage choisie.
        Args:
            data (pd.DataFrame): dataset à nettoyer.

        Returns:
            Any: Résultat de la stratégie -> (DataFrame ou tuple de DataFrames/Series).
        """
        return self.strategy.handle_data(data)






