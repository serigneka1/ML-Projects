from steps.ingest_data import get_data
from abc import ABC, abstractmethod
from typing import Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import logging




class CleanDataStrategy(ABC):
    """
    Appliquer la strategy abstract
    """
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> Any:
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
        CleanDataStrategy (_type_): recoit en entrée la stratégie abstract
    """
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et encode le DataFrame fourni.

        Args:
            data (pd.DataFrame): DataFrame original.

        Returns:
            pd.DataFrame: DataFrame nettoyé et encodé.
        """

        df = data.copy()
         # Remplacer les valeurs manquantes
        try:
            df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].bfill()
            logging.info("Valeur manquante remplacée par la valeur suivante dans la même colonne ...")
        except Exception as e:
            logging.error(f"Il y a une erreur pendant la gestion des valeurs manquantes: \n {e}")
            raise
       

        # Renommer les colonnes en français
        colonnes_fr = [
            'satisfaction', 'genre', 'type_client', 'age', 'type_voyage', 'classe',
            'distance_vol', 'confort_siege', 'horaire_dep_arr', 'repas_boisson',
            'emplacement_porte', 'wifi_vol', 'divertissement_vol', 'support_en_ligne',
            'facilité_reservation', 'service_bord', 'espace_jambes', 'gestion_bagages',
            'service_enregistrement', 'propreté', 'embarquement_en_ligne',
            'retard_depart_min', 'retard_arrivée_min'
        ]
        try:
            df.columns = colonnes_fr
            logging.info("Renommage des colonnes réussit ...")
        except Exception as e:
            logging.info(f"Il y a un problème pendant le renommage des colonnes en francais: {e}")
            raise

        # Encodage
        try:
            df["satisfaction"] = df["satisfaction"].apply(lambda x: 1 if x == 'satisfied' else 0)
            df["genre"] = df["genre"].apply(lambda x: 1 if x == 'Male' else 0)
            df["type_client"] = df["type_client"].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
            df["type_voyage"] = df["type_voyage"].apply(lambda x: 1 if x == 'Personal Travel' else 0)
            df["classe"] = df["classe"].map({'Eco': 1, 'Business': 2, 'Eco Plus': 3})
            logging.info("Encodage du dataset terminé !!!")
        except Exception as e:
            logging.error(f"Il y a un problème pendant l'encodage des colonnes catégorielles: {e}")
            raise


        return df




class DivideDataStrategy(CleanDataStrategy):
    """Applique la stratégie de division des données

    Args:
        CleanDataStrategy (_type_): s'applique sur les données propres
    """
    def __init__(self, preprocessing_strategy: PreprocessingStrategy):
        self.preprocessing_strategy = preprocessing_strategy

    def clean_data(self, data) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ applique les trannsformations nécessaires pour faire la division des données

        Args:
            data (_type_): recoit en entrée un dataset

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: retourne des échantillons de données d'entraienement et de tests
        """
        df = self.preprocessing_strategy.clean_data(data)
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

    def transform_data(self, data: pd.DataFrame) -> Any:
        """
        Exécute la stratégie de nettoyage choisie.
        Args:
            data (pd.DataFrame): dataset à nettoyer.

        Returns:
            Any: Résultat de la stratégie -> (DataFrame ou tuple de DataFrames/Series).
        """
        return self.strategy.clean_data(data)




# Utilisation
# data = get_data()

# preprocess_strategy = PreprocessingStrategy()
# divide_data_strategy = DivideDataStrategy(preprocess_strategy)

# data_cleansing = DataCleansingStrategy(divide_data_strategy)  
# X_train, X_test, y_train, y_test = data_cleansing.transform_data(data)