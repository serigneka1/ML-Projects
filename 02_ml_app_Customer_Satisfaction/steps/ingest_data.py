import pandas as pd
import os
from zenml import step



class IngestData:
    def __init__(self, data_base_dir):
        self.data_base_dir = data_base_dir
    
    def get_data_path(self):
        # data_base_dir = "../data"
        for file_name in os.listdir(self.data_base_dir):
            if file_name.endswith(".csv"):
                return os.path.join(self.data_base_dir, file_name)

        # si aucun fichier csv n'est trouvé après avoir parcouru le dossier
        raise ValueError(f"Il n 'a pas de {file_name}.csv dans le dossier {self.data_base_dir}")
    


@step
def get_data(data_base_dir):
    ingest_data = IngestData(data_base_dir)
    data_path = ingest_data.get_data_path()
    print(data_path)
    return pd.read_csv(data_path) 

get_data("data")