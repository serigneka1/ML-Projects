from zenml import pipeline
from steps.ingest_data import get_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.validate_model import validate_model

@pipeline
def train_pipeline(data_base_dir):
    # Ingest data
    df = get_data(data_base_dir)

    # Clean data
    clean_data(df)
    df_propres = clean_data(df)

    # Train and validate model
    train_model(df_propres)
    validate_model(df_propres)
    

