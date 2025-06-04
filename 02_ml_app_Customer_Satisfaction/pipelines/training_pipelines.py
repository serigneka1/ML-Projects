from zenml import pipeline

from steps.ingest_data import get_data
from steps.clean_data import clean_data
from steps.training_model import train_model
from steps.validate_model import validate_model
from configuration import ModelNameConfig


@pipeline(enable_cache=True)
def train_pipeline(data_base_dir: str):
    # Ingest data
    df = get_data(data_base_dir)

    # Clean data
    X_train, X_test, y_train, y_test = clean_data(df)

    # Train model 
    model = train_model(X_train, y_train)

    # Validate model
    accuracy = validate_model(model, X_test, y_test)
    

