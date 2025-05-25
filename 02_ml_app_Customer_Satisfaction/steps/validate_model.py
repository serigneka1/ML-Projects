from zenml import step
import pandas as pd



@step
def validate_model(data: pd.DataFrame) -> None:
    pass
