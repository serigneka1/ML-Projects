import numpy as np
import pandas as pd

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers import (MLFlowModelDeployer, )
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.ingest_data import get_data
from steps.clean_data import clean_data
from steps.training_model import train_model
from steps.validate_model import validate_model


docker_settings = DockerSettings(required_integrations=[MLFLOW])



class DeploymentTriggerConfig(BaseParameters):
    """ Deployment trigger config """
    min_accuracy:float = 0.92

@step(enable_cache=False)
def deployment_trigger( 
    accuracy: float,
    config: DeploymentTriggerConfig
):
    """ Implement a simple model deployment trigger that looks at the input model accuracy and decide 
    if it is good enough to deploy or not"""
    return accuracy >= config.min_accuracy




@pipeline(enable_cache=True, settings={"docker_settings": docker_settings})

def continuous_deployement_pipeline(
    min_accuracy:float= 0.92,
    workers:int = 1,
    timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT ):

        # Ingest data
    df = get_data(data_base_dir= "data")

    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    accuracy = validate_model(model, X_test, y_test)

    deployment_config = DeploymentTriggerConfig(min_accuracy= min_accuracy)
    deployment_decission = deployment_trigger(accuracy, config = deployment_config )

    mlflow_model_deployer_step(
        model = model,
        deployment_decission = deployment_decission,
        workers = workers,
        timeout = timeout,
        )
    