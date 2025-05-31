# from zenml.steps import BaseStep

# # "BaseStep",
# # "ResourceSettings",
# # "StepContext",
# # "step",
# # "get_step_context"

# class ModelNameConfig(BaseStep):
#     model_name: str = "LinearRegression"

from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: str = "LogisticRegression"
