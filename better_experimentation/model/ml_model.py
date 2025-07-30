from pydantic import BaseModel
from enum import Enum
from typing import Union
from sklearn.base import BaseEstimator
import onnxruntime
import mlflow.pyfunc


class ModelTechnology(str, Enum):
    """Supported Technology Types for Supervised Machine Learning Models
    """
    sklearn = "sklearn"
    general_from_onnx = "general_from_onnx"
    mlflow_sklearn = "mlflow_sklearn"

class ModelType(str, Enum):
    """Supported Models Types for Supervised Machine Learning Models
    """
    undefined = "undefined"
    classifier = "classifier"
    regressor = "regressor"

class MLModel(BaseModel):
    """A generic representation of a trained and loaded model
    """
    model_name: str
    model_object: Union[BaseEstimator, onnxruntime.InferenceSession, mlflow.pyfunc.PyFuncModel]
    model_technology: ModelTechnology
    model_type: ModelType

    class Config:
        arbitrary_types_allowed = True