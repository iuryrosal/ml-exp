from pydantic import BaseModel
from enum import Enum
from typing import Union
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Model, Sequential


class ModelTechnology(str, Enum):
    """Supported Technology Types for Supervised Machine Learning Models
    """
    sklearn = "sklearn"
    tensorflow = "tensorflow"

class ModelType(str, Enum):
    """Supported Models Types for Supervised Machine Learning Models
    """
    classifier = "classifier"
    regressor = "regressor"

class MLModel(BaseModel):
    """A generic representation of a trained and loaded model
    """
    model_index: int
    model_name: str
    model_object: Union[BaseEstimator, Model, Sequential]
    model_technology: ModelTechnology
    model_type: ModelType

    class Config:
        arbitrary_types_allowed = True