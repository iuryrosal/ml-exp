from pathlib import Path
import onnxruntime as ort
import mlflow.pyfunc

from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType


class GeneralModelRepository(IModelRepository):
    """Repository to handle loading models (via object and file path) in the context of general machine learning model
    Args:
        IModelRepository (ABC): Interface for repositories responsible for loading machine learning models
    """
    def __init__(self):
        super().__init__()

    def load_model_by_obj(self, model_name: str, model_obj) -> MLModel:
        """Loads models from the instantiated object

        Args:
            model_idx (int): Model incidence for identification
            model_obj (BaseEstimator): Object that is parked the trained model to be loaded

        Raises:
            ValueError: If the instance type is not within the mapping

        Returns:
            MLModel: Model loaded and processed to be used in the testing phases
        """
        if isinstance(model_obj, mlflow.pyfunc.PyFuncModel):
            return MLModel(
                model_name=model_name,
                model_object=model_obj,
                model_technology=ModelTechnology.mlflow_sklearn.value,
                model_type=ModelType.undefined.value
            )

        return MLModel(
            model_name=model_name,
            model_object=model_obj,
            model_technology=ModelTechnology.general_from_onnx.value,
            model_type=ModelType.undefined.value
        )

    def load_model_by_path(self, pathlib_obj: Path, model_name:str) -> MLModel:
        """Loads models from the path that model is stored, considering onnx format.

        Args:
            pathlib_obj (Path): Base path where stored models exist loaded in PathLib

        Raises:
            ValueError: If the instance type is not within the mapping

        Returns:
            list[MLModel]: List of models loaded and processed to be used in the testing phases
        """
        model_loaded = ort.InferenceSession(pathlib_obj)

        return MLModel(
            model_name=model_name,
            model_object=model_loaded,
            model_technology=ModelTechnology.general_from_onnx.value,
            model_type=ModelType.undefined.value
        )