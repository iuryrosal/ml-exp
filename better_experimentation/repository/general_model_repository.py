from pathlib import Path
import onnxruntime as ort

from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType


class GeneralModelRepository(IModelRepository):
    """Repository to handle loading models (via object and file path) in the context of Sklearn dependency

    Args:
        IModelRepository (ABC): Interface for repositories responsible for loading machine learning models
    """
    def __init__(self):
        super().__init__()

    def load_model_by_obj(self, model_idx: int, model_obj) -> MLModel:
        """Loads models from the instantiated object

        Args:
            model_idx (int): Model incidence for identification
            model_obj (BaseEstimator): Object that is parked the trained model to be loaded

        Raises:
            ValueError: If the instance type is not within the mapping

        Returns:
            MLModel: Model loaded and processed to be used in the testing phases
        """

        return MLModel(
            model_index=model_idx,
            model_name=f"{type(model_obj).__name__}_{id(model_obj)}",
            model_object=model_obj,
            model_technology=ModelTechnology.general_from_onnx.value,
            model_type=ModelType.undefined.value
        )

    def load_model_by_path(self, pathlib_obj: Path) -> list[MLModel]:
        """Loads models from the path that model is stored

        Args:
            pathlib_obj (Path): Base path where stored models exist loaded in PathLib

        Raises:
            ValueError: If the instance type is not within the mapping

        Returns:
            list[MLModel]: List of models loaded and processed to be used in the testing phases
        """
        models = []
        list_of_models_path = list(pathlib_obj.glob("**/*.onnx"))
        for model_idx, model_path in enumerate(list_of_models_path):
            model_loaded = ort.InferenceSession(model_path)

            models.append(
                MLModel(
                    model_index=model_idx,
                    model_name=f"{model_path.name}",
                    model_object=model_loaded,
                    model_technology=ModelTechnology.general_from_onnx.value,
                    model_type=ModelType.undefined.value
                )
            )
        return models