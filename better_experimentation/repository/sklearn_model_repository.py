from pathlib import Path
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType
from better_experimentation.utils.log_config import LogService, handle_exceptions


class SklearnModelRepository(IModelRepository):
    """Repository to handle loading models (via object and file path) in the context of Sklearn dependency

    Args:
        IModelRepository (ABC): Interface for repositories responsible for loading machine learning models
    """
    def __init__(self):
        super().__init__()

    def load_model_by_obj(self, model_name: str, model_obj: BaseEstimator) -> MLModel:
        """Loads models from the instantiated object

        Args:
            model_name (str): Model name for identification
            model_obj (BaseEstimator): Object that is parked the trained model to be loaded

        Raises:
            ValueError: If the instance type is not within the mapping

        Returns:
            MLModel: Model loaded and processed to be used in the testing phases
        """
        model_type = None
        if isinstance(model_obj, ClassifierMixin):
            model_type = ModelType.classifier.value
        elif isinstance(model_obj, RegressorMixin):
            model_type = ModelType.regressor.value
        else:
            raise ValueError(
                f"Model have invalid type. Current model type: {type(model_obj)}"
            )
        return MLModel(
            model_name=model_name,
            model_object=model_obj,
            model_technology=ModelTechnology.sklearn.value,
            model_type=model_type
        )
    

    def load_model_by_path(self, pathlib_obj: Path, model_name: str) -> list[MLModel]:
        """Loads models from the path that model is stored

        Args:

            pathlib_obj (Path): Base path where stored models exist loaded in PathLib

        Raises:
            ValueError: If the instance type is not within the mapping

        Returns:
            list[MLModel]: List of models loaded and processed to be used in the testing phases
        """
        with open(pathlib_obj, 'rb') as fp:
            model_loaded = pickle.load(fp)
            model_type = None
            if isinstance(model_loaded, ClassifierMixin):
                model_type = ModelType.classifier.value
            elif isinstance(model_loaded, RegressorMixin):
                model_type = ModelType.regressor.value
            else:
                raise ValueError(
                    f"Model have invalid type. Current model type: {type(model_loaded)}"
                )
        return MLModel(
                model_name=model_name,
                model_object=model_loaded,
                model_technology=ModelTechnology.sklearn.value,
                    model_type=model_type
                )