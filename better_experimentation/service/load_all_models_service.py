from tensorflow.keras.models import Model, Sequential
from sklearn.base import BaseEstimator
import itertools
import tensorflow
tensorflow.get_logger().setLevel('ERROR')  

from better_experimentation.repository.sklearn_model_repository import SklearnModelRepository
from better_experimentation.repository.tensorflow_model_repository import TensorflowModelRepository
from better_experimentation.service.load_model_service import LoadModelService
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType
from better_experimentation.utils.log_config import LogService, handle_exceptions


class LoadAllModelsService:
    __log_service = LogService()
    scores_classifier = ["accuracy", "f1", "precision", "recall"]
    scores_regression = ["mae", "mse", "r2"]

    def __init__(self, models_trained: list) -> None:
        self.models_trained = models_trained
        self.models = []
        self.__logger = self.__log_service.get_logger(__name__)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def _flatten_list(self, _list):
        return list(itertools.chain.from_iterable(_list))

    @handle_exceptions(__log_service.get_logger(__name__))
    def load_models(self) -> list[MLModel]:
        """Loading trained models from objects or file paths for application in the experimentation pipeline

        Args:
            models_trained (list): List containing the instantiated objects of the trained machine learning models and/or the file paths where the models are compressed.

        Raises:
            ValueError: Unrecognized instance model

        Returns:
            list[MLModel]: List with loaded models
        """
        sklearn_repo = SklearnModelRepository()
        tensorflow_repo = TensorflowModelRepository()

        if isinstance(self.models_trained, str):
            self.models_trained = [self.models_trained]

        for model_idx, model in enumerate(self.models_trained):
            if isinstance(model, str): # load model by path (can generate a new list, )
                self.models.append(LoadModelService(sklearn_repo).load_model_by_path(model))
                self.models.append(LoadModelService(tensorflow_repo).load_model_by_path(model))
            else: # load models_objects to combine with model list
                if isinstance(model, BaseEstimator):
                    self.models.append([LoadModelService(sklearn_repo).load_model_by_obj(model_idx, model)])
                elif isinstance(model, (Model, Sequential)):
                    self.models.append([LoadModelService(tensorflow_repo).load_model_by_obj(model_idx, model)])
                else:
                    raise ValueError(
                        "Invalid Model Technology."
                    )
        self.models = self._flatten_list(self.models)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def validate_models(self):
        """Checks whether all models are classifiers or regressors.

        Args:
            models (list[MLModel]): List of trained and loaded models containing information about each model

        Raises:
            ValueError: If there are models of different types in the same model list to apply in the experiment
        """
        if (not all(model.model_type == ModelType.classifier.value for model in self.models)
            and not all(model.model_type == ModelType.regressor.value for model in self.models)):
            raise ValueError("models must need all models to be classifiers or regressors and not a mixture of them, so a comparison is not possible.")
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def validate_scores_target(self, scores_target: list[str]):
        """Checks whether the performance metric exists and whether it makes sense according to the type of Machine Learning model that will be used

        Args:
            models (list[MLModel]): List of trained and loaded models containing information about each model
            scores_target (ist[str]): Performance metric selected to be used as a basis for comparing models.

        Raises:
            ValueError: If there are models of different types in the same model list to apply in the experiment
        """
        if all(model.model_type == ModelType.classifier.value for model in self.models):
            if all([score not in self.scores_classifier for score in scores_target]):
                raise ValueError(f"scores_target must be valid between them {self.scores_classifier}")
        if all(model.model_type == ModelType.regressor.value for model in self.models):
            if all([score not in self.scores_regression for score in scores_target]):
                raise ValueError(f"scores_target must be valid between them {self.scores_regression}")
    
    def get_models(self):
        return self.models