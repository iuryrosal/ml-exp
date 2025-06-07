from pathlib import Path
import pickle
from sklearn.base import ClassifierMixin, RegressorMixin

import tensorflow
tensorflow.get_logger().setLevel('ERROR')  

from tensorflow import keras

from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType
from better_experimentation.utils.log_config import LogService, handle_exceptions


class LoadModelByPath:
    __log_service = LogService()

    def __init__(self, model_repository: IModelRepository) -> None:
        self.model_repository = model_repository
        self.__logger = self.__log_service.get_logger(__name__)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def load_all_models(self, model_path):
        pathlib_obj_with_model = Path(model_path)
        return self.model_repository.load_model_by_path(pathlib_obj_with_model)