from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from tensorflow.keras.models import Model, Sequential

from model.ml_model import MLModel, ModelTechnology, ModelType
from utils.log_config import LogService, handle_exceptions


class LoadModelByObject:
    __log_service = LogService()
    def __init__(self, models_trained) -> None:
        self.models_obj = models_trained
        self.__logger = self.__log_service.get_logger(__name__)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def load_all_models(self):
        all_models = []
        for model_obj in self.models_obj:
            if isinstance(model_obj, BaseEstimator):
                all_models.append(self.load_sklearn_model(model_obj))
            elif isinstance(model_obj, (Model, Sequential)):
                all_models.append(self.load_tensorflow_model(model_obj))
            else:
                raise ValueError(
                    "Invalid Model Technology."
                )
        return all_models

    @handle_exceptions(__log_service.get_logger(__name__))
    def load_sklearn_model(self, model_obj):
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
            model_object=model_obj,
            model_technology=ModelTechnology.sklearn.value,
            model_type=model_type
        )

    @handle_exceptions(__log_service.get_logger(__name__))
    def load_tensorflow_modeL(self, model_obj):
        loss = model_obj.loss
        activation = model_obj.layers[-1].activation.__name__

        model_type = None
        if 'categorical_crossentropy' in loss or 'binary_crossentropy' in loss:
            model_type = ModelType.classifier.value
        elif 'mse' in loss or 'mae' in loss:
            model_type = ModelType.regressor.value
        elif activation in ['softmax', 'sigmoid']:
            model_type = ModelType.classifier.value
        elif activation == 'linear':
            model_type = ModelType.regressor.value
        else:
            raise ValueError(
                f"Model have invalid type. Current model type: {type(model_obj)}"
            )
        return MLModel(
            model_object=model_obj,
            model_technology=ModelTechnology.tensorflow.value,
            model_type=model_type
        )