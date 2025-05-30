from pathlib import Path
import pickle
from sklearn.base import ClassifierMixin, RegressorMixin

import tensorflow
tensorflow.get_logger().setLevel('ERROR')  

from tensorflow import keras
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType
from better_experimentation.utils.log_config import LogService, handle_exceptions


class LoadModelByPath:
    __log_service = LogService()

    def __init__(self, models_trained) -> None:
        self.models_path = models_trained
        self.__logger = self.__log_service.get_logger(__name__)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def load_all_models(self):
        all_models = []
        for model_path in self.models_path:
            pathlib_obj_with_model = Path(model_path)
            all_models += self.load_sklearn_model(pathlib_obj_with_model)
            all_models += self.load_tensorflow_model(pathlib_obj_with_model)
        return all_models

    @handle_exceptions(__log_service.get_logger(__name__))
    def load_sklearn_model(self, pathlib_obj):
        models = []
        list_of_models_path = list(pathlib_obj.glob("**/*.obj")) + list(pathlib_obj.glob("**/*.pkl"))
        for model_idx, model_path in enumerate(list_of_models_path):
            with open(model_path, 'rb') as fp:
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
                models.append(
                    MLModel(
                        model_index=model_idx,
                        model_name=f"{model_path.name}",
                        model_object=model_loaded,
                        model_technology=ModelTechnology.sklearn.value,
                        model_type=model_type
                    )
                )
        return models

    @handle_exceptions(__log_service.get_logger(__name__))
    def load_tensorflow_model(self, pathlib_obj):
        models = []
        list_of_models_path = list(pathlib_obj.glob("**/*.h5"))
        for model_idx, model_path in enumerate(list_of_models_path):
            model_loaded = keras.models.load_model(model_path)
            loss = model_loaded.loss
            activation = model_loaded.layers[-1].activation.__name__

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
                    f"Model have invalid type. Current model type: {type(model_loaded)}"
                )
            
            models.append(
                MLModel(
                    model_index=model_idx,
                    model_name=f"{model_path.name}",
                    model_object=model_loaded,
                    model_technology=ModelTechnology.tensorflow.value,
                    model_type=model_type
                )
            )
        return models