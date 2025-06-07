from pathlib import Path
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType


class TensorflowModelRepository(IModelRepository):
    def __init__(self):
        super().__init__()

    def load_model_by_obj(self, model_idx, model_obj):
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
            model_index=model_idx,
            model_name=f"{type(model_obj).__name__}_{id(model_obj)}",
            model_object=model_obj,
            model_technology=ModelTechnology.tensorflow.value,
            model_type=model_type
        )

    def load_model_by_path(self, pathlib_obj):
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