from tensorflow.keras.models import Model, Sequential

from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType

class LoadModelByObject:
    def __init__(self, model_repository: IModelRepository) -> None:
        self.model_repository = model_repository
    

    def load_all_models(self, model_idx, model_obj):
        return self.model_repository.load_model_by_obj(model_idx, model_obj)