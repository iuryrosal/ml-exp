from better_experimentation.repository.interfaces.model_repository import IModelRepository
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType
from pathlib import Path

class LoadModelService:
    def __init__(self, model_repository: IModelRepository) -> None:
        self.model_repository = model_repository

    def load_model_by_obj(self, context_name: str, model_obj):
        return self.model_repository.load_model_by_obj(context_name=context_name,
                                                       model_obj=model_obj)

    def load_model_by_path(self, model_path: str, context_name: str) -> MLModel:
        pathlib_obj_with_model = Path(model_path)
        return self.model_repository.load_model_by_path(pathlib_obj=pathlib_obj_with_model,
                                                        context_name=context_name)