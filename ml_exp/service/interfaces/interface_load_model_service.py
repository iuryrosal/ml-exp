from abc import abstractmethod, ABC

from ml_exp.model.ml_model import MLModel


class ILoadModelService(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def load_model_by_obj(self, context_name: str, model_obj):
        pass

    @abstractmethod
    def load_model_by_path(self, model_path: str, context_name: str) -> MLModel:
        pass