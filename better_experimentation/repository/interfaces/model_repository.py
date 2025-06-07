from abc import abstractmethod, ABC

class IModelRepository(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def load_model_by_obj(self, model_idx, model_obj):
        pass

    @abstractmethod
    def load_model_by_path(self, pathlib_obj):
        pass