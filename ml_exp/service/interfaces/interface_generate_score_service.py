from abc import abstractmethod, ABC


class IGenerateScoreService(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_scores_data(self):
        pass