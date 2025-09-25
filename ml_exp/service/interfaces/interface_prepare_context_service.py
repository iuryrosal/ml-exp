from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
from typing import Union


class IPrepareContextService(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def _get_ml_models(self):
        pass

    @abstractmethod
    def check_if_context_exists(self, context_name: str) -> None:
        pass

    @abstractmethod
    def get_contexts(self):
        pass

    @abstractmethod
    def load_ml_model(self,
                   context_name: str,
                   model_trained: Union[str, BaseEstimator]):
        pass

    @abstractmethod
    def add_context(self,
                    context_name: str,
                    model_trained: Union[str, BaseEstimator],
                    ref_data_test: str):
        pass

    @abstractmethod
    def validate_all_contexts(self):
        """Validates all experiments by checking if all models are of the same type and if the scores_target are valid.
        """
        pass

    @abstractmethod
    def validate_models(self):
        """Checks whether all models are classifiers or regressors.

        Raises:
            ValueError: If there are models of different types in the same model list to apply in the experiment
        """
        pass

    @abstractmethod
    def validate_scores_target(self):
        """Checks whether the performance metric exists and whether it makes sense according to the type of Machine Learning model that will be used

        Raises:
            ValueError: If there are models of different types in the same model list to apply in the experiment
        """
        pass