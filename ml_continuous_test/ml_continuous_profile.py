from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import pandas as pd

from ml_continuous_test.service.experimental_pipeline_service import ExperimentalPipelineService
from ml_continuous_test.service.prepare_data_service import PrepareDataService


class MLContinuousProfile:
    """Generate a profile report related a tests from Models.

    Used as is, it will output its content as an HTML report in a Jupyter notebook.
    """

    def __init__(self, models_trained: list[str, BaseEstimator],
                 X_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 scores_target: list[str],
                 n_splits: int = 100) -> None:
        self.scores = PrepareDataService(
            models_trained=models_trained,
            X_test=X_test,
            y_test=y_test,
            scores_target=scores_target,
            n_splits=n_splits).get_scores_data()
        # Persist result
        exp_pipe = ExperimentalPipelineService(scores_data=self.scores)
        exp_pipe.run_pipeline()

    def _validate_models_trained_sklearn(self, models_trained: list[BaseEstimator]):
        """Validate the trained models."""
        if (not all(isinstance(model, ClassifierMixin) for model in models_trained)
            and not all(isinstance(model, RegressorMixin) for model in models_trained)):
            raise ValueError("models_trained must need all models to be classifiers or regressors and not a mixture of them, so a comparison is not possible.")
    
    def _validate_scores_target(self, scores_target: list[str]):
        """Validate the scores target."""
        if not isinstance(scores_target, list):
            raise ValueError("scores_target must be a list of strings.")