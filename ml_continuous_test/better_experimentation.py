from sklearn.base import BaseEstimator
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ml_continuous_test.service.load_model_by_path import LoadModelByPath
from ml_continuous_test.service.load_model_by_obj import LoadModelByObject
from ml_continuous_test.model.ml_model import MLModel, ModelTechnology, ModelType
from ml_continuous_test.service.experimental_pipeline_service import ExperimentalPipelineService
from ml_continuous_test.service.prepare_data_service import PrepareDataService


class BetterExperimentation:
    scores_classifier = ["accuracy", "f1", "precision", "recall"]
    scores_regression = ["mae", "mse", "r2"]

    def __init__(self,
                 models_trained: list[str, BaseEstimator],
                 X_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 scores_target: list[str],
                 n_splits: int = 100) -> None:

        if all(isinstance(model, str) for model in models_trained):
            models = LoadModelByPath(models_trained).load_all_models()
        else:
            models = LoadModelByObject(models_trained).load_all_models()

        self._validate_models(models=models)
        self._validate_scores_target(scores_target=scores_target, models=models)

        self.scores = PrepareDataService(
            models_trained=models_trained,
            X_test=X_test,
            y_test=y_test,
            scores_target=scores_target,
            n_splits=n_splits).get_scores_data()
        self.exp_pipe = ExperimentalPipelineService(scores_data=self.scores)

    def _validate_models(self, models: list[MLModel]):
        if (not all(model.model_type == ModelType.classifier.value for model in models)
            and not all(model.model_type == ModelType.regressor.value for model in models)):
            raise ValueError("models must need all models to be classifiers or regressors and not a mixture of them, so a comparison is not possible.")
    
    def _validate_scores_target(self, scores_target: list[str], models: list[MLModel]):
        if not isinstance(scores_target, list) and not all(isinstance(score, str) for score in scores_target):
            raise ValueError("scores_target must be a list of strings.")
        if all(model.model_type == ModelType.classifier.value for model in models):
            if all([score not in self.scores_classifier for score in scores_target]):
                print(scores_target)
                raise ValueError(f"scores_target must be valid between them {self.scores_classifier}")
            if all(model.model_type == ModelType.regressor.value for model in models):
                if all([score not in self.scores_regression for score in scores_target]):
                    raise ValueError(f"scores_target must be valid between them {self.scores_regression}")
    
    def run(self):
        self.exp_pipe.run_pipeline()