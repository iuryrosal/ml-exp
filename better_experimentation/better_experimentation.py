from sklearn.base import BaseEstimator
import pandas as pd
from datetime import datetime
from typing import Union

from better_experimentation.service.load_model_by_path import LoadModelByPath
from better_experimentation.service.load_model_by_obj import LoadModelByObject
from better_experimentation.model.ml_model import MLModel, ModelTechnology, ModelType
from better_experimentation.service.experimental_pipeline_service import ExperimentalPipelineService
from better_experimentation.service.report_generator_service import ReportGeneratorService
from better_experimentation.service.prepare_data_service import PrepareDataService
from better_experimentation.utils.dataframe import read_pandas


class BetterExperimentation:
    scores_classifier = ["accuracy", "f1", "precision", "recall"]
    scores_regression = ["mae", "mse", "r2"]

    def __init__(self,
                 models_trained: list[str, BaseEstimator],
                 X_test: Union[pd.DataFrame, str],
                 y_test: Union[pd.DataFrame, str],
                 scores_target: Union[list[str], str],
                 n_splits: int = 100,
                 report_path: str = None,
                 report_name: str = None,
                 export_json_data: bool = True,
                 export_html_report: bool = True,
                 return_best_model: bool = False,
                 **kwargs) -> None:
        """It will apply the logic of continuous experimentation to a set of models, using test data, around performance metrics.

        Args:
            models_trained (list[str, BaseEstimator]): List of trained and loaded models containing information about each model
            X_test (Union[pd.DataFrame, str]): Test data involving only the features. It can be the Pandas Dataframe or the Path that contains the data file (supports pandas formats).
            y_test (Union[pd.DataFrame, str]): Test data involving only the target. It can be the Pandas Dataframe or the Path that contains the data file (supports pandas formats).
            scores_target (Union[list[str], str]): Performance metrics that will be used as a basis for generating comparison
            n_splits (int, optional): Number of performance metric data groups to be generated. This value will imply the number of values ​​for each model and for each performance metric. For more consistent results, it is recommended that the number of groups be equivalent to at least 10% of the total test data. Defaults to 100.
            report_path (str, optional): Folder where all reports to be generated will be stored. A None value will generate in the default /reports folder. Defaults to None.
            report_name (str, optional): Name of the folder that will be generated within the report_path containing all reports related to the given report, separated by timestamp. A value of None will use the default name of general_report. Defaults to None.
            export_json_data (bool, optional): It will save in report_path/report_name inside the timestamp folder the JSON containing all the performance metric values ​​collected before the application of the statistical tests. For each performance metric we will have a json. Defaults to True.
            export_html_report (bool, optional): It will generate the HTML report (n report_path/report_name) containing a summary of the results of the statistical tests for all selected performance metrics, as well as the best model around each metric (if any). Defaults to True.
            return_best_model (bool, optional): When the function that activates the pipeline is executed, the best model around the performance metric will be returned to the API. This only works if you define only one performance metric. Defaults to False.
        """

        self.__export_json_data = export_json_data
        self.__export_html_report = export_html_report
        self.__return_best_model = return_best_model
        self.__n_splits = n_splits

        # check data type of scores_target
        if isinstance(models_trained, list) and all(isinstance(model, str) for model in models_trained):
            self.models = LoadModelByPath(models_trained).load_all_models()
        elif isinstance(models_trained, str):
            self.models = LoadModelByPath([models_trained]).load_all_models()
        else:
            self.models = LoadModelByObject(models_trained).load_all_models()
        # check data type of scores_target
        if isinstance(scores_target, str):
            self.scores_target = [scores_target]
        elif isinstance(scores_target, list) and all([isinstance(score, str) for score in scores_target]):
            self.scores_target = scores_target
        else:
            raise ValueError(f"scores_target need to be string or list of strings. Current type of scores_target: {type(scores_target)}")

        # check values from models and score target
        self._validate_models(models=self.models)
        self._validate_scores_target(scores_target=self.scores_target, models=self.models)

        # check best_model flag with number os scores_target
        if self.__return_best_model and len(self.scores_target) > 1:
            raise ValueError("To find the best model of all, you only need to define one score_target to be evaluated and be the central parameter to define the best model. If you want to generate a report comparing the models around different metrics (score_target), disable the return_best_model parameter.")

        # check report_path
        if not report_path:
            report_base_path = "reports"
        else:
            report_base_path = report_path

        # check report_name
        if not report_name:
            self.report_base_name = "general_report"
        else:
            self.report_base_name = report_name

        self.report_base_path = report_base_path + "/" + self.report_base_name + "/" + datetime.now().strftime("%Y%m%d%H%M%S")

        # check data type of X_test
        if isinstance(X_test, pd.DataFrame):
            self.X_test = X_test
        elif isinstance(X_test, str):
            self.X_test = read_pandas(X_test)
        else:
            raise ValueError(f"X_test need to be Pandas Dataframe or string path to file. Current type of X_test: {type(X_test)}")

        # check data type of y_test
        if isinstance(y_test, pd.DataFrame):
            self.y_test = y_test
        elif isinstance(X_test, str):
            self.y_test = read_pandas(y_test)
        else:
            raise ValueError(f"y_test need to be Pandas Dataframe or string path to file. Current type of y_test: {type(y_test)}")

    def _validate_models(self, models: list[MLModel]):
        """Checks whether all models are classifiers or regressors.

        Args:
            models (list[MLModel]): List of trained and loaded models containing information about each model

        Raises:
            ValueError: If there are models of different types in the same model list to apply in the experiment
        """
        if (not all(model.model_type == ModelType.classifier.value for model in models)
            and not all(model.model_type == ModelType.regressor.value for model in models)):
            raise ValueError("models must need all models to be classifiers or regressors and not a mixture of them, so a comparison is not possible.")
    
    def _validate_scores_target(self, scores_target: list[str], models: list[MLModel]):
        """Checks whether the performance metric exists and whether it makes sense according to the type of Machine Learning model that will be used

        Args:
            models (list[MLModel]): List of trained and loaded models containing information about each model
            scores_target (ist[str]): Performance metric selected to be used as a basis for comparing models.

        Raises:
            ValueError: If there are models of different types in the same model list to apply in the experiment
        """
        if all(model.model_type == ModelType.classifier.value for model in models):
            if all([score not in self.scores_classifier for score in scores_target]):
                raise ValueError(f"scores_target must be valid between them {self.scores_classifier}")
        if all(model.model_type == ModelType.regressor.value for model in models):
            if all([score not in self.scores_regression for score in scores_target]):
                raise ValueError(f"scores_target must be valid between them {self.scores_regression}")
    
    def run(self):
        """Runs the continuous experimentation pipeline and Generates Reports
        """
        self.scores = PrepareDataService(
            models=self.models,
            X_test=self.X_test,
            y_test=self.y_test,
            scores_target=self.scores_target,
            n_splits=self.__n_splits).get_scores_data()
        
        self.exp_pipe = ExperimentalPipelineService(scores_data=self.scores)
        
        self.exp_pipe.run_pipeline()

        if self.__export_json_data:
            self.exp_pipe.export_json_results(report_path=self.report_base_path)

        general_report_generated = self.exp_pipe.get_general_report()
        
        if self.__export_html_report:
            ReportGeneratorService(
                reports=general_report_generated,
                report_base_path=self.report_base_path,
                report_name=self.report_base_name
            )

        if self.__return_best_model:
            best_model_index = general_report_generated.best_model_index
            if best_model_index:
                return self.models[best_model_index].model_name
            else:
                return "None"
        return None
