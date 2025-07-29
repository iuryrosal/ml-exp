import pandas as pd
from datetime import datetime
from typing import Union
from sklearn.base import BaseEstimator
from pathlib import Path
import numpy as np

from better_experimentation.repository.pandas_data_file_repository import PandasDataFileRepository

from better_experimentation.service.prepare_experiment_service import PrepareExperimentService
from better_experimentation.service.experimental_pipeline_service import ExperimentalPipelineService
from better_experimentation.service.report_generator_service import ReportGeneratorService
from better_experimentation.service.generate_score_service import GenerateScoreService
from better_experimentation.service.load_test_data_service import LoadTestDataService

class BetterExperimentation:

    def __init__(self,
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

        # Repositories
        self.pandas_data_file_repository = PandasDataFileRepository()
        
        # Services
        self.load_test_data_service_using_pandas = LoadTestDataService(self.pandas_data_file_repository)
        self.prepare_experiment_service = PrepareExperimentService(scores_target=scores_target)

        self.scores_target = None
        self.report_base_path = None
        self.test_data = {}
        self.scores = None
        self.report_base_name = None

        # check data type of scores_target
        if isinstance(scores_target, str):
            self.scores_target = [scores_target]
        elif isinstance(scores_target, list) and all([isinstance(score, str) for score in scores_target]):
            self.scores_target = scores_target
        else:
            raise ValueError(f"scores_target need to be string or list of strings. Current type of scores_target: {type(scores_target)}")

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

    def add_test_data(self,
                      test_data_name: str,
                      X_test: Union[pd.DataFrame, str],
                      y_test: Union[pd.DataFrame, str]):
        """Add test data by name

        Args:
            test_data_name (str): Name of the test data to be added
        """
        self.load_test_data_service_using_pandas.add_test_data(
            test_data_name=test_data_name,
            X_test=X_test,
            y_test=y_test
        )
    
    def add_model(self,
                   model_name: str,
                   model_trained: list[str, BaseEstimator],
                   ref_test_data: str):
        """Add models to be evaluated

        Args:
            models_trained (list[str, BaseEstimator]): List of trained and loaded models containing information about each model
        """
        self.prepare_experiment_service.add_experiment(
            model_name=model_name,
            model_trained=model_trained,
            ref_data_test=ref_test_data
        )

    def run(self):
        """Runs the continuous experimentation pipeline and Generates Reports
        """
        self.scores = GenerateScoreService(
            experiments=self.prepare_experiment_service.get_experiments(),
            test_data=self.load_test_data_service_using_pandas.get_all_test_data(),
            scores_target=self.scores_target,
            n_splits=self.__n_splits).get_scores_data()
        
        exp_pipe = ExperimentalPipelineService(scores_data=self.scores)
        
        exp_pipe.run_pipeline()

        if self.__export_json_data:
            exp_pipe.export_json_results(report_path=self.report_base_path)

        general_report_generated = exp_pipe.get_general_report()
        
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
