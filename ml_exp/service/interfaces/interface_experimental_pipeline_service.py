from abc import abstractmethod, ABC
from ml_exp.model.report import GeneralReport, GeneralReportByScore


class IExperimentalPipelineService(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def _process_ab_tests_results(self, report_by_score: GeneralReportByScore) -> None:
        """Generates a report based on test results related with specific metric, indicating whether the model needs to be adjusted.

        Args:
            general_report (GeneralReportByScore): Result of Hypho tests applied in the logic of the continuous experimentation treadmill around some specific metric
        """
        pass
    
    @abstractmethod
    def _verify_best_model_with_significant_result(self, report_by_score: GeneralReportByScore) -> tuple[int, str]:
        """Based on models that have significant differences in the tests to compare the median of the results and decide the best model

        Args:
            general_report (GeneralReportByScore): Result of Hypho tests applied in the logic of the continuous experimentation treadmill around some specific metric

        Returns:
            tuple[int, str]: index of the best model and string with details of the values that led to the decision of the best model around a given metric
        """
        pass

    @abstractmethod
    def _process_mannwhitney_results(self, report_by_score: GeneralReportByScore) -> tuple[int, str]:
        """Based on models that have significant differences (by Mann Whitney result) to compare the median of the results and decide the best model

        Args:
            general_report (GeneralReportByScore): Result of Hypho tests applied in the logic of the continuous experimentation treadmill around some specific metric

        Returns:
            tuple[int, str]: index of the best model and string with details of the values that led to the decision of the best model around a given metric
        """
        pass
    
    @abstractmethod
    def run_pipeline(self):
        """Apply the Hypho testing pipeline service that will perform the orchestration according to the adopted methodology, after which it will process the results of these tests to generate a suggestion about better models around each metric.
        """
        pass
    
    @abstractmethod
    def get_general_report(self) -> GeneralReport:
        """Return general report generated and enriched

        Returns:
            GeneralReport: General report with details of the results of the Hypho tests applied to the model test data around performance metrics.
        """
        pass
    
    def export_json_results(self, report_path: str = "reports") -> None:
        """Export details of results collected from Hypho testing for each performance metric

        Args:
            report_path (str, optional): Location where JSON will be generated. Defaults to "reports".
        """
        pass