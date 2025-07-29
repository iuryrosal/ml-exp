from pathlib import Path

from better_experimentation.service.ab_pipeline_service import ABPipelineService
from better_experimentation.model.report import GeneralReport, GeneralReportByScore
from better_experimentation.utils.log_config import LogService, handle_exceptions


class ExperimentalPipelineService:
    """Based on the results of the AB pipeline tests, it processes and generates final outputs to define the best model around each metric.
    """
    __log_service = LogService()
    def __init__(self,
                 scores_data: list) -> None:
        self.general_report = GeneralReport()
        self.scores_data = scores_data
        self.__logger = self.__log_service.get_logger(__name__)

    @handle_exceptions(__log_service.get_logger(__name__))
    def __process_ab_tests_results(self, report_by_score: GeneralReportByScore) -> None:
        """Generates a report based on test results related with specific metric, indicating whether the model needs to be adjusted.

        Args:
            general_report (GeneralReportByScore): Result of AB tests applied in the logic of the continuous experimentation treadmill around some specific metric
        """
        significant_differences = False

        # Check ANOVA or Kruskal-Wallis for decision
        if "perform_anova" in report_by_score.ab_tests.pipeline_track and report_by_score.ab_tests.anova.is_significant:
            significant_differences = True
            message = f"Diferença significativa detectada entre modelos (ANOVA) em torno de {report_by_score.score_target}."
            self.general_report.message_about_significancy.append(message)
            best_model, msg_best_model = self.__verify_best_model_with_significant_result(report_by_score)
            self.general_report.better_model_by_score.append(msg_best_model)
            self.general_report.best_model_index = best_model

        elif 'perform_kurskalwallis' in report_by_score.ab_tests.pipeline_track and report_by_score.ab_tests.kurskalwallis.is_significant:
            significant_differences = True
            message = f"Diferença significativa detectada entre modelos (Kruskal-Wallis) em torno de {report_by_score.score_target}."
            self.general_report.message_about_significancy.append(message)
            best_model, msg_best_model = self.__process_mannwhitney_results(report_by_score)
            self.general_report.better_model_by_score.append(msg_best_model)
            self.general_report.best_model_index = best_model
        else:
            message = f"Nenhuma diferença significativa detectada entre modelos em torno de {report_by_score.score_target}."
            self.general_report.message_about_significancy.append(message)
            self.general_report.better_model_by_score.append(f"Não existe modelo melhor em torno de {report_by_score.score_target} devido a falta de significância.")
            self.general_report.best_model_index = None
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def __verify_best_model_with_significant_result(self, report_by_score: GeneralReportByScore) -> tuple[int, str]:
        """Based on models that have significant differences in the tests to compare the median of the results and decide the best model

        Args:
            general_report (GeneralReportByScore): Result of AB tests applied in the logic of the continuous experimentation treadmill around some specific metric

        Returns:
            tuple[int, str]: index of the best model and string with details of the values that led to the decision of the best model around a given metric
        """
        max_result = 0
        model_with_max_result = None

        for model_result in report_by_score.score_described:
            median_model = model_result.median
            if median_model > max_result:
                max_result = median_model
                model_with_max_result = model_result.model_name
            else:
                continue

        if model_with_max_result is None:
            return model_with_max_result, f"Não existe modelo melhor em torno de {report_by_score.score_target}"
        else:
            return model_with_max_result, f"Melhor modelo baseado na mediana: {model_with_max_result} com mediana {max_result} em torno de {report_by_score.score_target}"

    @handle_exceptions(__log_service.get_logger(__name__))
    def __process_mannwhitney_results(self, report_by_score: GeneralReportByScore) -> tuple[int, str]:
        """Based on models that have significant differences (by Mann Whitney result) to compare the median of the results and decide the best model

        Args:
            general_report (GeneralReportByScore): Result of AB tests applied in the logic of the continuous experimentation treadmill around some specific metric

        Returns:
            tuple[int, str]: index of the best model and string with details of the values that led to the decision of the best model around a given metric
        """
        def get_index_from_score_describe_by_model_name(model_name: str, scores_described: list) -> int:
            return next((i for i, score in enumerate(scores_described) if score.model_name == model_name), None)

        if report_by_score.ab_tests.mannwhitney:
            max_result = 0
            model_with_max_result = None

            max_median_between_models = 0
            model_with_max_median = None
            for result in report_by_score.ab_tests.mannwhitney:
                if result.is_significant:
                    index_model_1 = get_index_from_score_describe_by_model_name(result.model_name_1, report_by_score.score_described)
                    index_model_2 = get_index_from_score_describe_by_model_name(result.model_name_2, report_by_score.score_described)
                    median_model_1 = report_by_score.score_described[index_model_1].median
                    median_model_2 = report_by_score.score_described[index_model_2].median
                    if median_model_1 > median_model_2:
                        max_median_between_models = median_model_1
                        model_with_max_median = result.model_name_1
                    else:
                        max_median_between_models = median_model_2
                        model_with_max_median = result.model_name_2

                    if max_median_between_models > max_result:
                        max_result = max_median_between_models
                        model_with_max_result = model_with_max_median
            return model_with_max_result, f"Melhor modelo baseado na mediana: {model_with_max_result} com mediana {max_result} em torno de {report_by_score.score_target}"
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def run_pipeline(self):
        """Apply the AB testing pipeline service that will perform the orchestration according to the adopted methodology, after which it will process the results of these tests to generate a suggestion about better models around each metric.
        """
        for score_name, scores in self.scores_data.items():
            exp_cont = ABPipelineService(scores_data=scores, score_target=score_name)
            exp_cont.run_pipeline()
            report_by_score = exp_cont.get_report()
            self.general_report.reports_by_score.append(report_by_score)

        for report in self.general_report.reports_by_score:
            self.__process_ab_tests_results(report)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def get_general_report(self) -> GeneralReport:
        """Return general report generated and enriched

        Returns:
            GeneralReport: General report with details of the results of the AB tests applied to the model test data around performance metrics.
        """
        return self.general_report
    
    def export_json_results(self, report_path: str = "reports") -> None:
        """Export details of results collected from AB testing for each performance metric

        Args:
            report_path (str, optional): Location where JSON will be generated. Defaults to "reports".
        """
        for general_report_by_score in self.general_report.reports_by_score:
            base_path = Path("./") 
            report_folder = base_path / report_path
            report_folder.mkdir(parents=True, exist_ok=True)

            report_name = f"{general_report_by_score.score_target}.json"
            filepath = report_folder / report_name
            with filepath.open("w", encoding ="utf-8") as f:
                f.write(general_report_by_score.json())
