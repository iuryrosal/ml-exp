from pathlib import Path

from better_experimentation.service.ab_pipeline_service import ABPipelineService
from better_experimentation.model.report import GeneralReport
from better_experimentation.utils.log_config import LogService, handle_exceptions


class ExperimentalPipelineService:
    __log_service = LogService()
    def __init__(self,
                 scores_data: list) -> None:
        self.general_report = GeneralReport()
        self.scores_data = scores_data
        self.__logger = self.__log_service.get_logger(__name__)

    @handle_exceptions(__log_service.get_logger(__name__))
    def __process_ab_tests_results(self, general_report):
        """Gera um relatório dos resultados dos testes, indicando se o modelo precisa ser ajustado."""
        # Análise do resultado dos testes e decisões
        significant_differences = False

        # Verificar ANOVA ou Kruskal-Wallis para decisão
        if "perform_anova" in general_report.ab_tests.pipeline_track and general_report.ab_tests.anova.is_significant:
            significant_differences = True
            message = f"Diferença significativa detectada entre modelos (ANOVA) em torno de {general_report.score_target}."
            self.general_report.message_about_significancy.append(message)
            best_model, msg_best_model = self.__verify_best_model_with_significant_result(general_report)
            self.general_report.better_model_by_score.append(msg_best_model)
            self.general_report.best_model_index = best_model

        elif 'perform_kurskalwallis' in general_report.ab_tests.pipeline_track and general_report.ab_tests.kurskalwallis.is_significant:
            significant_differences = True
            message = f"Diferença significativa detectada entre modelos (Kruskal-Wallis) em torno de {general_report.score_target}."
            self.general_report.message_about_significancy.append(message)
            best_model, msg_best_model = self.__process_mannwhitney_results(general_report)
            self.general_report.better_model_by_score.append(msg_best_model)
            self.general_report.best_model_index = best_model
        else:
            message = f"Nenhuma diferença significativa detectada entre modelos em torno de {general_report.score_target}."
            self.general_report.message_about_significancy.append(message)
            self.general_report.better_model_by_score.append(f"Não existe modelo melhor em torno de {general_report.score_target} devido a falta de significância.")
            self.general_report.best_model_index = None
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def __verify_best_model_with_significant_result(self, general_report):
        max_result = 0
        model_with_max_result = None

        for model_result in general_report.score_described:
            median_model = model_result.median
            if median_model > max_result:
                max_result = median_model
                model_with_max_result = model_result.model_name
            else:
                continue

        if model_with_max_result is None:
            return model_with_max_result, f"Não existe modelo melhor em torno de {general_report.score_target}"
        else:
            return model_with_max_result, f"Melhor modelo baseado na mediana: {model_with_max_result} com mediana {max_result} em torno de {general_report.score_target}"

    @handle_exceptions(__log_service.get_logger(__name__))
    def __process_mannwhitney_results(self, general_report):
        """Processa os resultados do teste de Mann-Whitney e gera um relatório."""
        if general_report.ab_tests.mannwhitney:
            max_result = 0
            model_with_max_result = None

            max_median_between_models = 0
            model_with_max_median = None
            for result in general_report.ab_tests.mannwhitney:
                if result.is_significant:
                    median_model_1 = general_report.score_described[int(result.model_index_1)].median
                    median_model_2 = general_report.score_described[int(result.model_index_2)].median
                    if median_model_1 > median_model_2:
                        max_median_between_models = median_model_1
                        model_with_max_median = result.model_index_1
                    else:
                        max_median_between_models = median_model_2
                        model_with_max_median = result.model_index_2

                    if max_median_between_models > max_result:
                        max_result = max_median_between_models
                        model_with_max_result = model_with_max_median
            return model_with_max_result, f"Melhor modelo baseado na mediana: {model_with_max_result} com mediana {max_result} em torno de {general_report.score_target}"
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def run_pipeline(self):
        for score_name, scores in self.scores_data.items():
            exp_cont = ABPipelineService(scores_data=scores, score_target=score_name)
            exp_cont.run_pipeline()
            general_report_by_score = exp_cont.get_report()
            self.general_report.reports_by_score.append(general_report_by_score)

        for report in self.general_report.reports_by_score:
            self.__process_ab_tests_results(report)
    
    @handle_exceptions(__log_service.get_logger(__name__))
    def get_general_report(self):
        return self.general_report
    
    def export_json_results(self, report_path: str = "reports"):
        for general_report_by_score in self.general_report.reports_by_score:
            base_path = Path("./") 
            report_folder = base_path / report_path
            report_folder.mkdir(parents=True, exist_ok=True)

            report_name = f"{general_report_by_score.score_target}.json"
            filepath = report_folder / report_name
            with filepath.open("w", encoding ="utf-8") as f:
                f.write(general_report_by_score.json())
