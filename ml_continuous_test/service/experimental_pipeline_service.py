from ml_continuous_test.service.ab_pipeline_service import ABPipelineService
from ml_continuous_test.service.report_generator_service import ReportGeneratorService
from ml_continuous_test.model.report import GeneralReport


class ExperimentalPipelineService:
    def __init__(self, scores_data: list) -> None:
        self.general_report = GeneralReport()
        for score_name, scores in scores_data.items():
            exp_cont = ABPipelineService(scores_data=scores, score_target=score_name)
            exp_cont.run_pipeline()
            exp_cont.export_report(f"report_{score_name}.json")
            general_report_by_score = exp_cont.get_report()
            self.general_report.reports_by_score.append(general_report_by_score)

    def __process_ab_tests_results(self, general_report):
        """Gera um relatório dos resultados dos testes, indicando se o modelo precisa ser ajustado."""
        # Análise do resultado dos testes e decisões
        significant_differences = False

        # Verificar ANOVA ou Kruskal-Wallis para decisão
        if "perform_anova" in general_report.ab_tests.pipeline_track and general_report.ab_tests.anova.is_significant:
            significant_differences = True
            message = "Diferença significativa detectada entre modelos (ANOVA)."
            print(message)
        elif 'perform_kurskalwallis' in general_report.ab_tests.pipeline_track and general_report.ab_tests.kurskalwallis.is_significant:
            significant_differences = True
            message = "Diferença significativa detectada entre modelos (Kruskal-Wallis)."
            self.__process_mannwhitney_results(general_report)
            print(message)
        else:
            message = "Nenhuma diferença significativa detectada entre modelos."
            print(message)

        # Gerar relatório detalhado
        if significant_differences:
            print("\n--- Ajustar modelos com base nos resultados ---")
        else:
            print("\n--- Não há diferenças estatisticamente significativas entre os modelos ---")
    
    def __process_mannwhitney_results(self, general_report):
        """Processa os resultados do teste de Mann-Whitney e gera um relatório."""
        if general_report.ab_tests.mannwhitney:
            print("\n--- Resultados do teste de Mann-Whitney ---")
            max_result = 0
            model_with_max_result = None

            max_median_between_models = 0
            model_with_max_median = None
            for result in general_report.ab_tests.mannwhitney:
                if result.is_significant:
                    median_model_1 = general_report.score_described[int(result.context_1)].median
                    median_model_2 = general_report.score_described[int(result.context_2)].median
                    if median_model_1 > median_model_2:
                        max_median_between_models = median_model_1
                        model_with_max_median = result.context_1
                    else:
                        max_median_between_models = median_model_2
                        model_with_max_median = result.context_2

                    if max_median_between_models > max_result:
                        max_result = max_median_between_models
                        model_with_max_result = model_with_max_median
            print(f"Melhor modelo baseado na mediana: {model_with_max_result} com mediana {max_result} em torno de {general_report.score_target}")
    
    def run_pipeline(self):
        for report in self.general_report.reports_by_score:
            self.__process_ab_tests_results(report)
        ReportGeneratorService(reports=self.general_report)