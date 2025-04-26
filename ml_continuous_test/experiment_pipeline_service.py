import numpy as np
from itertools import chain

from ab_test_repository import ABTestRepository
from model.report import ABTestReport


class ExperimentPipelineService:
    def __init__(self, scores_data, score_target, alpha=0.05):
        """
        Inicializa a pipeline com os dados e o nível de significância.
        
        Parâmetros:
        scores_data (dict): Um dicionário contendo os dados para cada campanha.
        alpha (float): O nível de significância para os testes estatísticos.
        """
        self.scores_data = scores_data[score_target]
        self.ab_test_repo = ABTestRepository(alpha=alpha)
        self.ab_test_report_obj = ABTestReport(score_target=score_target)

    def __check_normality(self):
        """Verifica a normalidade dos dados para cada campanha usando Shapiro-Wilk."""
        shapiro_results = []
        for campaign, values in self.scores_data.items():
            result = self.ab_test_repo.apply_shapiro(context=campaign, values=values)
            shapiro_results.append(result)
        self.ab_test_report_obj.shapirowilk = shapiro_results
    
    def __group_all_values(self):
        all_values = []
        for campaign, values in self.scores_data.items():
            all_values.append(values)
        result = list(chain.from_iterable(all_values))
        return result

    def __check_homocedasticity(self):
        """Verifica homocedasticidade entre os grupos usando os testes de Levene e Bartlett."""
        values = self.__group_all_values()
        levene_result = self.ab_test_repo.apply_levene(context="all_models", values=values)
        bartlett_result = self.ab_test_repo.apply_bartlett(context="all_models", values=values)

        self.ab_test_report_obj.levene = levene_result
        self.ab_test_report_obj.bartlett = bartlett_result

    def __perform_anova(self):
        """Realiza ANOVA se os dados forem normais e homocedásticos."""
        values = self.__group_all_values()
        self.ab_test_report_obj.anova = self.ab_test_repo.apply_anova(context="all_models", values=values)
        
        # Se ANOVA for significativa, realiza o teste de Tukey para comparações post-hoc
        if self.ab_test_report_obj.anova.is_significant:
            combined_data = np.concatenate(values)
            labels = np.concatenate([[campaign] * len(vals[self.score_target]) for campaign, vals in self.scores_data.items()])
            turkey_result = self.ab_test_repo.apply_turkey(context="all_models", values=combined_data, labels=labels)
            self.ab_test_report_obj.turkey = turkey_result

    def __perform_non_parametric_tests(self):
        """Realiza testes não paramétricos para dados não normais ou não homocedásticos."""
        values = self.__group_all_values()
        kruskal_result = self.ab_test_repo.apply_kruskal(context="all_models", values=values)
        self.ab_test_report_obj.kurskalwallis = kruskal_result

        # Realiza comparações post-hoc com Mann-Whitney se Kruskal-Wallis for significativo
        if self.ab_test_report_obj.kurskalwallis.is_significant:
            mannwhitney_results = []
            models = list(self.scores_data.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    result = self.ab_test_repo.apply_mannwhitney(context_1=model1, context_2=model2, values=values)
                    mannwhitney_results.append(result)
            self.ab_test_report_obj.mannwhitney = mannwhitney_results

    def run_pipeline(self):
        """Executa toda a pipeline de experimentação."""
        self.__check_normality()
        self.__check_homocedasticity()

        normal_result_list = [shapiro_result.is_normal for shapiro_result in self.ab_test_report_obj.shapirowilk]

        # Verifica se ANOVA é aplicável (normalidade e homocedasticidade)
        if all(normal_result_list) and self.ab_test_report_obj.levene.is_homoscedastic:
            self.__perform_anova()
        else:
            self.__perform_non_parametric_tests()

        print(self.ab_test_report_obj.__dict__)

        return self.ab_test_report_obj