import numpy as np
import statistics

from better_experimentation.repository.ab_test_repository import ABTestRepository
from better_experimentation.model.report import ABTestReport, GeneralReportByScore, ScoreDescribed
from better_experimentation.utils.log_config import LogService, handle_exceptions


class ABPipelineService:
    """Orchestrates the methodology adopted to articulate AB tests based on test results collected from models around a metric. Uses the AB test repository to apply the tests.
    """
    __log_service = LogService()
    def __init__(self, scores_data, score_target, alpha=0.05):
        """
        Initializes the pipeline with the data and significance level.

        Parameters:
        scores_data (dict): A dictionary containing the data for each campaign.
        alpha (float): The significance level for the statistical tests.
        """
        self.scores_data = scores_data
        self.ab_test_repo = ABTestRepository(alpha=alpha)
        self.ab_test_report_obj = ABTestReport(score_target=score_target)
        self.report_by_score = GeneralReportByScore(score_target=score_target)
        self.__logger = self.__log_service.get_logger(__name__)

    @handle_exceptions(__log_service.get_logger(__name__))
    def __collect_statistical_results(self):
        for model_name, scores in self.scores_data.items():
            score_model = ScoreDescribed(
                model_name=model_name,
                mean=statistics.mean(scores),
                std=statistics.stdev(scores),
                median=statistics.median(scores),
                minimum=min(scores),
                maximum=max(scores),
                mode=statistics.mode(scores)
            )
            self.report_by_score.score_described.append(score_model)

    @handle_exceptions(__log_service.get_logger(__name__))
    def __check_normality(self):
        """Checks the normality of data for each campaign using Shapiro-Wilk."""
        shapiro_results = []
        for campaign, values in self.scores_data.items():
            result = self.ab_test_repo.apply_shapiro(context=campaign, values=values)
            shapiro_results.append(result)
        self.ab_test_report_obj.shapirowilk = shapiro_results

    @handle_exceptions(__log_service.get_logger(__name__))
    def __group_all_values(self):
        all_values = []
        for campaign, values in self.scores_data.items():
            all_values.append(values)
        return all_values

    @handle_exceptions(__log_service.get_logger(__name__))
    def __check_homocedasticity(self):
        """Checks homoscedasticity between groups using Levene and Bartlett tests."""
        values = self.__group_all_values()
        levene_result = self.ab_test_repo.apply_levene(context="all_models", values=values)

        self.ab_test_report_obj.levene = levene_result

    @handle_exceptions(__log_service.get_logger(__name__))
    def __perform_t_student(self):
        """Performs Student's t-test between models."""
        values = self.__group_all_values()
        models = list(self.scores_data.keys())
        context = f"T Student between {models[0]} and {models[1]}"
        result = self.ab_test_repo.apply_t_student(context=context,
                                                    model_name_1=models[0],
                                                    model_name_2=models[1],
                                                    values=self.scores_data)
        self.ab_test_report_obj.tstudent = result

    @handle_exceptions(__log_service.get_logger(__name__))
    def __perform_anova(self):
        """Performs ANOVA if data are normal and homoscedastic."""
        values = self.__group_all_values()
        self.ab_test_report_obj.anova = self.ab_test_repo.apply_anova(context="all_models", values=values)

    @handle_exceptions(__log_service.get_logger(__name__))
    def __perform_turkey(self):
        values = self.__group_all_values()
        combined_data = np.concatenate(values)
        labels = np.concatenate([[campaign] * len(vals) for campaign, vals in self.scores_data.items()])
        turkey_result = self.ab_test_repo.apply_turkey(context="all_models", values=combined_data, labels=labels)
        self.ab_test_report_obj.turkey = turkey_result

    @handle_exceptions(__log_service.get_logger(__name__))
    def _perform_parametric_tests(self):
        """Performs ANOVA if data are normal and homoscedastic."""
        self.__perform_anova()
        self.ab_test_report_obj.pipeline_track.append("perform_anova")
        
        # If ANOVA is significant, perform Tukey's test for post-hoc comparisons
        if self.ab_test_report_obj.anova.is_significant:
            self.ab_test_report_obj.pipeline_track.append("anova_is_significant")
            self.__perform_turkey()
            self.ab_test_report_obj.pipeline_track.append("perform_turkey")

    @handle_exceptions(__log_service.get_logger(__name__))
    def __perform_kruskal(self):
        values = self.__group_all_values()
        kruskal_result = self.ab_test_repo.apply_kruskal(context="all_models", values=values)
        self.ab_test_report_obj.kurskalwallis = kruskal_result

    @handle_exceptions(__log_service.get_logger(__name__))
    def __perform_mann_whitney(self):
        mannwhitney_results = []
        models = list(self.scores_data.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                context = f"Mann-Whitney between {model1=} and {model2=}"
                result = self.ab_test_repo.apply_mannwhitney(context=context,
                                                            model_name_1=model1,
                                                            model_name_2=model2,
                                                            values=self.scores_data)
                mannwhitney_results.append(result)
        self.ab_test_report_obj.mannwhitney = mannwhitney_results

    @handle_exceptions(__log_service.get_logger(__name__))
    def _perform_non_parametric_tests(self):
        """Performs nonparametric tests for nonnormal or nonhomoscedastic data."""
        self.__perform_kruskal()
        self.ab_test_report_obj.pipeline_track.append("perform_kurskalwallis")

        # Performs post-hoc comparisons with Mann-Whitney if Kruskal-Wallis is significant
        if self.ab_test_report_obj.kurskalwallis.is_significant:
            self.ab_test_report_obj.pipeline_track.append("kurskalwallis_is_significant")
            self.__perform_mann_whitney()
            self.ab_test_report_obj.pipeline_track.append("perform_mannwhitney")

    @handle_exceptions(__log_service.get_logger(__name__))
    def run_pipeline(self):
        """Executes the entire AB testing flow according to the adopted methodology.
        """
        self.__collect_statistical_results()
        self.__check_normality()
        self.ab_test_report_obj.pipeline_track.append("check_normality_with_shapiro")

        normal_result_list = [shapiro_result.is_normal for shapiro_result in self.ab_test_report_obj.shapirowilk]
        if len(list(self.scores_data.keys())) > 2: # 3 or more models
            self.ab_test_report_obj.pipeline_track.append("3_or_more_models_is_true")
            self.__check_homocedasticity()
            self.ab_test_report_obj.pipeline_track.append("check_homocedasticity_with_levene")

            # Verifica se ANOVA é aplicável (normalidade e homocedasticidade)
            if all(normal_result_list) and self.ab_test_report_obj.levene.is_homoscedastic:
                self.ab_test_report_obj.pipeline_track.append("data_normal_and_homocedasticity_is_true")
                self._perform_parametric_tests()
            else:
                self.ab_test_report_obj.pipeline_track.append("data_normal_and_homocedasticity_is_false")
                self._perform_non_parametric_tests()
        
        else:
            self.ab_test_report_obj.pipeline_track.append("3_or_more_models_is_false")
            if all(normal_result_list):
                self.ab_test_report_obj.pipeline_track.append("data_normal_is_true")
                self.__perform_t_student()
                self.ab_test_report_obj.pipeline_track.append("perform_t_student")
            else:
                self.ab_test_report_obj.pipeline_track.append("data_normal_is_false")
                self.__perform_mann_whitney()
                self.ab_test_report_obj.pipeline_track.append("perform_mannwhitney")

        self.ab_test_report_obj.pipeline_track.append("done")
        self.report_by_score.ab_tests = self.ab_test_report_obj    

    @handle_exceptions(__log_service.get_logger(__name__))
    def get_report(self) -> GeneralReportByScore:
        return self.report_by_score