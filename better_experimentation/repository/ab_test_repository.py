import numpy as np
from scipy.stats import shapiro, anderson, kstest, levene, bartlett, ttest_ind, f_oneway, mannwhitneyu, wilcoxon, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from better_experimentation.model.ab_test_results import ShapiroWilkTestResult, LeveneTestResult, TStudentTestResult, AnovaTestResult, TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult


class ABTestRepository:
    """Repository responsible to define the logic to apply each AB tests independently and linked to the models
    """
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def apply_shapiro(self, context: str, values: list) -> ShapiroWilkTestResult:
        """Apply the Shapiro-Wilk test to check the normality in the distribution

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            ShapiroWilkTestResult: Test result
        """
        stat, p_value = shapiro(values)
        is_normal = p_value >= self.alpha
        ab_test_result = ShapiroWilkTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_normal=is_normal
        )
        return ab_test_result

    def apply_levene(self, context: str, values:list) -> LeveneTestResult:
        """Apply the Levene test to check if the distribution data are homoscedastic

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            LeveneTestResult: Test result
        """
        stat, p_value = levene(*values)
        is_homoscedastic = p_value >= self.alpha
        ab_test_result = LeveneTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_homoscedastic=is_homoscedastic
        )
        return ab_test_result

    def apply_anova(self, context:str, values: list) -> AnovaTestResult:
        """Apply ANOVA test to validate whether there are significant differences between the metric results between the models

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            AnovaTestResult: Test result
        """
        stat, p_value = f_oneway(*values)
        is_significant = p_value < self.alpha
        ab_test_result = AnovaTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result

    def apply_turkey(self, context: str, values: list, labels: list) -> TurkeyTestResult:
        """Apply Turkey Test to validate whether there are significant differences between the metric results between the models

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected
            labels (list): Labels indicating the model index related to the data

        Returns:
            TurkeyTestResult: Test result
        """
        turkey_result = pairwise_tukeyhsd(values, labels, alpha=self.alpha)
        ab_test_result = TurkeyTestResult(
            context=context,
            stat=None,
            p_value=turkey_result.pvalues,
            reject=turkey_result.reject,
            meandiffs=turkey_result.meandiffs,
            std_pairs=turkey_result.std_pairs,
            q_crit=turkey_result.q_crit
        )
        return ab_test_result
    
    def apply_kruskal(self, context: str, values: list) -> KruskalWallisTestResult:
        """Apply KruskalWallis test to validate whether there are significant differences between the metric results between the models

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            KruskalWallisTestResult: Test result
        """
        try: 
            stat, p_value = kruskal(*values)
            is_significant = p_value < self.alpha
            ab_test_result = KruskalWallisTestResult(
                context=context,
                stat=stat,
                p_value=p_value,
                is_significant=is_significant
            )
        except ValueError as e:
            if "All numbers are identical" in str(e):
                ab_test_result = KruskalWallisTestResult(
                    context=context,
                    stat=float('nan'),
                    p_value=float('nan') ,
                    is_significant=False
                )
        return ab_test_result

    def apply_mannwhitney(self, context: str, model_name_1: str, model_name_2: str, values: list) -> MannWhitneyTestResult:
        """Apply the Mann-Whitney test to validate whether there are significant differences between the metric results between pair of models

        Args:
            context (str): General description of the comparative context
            model_name_1 (str): Index 1 of one of the models of the pair being used in the comparison
            model_name_2 (str): Index 2 of one of the models of the pair being used in the comparison
            values (list): Model metric values ​​to be used in testing

        Returns:
            MannWhitneyTestResult: Test result
        """
        stat, p_value = mannwhitneyu(values[f"{model_name_1}"], values[f"{model_name_2}"])
        is_significant = p_value < self.alpha
        ab_test_result = MannWhitneyTestResult(
            context=context,
            model_name_1=model_name_1,
            model_name_2=model_name_2,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result

    def apply_t_student(self, context: str, model_name_1: str, model_name_2: str, values: list) -> TStudentTestResult:
        """Apply the T-Student test to validate whether there are significant differences between the metric results between pair of models

        Args:
            context (str): General description of the comparative context
            model_name_1 (str): Name of one of the models of the pair being used in the comparison
            model_name_2 (str): Index 2 of one of the models of the pair being used in the comparison
            values (list): Model metric values ​​to be used in testing

        Returns:
            TStudentTestResult: Test result
        """
        stat, p_value = ttest_ind(values[f"{model_name_1}"], values[f"{model_name_2}"])
        is_significant = p_value < self.alpha
        ab_test_result = TStudentTestResult(
            context=context,
            model_name_1=model_name_1,
            model_name_2=model_name_2,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result