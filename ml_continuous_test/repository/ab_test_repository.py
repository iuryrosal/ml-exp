import numpy as np
from itertools import chain
from scipy.stats import shapiro, anderson, kstest, levene, bartlett, ttest_ind, f_oneway, mannwhitneyu, wilcoxon, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from model.ab_test_results import ShapiroWilkTestResult, LeveneTestResult, TStudentTestResult, AnovaTestResult, TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult


class ABTestRepository:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def apply_shapiro(self, context, values):
        stat, p_value = shapiro(values)
        is_normal = p_value >= self.alpha
        ab_test_result = ShapiroWilkTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_normal=is_normal
        )
        return ab_test_result

    def apply_levene(self, context, values):
        stat, p_value = levene(*values)
        is_homoscedastic = p_value >= self.alpha
        ab_test_result = LeveneTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_homoscedastic=is_homoscedastic
        )
        return ab_test_result
    
    def apply_bartlett(self, context, values):
        stat, p_value = bartlett(*values)
        is_homoscedastic = p_value >= self.alpha
        ab_test_result = LeveneTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_homoscedastic=is_homoscedastic
        )
        return ab_test_result

    def apply_anova(self, context, values):
        stat, p_value = f_oneway(*values)
        is_significant = p_value < self.alpha
        ab_test_result = AnovaTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result

    def apply_turkey(self, context, values, labels):
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
    
    def apply_kruskal(self, context, values):
        stat, p_value = kruskal(*values)
        is_significant = p_value < self.alpha
        ab_test_result = KruskalWallisTestResult(
            context=context,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result

    def apply_mannwhitney(self, context, context_1, context_2, values):
        stat, p_value = mannwhitneyu(values[f"{context_1}"], values[f"{context_2}"])
        is_significant = p_value < self.alpha
        ab_test_result = MannWhitneyTestResult(
            context=context,
            context_1=context_1,
            context_2=context_2,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result
    
    def apply_t_student(self, context, context_1, context_2, values):
        stat, p_value = ttest_ind(values[f"{context_1}"], values[f"{context_2}"])
        is_significant = p_value < self.alpha
        ab_test_result = TStudentTestResult(
            context=context,
            context_1=context_1,
            context_2=context_2,
            stat=stat,
            p_value=p_value,
            is_significant=is_significant
        )
        return ab_test_result