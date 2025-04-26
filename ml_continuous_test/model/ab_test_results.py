from pydantic import BaseModel
from typing import Union


class ABTestResult(BaseModel):
    """Generic representation of a result generated from an AB test
    """
    test_name: str
    context: Union[int, str, None]
    stat: Union[float, None]
    p_value: float

class ShapiroWilkTestResult(ABTestResult):
    """Result generated from an Shapiro-Wilk test
    """
    test_name: str = "shapirowilk"
    is_normal: bool

class LeveneTestResult(ABTestResult):
    """Result generated from an Levene test
    """
    test_name: str = "levene"
    is_homoscedastic: bool

class BartlettTestResult(ABTestResult):
    """Result generated from an Bartlett test
    """
    test_name: str = "barlett"
    is_homoscedastic: bool

class AnovaTestResult(ABTestResult):
    """Result generated from an Anova test
    """
    test_name: str = "anova"

class TurkeyTestResult(ABTestResult):
    """Result generated from an Tukey test
    """
    test_name: str = "turkey"
    reject: list[bool]
    meandiffs: float
    confint: float
    std_pairs: float
    q_crit: float
    halfwidths: float

class KruskalWallisTestResult(ABTestResult):
    """Result generated from an Kruskal-Wallis test
    """
    test_name: str = "kruskalwallis"
    is_significant: bool

class MannWhitneyTestResult(ABTestResult):
    """Result generated from an Mann-Whitney test
    """
    test_name: str = "mannwhitney"
    context_1: str
    context_2: str