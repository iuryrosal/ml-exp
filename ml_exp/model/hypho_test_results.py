from pydantic import BaseModel
from typing import Union


class HypoTestResult(BaseModel):
    """Generic representation of a result generated from an Hypho test
    """
    test_name: str
    context: Union[int, str, None]
    stat: Union[float, None]
    p_value: float

class ShapiroWilkTestResult(HypoTestResult):
    """Result generated from an Shapiro-Wilk test
    """
    test_name: str = "shapirowilk"
    is_normal: bool

class LeveneTestResult(HypoTestResult):
    """Result generated from an Levene test
    """
    test_name: str = "levene"
    is_homoscedastic: bool

class BartlettTestResult(HypoTestResult):
    """Result generated from an Bartlett test
    """
    test_name: str = "barlett"
    is_homoscedastic: bool

class AnovaTestResult(HypoTestResult):
    """Result generated from an Anova test
    """
    test_name: str = "anova"
    is_significant: bool

class TurkeyTestResult(HypoTestResult):
    """Result generated from an Tukey test
    """
    test_name: str = "turkey"
    p_value: list[float]
    reject: list[bool]
    meandiffs: list[float]
    std_pairs: list[float]
    q_crit: float

class KruskalWallisTestResult(HypoTestResult):
    """Result generated from an Kruskal-Wallis test
    """
    test_name: str = "kruskalwallis"
    is_significant: bool

class MannWhitneyTestResult(HypoTestResult):
    """Result generated from an Mann-Whitney test
    """
    test_name: str = "mannwhitney"
    context_name_1: str
    context_name_2: str
    is_significant: bool
    corrected_p_value: float | None = None # if apply benjamini_hochberg

class TStudentTestResult(HypoTestResult):
    """Result generated from an Shapiro-Wilk test
    """
    test_name: str = "tstudent"
    context_name_1: str
    context_name_2: str
    is_significant: bool

class WelchTestResult(HypoTestResult):
    test_name: str = "welch"
    context_name_1: str
    context_name_2: str
    is_significant: bool