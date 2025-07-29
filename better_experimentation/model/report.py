from pydantic import BaseModel
from typing import Union
import datetime

from better_experimentation.model.ab_test_results import ShapiroWilkTestResult, LeveneTestResult, BartlettTestResult, AnovaTestResult, TStudentTestResult, TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult

class ScoreDescribed(BaseModel):
    """Generates a statistical summary of the data distribution of a performance metric around a ML model
    """
    model_name: str
    mean: float = None
    std: float = None
    median: float = None
    minimum: float = None
    maximum: float = None
    mode: float = None

class ABTestReport(BaseModel):
    """Represents the result of the application of statistical tests to validate the significance of the models, involving performance data after testing
    """
    pipeline_track: list[str] = []
    shapirowilk: list[ShapiroWilkTestResult] = []
    levene: list[LeveneTestResult] = []
    bartlett: list[BartlettTestResult] = []
    anova: AnovaTestResult = None
    turkey: list[TurkeyTestResult] = []
    kurskalwallis: KruskalWallisTestResult = None
    mannwhitney: list[MannWhitneyTestResult] = []
    tstudent: TStudentTestResult = None

class GeneralReportByScore(BaseModel):
    """Groups all relevant information about statistics and comparison of model results with statistical tests around a given performance metric
    """
    score_target: str
    score_described: list[ScoreDescribed] = []
    ab_tests: ABTestReport = None

class GeneralReport(BaseModel):
    """The General Report that aggregates all the details of the statistical test results for the specified performance metrics
    """
    reports_by_score: list[GeneralReportByScore] = []
    better_model_by_score: list[str] = []
    best_model_index: Union[int, None] = None
    message_about_significancy: list[str] = []
    created_at: datetime.datetime = datetime.datetime.now()