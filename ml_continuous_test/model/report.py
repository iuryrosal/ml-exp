from pydantic import BaseModel
from typing import Union
import datetime

from model.ab_test_results import ShapiroWilkTestResult, LeveneTestResult, BartlettTestResult, AnovaTestResult, TStudentTestResult, TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult


class ScoreDescribed(BaseModel):
    model_id: int
    mean: float = None
    std: float = None
    median: float = None
    minimum: float = None
    maximum: float = None
    mode: float = None

class ABTestReport(BaseModel):
    """Represents the result of the application of statistical tests to validate the significance of the models."""
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
    """It groups together all relevant information about statistics and comparison of model results with statistical tests."""
    score_target: str
    score_described: list[ScoreDescribed] = []
    ab_tests: ABTestReport = None

class GeneralReport(BaseModel):
    reports_by_score: list[GeneralReportByScore] = []
    better_model_by_score: list[str] = []
    message_about_significancy: list[str] = []
    created_at: datetime.datetime = datetime.datetime.now()