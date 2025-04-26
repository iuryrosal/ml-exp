from pydantic import BaseModel
from typing import Union
import datetime

from model.ab_test_results import ShapiroWilkTestResult, LeveneTestResult, BartlettTestResult, AnovaTestResult, TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult


class ScoreDescribed(BaseModel):
    mean: float = None
    std: float = None
    median: float = None
    minimum: float = None
    maximum: float = None
    mode: float = None

class ScoresRegressionModel(BaseModel):
    """Groups the scores collected from a prediction made around a given regression model"""
    mean_squared_error: list[float] = None
    root_mean_squared_error: list[float] = None
    mean_absolute_error: list[float] = None
    r_2: list[float] = None
    details_statistic: dict[str, ScoreDescribed] = {}

class ScoresClassificationModel(BaseModel):
    """Groups the scores collected from a prediction made around a given classification model"""
    accuracy: list[float] = None
    precision: list[float] = None
    recall: list[float] = None
    f1_score: list[float] = None
    auc_roc: list[float] = None
    details_statistic: dict[str, ScoreDescribed] = {}

class ModelReport(BaseModel):
    """Group information around a given model."""
    id: int
    module: str
    name: str
    model_type: str
    score_collected: Union[ScoresRegressionModel, ScoresClassificationModel]

class ABTestReport(BaseModel):
    """Represents the result of the application of statistical tests to validate the significance of the models."""
    score_target: str
    shapirowilk: list[ShapiroWilkTestResult] = []
    levene: list[LeveneTestResult] = []
    bartlett: list[BartlettTestResult] = []
    anova: list[AnovaTestResult] = []
    turkey: list[TurkeyTestResult] = []
    kurskalwallis: list[KruskalWallisTestResult] = []
    mannwhitney: list[MannWhitneyTestResult] = []

class GeneralReport(BaseModel):
    """It groups together all relevant information about statistics and comparison of model results with statistical tests."""
    timestamp: datetime.time
    models: list[ModelReport]
    ab_tests: ABTestReport