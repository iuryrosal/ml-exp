import numpy as np

from tests.config.general_fixtures import ab_test_repository
from better_experimentation.model.ab_test_results import (
    ShapiroWilkTestResult, LeveneTestResult, AnovaTestResult,
    TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult,
    TStudentTestResult
)


def test_apply_shapiro_returns_expected_type(ab_test_repository):
    values = np.random.normal(loc=0, scale=1, size=50)
    result = ab_test_repository.apply_shapiro(context="test", values=values)
    assert isinstance(result, ShapiroWilkTestResult)
    assert result.context == "test"
    assert isinstance(result.stat, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.is_normal, bool)


def test_apply_levene_returns_expected_type(ab_test_repository):
    values = [np.random.normal(0, 1, 30), np.random.normal(0, 1, 30)]
    result = ab_test_repository.apply_levene(context="test", values=values)
    assert isinstance(result, LeveneTestResult)
    assert isinstance(result.is_homoscedastic, bool)


def test_apply_anova_returns_expected_type(ab_test_repository):
    values = [np.random.normal(0, 1, 30), np.random.normal(1, 1, 30)]
    result = ab_test_repository.apply_anova(context="test", values=values)
    assert isinstance(result, AnovaTestResult)
    assert isinstance(result.is_significant, bool)


def test_apply_turkey_returns_expected_type(ab_test_repository):
    values = list(np.random.normal(loc=0, scale=1, size=90))
    labels = ["A"] * 30 + ["B"] * 30 + ["C"] * 30
    result = ab_test_repository.apply_turkey(context="test", values=values, labels=labels)
    assert isinstance(result, TurkeyTestResult)
    assert len(result.p_value) == len(result.reject)


def test_apply_kruskal_returns_expected_type(ab_test_repository):
    values = [np.random.normal(0, 1, 30), np.random.normal(1, 1, 30)]
    result = ab_test_repository.apply_kruskal(context="test", values=values)
    assert isinstance(result, KruskalWallisTestResult)
    assert isinstance(result.is_significant, bool)


def test_apply_mannwhitney_returns_expected_type(ab_test_repository):
    values = {
        "0": np.random.normal(0, 1, 30),
        "1": np.random.normal(1, 1, 30)
    }
    result = ab_test_repository.apply_mannwhitney(context="test", model_index_1=0, model_index_2=1, values=values)
    assert isinstance(result, MannWhitneyTestResult)
    assert result.model_index_1 == 0
    assert result.model_index_2 == 1


def test_apply_t_student_returns_expected_type(ab_test_repository):
    values = {
        "0": np.random.normal(0, 1, 30),
        "1": np.random.normal(1, 1, 30)
    }
    result = ab_test_repository.apply_t_student(context="test", model_index_1=0, model_index_2=1, values=values)
    assert isinstance(result, TStudentTestResult)
    assert isinstance(result.is_significant, bool)