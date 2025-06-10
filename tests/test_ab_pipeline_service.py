import pytest
from unittest.mock import MagicMock, patch

from tests.config.general_fixtures import mock_repository_result
from better_experimentation.service.ab_pipeline_service import ABPipelineService


@patch("better_experimentation.service.ab_pipeline_service.ABTestRepository")
def test_run_pipeline_t_student_path(mock_repo_class, mock_repository_result):
    """Test 2 models with normal data"""
    mock_data = {
        "0": [0.5] * 30,
        "1": [0.5] * 30,
    }

    mock_repo = MagicMock()
    mock_repo.apply_shapiro.return_value = mock_repository_result(is_normal=True)
    mock_repo.apply_t_student.return_value = mock_repository_result(is_significant=False)

    mock_repo_class.return_value = mock_repo

    service = ABPipelineService(scores_data=mock_data, score_target="accuracy")
    service.run_pipeline()
    report = service.get_report()

    assert "check_normality_with_shapiro" in report.ab_tests.pipeline_track
    assert "data_normal_is_true" in report.ab_tests.pipeline_track
    assert "perform_t_student" in report.ab_tests.pipeline_track
    assert "done" in report.ab_tests.pipeline_track

@patch("better_experimentation.service.ab_pipeline_service.ABTestRepository")
def test_run_pipeline_anova_tukey_path(mock_repo_class, mock_repository_result):
    """Test 3 models with normal and homoscedastic data"""
    mock_data = {
        "0": [0.5] * 30,
        "1": [0.5] * 30,
        "2": [0.5] * 30
    }

    mock_repo = MagicMock()
    mock_repo.apply_shapiro.side_effect = [
        mock_repository_result(is_normal=True),
        mock_repository_result(is_normal=True),
        mock_repository_result(is_normal=True)
    ]
    mock_repo.apply_levene.return_value = mock_repository_result(is_homoscedastic=True)
    mock_repo.apply_anova.return_value = mock_repository_result(is_significant=True)
    mock_repo.apply_turkey.return_value = mock_repository_result()

    mock_repo_class.return_value = mock_repo

    service = ABPipelineService(scores_data=mock_data, score_target="accuracy")
    service.run_pipeline()
    report = service.get_report()

    assert report.ab_tests.anova.is_significant is True
    assert "check_normality_with_shapiro" in report.ab_tests.pipeline_track
    assert "check_homocedasticity_with_levene" in report.ab_tests.pipeline_track
    assert "perform_anova" in report.ab_tests.pipeline_track
    assert "anova_is_significant" in report.ab_tests.pipeline_track
    assert "perform_turkey" in report.ab_tests.pipeline_track
    assert "done" in report.ab_tests.pipeline_track

@patch("better_experimentation.service.ab_pipeline_service.ABTestRepository")
def test_run_pipeline_kruskal_mannwhitney_path(mock_repo_class, mock_repository_result):
    """3 models with not normal data"""
    mock_data = {
        "0": [0.5] * 30,
        "1": [0.6] * 30,
        "2": [0.7] * 30
    }

    mock_repo = MagicMock()
    mock_repo.apply_shapiro.side_effect = [
        mock_repository_result(is_normal=False),
        mock_repository_result(is_normal=False),
        mock_repository_result(is_normal=False)
    ]
    mock_repo.apply_kruskal.return_value = mock_repository_result(is_significant=True)
    mock_repo.apply_mannwhitney.return_value = mock_repository_result()

    mock_repo_class.return_value = mock_repo

    service = ABPipelineService(scores_data=mock_data, score_target="accuracy")
    service.run_pipeline()
    report = service.get_report()

    assert report.ab_tests.kurskalwallis.is_significant is True
    assert "check_normality_with_shapiro" in report.ab_tests.pipeline_track
    assert "data_normal_and_homocedasticity_is_false" in report.ab_tests.pipeline_track
    assert "perform_kurskalwallis" in report.ab_tests.pipeline_track
    assert "kurskalwallis_is_significant" in report.ab_tests.pipeline_track
    assert "perform_mannwhitney" in report.ab_tests.pipeline_track
    assert "done" in report.ab_tests.pipeline_track


@patch("better_experimentation.service.ab_pipeline_service.ABTestRepository")
def test_run_pipeline_mannwhitney_only_for_non_normal_data(mock_repo_class, mock_repository_result):
    """2 models with not normal data"""
    mock_data = {
        "0": [0.4] * 20,
        "1": [0.6] * 20,
    }

    mock_repo = MagicMock()
    mock_repo.apply_shapiro.side_effect = [
        mock_repository_result(is_normal=False),
        mock_repository_result(is_normal=False)
    ]
    mock_repo.apply_mannwhitney.return_value = mock_repository_result()
    mock_repo_class.return_value = mock_repo

    service = ABPipelineService(scores_data=mock_data, score_target="recall")
    service.run_pipeline()
    report = service.get_report()

    assert report.ab_tests.shapirowilk[0].is_normal is False
    assert report.ab_tests.shapirowilk[1].is_normal is False
    assert report.ab_tests.mannwhitney is not None
    assert "check_normality_with_shapiro" in report.ab_tests.pipeline_track
    assert "3_or_more_models_is_false" in report.ab_tests.pipeline_track
    assert "data_normal_is_false" in report.ab_tests.pipeline_track
    assert "perform_mannwhitney" in report.ab_tests.pipeline_track
    assert "done" in report.ab_tests.pipeline_track