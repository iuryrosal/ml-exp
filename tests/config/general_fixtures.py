import pytest

from better_experimentation.repository.ab_test_repository import ABTestRepository
from better_experimentation.repository.sklearn_model_repository import SklearnModelRepository
from better_experimentation.repository.pandas_data_file_repository import PandasDataFileRepository



@pytest.fixture
def ab_test_repository():
    return ABTestRepository(alpha=0.05)

@pytest.fixture
def mock_repository_result():
    class MockResult:
        def __init__(self, is_significant=False, is_normal=True, is_homoscedastic=True):
            self.is_significant = is_significant
            self.is_normal = is_normal
            self.is_homoscedastic = is_homoscedastic
    return MockResult

@pytest.fixture
def sklearn_model_repository():
    return SklearnModelRepository()

@pytest.fixture
def pandas_data_file_repository():
    return PandasDataFileRepository()