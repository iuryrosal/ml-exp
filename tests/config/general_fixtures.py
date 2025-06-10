import pytest

from better_experimentation.repository.ab_test_repository import ABTestRepository


@pytest.fixture
def ab_test_repository():
    return ABTestRepository(alpha=0.05)

# Fixture de retorno padrão para os métodos do ABTestRepository
@pytest.fixture
def mock_repository_result():
    class MockResult:
        def __init__(self, is_significant=False, is_normal=True, is_homoscedastic=True):
            self.is_significant = is_significant
            self.is_normal = is_normal
            self.is_homoscedastic = is_homoscedastic
    return MockResult