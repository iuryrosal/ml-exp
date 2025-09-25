from abc import abstractmethod, ABC
from ml_exp.model.report import GeneralReportByScore


class IABPipelineService(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def _collect_statistical_results(self):
        pass

    @abstractmethod
    def _check_normality(self):
        """Checks the normality of data for each campaign using Shapiro-Wilk."""
        pass

    @abstractmethod
    def _group_all_values(self):
        pass

    @abstractmethod
    def _check_homocedasticity_more_than_2(self):
        """Checks homoscedasticity between groups using Levene and Bartlett tests."""
        pass

    @abstractmethod
    def _check_homocedasticity(self):
        """Checks homoscedasticity between groups using Levene and Bartlett tests."""
        pass

    @abstractmethod
    def _perform_parametric_tests(self):
        """Performs ANOVA if data are normal and homoscedastic."""
        pass

    @abstractmethod
    def _perform_non_parametric_tests(self):
        """Performs nonparametric tests for nonnormal or nonhomoscedastic data."""
        pass

    @abstractmethod
    def run_pipeline(self):
        """Executes the entire AB testing flow according to the adopted methodology.
        """
        pass 

    @abstractmethod
    def get_report(self) -> GeneralReportByScore:
        pass