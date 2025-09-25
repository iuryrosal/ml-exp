from abc import abstractmethod, ABC
from ml_exp.model.ab_test_results import ShapiroWilkTestResult, LeveneTestResult, TStudentTestResult, AnovaTestResult, TurkeyTestResult, KruskalWallisTestResult, MannWhitneyTestResult


class IABTestRepository(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def apply_shapiro(self, context: str, values: list) -> ShapiroWilkTestResult:
        """Apply the Shapiro-Wilk test to check the normality in the distribution

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            ShapiroWilkTestResult: Test result
        """
        pass
    
    @abstractmethod
    def apply_levene(self, context: str, values:list) -> LeveneTestResult:
        """Apply the Levene test to check if the distribution data are homoscedastic

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            LeveneTestResult: Test result
        """
        pass

    @abstractmethod
    def apply_anova(self, context:str, values: list) -> AnovaTestResult:
        """Apply ANOVA test to validate whether there are significant differences between the metric results between the models

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            AnovaTestResult: Test result
        """
        pass

    @abstractmethod
    def apply_turkey(self, context: str, values: list, labels: list) -> TurkeyTestResult:
        """Apply Turkey Test to validate whether there are significant differences between the metric results between the models

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected
            labels (list): Labels indicating the model index related to the data

        Returns:
            TurkeyTestResult: Test result
        """
        pass
    
    @abstractmethod
    def apply_kruskal(self, context: str, values: list) -> KruskalWallisTestResult:
        """Apply KruskalWallis test to validate whether there are significant differences between the metric results between the models

        Args:
            context (str): Model Index related to metrics data collected after testing
            values (list): Performance metric values ​​collected

        Returns:
            KruskalWallisTestResult: Test result
        """
        pass

    @abstractmethod
    def apply_mannwhitney(self, context: str, context_name_1: str, context_name_2: str, values: list) -> MannWhitneyTestResult:
        """Apply the Mann-Whitney test to validate whether there are significant differences between the metric results between pair of models

        Args:
            context (str): General description of the comparative context
            context_name_1 (str): Index 1 of one of the models of the pair being used in the comparison
            context_name_2 (str): Index 2 of one of the models of the pair being used in the comparison
            values (list): Model metric values ​​to be used in testing

        Returns:
            MannWhitneyTestResult: Test result
        """
        pass

    @abstractmethod
    def apply_t_student(self, context: str, context_name_1: str, context_name_2: str, values: list) -> TStudentTestResult:
        """Apply the T-Student test to validate whether there are significant differences between the metric results between pair of models

        Args:
            context (str): General description of the comparative context
            context_name_1 (str): Name of one of the models of the pair being used in the comparison
            context_name_2 (str): Index 2 of one of the models of the pair being used in the comparison
            values (list): Model metric values ​​to be used in testing

        Returns:
            TStudentTestResult: Test result
        """
        pass