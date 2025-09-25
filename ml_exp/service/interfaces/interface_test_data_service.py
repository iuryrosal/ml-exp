from abc import abstractmethod, ABC
import pandas as pd
from pathlib import Path
from typing import Union


class ILoadTestDataService(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def generate_dataframe(self, file_name: Path) -> pd.DataFrame:
        """Generate pandas dataframe from some path object 

        Args:
            file_name (Path): Path related with data file to load

        Returns:
            pd.DataFrame: Pandas Dataframe with data from data file (file_name path)
        """
        pass
    
    @abstractmethod
    def add_test_data(self,
                      test_data_name: str,
                      X_test: Union[pd.DataFrame, str],
                      y_test: Union[pd.DataFrame, str]):
        pass
    
    @abstractmethod
    def get_all_test_data(self) -> dict:
        """Get all test data

        Returns:
            dict: Dictionary with all test data
        """
        pass

    @abstractmethod
    def get_test_data(self, test_data_name: str) -> dict:
        """Get test data by name

        Args:
            test_data_name (str): Name of the test data to be retrieved

        Returns:
            dict: Dictionary with test data
        """
        pass
    
    @abstractmethod
    def remove_test_data(self, test_data_name: str):
        """Remove test data by name

        Args:
            test_data_name (str): Name of the test data to be removed
        """
        pass