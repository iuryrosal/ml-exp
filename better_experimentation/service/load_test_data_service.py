import pandas as pd
from pathlib import Path
from typing import Union
import numpy as np
from better_experimentation.repository.interfaces.data_file_repository import IDataFileRepository


class LoadTestDataService:
    """Load Data File from some file path using repository
    """
    def __init__(self, data_file_repository: IDataFileRepository) -> None:
        self.data_file_repo = data_file_repository

        self.test_data = {}
    
    def generate_dataframe(self, file_name: Path) -> pd.DataFrame:
        """Generate pandas dataframe from some path object 

        Args:
            file_name (Path): Path related with data file to load

        Returns:
            pd.DataFrame: Pandas Dataframe with data from data file (file_name path)
        """
        return self.data_file_repo.read(file_name)
    
    def add_test_data(self,
                      test_data_name: str,
                      X_test: Union[pd.DataFrame, str],
                      y_test: Union[pd.DataFrame, str]):
        
        if test_data_name in self.test_data:
            raise ValueError(f"Test data '{test_data_name}' already exists. Please use a different name or remove the existing test data before adding new one.")

        # check data type of X_test
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test
        elif isinstance(X_test, str):
            X_test = self.generate_dataframe(Path(X_test))
        elif isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test).reset_index(drop=True)
        else:
            raise ValueError(f"X_test need to be Pandas Dataframe or string path to file. Current type of X_test: {type(X_test)}")

        # check data type of y_test
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test
        elif isinstance(y_test, str):
            y_test = self.generate_dataframe(Path(y_test))
        elif isinstance(y_test, np.ndarray):
            y_test = pd.DataFrame(y_test).reset_index(drop=True)
        else:
            raise ValueError(f"y_test need to be Pandas Dataframe or string path to file. Current type of y_test: {type(y_test)}")

        self.test_data[test_data_name] = {"x_test": X_test, "y_test": y_test}
    
    def get_all_test_data(self) -> dict:
        """Get all test data

        Returns:
            dict: Dictionary with all test data
        """
        return self.test_data

    def get_test_data(self, test_data_name: str) -> dict:
        """Get test data by name

        Args:
            test_data_name (str): Name of the test data to be retrieved

        Returns:
            dict: Dictionary with test data
        """
        if test_data_name in self.test_data:
            return self.test_data[test_data_name]
        else:
            raise ValueError(f"Test data '{test_data_name}' not found. Please add the test data before retrieving it.")
        
    def remove_test_data(self, test_data_name: str):
        """Remove test data by name

        Args:
            test_data_name (str): Name of the test data to be removed
        """
        if test_data_name in self.test_data:
            del self.test_data[test_data_name]
        else:
            raise ValueError(f"Test data '{test_data_name}' not found. Please add the test data before removing it.")