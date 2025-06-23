import pandas as pd
from pathlib import Path
from better_experimentation.repository.interfaces.data_file_repository import IDataFileRepository


class LoadDataFileService:
    """Load Data File from some file path using repository
    """
    def __init__(self, data_file_repository: IDataFileRepository) -> None:
        self.data_file_repo = data_file_repository
    
    def generate_dataframe(self, file_name: Path) -> pd.DataFrame:
        """Generate pandas dataframe from some path object 

        Args:
            file_name (Path): Path related with data file to load

        Returns:
            pd.DataFrame: Pandas Dataframe with data from data file (file_name path)
        """
        return self.data_file_repo.read(file_name)