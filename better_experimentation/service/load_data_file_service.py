import pandas as pd

from better_experimentation.repository.data_file_repository import DataFileRepository


class LoadDataFileService:
    def __init__(self, file_name: str) -> None:
        self.data_file_repo = DataFileRepository(file_name)
    
    def generate_pandas_dataframe(self) -> pd.DataFrame:
        return self.data_file_repo.read_pandas()