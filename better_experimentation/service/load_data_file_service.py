import pandas as pd

from better_experimentation.repository.interfaces.data_file_repository import IDataFileRepository


class LoadDataFileService:
    def __init__(self, data_file_repository: IDataFileRepository) -> None:
        self.data_file_repo = data_file_repository
    
    def generate_dataframe(self, file_name: str) -> pd.DataFrame:
        return self.data_file_repo.read(file_name)