from abc import abstractmethod, ABC
from pathlib import Path
import pandas as pd

class IDataFileRepository(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def read(self, file_path: Path) -> pd.DataFrame:
        pass