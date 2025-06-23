import warnings
from pathlib import Path
from better_experimentation.repository.interfaces.data_file_repository import IDataFileRepository
import pandas as pd

class PandasDataFileRepository(IDataFileRepository):
    """Repository to load data file to pandas dataframe

    Args:
        IDataFileRepository (ABC): Interface responsible to define the main logic to load data files
    """
    def __init__(self) -> None:
        pass

    def warn_read(self) -> None:
        """Warn the user when an extension is not supported.
        """
        warnings.warn(
            f"""There was an attempt to read a file with extension {self.file_extension}, we assume it to be in CSV format.
            To prevent this warning from showing up, please rename the file to any of the extensions supported by pandas
            (docs: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
            If you think this extension should be supported, please report this as an issue:
            https://github.com/ydataai/ydata-profiling/issues"""
        )

    @staticmethod
    def is_supported_compression(extension) -> bool:
        """Determine if the given file extension indicates a compression format that pandas can handle automatically.

        Args:
            file_extension (str): the file extension to test

        Returns:
            bool: True if the extension indicates a compression format that pandas handles automatically and False otherwise

        Notes:
            Pandas can handle on the fly decompression from the following extensions: ‘.bz2’, ‘.gz’, ‘.zip’, or ‘.xz’
            (otherwise no decompression). If using ‘.zip’, the ZIP file must contain exactly one data file to be read in.
        """
        return extension.lower() in [".bz2", ".gz", ".xz", ".zip"]

    @staticmethod
    def remove_suffix(text: str, suffix: str) -> str:
        """Removes the given suffix from the given string.

        Args:
            text (str): the string to remove the suffix from
            suffix (str): the suffix to remove from the string

        Returns:
            str: the string with the suffix removed, if the string ends with the suffix, otherwise the unmodified string
        """
        return text[: -len(suffix)] if suffix and text.endswith(suffix) else text

    def uncompressed_extension(self, file_path: Path) -> str:
        """Returns the uncompressed extension of the given file name.

        Args:
            file_path (Path): the file name to get the uncompressed extension of

        Returns:
            str: the uncompressed extension, or the original extension if pandas doesn't handle it automatically
        """
        extension = file_path.suffix.lower()
        return (
            Path(self.remove_suffix(str(file_path).lower(), extension)).suffix
            if self.is_supported_compression(extension)
            else extension
        )


    def read(self, file_path: Path) -> pd.DataFrame:
        """Read DataFrame based on the file extension. This function is used when the file is in a standard format.
        Various file types are supported (.csv, .json, .jsonl, .data, .tsv, .xls, .xlsx, .xpt, .sas7bdat, .parquet)

        Args:
            file_name (Path): the file to read

        Returns:
            DataFrame
        """
        self.file_extension = self.uncompressed_extension(file_path)

        if self.file_extension == ".json":
            df = pd.read_json(str(file_path))
        elif self.file_extension == ".jsonl":
            df = pd.read_json(str(file_path), lines=True)
        elif self.file_extension == ".dta":
            df = pd.read_stata(str(file_path))
        elif self.file_extension == ".tsv":
            df = pd.read_csv(str(file_path), sep="\t")
        elif self.file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(str(file_path))
        elif self.file_extension in [".hdf", ".h5"]:
            df = pd.read_hdf(str(file_path))
        elif self.file_extension in [".sas7bdat", ".xpt"]:
            df = pd.read_sas(str(file_path))
        elif self.file_extension == ".parquet":
            df = pd.read_parquet(str(file_path))
        elif self.file_extension in [".pkl", ".pickle"]:
            df = pd.read_pickle(str(file_path))
        elif self.file_extension == ".tar":
            raise ValueError(
                "tar compression is not supported directly by pandas, please use the 'tarfile' module"
            )
        else:
            if self.file_extension != ".csv":
                self.warn_read()

            df = pd.read_csv(str(file_path))
        return df