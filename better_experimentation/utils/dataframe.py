import warnings
from pathlib import Path
from typing import Any, Optional
import pandas as pd


def warn_read(extension: str) -> None:
    """Warn the user when an extension is not supported.

    Args:
        extension: The extension that is not supported.
    """
    warnings.warn(
        f"""There was an attempt to read a file with extension {extension}, we assume it to be in CSV format.
        To prevent this warning from showing up, please rename the file to any of the extensions supported by pandas
        (docs: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
        If you think this extension should be supported, please report this as an issue:
        https://github.com/ydataai/ydata-profiling/issues"""
    )

def is_supported_compression(file_extension: str) -> bool:
    """Determine if the given file extension indicates a compression format that pandas can handle automatically.

    Args:
        file_extension (str): the file extension to test

    Returns:
        bool: True if the extension indicates a compression format that pandas handles automatically and False otherwise

    Notes:
        Pandas can handle on the fly decompression from the following extensions: ‘.bz2’, ‘.gz’, ‘.zip’, or ‘.xz’
        (otherwise no decompression). If using ‘.zip’, the ZIP file must contain exactly one data file to be read in.
    """
    return file_extension.lower() in [".bz2", ".gz", ".xz", ".zip"]

def remove_suffix(text: str, suffix: str) -> str:
    """Removes the given suffix from the given string.

    Args:
        text (str): the string to remove the suffix from
        suffix (str): the suffix to remove from the string

    Returns:
        str: the string with the suffix removed, if the string ends with the suffix, otherwise the unmodified string

    Notes:
        In python 3.9+, there is a built-in string method called removesuffix() that can serve this purpose.
    """
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text

def uncompressed_extension(file_path: Path) -> str:
    """Returns the uncompressed extension of the given file name.

    Args:
        file_path (Path): the file name to get the uncompressed extension of

    Returns:
        str: the uncompressed extension, or the original extension if pandas doesn't handle it automatically
    """
    extension = file_path.suffix.lower()
    return (
        Path(remove_suffix(str(file_path).lower(), extension)).suffix
        if is_supported_compression(extension)
        else extension
    )


def read_pandas(file_name: str) -> pd.DataFrame:
    """Read DataFrame based on the file extension. This function is used when the file is in a standard format.
    Various file types are supported (.csv, .json, .jsonl, .data, .tsv, .xls, .xlsx, .xpt, .sas7bdat, .parquet)

    Args:
        file_name: the file to read

    Returns:
        DataFrame

    Notes:
        This function is based on pandas IO tools:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
        https://pandas.pydata.org/pandas-docs/stable/reference/io.html

        This function is not intended to be flexible or complete. The main use case is to be able to read files without
        user input, which is currently used in the editor integration. For more advanced use cases, the user should load
        the DataFrame in code.
    """
    file_path = Path(file_name)
    extension = uncompressed_extension(file_path)
    if extension == ".json":
        df = pd.read_json(str(file_path))
    elif extension == ".jsonl":
        df = pd.read_json(str(file_path), lines=True)
    elif extension == ".dta":
        df = pd.read_stata(str(file_path))
    elif extension == ".tsv":
        df = pd.read_csv(str(file_path), sep="\t")
    elif extension in [".xls", ".xlsx"]:
        df = pd.read_excel(str(file_path))
    elif extension in [".hdf", ".h5"]:
        df = pd.read_hdf(str(file_path))
    elif extension in [".sas7bdat", ".xpt"]:
        df = pd.read_sas(str(file_path))
    elif extension == ".parquet":
        df = pd.read_parquet(str(file_path))
    elif extension in [".pkl", ".pickle"]:
        df = pd.read_pickle(str(file_path))
    elif extension == ".tar":
        raise ValueError(
            "tar compression is not supported directly by pandas, please use the 'tarfile' module"
        )
    else:
        if extension != ".csv":
            warn_read(extension)

        df = pd.read_csv(str(file_path))
    return df