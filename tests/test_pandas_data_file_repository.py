import pytest
import pandas as pd
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.config.general_fixtures import pandas_data_file_repository

def test_remove_suffix(pandas_data_file_repository):
    assert pandas_data_file_repository.remove_suffix("file.csv", ".csv") == "file"
    assert pandas_data_file_repository.remove_suffix("file.csv", ".txt") == "file.csv"
    assert pandas_data_file_repository.remove_suffix("file", ".csv") == "file"


@pytest.mark.parametrize("ext,expected", [
    (".csv", False),
    (".gz", True),
    (".zip", True),
    (".bz2", True),
    (".xz", True),
    (".txt", False),
])
def test_is_supported_compression(pandas_data_file_repository, ext, expected):
    assert pandas_data_file_repository.is_supported_compression(ext) == expected


@pytest.mark.parametrize("filename,expected_ext", [
    ("data.csv", ".csv"),
    ("data.csv.gz", ".csv"),
    ("data.json.xz", ".json"),
    ("data.json", ".json"),
])
def test_uncompressed_extension(pandas_data_file_repository, filename, expected_ext):
    path = Path(filename)
    assert pandas_data_file_repository.uncompressed_extension(path) == expected_ext


def test_warn_read_triggers_warning(pandas_data_file_repository):
    pandas_data_file_repository.file_extension = ".unknown"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pandas_data_file_repository.warn_read()
        assert len(w) == 1
        assert "we assume it to be in CSV format" in str(w[-1].message)


def test_read_csv_file(pandas_data_file_repository):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "file.csv"
        df_original = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_original.to_csv(path, index=False)

        df_loaded = pandas_data_file_repository.read(path)
        pd.testing.assert_frame_equal(df_loaded, df_original)


def test_read_json_file(pandas_data_file_repository):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "file.json"
        df_original = pd.DataFrame({"x": [10, 20]})
        df_original.to_json(path, orient="records")

        df_loaded = pandas_data_file_repository.read(path)
        pd.testing.assert_frame_equal(df_loaded, pd.read_json(path))


def test_read_parquet_file(pandas_data_file_repository):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "file.parquet"
        df_original = pd.DataFrame({"c": [5, 6]})
        df_original.to_parquet(path)

        df_loaded = pandas_data_file_repository.read(path)
        pd.testing.assert_frame_equal(df_loaded, df_original)


def test_read_unsupported_tar_raises(pandas_data_file_repository):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "file.tar"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="tar compression is not supported directly"):
            pandas_data_file_repository.read(path)