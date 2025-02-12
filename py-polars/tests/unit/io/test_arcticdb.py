import os
from pathlib import Path
import pandas as pd
import shutil

import pytest

import polars as pl
from polars.io.arcticdb import SymbolIdentifier

from arcticdb import Arctic

# TODO: Consider writing the dataframes to use in the files folder
# Also the pandas dependency is kind of nasty
@pytest.fixture
def arctic_identifier(io_files_path: Path) -> SymbolIdentifier:
    lmdb_path = "/tmp/arcticdb"
    Path(lmdb_path).mkdir(parents=True, exist_ok=True)

    uri = f"lmdb://{lmdb_path}"
    lib_name = "test_lib"
    symbol = "sym"

    ac = Arctic(uri)
    lib = ac.create_library(lib_name)
    demo_df = pd.DataFrame({"col": [1, 2, 3]}, index=pd.date_range(start=pd.Timestamp(2025, 1, 1), periods=3))
    lib.write(symbol, demo_df)
    del lib
    del ac

    yield SymbolIdentifier(uri, lib_name, symbol)

    shutil.rmtree(lmdb_path)

# TODO: Add necessary marks
class TestArcticdbScanIO:
    """Test coverage for `arcticdb` scan ops."""

    def test_scan_arcticdb_plain(self, arctic_identifier):
        df = pl.scan_arcticdb(arctic_identifier)
        assert len(df.collect()) == 3
        assert df.collect_schema() == {
            "__index_level_0__": pl.Time,
            "col": pl.Int64,
        }