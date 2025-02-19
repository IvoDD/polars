import os
from datetime import datetime
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
    num_rows = 100
    demo_df = pd.DataFrame(
        {
            "int_col": range(num_rows),
            "float_col": [2.0*i  for i in range(num_rows)],
            "str_col": [f"str_{i}" for i in range(num_rows)]
        },
        index=pd.date_range(start=pd.Timestamp(2025, 1, 1), periods=num_rows))
    lib.write(symbol, demo_df)
    del lib
    del ac

    yield SymbolIdentifier(uri, lib_name, symbol)

    shutil.rmtree(lmdb_path)

# TODO: Add necessary marks
class TestArcticdbScanIO:
    """Test coverage for `arcticdb` scan ops."""

    def test_scan_arcticdb_plain(self, arctic_identifier):
        lf = pl.scan_arcticdb(arctic_identifier)
        # TODO: len here and in other places is not a good test
        assert len(lf.collect()) == 100
        assert lf.collect_schema() == {
            "__index_level_0__": pl.Time,
            "int_col": pl.Int64,
            "float_col": pl.Float64,
            "str_col": pl.String,
        }


    def test_scan_arcticdb_complex_processing(self, arctic_identifier):
        # Using multiple filters and projection which will push down to arcticdb layer.
        lf = pl.scan_arcticdb(arctic_identifier)
        lf = lf.filter((10 <= pl.col("int_col")) & (pl.col("int_col") < 50)) # After this we will have rows between 10 and 50
        lf = lf.filter(pl.col("float_col") <= 40.1) # After this we will have rows between 10 and 20 incl
        lf = lf.filter(pl.col("str_col").is_in(["str_1", "str_10", "str_13", "str_17", "str_25"])) # After this we will have only rows 10, 13, 17
        lf = lf.select(["int_col"])
        assert len(lf.collect()) == 3
        assert lf.collect_schema() == {
            "int_col": pl.Int64,
        }


    def test_scan_arcticdb_basic_processing(self, arctic_identifier):
        lf = pl.scan_arcticdb(arctic_identifier)

        res = lf.filter(pl.col("int_col") <= 20)
        assert len(res.collect()) == 21

        res = lf.filter(pl.col("int_col") < 20)
        assert len(res.collect()) == 20

        res = lf.filter(pl.col("int_col") > 20)
        assert len(res.collect()) == 79

        res = lf.filter(pl.col("int_col") >= 20)
        assert len(res.collect()) == 80

        res = lf.filter(pl.col("int_col").is_in([10, 20, 30, 105, 40]))
        assert len(res.collect()) == 4

        res = lf.filter(pl.col("float_col").is_not_nan())
        assert len(res.collect()) == 100

        res = lf.filter(pl.col("str_col").is_not_null())
        assert len(res.collect()) == 100

        # Checks below don't do predicate pushdown.
        res = lf.filter(pl.col("__index_level_0__") >= datetime(2025, 1, 11))
        assert len(res.collect()) == 90

        res = lf.filter(pl.col("__index_level_0__") < datetime(2025, 1, 11))
        assert len(res.collect()) == 10
