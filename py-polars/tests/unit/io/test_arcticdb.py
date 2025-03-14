import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import shutil

import pytest

import polars as pl

from arcticdb import Arctic

# TODO: Consider writing the dataframes to use in the files folder
# Also the pandas dependency is kind of nasty
# Also returning a tuple is not great as well
@pytest.fixture
def arcticdb_identifier(io_files_path: Path) -> tuple[str, str, str]:
    lmdb_path = "/tmp/arcticdb"
    Path(lmdb_path).mkdir(parents=True, exist_ok=True)

    uri = f"lmdb://{lmdb_path}"
    lib_name = "test_lib"
    symbol = "sym"

    ac = Arctic(uri)
    lib = ac.create_library(lib_name)
    num_rows = 200
    demo_df = pd.DataFrame(
        {
            "int_col": range(num_rows),
            "float_col": [2.0*i  for i in range(num_rows)],
            "str_col": [f"str_{i}" for i in range(num_rows)]
        },
        index=pd.date_range(start=pd.Timestamp(2025, 1, 1), periods=num_rows, freq="s"))
    lib.write(symbol, demo_df)
    del lib
    del ac

    yield (uri, lib_name, symbol)

    shutil.rmtree(lmdb_path)

@pytest.fixture
def arcticdb_lazy_frame(arcticdb_identifier):
    uri, lib_name, sym = arcticdb_identifier
    ac = Arctic(uri)
    lib = ac[lib_name]
    yield lib.read(sym, lazy=True)

# TODO: Add necessary marks
class TestArcticdbScanIO:
    """Test coverage for `arcticdb` scan ops."""

    def test_scan_arcticdb_plain(self, arcticdb_identifier):
        uri, lib_name, sym = arcticdb_identifier
        lf = pl.scan_arcticdb(uri, lib_name, sym)
        # TODO: len here and in other places is not a good test
        assert len(lf.collect()) == 200
        assert lf.collect_schema() == {
            "__index_level_0__": pl.Time,
            "int_col": pl.Int64,
            "float_col": pl.Float64,
            "str_col": pl.String,
        }


    def test_scan_arcticdb_complex_processing(self, arcticdb_identifier):
        uri, lib_name, sym = arcticdb_identifier
        # Using multiple filters and projection which will push down to arcticdb layer.
        lf = pl.scan_arcticdb(uri, lib_name, sym)
        lf = lf.filter((10 <= pl.col("int_col")) & (pl.col("int_col") < 50)) # After this we will have rows between 10 and 50
        lf = lf.filter(pl.col("float_col") <= 40.1) # After this we will have rows between 10 and 20 incl
        lf = lf.filter(pl.col("str_col").is_in(["str_1", "str_10", "str_13", "str_17", "str_25"])) # After this we will have only rows 10, 13, 17
        lf = lf.select(["int_col"])
        assert len(lf.collect()) == 3
        assert lf.collect_schema() == {
            "int_col": pl.Int64,
        }


    def test_scan_arcticdb_basic_processing(self, arcticdb_identifier):
        uri, lib_name, sym = arcticdb_identifier
        lf = pl.scan_arcticdb(uri, lib_name, sym)

        res = lf.filter(pl.col("int_col") <= 20)
        assert len(res.collect()) == 21

        res = lf.filter(pl.col("int_col") < 20)
        assert len(res.collect()) == 20

        res = lf.filter(pl.col("int_col") > 20)
        assert len(res.collect()) == 179

        res = lf.filter(pl.col("int_col") >= 20)
        assert len(res.collect()) == 180

        res = lf.filter(pl.col("int_col").is_in([10, 20, 30, 205, 40]))
        assert len(res.collect()) == 4

        res = lf.filter(pl.col("float_col").is_not_nan())
        assert len(res.collect()) == 200

        res = lf.filter(pl.col("str_col").is_not_null())
        assert len(res.collect()) == 200

        # Checks below don't do predicate pushdown.
        res = lf.filter(pl.col("__index_level_0__") >= datetime(2025, 1, 1, 0, 0, 10))
        assert len(res.collect()) == 190

        res = lf.filter(pl.col("__index_level_0__") < datetime(2025, 1, 1, 0, 0, 10))
        assert len(res.collect()) == 10

    def test_scan_arcticdb_combined_processing(self, arcticdb_lazy_frame):
        adb_lf = arcticdb_lazy_frame
        adb_lf["int_col_2"] = adb_lf["int_col"] * 2
        adb_lf.resample("30s").agg({"int_col": "mean", "int_col_2": "sum"})
        lf = pl.scan_arcticdb(adb_lf)
        lf = lf.filter((50 <= pl.col("int_col")) & (pl.col("int_col") < 150))
        assert len(lf.collect()) == 3
        # TODO: Check schema. Currently is incorrect because of WIP adb.LazyDataFrame.collect_schema
