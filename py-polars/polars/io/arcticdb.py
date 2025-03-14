from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, Callable

import polars._reexport as pl
from polars._utils.convert import to_py_date, to_py_datetime
from polars.dependencies import arcticdb

if TYPE_CHECKING:
    from datetime import date, datetime

    from arcticdb import LazyDataFrame

    from polars import DataFrame, LazyFrame, Series

__all__ = ["scan_arcticdb"]

_temporal_conversions: dict[str, Callable[..., datetime | date]] = {
    "to_py_date": to_py_date,
    "to_py_datetime": to_py_datetime,
}

def scan_arcticdb(
    source: arcticdb.LazyDataFrame | str,
    lib_name: Optional[str] = None,
    symbol: Optional[str] = None,
) -> LazyFrame:
    """
    Lazily read dataframe from an ArcticDB library.
    It can either scan a symbol lazily from an ArcticDB library by providing an ArcticDB uri, the library name and the symbol name.
    Or it can read from an arcticdb.LazyDataFrame. When collecting from an arcticdb.LazyDataFrame its clauses will be
    applied before the clauses from polars.

    Parameters
    ----------
    source
        An ArcticDB LazyDataFrame, or an ArcticDB URI pointing to the ArcticDB storage
    lib_name
        The library name from which to read the symbol. Should be passed if and only if reading from an URI.
    symbol
        The ArcticDB symbol to read from the library. Should be passed if and only if reading from an URI.

    Returns
    -------
    LazyFrame

    Examples
    --------
    TODO: Write examples with both arcticdb uri and arcticdb.LazyDataFrame

    """
    from arcticdb import Arctic, OutputFormat

    if isinstance(source, str):
        if lib_name is None or symbol is None:
            raise ValueError("If using an ArcticDB uri as source, lib_name and symbol must also be provided.")
        ac = Arctic(source)
        lib = ac[lib_name]
        adb_lf = lib.read(symbol, lazy=True)
    else:
        if lib_name is not None or symbol is not None:
            raise ValueError("If using an ArcticDB LazyDataFrame as source, lib_name and symbol must NOT be provided.")
        adb_lf = source

    # TODO: To use a `schema_fn` we need a good way to convert `pa.Schema` to `pl.SchemaDict`
    # To do it cleanly we'll need https://github.com/pola-rs/polars/issues/15563
    # schema_fn = partial(_collect_pyarrow_schema, source)
    arrow_schema = _collect_pyarrow_schema(adb_lf)
    scan_fn = partial(_scan_pyarrow_dataset_impl, adb_lf)
    return pl.LazyFrame._scan_python_function(arrow_schema, scan_fn, pyarrow=True)


def _collect_pyarrow_schema(lazy_df: LazyDataFrame) -> pa.Schema:
    from arcticdb import OutputFormat
    return lazy_df.collect_schema(output_format=OutputFormat.ARROW)


def _scan_pyarrow_dataset_impl(
    lazy_df: LazyDataFrame,
    with_columns: list[str] | None = None,
    predicate: str = "",
    n_rows: int | None = None,
) -> DataFrame | Series:
    from polars import from_arrow
    from arcticdb import OutputFormat
    from arcticdb.version_store.processing import ExpressionNode

    if with_columns is not None:
        lazy_df = lazy_df.select_columns(with_columns)

    if predicate is not None:
        print("Predicate", predicate)
        adb_expression = ExpressionNode._from_pyarrow_expression_str(predicate, function_map=_temporal_conversions)
        lazy_df = lazy_df[adb_expression]

    if n_rows is not None:
        lazy_df = lazy_df.head(n_rows)

    arrow_df = lazy_df.collect(output_format=OutputFormat.ARROW).data
    result = from_arrow(arrow_df)
    return result