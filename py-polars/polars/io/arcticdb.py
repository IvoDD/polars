from __future__ import annotations

import ast
from _ast import GtE, Lt, LtE
from ast import (
    Attribute,
    BinOp,
    BitAnd,
    BitOr,
    Call,
    Compare,
    Constant,
    Eq,
    Gt,
    Invert,
    List,
    Name,
    UnaryOp,
)
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any, Callable

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

# TODO: Maybe move in arcticdb
from typing import NamedTuple
class SymbolIdentifier(NamedTuple):
    uri : str
    lib_name : str
    symbol : str


def scan_arcticdb(
    source: LazyDataFrame | SymbolIdentifier,
) -> LazyFrame:
    """
    Lazily read from an ArcticDB library.

    Parameters
    ----------
    source
        An ArcticDB LazyDataFrame, or a direct path to an arcticdb symbol.


    Returns
    -------
    LazyFrame

    """
    from arcticdb import Arctic, OutputFormat

    if isinstance(source, SymbolIdentifier):
        ac = Arctic(source.uri)
        lib = ac[source.lib_name]
        source = lib.read(source.symbol, lazy=True)

    # TODO: To use a lazy schema function we need a good way to convert `pa.Schema` to `pl.SchemaDict`
    # To do it cleanly we'll need https://github.com/pola-rs/polars/issues/15563
    # Just for demo we load the schema eagerly.
    # schema_fn = partial(_collect_pyarrow_schema, source)
    arrow_schema = _collect_pyarrow_schema(source)
    scan_fn = partial(_scan_pyarrow_dataset_impl, source)
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

    if with_columns is not None:
        # TODO
        raise ValueError("Unsupported column filter")
        lazy_df = lazy_df[with_columns]

    if predicate is not None:
        # TODO
        raise ValueError("Unsupported predicate")
        lazy_df = lazy_df[predicate]

    if n_rows is not None:
        # TODO
        raise ValueError("Unsupported predicate")
        lazy_df = lazy_df.head(n_rows)


    arrow_df = lazy_df.collect(output_format=OutputFormat.ARROW).data
    print(arrow_df)
    result = from_arrow(arrow_df)
    print(result)
    return result