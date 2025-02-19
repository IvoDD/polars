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

# TODO: Better imports
from arcticdb.version_store.processing import ExpressionNode

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
        lazy_df = lazy_df.select_columns(with_columns)

    if predicate is not None:
        print("Predicate", predicate)
        expr_ast = _to_ast(predicate)
        adb_expression = _convert_predicate(expr_ast)
        lazy_df = lazy_df[adb_expression]

    if n_rows is not None:
        lazy_df = lazy_df.head(n_rows)

    arrow_df = lazy_df.collect(output_format=OutputFormat.ARROW).data
    result = from_arrow(arrow_df)
    return result


def _to_ast(expr: str) -> ast.expr:
    """
    Converts a Python string to an AST.

    This will take the Python Arrow expression (as a string), and it will
    be converted into a Python AST that can be traversed to convert it to an
    ArcticDB ExpressionNode.

    The reason to convert it to an AST is because the PyArrow expression
    itself doesn't have any methods/properties to traverse the expression.
    We need this to convert it into an ArcticDB expression.

    Parameters
    ----------
    expr
        The string expression

    Returns
    -------
    The AST representing the Arrow expression
    """
    return ast.parse(expr, mode="eval").body


@singledispatch
def _convert_predicate(a: Any) -> Any:
    """Walks the AST to convert the PyArrow expression to an ArcticDB expression."""
    msg = f"Unexpected symbol: {a}"
    raise ValueError(msg)


@_convert_predicate.register(Constant)
def _(a: Constant) -> Any:
    print("Constant", a.value)
    return a.value


@_convert_predicate.register(Name)
def _(a: Name) -> Any:
    print("Name", a.id)
    return a.id


@_convert_predicate.register(UnaryOp)
def _(a: UnaryOp) -> Any:
    print("Unary", a.op, a.operand)
    if isinstance(a.op, Invert):
        return ~_convert_predicate(a.operand)
    else:
        msg = f"Unexpected UnaryOp: {a}"
        raise TypeError(msg)


@_convert_predicate.register(Call)
def _(a: Call) -> Any:
    print("Call: ", a.func, a.args)
    args = [_convert_predicate(arg) for arg in a.args]
    f = _convert_predicate(a.func) # TODO: Needed to uncover (expr).isin(list)
    if f == "field":
        print("Field: ", args, f)
        return ExpressionNode.column_ref(args[0])
    elif f == "scalar": # TODO: Understand this
        return args[0]
    elif f in _temporal_conversions: # TODO: Understand this
        # convert from polars-native i64 to ISO8601 string
        return _temporal_conversions[f](*args).isoformat()
    else:
        ref = _convert_predicate(a.func.value)  # type: ignore[attr-defined]
        if f == "isin":
            return ref.isin(args[0])
        elif f == "is_null":
            return ref.isnull()
        elif f == "is_nan":
            return ref.isna()

    msg = f"Unknown call: {f!r}"
    raise ValueError(msg)


@_convert_predicate.register(Attribute)
def _(a: Attribute) -> Any:
    print("Attribute", a.attr, a.value)
    return a.attr


@_convert_predicate.register(BinOp)
def _(a: BinOp) -> Any:
    print("BinOp", a.op, a.left, a.right)
    lhs = _convert_predicate(a.left)
    rhs = _convert_predicate(a.right)

    op = a.op
    if isinstance(op, BitAnd):
        return lhs & rhs
    if isinstance(op, BitOr):
        return lhs | rhs
    else:
        msg = f"Unknown: {lhs} {op} {rhs}"
        raise TypeError(msg)


@_convert_predicate.register(Compare)
def _(a: Compare) -> Any:
    print("Compare", a.ops, a.left, a.comparators)
    # Compares returned by polars only ever contain a single comparison
    op = a.ops[0]
    left = a.left
    right = a.comparators[0]
    lhs = _convert_predicate(left)
    rhs = _convert_predicate(right)
    print("Strung op: ", str(op))

    # TODO: More options. Also consider using ExpressionNode.compose(str_op)
    if isinstance(op, Gt):
        return lhs > rhs
    if isinstance(op, GtE):
        return lhs >= rhs
    if isinstance(op, Eq):
        return lhs == rhs
    if isinstance(op, Lt):
        return lhs < rhs
    if isinstance(op, LtE):
        return lhs <= rhs
    else:
        msg = f"Unknown comparison: {op}"
        raise TypeError(msg)


@_convert_predicate.register(List)
def _(a: List) -> Any:
    return [_convert_predicate(e) for e in a.elts]
