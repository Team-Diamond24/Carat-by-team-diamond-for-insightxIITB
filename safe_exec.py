"""
safe_exec.py - Deterministic Safe Execution Layer for Pandas Operations

This module provides a sandboxed, deterministic way to execute common
pandas operations WITHOUT using exec() or eval(). Operations are
described as structured dictionaries (plans) and dispatched to
dedicated handler functions.

Supported operations:
  - "filter"  : Equality filter on a column
  - "groupby" : Group by a column with an aggregation function
  - "mean"    : Compute the mean of a single column

Usage:
    from safe_exec import execute_plan
    result = execute_plan({"operation": "mean", "column": "amount_inr"}, df)
"""

import pandas as pd
from typing import Any, Dict, Union


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------

def _handle_filter(plan: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where a column equals a given value.

    Plan keys:
        column (str): Column name to filter on.
        value  (Any): The value to match for equality.

    Returns:
        pd.DataFrame containing only the matching rows.

    Example plan:
        {
            "operation": "filter",
            "column": "transaction_status",
            "value": "SUCCESS"
        }
    """
    column = plan["column"]
    value = plan["value"]

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe. "
                         f"Available columns: {list(df.columns)}")

    return df[df[column] == value]


def _handle_groupby(plan: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """
    Group by a column and apply an aggregation function.

    Plan keys:
        group_by   (str): Column name to group by.
        agg_column (str): Column name to aggregate.
        agg_func   (str): Aggregation function name. One of:
                          "sum", "mean", "count", "min", "max",
                          "median", "std", "var", "nunique", "first", "last".

    Returns:
        pd.Series with the group labels as the index and aggregated values.

    Example plan:
        {
            "operation": "groupby",
            "group_by": "sender_bank",
            "agg_column": "amount_inr",
            "agg_func": "mean"
        }
    """
    ALLOWED_AGG_FUNCS = {
        "sum", "mean", "count", "min", "max",
        "median", "std", "var", "nunique", "first", "last",
    }

    group_by = plan["group_by"]
    agg_column = plan["agg_column"]
    agg_func = plan["agg_func"]

    if group_by not in df.columns:
        raise ValueError(f"Group-by column '{group_by}' not found. "
                         f"Available columns: {list(df.columns)}")
    if agg_column not in df.columns:
        raise ValueError(f"Aggregation column '{agg_column}' not found. "
                         f"Available columns: {list(df.columns)}")
    if agg_func not in ALLOWED_AGG_FUNCS:
        raise ValueError(f"Aggregation function '{agg_func}' is not allowed. "
                         f"Allowed functions: {sorted(ALLOWED_AGG_FUNCS)}")

    grouped = df.groupby(group_by)[agg_column]
    result = getattr(grouped, agg_func)()

    return result


def _handle_mean(plan: Dict[str, Any], df: pd.DataFrame) -> float:
    """
    Compute the mean of a single column.

    Plan keys:
        column (str): Column name to compute the mean of.

    Returns:
        float — the arithmetic mean of the column.

    Example plan:
        {
            "operation": "mean",
            "column": "amount_inr"
        }
    """
    column = plan["column"]

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe. "
                         f"Available columns: {list(df.columns)}")

    return float(df[column].mean())


# ---------------------------------------------------------------------------
# Operation dispatch registry
# ---------------------------------------------------------------------------

_OPERATION_HANDLERS = {
    "filter":  _handle_filter,
    "groupby": _handle_groupby,
    "mean":    _handle_mean,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_plan(
    plan: Dict[str, Any],
    df: pd.DataFrame,
) -> Union[pd.DataFrame, pd.Series, float]:
    """
    Execute a structured operation plan against a pandas DataFrame.

    This function acts as a safe, deterministic dispatcher. It does NOT
    use exec() or eval() at any point. Instead, it maps the "operation"
    key in the plan dictionary to a dedicated handler function.

    Args:
        plan: A dictionary describing the operation to perform.
              Must contain an "operation" key whose value is one of:
              "filter", "groupby", "mean".
              Additional keys depend on the operation (see individual
              handler docstrings above).
        df:   The pandas DataFrame to operate on.

    Returns:
        The result of the operation — a DataFrame, Series, or scalar,
        depending on the operation type.

    Raises:
        ValueError: If the operation is unknown or required keys are
                    missing / invalid.
        KeyError:   If required plan keys are absent.

    Examples:
        >>> execute_plan({"operation": "mean", "column": "amount_inr"}, df)
        1311.42

        >>> execute_plan({
        ...     "operation": "filter",
        ...     "column": "transaction_status",
        ...     "value": "FAILED"
        ... }, df)
           transaction_type  amount_inr transaction_status  ...
        3            P2P         1100              FAILED  ...

        >>> execute_plan({
        ...     "operation": "groupby",
        ...     "group_by": "sender_bank",
        ...     "agg_column": "amount_inr",
        ...     "agg_func": "sum"
        ... }, df)
        sender_bank
        Axis        12345678
        HDFC        23456789
        ...
        Name: amount_inr, dtype: int64
    """
    if not isinstance(plan, dict):
        raise TypeError(f"Plan must be a dict, got {type(plan).__name__}")

    if "operation" not in plan:
        raise KeyError("Plan dictionary must contain an 'operation' key. "
                       f"Received keys: {list(plan.keys())}")

    operation = plan["operation"]

    if operation not in _OPERATION_HANDLERS:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Supported operations: {sorted(_OPERATION_HANDLERS.keys())}"
        )

    handler = _OPERATION_HANDLERS[operation]
    return handler(plan, df)
