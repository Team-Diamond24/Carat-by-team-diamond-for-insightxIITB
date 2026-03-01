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

    result = df[df[column] == value]

    # Case-insensitive fallback for string values
    if len(result) == 0 and isinstance(value, str):
        result = df[df[column].astype(str).str.lower() == value.lower()]

    return result


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
        "size", "idxmax", "idxmin"
    }

    group_by = plan["group_by"]
    agg_column = plan["agg_column"]
    agg_func = plan["agg_func"]

    if isinstance(group_by, list):
        for col in group_by:
            if col not in df.columns:
                raise ValueError(f"Group-by column '{col}' not found. Available: {list(df.columns)}")
    else:
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
    
    # Flatten MultiIndex to simple strings so downstream charting doesn't break
    if isinstance(result.index, pd.MultiIndex):
         result.index = result.index.map(lambda x: ' - '.join(str(e) for e in x))

    return result


def _handle_mean(plan: Dict[str, Any], df: pd.DataFrame) -> float:
    """Compute the mean of a single column."""
    column = plan["column"]
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    return float(df[column].mean())


def _handle_single_agg(func_name: str):
    """Factory for single-column aggregation handlers (sum, count, max, min, etc.)."""
    def handler(plan: Dict[str, Any], df: pd.DataFrame):
        column = plan["column"]
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        return getattr(df[column], func_name)()
    handler.__doc__ = f"Compute {func_name} of a single column."
    return handler


def _handle_filter_multi(plan: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows for multiple conditions using AND logic.

    Plan keys:
        filters (list of dict): List of feature dictionaries with 'column' and 'value'.
    """
    filters = plan.get("filters", [])
    if not filters:
        return df
        
    mask = pd.Series(True, index=df.index)
    for f in filters:
        column = f["column"]
        value = f["value"]
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        mask &= (df[column] == value)
        
    result = df[mask]
    
    # Case insensitive fallback
    if len(result) == 0:
        mask_fallback = pd.Series(True, index=df.index)
        for f in filters:
             column = f["column"]
             value = f["value"]
             if isinstance(value, str):
                  mask_fallback &= (df[column].astype(str).str.lower() == value.lower())
             else:
                  mask_fallback &= (df[column] == value)
        result = df[mask_fallback]
        
    return result

# ---------------------------------------------------------------------------
# Operation dispatch registry
# ---------------------------------------------------------------------------

_OPERATION_HANDLERS = {
    "filter":  _handle_filter,
    "filter_multi": _handle_filter_multi,
    "groupby": _handle_groupby,
    "mean":    _handle_mean,
    "sum":     _handle_single_agg("sum"),
    "count":   _handle_single_agg("count"),
    "max":     _handle_single_agg("max"),
    "min":     _handle_single_agg("min"),
    "median":  _handle_single_agg("median"),
    "std":     _handle_single_agg("std"),
    "var":     _handle_single_agg("var"),
    "nunique": _handle_single_agg("nunique"),
    "size":    _handle_single_agg("size"),
    "idxmax":  _handle_single_agg("idxmax"),
    "idxmin":  _handle_single_agg("idxmin"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_plan(
    plan: Dict[str, Any],
    df: pd.DataFrame,
) -> Union[pd.DataFrame, pd.Series, float]:
    """Execute a structured operation plan against a pandas DataFrame."""
    if not isinstance(plan, dict):
        raise TypeError(f"Plan must be a dict, got {type(plan).__name__}")

    # Accept both 'op' (from planner) and 'operation' as the key
    if "op" in plan and "operation" not in plan:
        plan = dict(plan, operation=plan["op"])

    # Translate planner's groupby format to safe_exec's expected keys
    if plan.get("operation") == "groupby" and "by" in plan and "agg" in plan:
        by_cols = plan["by"]
        agg_dict = plan["agg"]
        first_col = list(agg_dict.keys())[0]
        first_func = list(agg_dict.values())[0]
        plan = dict(plan,
            group_by=by_cols,  # Keep the list untouched
            agg_column=first_col,
            agg_func=first_func
        )

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

