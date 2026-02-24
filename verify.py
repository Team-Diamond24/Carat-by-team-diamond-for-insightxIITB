"""
verify.py - Result Verification Layer

Provides basic sanity checks on the DataFrame used and the
result produced by the safe execution pipeline.

Usage:
    from verify import verify_result
    report = verify_result(df, result)
    # => {"rows_used": 250000, "status": "ok"}
"""

import pandas as pd
from typing import Any, Dict


def verify_result(df_used: pd.DataFrame, result: Any) -> Dict[str, Any]:
    """
    Verify that the DataFrame and result are valid.

    Checks:
        1. df_used is not empty.
        2. result is not None.

    Args:
        df_used: The pandas DataFrame that was operated on.
        result:  The output of the execution (DataFrame, Series, scalar, etc.).

    Returns:
        A dict with:
            - "rows_used" (int): Number of rows in df_used.
            - "status" (str): "ok" if all checks pass, "error" otherwise.
            - "error" (str, optional): Description of what failed.

    Examples:
        >>> verify_result(df, 1311.42)
        {"rows_used": 250000, "status": "ok"}

        >>> verify_result(pd.DataFrame(), None)
        {"rows_used": 0, "status": "error", "error": "DataFrame is empty; Result is None"}
    """
    errors = []

    if df_used is None or (isinstance(df_used, pd.DataFrame) and df_used.empty):
        errors.append("DataFrame is empty")

    if result is None:
        errors.append("Result is None")
    elif isinstance(result, float) and (result != result or result == float('inf') or result == float('-inf')):
        errors.append("Result is NaN or infinite")
    elif isinstance(result, pd.Series) and result.empty:
        errors.append("Result Series is empty")
    elif isinstance(result, pd.DataFrame) and result.empty:
        errors.append("Result DataFrame is empty")

    if errors:
        return {
            "rows_used": len(df_used) if df_used is not None else 0,
            "status": "error",
            "error": "; ".join(errors),
        }

    return {
        "rows_used": len(df_used),
        "status": "ok",
    }
