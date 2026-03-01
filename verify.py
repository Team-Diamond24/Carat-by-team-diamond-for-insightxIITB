import pandas as pd
from typing import Any, Dict

def verify_result(df_used: pd.DataFrame, plan: dict, result: Any) -> dict:
    """
    Returns: { "valid": bool, "warnings": list[str], "errors": list[str] }
    """
    warnings = []
    errors = []

    # 1. DataFrame is not None and not empty -> error if so
    if result is None:
        errors.append("Result is None")
    elif isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
        errors.append("Result DataFrame/Series is empty")
    elif isinstance(result, pd.DataFrame) and len(result.columns) == 0:
        errors.append("Result DataFrame has no columns")

    # 2. All numeric result columns are not entirely null
    if isinstance(result, pd.DataFrame):
        numeric_cols = result.select_dtypes(include="number").columns
        for col in numeric_cols:
            if result[col].isnull().all():
                errors.append(f"Numeric column '{col}' is entirely null")
    elif isinstance(result, pd.Series):
        if pd.api.types.is_numeric_dtype(result) and result.isnull().all():
            errors.append("Numeric Series is entirely null")

    # 3. If plan had a filter, verify the filtered column values actually match
    if isinstance(result, pd.DataFrame) and not result.empty:
        # Check single filter
        if plan.get("operation") == "filter" or plan.get("op") == "filter":
            col = plan.get("column")
            val = plan.get("value")
            if col and col in result.columns:
                # spot check first 5 rows
                head_vals = result[col].head(5)
                # allowing type coercion or case-insensitive matches roughly
                if not all(str(v).lower() == str(val).lower() for v in head_vals):
                     warnings.append(f"Filter mismatch: expected '{val}' in column '{col}'")
        
        # Check multi filter
        if plan.get("operation") == "filter_multi" or plan.get("op") == "filter_multi":
            filters = plan.get("filters", [])
            for f in filters:
                col = f.get("column")
                val = f.get("value")
                if col and col in result.columns:
                    head_vals = result[col].head(5)
                    if not all(str(v).lower() == str(val).lower() for v in head_vals):
                         warnings.append(f"Multiple filter mismatch: expected '{val}' in column '{col}'")

    # 4. If plan was a groupby, verify the groupby column exists in result
    if plan.get("operation") == "groupby" or plan.get("op") == "groupby":
        # Ensure we have more than 1 unique value representing groups
        # If result is Series (like df.groupby('col')['agg'].mean()), the index is the groups
        if isinstance(result, pd.Series):
            if len(result.index.unique()) <= 1:
                warnings.append("Groupby result has 1 or fewer unique groups returned")
        elif isinstance(result, pd.DataFrame):
            group_by_col = plan.get("group_by") or (plan.get("by") if isinstance(plan.get("by"), str) else plan.get("by", [None])[0])
            if group_by_col and group_by_col in result.columns:
                 if result[group_by_col].nunique() <= 1:
                     warnings.append("Groupby result has 1 or fewer unique groups returned")
            elif len(result.index.unique()) <= 1:
                 warnings.append("Groupby result has 1 or fewer unique groups returned")

    # 5. Row count sanity: if result has > 10,000 rows, add a warning
    if isinstance(result, (pd.DataFrame, pd.Series)) and len(result) > 10000:
        warnings.append("Result may be too large to display meaningfully")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors
    }
