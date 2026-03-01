"""
db.py - Centralized SQLite Access Layer

Manages SQLite connections and exposes a function
to load the upi_transactions_2024 table as a pandas DataFrame.

The database path is read from environment variables:
    SQLITE_DB_PATH (default: carat.db)

Usage:
    from db import load_transactions
    df = load_transactions()
"""

import os
import sqlite3
import pandas as pd
from typing import Optional


def get_connection():
    """
    Open and return a new sqlite3 connection.

    Environment variables used:
        SQLITE_DB_PATH (default: carat.db)

    Returns:
        sqlite3 connection object.
    """
    db_path = os.environ.get("SQLITE_DB_PATH", "carat.db")
    return sqlite3.connect(db_path)


def load_transactions(
    columns: Optional[list] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the upi_transactions_2024 table into a pandas DataFrame.

    Args:
        columns: Optional list of column names to SELECT.
                 If None, selects all columns (*).
        limit:   Optional row limit. If None, loads all rows.

    Returns:
        pd.DataFrame with the requested data.
    """
    conn = get_connection()
    try:
        # Validate column names against actual table columns to prevent injection
        if columns:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info(upi_transactions_2024)")
            valid_cols = {row[1] for row in cur.fetchall()}
            cur.close()
            for c in columns:
                if c not in valid_cols:
                    raise ValueError(f"Invalid column '{c}'. Valid columns: {sorted(valid_cols)}")

        col_clause = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_clause} FROM upi_transactions_2024"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    return df
