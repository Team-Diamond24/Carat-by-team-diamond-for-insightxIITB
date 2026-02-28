"""
db.py - Centralized PostgreSQL Access Layer

Manages PostgreSQL connections via psycopg2 and exposes a function
to load the upi_transactions_2024 table as a pandas DataFrame.

Credentials are read from environment variables:
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Usage:
    from db import load_transactions
    df = load_transactions()
    df = load_transactions(columns=["amount_inr", "sender_bank"], limit=100)
"""

import os
import psycopg2
import pandas as pd
from typing import Optional


def get_connection():
    """
    Open and return a new psycopg2 connection using environment variables.

    Environment variables used:
        PGHOST     (default: localhost)
        PGPORT     (default: 5432)
        PGDATABASE (default: gearguard)
        PGUSER     (default: postgres)
        PGPASSWORD (no default — must be set)

    Returns:
        psycopg2 connection object.

    Raises:
        psycopg2.OperationalError: If the database is unreachable.
    """
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", 5432)),
        database=os.environ.get("PGDATABASE", "gearguard"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD"),
    )


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

    Raises:
        psycopg2.OperationalError: If the database is unreachable.
        psycopg2.ProgrammingError: If the table or columns do not exist.
    """
    conn = get_connection()
    try:
        # Validate column names against actual table columns to prevent injection
        if columns:
            cur = conn.cursor()
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'upi_transactions_2024'"
            )
            valid_cols = {row[0] for row in cur.fetchall()}
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
