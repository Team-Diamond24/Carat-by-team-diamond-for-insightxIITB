"""
tests/test_db.py - Tests for the PostgreSQL database access layer.

Requires a running PostgreSQL instance with the upi_transactions_2024 table
in the gearguard database. Set PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
environment variables before running.

Run with:  pytest tests/test_db.py -v
"""

import os
import sys
import pytest
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import get_connection, load_transactions


EXPECTED_MIN_ROWS = 200_000

EXPECTED_COLUMNS = sorted([
    "transaction_type", "amount_inr", "transaction_status",
    "sender_age_group", "sender_state", "sender_bank",
    "receiver_bank", "device_type", "network_type",
    "fraud_flag", "hour_of_day", "day_of_week",
    "is_weekend", "merchant_category",
])


class TestGetConnection:
    def test_returns_connection(self):
        conn = get_connection()
        assert conn is not None
        conn.close()

    def test_connection_is_usable(self):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
        cur.close()
        conn.close()


class TestLoadTransactions:
    def test_returns_dataframe(self):
        df = load_transactions()
        assert isinstance(df, pd.DataFrame)

    def test_has_rows(self):
        df = load_transactions()
        assert len(df) >= EXPECTED_MIN_ROWS, (
            f"Expected >= {EXPECTED_MIN_ROWS} rows, got {len(df)}"
        )

    def test_columns_match_after_normalization(self):
        df = load_transactions()
        # Apply the same normalization engine.py does
        df.columns = (df.columns
                      .str.strip()
                      .str.replace(' ', '_')
                      .str.replace('(', '')
                      .str.replace(')', '')
                      .str.lower())
        assert sorted(df.columns.tolist()) == EXPECTED_COLUMNS, (
            f"Column mismatch: {sorted(df.columns.tolist())}"
        )

    def test_with_limit(self):
        df = load_transactions(limit=10)
        assert len(df) == 10

    def test_with_columns(self):
        df = load_transactions(columns=["amount_inr", "sender_bank"])
        assert list(df.columns) == ["amount_inr", "sender_bank"]
        assert len(df) > 0
