"""
tests/test_sql_analyst.py - Tests for the SQL analyst pipeline.

Run with:  pytest tests/test_sql_analyst.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sql_analyst import validate_sql, _clean_sql


class TestValidateSql:
    def test_select_is_valid(self):
        is_valid, err = validate_sql("SELECT * FROM upi_transactions_2024 LIMIT 10;")
        assert is_valid is True
        assert err == ""

    def test_with_cte_is_valid(self):
        sql = "WITH top AS (SELECT sender_bank FROM upi_transactions_2024) SELECT * FROM top;"
        is_valid, err = validate_sql(sql)
        assert is_valid is True

    def test_delete_is_rejected(self):
        is_valid, err = validate_sql("DELETE FROM upi_transactions_2024;")
        assert is_valid is False
        assert "DELETE" in err

    def test_drop_is_rejected(self):
        is_valid, err = validate_sql("DROP TABLE upi_transactions_2024;")
        assert is_valid is False

    def test_insert_is_rejected(self):
        is_valid, err = validate_sql("INSERT INTO upi_transactions_2024 VALUES ('x');")
        assert is_valid is False

    def test_update_is_rejected(self):
        is_valid, err = validate_sql("UPDATE upi_transactions_2024 SET amount_inr = 0;")
        assert is_valid is False

    def test_truncate_is_rejected(self):
        is_valid, err = validate_sql("TRUNCATE upi_transactions_2024;")
        assert is_valid is False

    def test_empty_is_rejected(self):
        is_valid, err = validate_sql("")
        assert is_valid is False


class TestCleanSql:
    def test_strips_markdown_fences(self):
        raw = "```sql\nSELECT * FROM t;\n```"
        assert _clean_sql(raw) == "SELECT * FROM t;"

    def test_strips_think_tags(self):
        raw = "<think>let me think...</think>SELECT 1;"
        assert _clean_sql(raw) == "SELECT 1;"

    def test_adds_semicolon(self):
        raw = "SELECT * FROM t"
        assert _clean_sql(raw).endswith(";")

    def test_preserves_valid_sql(self):
        raw = "SELECT sender_bank, COUNT(*) FROM upi_transactions_2024 GROUP BY sender_bank LIMIT 10;"
        assert _clean_sql(raw) == raw


class TestExecuteSql:
    """Integration tests — require a running PostgreSQL with the upi_transactions_2024 table."""

    @pytest.mark.skipif(
        not os.environ.get("PGPASSWORD"),
        reason="No PGPASSWORD set — skip DB tests"
    )
    def test_simple_select(self):
        from sql_analyst import execute_sql
        df = execute_sql("SELECT COUNT(*) AS total FROM upi_transactions_2024;")
        assert len(df) == 1
        assert df["total"].iloc[0] > 200000

    @pytest.mark.skipif(
        not os.environ.get("PGPASSWORD"),
        reason="No PGPASSWORD set — skip DB tests"
    )
    def test_rejects_delete(self):
        from sql_analyst import execute_sql
        with pytest.raises(ValueError, match="SQL validation failed"):
            execute_sql("DELETE FROM upi_transactions_2024;")
