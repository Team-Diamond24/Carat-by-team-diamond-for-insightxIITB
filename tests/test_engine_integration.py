"""
tests/test_engine_integration.py - Integration tests for engine.py with PostgreSQL backend.

Verifies that the engine module loads data correctly and
the core get_full_answer pipeline produces valid responses.

Run with:  pytest tests/test_engine_integration.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEngineDataLoading:
    def test_df_is_loaded(self):
        from engine import df
        assert df is not None
        assert len(df) > 200_000

    def test_df_has_expected_columns(self):
        from engine import df
        expected = [
            "transaction_type", "amount_inr", "transaction_status",
            "sender_age_group", "sender_state", "sender_bank",
            "receiver_bank", "device_type", "network_type",
            "fraud_flag", "hour_of_day", "day_of_week",
            "is_weekend", "merchant_category",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"


class TestGetFullAnswer:
    def test_fast_path_avg_amount(self):
        from engine import get_full_answer
        result = get_full_answer("What is the average transaction amount?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "headline" in result
        assert "stats" in result

    def test_fast_path_bank_volume(self):
        from engine import get_full_answer
        result = get_full_answer("Compare transactions by bank")
        assert isinstance(result, dict)
        assert result.get("headline") is not None


class TestStatsEndpoint:
    def test_stats_returns_valid_json(self):
        from server import app
        client = app.test_client()
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_transactions" in data
        assert "success_rate" in data
        assert "avg_amount" in data
