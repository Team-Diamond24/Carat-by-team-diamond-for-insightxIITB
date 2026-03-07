"""
warmup_cache.py - Pre-Saved Query System with Keyword-Based Matching

At server startup, runs pre-defined SQL for 22+ common queries,
generates charts and data-driven summaries (zero LLM calls),
and stores complete responses. When a user asks a question,
`match_presaved_query()` uses keyword matching to find the best
pre-saved answer — so "which bank fails most?" and
"which bank has the most failures" both return the same instant result.
"""

import pandas as pd
from db import get_connection
from sql_analyst import generate_sql_chart
from shared_utils import generate_followup_questions

# -----------------------------------------------------------------------
# Storage for pre-computed responses  (question_id -> response dict)
# -----------------------------------------------------------------------
_PRESAVED_RESPONSES = {}


# -----------------------------------------------------------------------
# Pre-saved query definitions
# Each entry has:
#   id         : unique key
#   keywords   : list of keyword sets — question must match ALL keywords
#                in at least ONE set to be considered a match
#   exclude    : optional keywords that, if present, disqualify a match
#   sql        : pre-written SQL query
#   summary_tpl: template for the data-driven summary
# -----------------------------------------------------------------------

PRESAVED_QUERIES = [
    # ======================== QUICK QUERIES (Dashboard Sidebar) ========================
    {
        "id": "bank_fails_most",
        "keywords": [
            ["bank", "fail"],
            ["bank", "failure"],
            ["bank", "failed"],
        ],
        "exclude": ["rate"],
        "sql": """
            SELECT sender_bank AS "Bank",
                   COUNT(*) AS "Failed Transactions"
            FROM upi_transactions_2024
            WHERE transaction_status = 'FAILED'
            GROUP BY sender_bank
            ORDER BY "Failed Transactions" DESC
        """,
        "summary_tpl": (
            "{top} has the highest number of failed transactions at {top_val:,}, "
            "followed by {second} with {second_val:,}. {bottom} has the fewest failures "
            "at {bottom_val:,}. This analysis covers all failed transactions across 8 major banks."
        ),
    },
    {
        "id": "peak_transaction_hours",
        "keywords": [
            ["peak", "hour"],
            ["peak", "transaction"],
            ["busiest", "hour"],
        ],
        "exclude": ["food", "grocery", "entertainment", "shopping", "category"],
        "sql": """
            SELECT hour_of_day AS "Hour",
                   COUNT(*) AS "Transaction Count"
            FROM upi_transactions_2024
            GROUP BY hour_of_day
            ORDER BY "Transaction Count" DESC
        """,
        "summary_tpl": (
            "Peak transaction activity occurs at {top}:00 with {top_val:,} transactions. "
            "The second busiest hour is {second}:00 with {second_val:,}. "
            "Activity is lowest at {bottom}:00 with only {bottom_val:,} transactions."
        ),
    },
    {
        "id": "fraud_flag_patterns",
        "keywords": [
            ["fraud", "pattern"],
            ["fraud", "flag"],
            ["fraud", "category"],
        ],
        "exclude": ["state", "bank", "5000", "high", "above"],
        "sql": """
            SELECT merchant_category AS "Category",
                   ROUND(AVG(fraud_flag) * 100, 2) AS "Fraud Rate %"
            FROM upi_transactions_2024
            GROUP BY merchant_category
            ORDER BY "Fraud Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has the highest fraud rate at {top_val}%, "
            "while {bottom} has the lowest at {bottom_val}%. "
            "Fraud patterns vary significantly across merchant categories, "
            "with a spread of {spread:.2f} percentage points."
        ),
    },
    {
        "id": "top_states_transactions",
        "keywords": [
            ["state", "most", "transaction"],
            ["state", "top"],
            ["state", "send"],
            ["state", "highest", "volume"],
            ["top", "performing", "state"],
        ],
        "exclude": ["fail", "failure", "fraud", "18-25"],
        "sql": """
            SELECT sender_state AS "State",
                   COUNT(*) AS "Total Transactions"
            FROM upi_transactions_2024
            GROUP BY sender_state
            ORDER BY "Total Transactions" DESC
        """,
        "summary_tpl": (
            "{top} dominates with {top_val:,} transactions, "
            "followed by {second} at {second_val:,}. "
            "{bottom} has the fewest at {bottom_val:,}. "
            "The top 3 states account for a significant share of total UPI volume."
        ),
    },
    {
        "id": "weekend_vs_weekday",
        "keywords": [
            ["weekend", "weekday"],
            ["weekend", "trend"],
            ["weekend", "compare"],
            ["weekend", "vs"],
        ],
        "exclude": [],
        "sql": """
            SELECT CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS "Day Type",
                   COUNT(*) AS "Transaction Count",
                   ROUND(AVG(amount_inr), 2) AS "Avg Amount (INR)"
            FROM upi_transactions_2024
            GROUP BY is_weekend
            ORDER BY "Transaction Count" DESC
        """,
        "summary_tpl": (
            "Weekday transactions total {top_val:,} compared to weekend volumes. "
            "The data reveals distinct patterns in both volume and average amounts, "
            "reflecting different spending behaviors between weekdays and weekends."
        ),
    },
    {
        "id": "p2p_by_age_group",
        "keywords": [
            ["p2p", "age"],
            ["p2p", "transfer"],
            ["age", "group", "breakdown"],
            ["age", "group", "p2p"],
        ],
        "exclude": [],
        "sql": """
            SELECT sender_age_group AS "Age Group",
                   COUNT(*) AS "P2P Count"
            FROM upi_transactions_2024
            WHERE transaction_type = 'P2P'
            GROUP BY sender_age_group
            ORDER BY "P2P Count" DESC
        """,
        "summary_tpl": (
            "The {top} age group leads P2P transfers with {top_val:,} transactions, "
            "followed by {second} at {second_val:,}. "
            "The {bottom} age group has the fewest at {bottom_val:,}, "
            "reflecting higher digital adoption among younger demographics."
        ),
    },
    {
        "id": "anomaly_fraud",
        "keywords": [
            ["anomaly", "fraud"],
            ["anomaly", "pattern"],
        ],
        "exclude": [],
        "sql": """
            SELECT sender_bank AS "Bank",
                   ROUND(AVG(fraud_flag) * 100, 2) AS "Fraud Rate %",
                   SUM(fraud_flag) AS "Total Frauds",
                   COUNT(*) AS "Total Transactions"
            FROM upi_transactions_2024
            GROUP BY sender_bank
            ORDER BY "Fraud Rate %" DESC
        """,
        "summary_tpl": (
            "{top} shows the highest fraud rate at {top_val}%, "
            "while {bottom} has the lowest at {bottom_val}%. "
            "Anomaly patterns suggest certain banks have higher exposure to fraudulent activity."
        ),
    },
    {
        "id": "merchant_volume",
        "keywords": [
            ["merchant", "volume"],
            ["merchant", "highest"],
            ["merchant", "transaction"],
            ["category", "volume"],
            ["top", "merchant"],
        ],
        "exclude": ["fraud", "fail", "average", "amount"],
        "sql": """
            SELECT merchant_category AS "Category",
                   COUNT(*) AS "Transaction Volume"
            FROM upi_transactions_2024
            GROUP BY merchant_category
            ORDER BY "Transaction Volume" DESC
        """,
        "summary_tpl": (
            "{top} leads transaction volume with {top_val:,} transactions, "
            "followed by {second} at {second_val:,}. "
            "{bottom} has the least at {bottom_val:,}."
        ),
    },
    {
        "id": "state_failure_rates",
        "keywords": [
            ["state", "failure", "rate"],
            ["state", "highest", "failure"],
            ["regional", "risk"],
            ["state", "fail", "rate"],
        ],
        "exclude": [],
        "sql": """
            SELECT sender_state AS "State",
                   ROUND(SUM(CASE WHEN transaction_status='FAILED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS "Failure Rate %"
            FROM upi_transactions_2024
            GROUP BY sender_state
            ORDER BY "Failure Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has the highest failure rate at {top_val}%, "
            "while {bottom} has the lowest at {bottom_val}%. "
            "The {spread:.2f} percentage point spread may indicate regional infrastructure differences."
        ),
    },

    # ======================== FAST-PATH QUERIES (from engine.py) ========================
    {
        "id": "weekend_traffic_pct",
        "keywords": [
            ["weekend", "percent"],
            ["weekend", "percentage"],
            ["weekend", "traffic"],
        ],
        "exclude": ["weekday", "vs"],
        "sql": """
            SELECT ROUND(AVG(is_weekend) * 100, 2) AS "Weekend Traffic %"
            FROM upi_transactions_2024
        """,
        "summary_tpl": "Weekend transactions account for {top_val}% of all UPI transactions.",
    },
    {
        "id": "overall_fraud_rate",
        "keywords": [
            ["fraud", "rate"],
            ["overall", "fraud"],
            ["fraud", "percentage"],
        ],
        "exclude": ["state", "bank", "5000", "high", "above", "category", "device", "pattern", "flag", "anomaly"],
        "sql": """
            SELECT ROUND(AVG(fraud_flag) * 100, 2) AS "Overall Fraud Rate %"
            FROM upi_transactions_2024
        """,
        "summary_tpl": "The overall fraud rate is {top_val}% across all 250,000 UPI transactions.",
    },
    {
        "id": "high_value_fraud",
        "keywords": [
            ["fraud", "5000"],
            ["fraud", "high", "value"],
            ["fraud", "above"],
        ],
        "exclude": [],
        "sql": """
            SELECT ROUND(AVG(fraud_flag) * 100, 2) AS "High-Value Fraud Rate %"
            FROM upi_transactions_2024
            WHERE amount_inr > 5000
        """,
        "summary_tpl": (
            "{top_val}% of transactions above ₹5,000 are flagged for fraud. "
            "High-value transactions show elevated risk warranting closer monitoring."
        ),
    },
    {
        "id": "failure_rate_by_device",
        "keywords": [
            ["failure", "rate", "device"],
            ["fail", "device"],
            ["device", "type", "failure"],
        ],
        "exclude": [],
        "sql": """
            SELECT device_type AS "Device",
                   ROUND(SUM(CASE WHEN transaction_status='FAILED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS "Failure Rate %"
            FROM upi_transactions_2024
            GROUP BY device_type
            ORDER BY "Failure Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has the highest failure rate at {top_val}%, "
            "while {bottom} has the lowest at {bottom_val}%. "
            "Device type plays a role in transaction reliability."
        ),
    },
    {
        "id": "5g_vs_wifi_failure",
        "keywords": [
            ["5g", "wifi", "failure"],
            ["5g", "wifi"],
            ["5g", "fail"],
            ["wifi", "fail"],
            ["network", "failure"],
        ],
        "exclude": ["device", "volume"],
        "sql": """
            SELECT network_type AS "Network",
                   ROUND(SUM(CASE WHEN transaction_status='FAILED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS "Failure Rate %"
            FROM upi_transactions_2024
            WHERE network_type IN ('5G', 'WiFi')
            GROUP BY network_type
            ORDER BY "Failure Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has a failure rate of {top_val}%, while {bottom} is at {bottom_val}%. "
            "Network type influences transaction success rates."
        ),
    },
    {
        "id": "ios_vs_android_success",
        "keywords": [
            ["ios", "android"],
            ["success", "ios"],
            ["success", "android"],
            ["ios", "success", "rate"],
            ["compare", "ios", "android"],
        ],
        "exclude": [],
        "sql": """
            SELECT device_type AS "Device",
                   ROUND(SUM(CASE WHEN transaction_status='SUCCESS' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS "Success Rate %"
            FROM upi_transactions_2024
            WHERE device_type IN ('iOS', 'Android')
            GROUP BY device_type
            ORDER BY "Success Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has a success rate of {top_val}%, compared to {bottom} at {bottom_val}%. "
            "Both platforms show strong reliability for UPI transactions."
        ),
    },
    {
        "id": "4g_vs_wifi_volume",
        "keywords": [
            ["4g", "wifi", "volume"],
            ["4g", "broadband"],
            ["4g", "volume"],
            ["network", "volume"],
            ["network", "type", "transaction"],
        ],
        "exclude": ["failure", "fail"],
        "sql": """
            SELECT network_type AS "Network",
                   COUNT(*) AS "Transaction Volume"
            FROM upi_transactions_2024
            GROUP BY network_type
            ORDER BY "Transaction Volume" DESC
        """,
        "summary_tpl": (
            "{top} leads in transaction volume with {top_val:,} transactions. "
            "{bottom} has the least at {bottom_val:,}. "
            "Network distribution shows clear preferences among UPI users."
        ),
    },
    {
        "id": "state_18_25_age",
        "keywords": [
            ["state", "18-25"],
            ["state", "18 to 25"],
            ["young", "state"],
        ],
        "exclude": [],
        "sql": """
            SELECT sender_state AS "State",
                   COUNT(*) AS "Transactions (18-25)"
            FROM upi_transactions_2024
            WHERE sender_age_group = '18-25'
            GROUP BY sender_state
            ORDER BY "Transactions (18-25)" DESC
        """,
        "summary_tpl": (
            "{top} leads with {top_val:,} transactions from the 18-25 age group, "
            "followed by {second} at {second_val:,}. "
            "{bottom} has the fewest young user transactions at {bottom_val:,}."
        ),
    },
    {
        "id": "avg_amount_by_category",
        "keywords": [
            ["average", "amount", "category"],
            ["avg", "amount", "category"],
            ["average", "transaction", "category"],
            ["average", "amount", "merchant"],
            ["mean", "amount"],
        ],
        "exclude": [],
        "sql": """
            SELECT merchant_category AS "Category",
                   ROUND(AVG(amount_inr), 2) AS "Avg Amount (INR)"
            FROM upi_transactions_2024
            GROUP BY merchant_category
            ORDER BY "Avg Amount (INR)" DESC
        """,
        "summary_tpl": (
            "{top} has the highest average transaction amount at ₹{top_val:,.2f}, "
            "while {bottom} has the lowest at ₹{bottom_val:,.2f}. "
            "Average amounts vary by ₹{spread:,.2f} across categories."
        ),
    },
    {
        "id": "peak_hours_all_categories",
        "keywords": [
            ["peak", "hour", "category"],
            ["peak", "hour", "all"],
            ["busiest", "hour", "category"],
        ],
        "exclude": [],
        "sql": """
            SELECT merchant_category AS "Category",
                   (SELECT hour_of_day FROM upi_transactions_2024 t2
                    WHERE t2.merchant_category = t1.merchant_category
                    GROUP BY hour_of_day ORDER BY COUNT(*) DESC LIMIT 1) AS "Peak Hour"
            FROM upi_transactions_2024 t1
            GROUP BY merchant_category
            ORDER BY merchant_category
        """,
        "summary_tpl": (
            "Peak transaction hours vary by merchant category. "
            "{top} peaks at hour {top_val}, while other categories show diverse peak times."
        ),
    },

    # ======================== ADDITIONAL COMMON QUERIES ========================
    {
        "id": "total_transactions",
        "keywords": [
            ["total", "transaction"],
            ["how", "many", "transaction"],
            ["count", "transaction"],
            ["number", "transaction"],
        ],
        "exclude": ["fail", "success", "state", "bank", "device"],
        "sql": "SELECT COUNT(*) AS \"Total Transactions\" FROM upi_transactions_2024",
        "summary_tpl": "There are {top_val:,} total UPI transactions in the 2024 dataset.",
    },
    {
        "id": "success_rate_overall",
        "keywords": [
            ["success", "rate"],
            ["overall", "success"],
        ],
        "exclude": ["ios", "android", "device", "bank", "state"],
        "sql": """
            SELECT ROUND(SUM(CASE WHEN transaction_status='SUCCESS' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                   AS "Success Rate %"
            FROM upi_transactions_2024
        """,
        "summary_tpl": "The overall success rate is {top_val}% across all 250,000 UPI transactions.",
    },
    {
        "id": "transaction_type_breakdown",
        "keywords": [
            ["transaction", "type", "breakdown"],
            ["transaction", "type", "distribution"],
            ["p2p", "p2m", "bill"],
            ["type", "transaction", "count"],
        ],
        "exclude": [],
        "sql": """
            SELECT transaction_type AS "Type",
                   COUNT(*) AS "Count"
            FROM upi_transactions_2024
            GROUP BY transaction_type
            ORDER BY "Count" DESC
        """,
        "summary_tpl": (
            "{top} is the most common transaction type with {top_val:,} transactions, "
            "followed by {second} at {second_val:,}. "
            "{bottom} has the least at {bottom_val:,}."
        ),
    },
    {
        "id": "bank_failure_rate",
        "keywords": [
            ["bank", "failure", "rate"],
            ["bank", "fail", "rate"],
            ["failure", "rate", "bank"],
        ],
        "exclude": [],
        "sql": """
            SELECT sender_bank AS "Bank",
                   ROUND(SUM(CASE WHEN transaction_status='FAILED' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS "Failure Rate %"
            FROM upi_transactions_2024
            GROUP BY sender_bank
            ORDER BY "Failure Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has the highest failure rate at {top_val}%, "
            "while {bottom} has the lowest at {bottom_val}%. "
            "The spread of {spread:.2f} percentage points highlights reliability differences across banks."
        ),
    },
    {
        "id": "fraud_by_state",
        "keywords": [
            ["fraud", "state"],
            ["fraud", "rate", "state"],
        ],
        "exclude": [],
        "sql": """
            SELECT sender_state AS "State",
                   ROUND(AVG(fraud_flag) * 100, 2) AS "Fraud Rate %"
            FROM upi_transactions_2024
            GROUP BY sender_state
            ORDER BY "Fraud Rate %" DESC
        """,
        "summary_tpl": (
            "{top} has the highest fraud rate at {top_val}%, "
            "{bottom} has the lowest at {bottom_val}%. "
            "Regional fraud patterns suggest geographic risk concentrations."
        ),
    },
    {
        "id": "avg_transaction_amount",
        "keywords": [
            ["average", "transaction", "amount"],
            ["avg", "transaction"],
            ["mean", "transaction", "amount"],
            ["overall", "average", "amount"],
        ],
        "exclude": ["category", "merchant", "bank", "state"],
        "sql": """
            SELECT ROUND(AVG(amount_inr), 2) AS "Average Amount (INR)"
            FROM upi_transactions_2024
        """,
        "summary_tpl": "The average UPI transaction amount is ₹{top_val:,.2f} across 250,000 transactions.",
    },
    {
        "id": "day_of_week_trend",
        "keywords": [
            ["day", "week", "trend"],
            ["daily", "trend"],
            ["day", "week", "transaction"],
            ["monday", "sunday"],
            ["busiest", "day"],
        ],
        "exclude": [],
        "sql": """
            SELECT day_of_week AS "Day",
                   COUNT(*) AS "Transaction Count"
            FROM upi_transactions_2024
            GROUP BY day_of_week
            ORDER BY "Transaction Count" DESC
        """,
        "summary_tpl": (
            "{top} is the busiest day with {top_val:,} transactions, "
            "while {bottom} is the quietest at {bottom_val:,}. "
            "Transaction patterns show clear day-of-week preferences."
        ),
    },
]


# -----------------------------------------------------------------------
# Keyword-based matching engine
# -----------------------------------------------------------------------

def match_presaved_query(question: str):
    """
    Match a user question against pre-saved query keyword patterns.
    Returns the pre-computed response dict or None if no match.

    Matching logic:
    - The question is lowercased and split into words
    - Each pre-saved query has one or more keyword sets
    - A match occurs if ALL keywords in at least ONE set are found in the question
    - Exclude keywords disqualify a match (prevents overlapping queries)
    - If multiple queries match, the one with the most specific (longest) keyword set wins
    """
    q = question.lower()
    q_words = set(q.split())

    best_match_id = None
    best_match_score = 0  # number of keywords in the matching set

    for entry in PRESAVED_QUERIES:
        query_id = entry["id"]
        exclude = entry.get("exclude", [])

        # Check exclusions first
        if any(ex in q for ex in exclude):
            continue

        # Check keyword sets
        for kw_set in entry["keywords"]:
            if all(kw in q for kw in kw_set):
                score = len(kw_set)
                if score > best_match_score:
                    best_match_score = score
                    best_match_id = query_id

    if best_match_id and best_match_id in _PRESAVED_RESPONSES:
        print(f"[FAST-PATH] Matched '{question}' → {best_match_id}")
        return _PRESAVED_RESPONSES[best_match_id]

    return None


# -----------------------------------------------------------------------
# Summary builder
# -----------------------------------------------------------------------

def _build_summary(template: str, df: pd.DataFrame) -> str:
    """Fill a summary template with top/bottom values from the query result."""
    if df is None or len(df) == 0:
        return "No data available."

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_cols:
        return f"Query returned {len(df)} rows."

    val_col = num_cols[0]

    # For single-row scalar results
    if len(df) == 1:
        top_val = df[val_col].iloc[0]
        try:
            return template.format(
                top=cat_cols[0] if cat_cols else "Result",
                top_val=top_val,
                second="", second_val=0,
                bottom="", bottom_val=0,
                spread=0,
            )
        except (KeyError, IndexError, ValueError):
            return f"The result is {top_val}."

    if not cat_cols:
        return f"Query returned {len(df)} rows."

    cat_col = cat_cols[0]
    top_row = df.iloc[0]
    bottom_row = df.iloc[-1]
    second_row = df.iloc[1] if len(df) > 1 else top_row

    try:
        return template.format(
            top=top_row[cat_col],
            top_val=top_row[val_col],
            second=second_row[cat_col],
            second_val=second_row[val_col],
            bottom=bottom_row[cat_col],
            bottom_val=bottom_row[val_col],
            spread=abs(float(top_row[val_col]) - float(bottom_row[val_col])),
        )
    except (KeyError, IndexError, ValueError):
        return f"Query returned {len(df)} rows across {len(df[cat_col].unique())} categories."


# -----------------------------------------------------------------------
# Startup warmup — pre-compute all responses
# -----------------------------------------------------------------------

def warmup_cache():
    """Pre-compute and store all pre-saved query results. Called at server startup."""
    print("[WARMUP] Pre-computing fast-path query responses...")
    success = 0

    conn = get_connection()
    try:
        for entry in PRESAVED_QUERIES:
            qid = entry["id"]
            sql = entry["sql"].strip()
            tpl = entry["summary_tpl"]

            try:
                df = pd.read_sql(sql.rstrip(";"), conn)

                if df is None or len(df) == 0:
                    print(f"  [SKIP] {qid} — empty result")
                    continue

                chart = generate_sql_chart(df, qid.replace("_", " "))
                summary = _build_summary(tpl, df)
                result_str = df.to_string(index=False)

                num_cols = df.select_dtypes(include="number").columns.tolist()
                stats = [{"label": "ROWS", "value": str(len(df))}]
                if num_cols:
                    col = num_cols[0]
                    stats.insert(0, {"label": f"MAX {col.upper()}", "value": f"{df[col].max():.2f}"})

                _PRESAVED_RESPONSES[qid] = {
                    "summary": summary,
                    "answer": summary,
                    "headline": "SQL Database Insight",
                    "stats": stats,
                    "result": result_str,
                    "data": result_str,
                    "sql": sql,
                    "code": sql,
                    "chart": chart,
                    "rows_returned": len(df),
                    "followups": generate_followup_questions(qid.replace("_", " ")),
                    "verification": {
                        "valid": True,
                        "warnings": [],
                        "errors": [],
                        "model_used": "pre-saved (no LLM)",
                        "fallback_used": False,
                    },
                }
                success += 1

            except Exception as e:
                print(f"  [ERROR] {qid}: {e}")
    finally:
        conn.close()

    print(f"[WARMUP] Pre-saved {success}/{len(PRESAVED_QUERIES)} query responses ✓")
    return success
