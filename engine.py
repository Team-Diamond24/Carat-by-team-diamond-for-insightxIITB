import pandas as pd
from openai import OpenAI
import time
import plotly.express as px
import plotly.utils
import json
from planner import code_to_plan
from safe_exec import execute_plan
from verify import verify_result

import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    timeout=30.0
)

USE_CSV = os.environ.get("USE_CSV", "false").lower() == "true"

if USE_CSV:
    df = pd.read_csv("upi_transactions_2024.csv")
    print("Dataset loaded from CSV.")
else:
    from db import load_transactions
    df = load_transactions()
    print("Dataset loaded from PostgreSQL.")

df.columns = (df.columns
              .str.strip()
              .str.replace(' ', '_')
              .str.replace('(', '')
              .str.replace(')', '')
              .str.lower())

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)


def call_ai(prompt):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"AI call error: {str(e)}")
                return "Error: " + str(e)
    return "Error: Too many retries"


def ask_question(user_question, chat_history=None):
    history_context = ""
    if chat_history:
        recent = chat_history[-4:]  # last 2 turns
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_context += f"{role}: {content}\n"
        history_context = f"Recent conversation:\n{history_context}\n"

    # Build column info dynamically from the actual DataFrame
    col_list = ", ".join(df.columns.tolist())
    row_count = len(df)

    # Build sample values for key categorical columns
    value_hints = []
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() <= 15:
            vals = ", ".join(str(v) for v in df[col].unique()[:10])
            value_hints.append(f"{col} ({vals})")
        elif col in ("fraud_flag", "is_weekend"):
            value_hints.append(f"{col} (0 or 1)")
        elif col in ("hour_of_day",):
            value_hints.append(f"{col} (0-23)")
        elif col in ("amount_inr",):
            value_hints.append(f"{col} (numeric, INR)")
        else:
            value_hints.append(col)
    col_detail = "\n".join(value_hints)

    prompt = (
        f"You are a data analyst. You have a pandas dataframe called 'df' with {row_count:,} UPI transactions.\n\n"
        f"DataFrame df columns and values:\n{col_detail}\n\n"
        + history_context +
        "Write ONLY executable pandas code to answer this question.\n"
        "Store the final answer in a variable called 'result'.\n"
        "If result is a comparison or ranking, store as a pandas Series with named index.\n"
        "Do not include any imports. Do not use print(). No markdown. No explanation.\n"
        "Just raw Python code that can be executed with exec().\n\n"
        "Question: " + user_question
    )
    raw = call_ai(prompt)
    print("Raw AI code response:", raw)
    return raw


def clean_code(raw_code):
    code = raw_code
    code = code.replace("```python", "")
    code = code.replace("```", "")
    code = code.strip()
    return code


def generate_chart(result, user_question):
    try:
        if result is None:
            return None

        # --- Handle dict → convert to Series first ---
        if isinstance(result, dict):
            result = pd.Series(result)

        # --- Handle pd.Series → bar chart directly ---
        if isinstance(result, pd.Series):
            chart_df = result.reset_index()
            chart_df.columns = ["Category", "Value"]
            x_label = result.index.name or "Category"
            y_label = result.name or "Value"

            chart_df = chart_df.sort_values("Value", ascending=False)
            chart_df["Category"] = chart_df["Category"].astype(str)
            fig = px.bar(
                chart_df, x="Value", y="Category",
                title=f"{y_label} by {x_label}", orientation="h",
                labels={"Category": x_label, "Value": y_label},
                color_discrete_sequence=["#b1b2ff"]
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))

        # --- Handle pd.DataFrame ---
        elif isinstance(result, pd.DataFrame) and len(result) > 0:
            numeric_cols = result.select_dtypes(include="number").columns.tolist()
            non_numeric_cols = result.select_dtypes(exclude="number").columns.tolist()

            # 4. Two numeric columns → scatter
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(
                    result, x=x_col, y=y_col,
                    title=user_question,
                    labels={x_col: x_col, y_col: y_col},
                    color_discrete_sequence=["#b1b2ff"]
                )

            # 5. One numeric column only → histogram
            elif len(numeric_cols) == 1:
                col = numeric_cols[0]
                fig = px.histogram(
                    result, x=col,
                    title=user_question,
                    labels={col: col},
                    color_discrete_sequence=["#b1b2ff"]
                )

            # 6. One categorical + one numeric → bar
            elif len(non_numeric_cols) >= 1 and len(numeric_cols) >= 1:
                fig = px.bar(
                    result, x=non_numeric_cols[0], y=numeric_cols[0],
                    title=user_question,
                    labels={non_numeric_cols[0]: non_numeric_cols[0], numeric_cols[0]: numeric_cols[0]},
                    color_discrete_sequence=["#b1b2ff"]
                )
            else:
                return None
        else:
            return None

        # Common layout
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#3730a3", family="JetBrains Mono"),
            title_font_size=12,
            margin=dict(t=30, r=20, b=40, l=120)
        )

        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

    except Exception as e:
        print(f"Chart generation error: {e}")
        return None


def generate_followup_questions(user_question, result):
    """Context-aware followup suggestions — no LLM call."""
    q = user_question.lower()
    followups = []

    # Context-specific suggestions
    if "bank" in q:
        followups.append("Compare failure rates across banks")
    if "state" in q:
        followups.append("Which state has the highest fraud rate?")
    if "fraud" in q or "flagged" in q:
        followups.append("Which banks have the most fraud?")
    if "age" in q:
        followups.append("Average amount by age group")
    if "device" in q or "android" in q or "ios" in q:
        followups.append("Network type usage by device")
    if "weekend" in q:
        followups.append("Peak hours on weekends vs weekdays")

    # Pad with generic options if needed
    generic = [
        "Break this down by state",
        "Compare by device type",
        "Show the trend over time",
        "What is the fraud rate for this?",
    ]
    for g in generic:
        if len(followups) >= 3:
            break
        if g not in followups:
            followups.append(g)

    return followups[:3]


def generate_insight(user_question, result):
    """Context-aware insight — detects result type and formats accordingly."""
    rows = len(df)
    q = user_question.lower()

    # Detect if the result represents a percentage
    is_pct = any(w in q for w in ["rate", "percent", "percentage", "ratio", "proportion", "fraud", "success", "failure"])
    # Detect if the result represents a count
    is_count = any(w in q for w in ["count", "total", "how many", "number of", "volume"])

    if isinstance(result, pd.Series) and len(result) > 0:
        top_idx = result.idxmax()
        top_val = result.max()
        bot_idx = result.idxmin()
        bot_val = result.min()

        if is_pct:
            headline = f"{top_idx}: {float(top_val):.2f}%"
            insight = f"{top_idx} leads at {float(top_val):.2f}%, while {bot_idx} is lowest at {float(bot_val):.2f}%. Based on {rows:,} transactions."
        elif is_count:
            headline = f"{top_idx}: {int(top_val):,}"
            insight = f"{top_idx} has the highest count with {int(top_val):,}. {bot_idx} has the least with {int(bot_val):,}. Based on {rows:,} transactions."
        else:
            headline = f"{top_idx}: ₹{float(top_val):,.2f}"
            insight = f"{top_idx} leads with ₹{float(top_val):,.2f}. {bot_idx} is lowest at ₹{float(bot_val):,.2f}. Based on {rows:,} transactions."

    elif isinstance(result, pd.DataFrame):
        headline = f"{len(result):,} rows computed"
        insight = f"Returned {len(result):,} rows from {rows:,} total transactions."

    elif isinstance(result, (int, float)):
        val = float(result)
        if is_pct:
            headline = f"{val:.2f}%"
            insight = f"The result is {val:.2f}%. Based on {rows:,} transactions."
        elif is_count:
            headline = f"{int(val):,}"
            insight = f"The count is {int(val):,}. Based on {rows:,} transactions."
        else:
            headline = f"₹{val:,.2f}"
            insight = f"The value is ₹{val:,.2f}. Based on {rows:,} transactions."
    else:
        headline = str(result)[:60]
        insight = f"Based on {rows:,} transactions."

    return headline, insight


def _build_fast_response(result, question, headline, insight):
    """Helper to build a standard response dict from a fast-path result."""
    chart = generate_chart(result, question)
    verification = verify_result(df, result)
    followups = generate_followup_questions(question, result)
    stats = []
    if isinstance(result, pd.Series) and len(result) > 0:
        stats = [
            {"label": "HIGHEST", "value": str(result.index[0]) + ": " + str(round(float(result.iloc[0]), 2))},
            {"label": "AVERAGE", "value": str(round(float(result.mean()), 2))},
            {"label": "LOWEST", "value": str(result.index[-1]) + ": " + str(round(float(result.iloc[-1]), 2))}
        ]
    elif result is not None and not isinstance(result, pd.Series):
        stats = [{"label": "RESULT", "value": str(result)}]
    return {
        "answer": insight,
        "headline": headline,
        "stats": stats,
        "data": str(result),
        "code": "# fast path — no LLM",
        "chart": chart,
        "followups": followups,
        "verification": verification,
        "skip_llm": True
    }


def try_fast_query(question, df):
    """Deterministic fast path — skips LLM for common queries.
    
    PATTERN ORDER: most-specific first, least-specific last.
    Cross-dimension queries (fraud×state) before single-dimension (top state).
    """
    q = question.lower()

    # ===== CROSS-DIMENSION: fraud/failure × dimension =====

    # 1) Fraud by state
    if any(w in q for w in ["fraud", "flagged"]) and "state" in q:
        fraud_by_state = df.groupby("sender_state")["fraud_flag"].mean().round(4) * 100
        fraud_by_state = fraud_by_state.sort_values(ascending=False)
        return _build_fast_response(
            fraud_by_state, question,
            f"{fraud_by_state.index[0]} has highest fraud",
            f"{fraud_by_state.index[0]} has the highest fraud rate at {fraud_by_state.iloc[0]:.2f}%. Based on {len(df):,} transactions."
        )

    # 2) Fraud by bank
    if any(w in q for w in ["fraud", "flagged"]) and "bank" in q:
        fraud_by_bank = df.groupby("sender_bank")["fraud_flag"].mean().round(4) * 100
        fraud_by_bank = fraud_by_bank.sort_values(ascending=False)
        return _build_fast_response(
            fraud_by_bank, question,
            f"{fraud_by_bank.index[0]} has highest fraud",
            f"{fraud_by_bank.index[0]} has the highest fraud rate at {fraud_by_bank.iloc[0]:.2f}%. Based on {len(df):,} transactions."
        )

    # 3) Fraud in high-value transactions
    if any(w in q for w in ["fraud", "flagged"]) and any(w in q for w in ["high-value", "over", "large", "above"]):
        high_value = df[df["amount_inr"] >= 5000]
        fraud_pct = round(high_value["fraud_flag"].mean() * 100, 2)
        return _build_fast_response(
            fraud_pct, question,
            f"{fraud_pct}% fraud rate",
            f"{fraud_pct}% of high-value transactions (≥₹5,000) are flagged. Based on {len(high_value):,} transactions."
        )

    # 4) Overall fraud rate (guarded: NOT state, NOT bank — those are handled above)
    if any(w in q for w in ["fraud", "flagged"]) and ("rate" in q or "percent" in q or "how" in q or "%" in q or "what" in q):
        fraud_pct = round(df["fraud_flag"].mean() * 100, 2)
        return _build_fast_response(
            fraud_pct, question,
            f"{fraud_pct}% fraud rate",
            f"Overall fraud flag rate is {fraud_pct}% across {len(df):,} transactions."
        )

    # 5) Failure by bank
    if ("fail" in q or "failed" in q or "failure" in q) and "bank" in q:
        failed = df[df["transaction_status"] == "FAILED"]
        counts = failed["sender_bank"].value_counts()
        return _build_fast_response(
            counts, question,
            f"{counts.index[0]} has the most failures",
            f"{counts.index[0]} leads in failed transactions with {counts.iloc[0]:,} failures."
        )

    # 6) Failure rate by device
    if ("failure" in q or "fail" in q) and any(w in q for w in ["device", "android", "ios"]):
        grouped = df.groupby("device_type")["transaction_status"].apply(
            lambda x: round((x == "FAILED").sum() / len(x) * 100, 2)
        )
        return _build_fast_response(
            grouped, question,
            "Failure rate by device type",
            f"{grouped.idxmax()} has the highest failure rate at {grouped.max():.2f}%."
        )

    # ===== SPECIFIC SINGLE-DIMENSION =====

    # 7) Age group + P2P
    if "age group" in q and "p2p" in q:
        data = df[df["transaction_type"] == "P2P"]
        counts = data["sender_age_group"].value_counts()
        return _build_fast_response(
            counts, question,
            "P2P transfers by age group",
            f"{counts.idxmax()} has the highest P2P transfers"
        )

    # 8) Average transaction amount
    if "average" in q and ("transaction" in q or "amount" in q):
        avg = round(float(df["amount_inr"].mean()), 2)
        return _build_fast_response(
            avg, question,
            f"₹{avg:,.2f} computed",
            f"The average transaction amount is ₹{avg:,.2f} across {len(df):,} transactions."
        )

    # 9) Peak merchant hours
    if "peak" in q and "merchant" in q:
        merchant_hours = df.groupby(["merchant_category", "hour_of_day"]).size().reset_index(name="count")
        peak = merchant_hours.loc[merchant_hours.groupby("merchant_category")["count"].idxmax()]
        result_series = peak.set_index("merchant_category")["hour_of_day"]
        return _build_fast_response(
            result_series, question,
            "Peak hours by merchant category",
            f"Peak transaction hours vary by merchant category. Based on {len(df):,} transactions."
        )

    # 10) Success rate
    if "success" in q and ("rate" in q or "percent" in q or "%" in q):
        rate = round((df["transaction_status"] == "SUCCESS").sum() / len(df) * 100, 2)
        return _build_fast_response(
            rate, question,
            f"{rate}% success rate",
            f"Overall success rate is {rate}% across {len(df):,} transactions."
        )

    # 11) Total transactions
    if "total" in q and ("transaction" in q or "count" in q or "how many" in q):
        total = len(df)
        return _build_fast_response(
            total, question,
            f"{total:,} transactions",
            f"The dataset contains {total:,} total transactions."
        )

    # 12) Weekend analysis
    if "weekend" in q and any(w in q for w in ["transaction", "compare", "volume", "how", "what"]):
        weekend = df[df["is_weekend"] == 1]
        weekday = df[df["is_weekend"] == 0]
        w_pct = round(len(weekend) / len(df) * 100, 2)
        return _build_fast_response(
            w_pct, question,
            f"{w_pct}% weekend traffic",
            f"{w_pct}% of transactions occur on weekends ({len(weekend):,}) vs {len(weekday):,} on weekdays."
        )

    # 13) Network type breakdown
    if "network" in q and any(w in q for w in ["type", "breakdown", "distribution", "compare", "usage"]):
        counts = df["network_type"].value_counts()
        return _build_fast_response(
            counts, question,
            f"{counts.index[0]} dominates",
            f"{counts.index[0]} is the most used network with {counts.iloc[0]:,} transactions."
        )

    # 14) Transaction type distribution
    if "transaction type" in q and any(w in q for w in ["breakdown", "distribution", "split", "compare", "how"]):
        counts = df["transaction_type"].value_counts()
        return _build_fast_response(
            counts, question,
            f"{counts.index[0]} leads",
            f"{counts.index[0]} is the most common transaction type with {counts.iloc[0]:,} transactions."
        )

    # ===== BROAD FALLBACKS (guarded) =====

    # 15) Top state — GUARD: skip if fraud/fail keywords present
    if ("top" in q or "highest" in q or "most" in q) and "state" in q:
        if not any(w in q for w in ["fraud", "fail", "flagged", "error"]):
            counts = df["sender_state"].value_counts()
            return _build_fast_response(
                counts, question,
                f"{counts.index[0]} leads",
                f"{counts.index[0]} has the most transactions with {counts.iloc[0]:,} total."
            )

    # 16) Bank volume — GUARD: skip if fraud/fail keywords present
    if "bank" in q and any(w in q for w in ["transaction", "volume", "count", "how many", "compare"]):
        if not any(w in q for w in ["fraud", "fail", "flagged", "error"]):
            counts = df["sender_bank"].value_counts()
            return _build_fast_response(
                counts, question,
                f"{counts.index[0]} leads by volume",
                f"{counts.index[0]} leads with {counts.iloc[0]:,} transactions."
            )

    return None


# --- Result Cache (bounded) ---
FAST_CACHE = {}
_CACHE_MAX = 200


def canonicalize(question):
    """Normalize question for cache key."""
    return question.strip().lower()


def get_cached(question):
    """Return cached result or None."""
    return FAST_CACHE.get(canonicalize(question))


def set_cached(question, value):
    """Store result in cache, evict oldest if full."""
    if len(FAST_CACHE) >= _CACHE_MAX:
        oldest = next(iter(FAST_CACHE))
        del FAST_CACHE[oldest]
    FAST_CACHE[canonicalize(question)] = value


# --- Question Validator ---
_DATA_KEYWORDS = {
    "transaction", "amount", "bank", "state", "fraud", "device", "network",
    "age", "group", "p2p", "p2m", "upi", "payment", "recharge", "bill",
    "success", "fail", "rate", "average", "mean", "total", "count",
    "compare", "top", "highest", "lowest", "most", "least", "peak",
    "hour", "day", "weekend", "merchant", "category", "trend", "distribution",
    "breakdown", "percentage", "how", "what", "which", "show", "list",
    "many", "much", "volume", "status", "sender", "receiver", "flagged",
    "android", "ios", "web", "wifi", "scatter", "chart", "plot",
    "income", "spending", "transfer", "value", "number", "ratio",
    "expensive", "cheap", "rich", "poor", "high", "low", "over", "under",
    "between", "across", "per", "each", "every", "monthly", "daily",
    "weekly", "type", "kind", "sort", "rank", "order", "sum", "median",
    "max", "min", "quartile", "analyze", "analyse", "insight", "pattern",
    "correlation", "split", "segment", "filter", "anomaly", "detect",
}


def validate_question(question):
    """Validate that a question is worth sending to the LLM.
    Returns (is_valid, error_message)."""
    q = question.strip()

    # Too short
    if len(q) < 10 or len(q.split()) < 3:
        return False, "Please ask a complete question about UPI transaction data (e.g. 'What is the average transaction amount?')."

    # Must contain at least one data-related keyword
    words = set(q.lower().split())
    # Also check substrings for compound words
    has_keyword = any(kw in q.lower() for kw in _DATA_KEYWORDS)
    if not has_keyword:
        return False, "That doesn't look like a data question. Try asking about transactions, banks, fraud rates, age groups, states, or device types."

    return True, ""


def get_full_answer(user_question, chat_history=None):
    start = time.time()

    # Validate question before wasting an API call
    is_valid, error_msg = validate_question(user_question)
    if not is_valid:
        print(f"Question rejected: '{user_question}' — {error_msg}")
        return {
            "answer": error_msg,
            "headline": "INVALID QUERY",
            "stats": [],
            "data": None,
            "code": None,
            "chart": None,
            "followups": ["What is the average transaction amount?", "Show fraud rate by bank", "Compare device types"],
            "verification": {"rows_used": 0, "status": "rejected", "error": "Question validation failed"}
        }

    # Check result cache first
    cached = get_cached(user_question)
    if cached is not None:
        print("Result cache hit for:", user_question)
        print("Query time:", time.time() - start)
        return cached

    # Try deterministic fast path
    fast = try_fast_query(user_question, df)
    if fast is not None:
        set_cached(user_question, fast)
        print("Query time:", time.time() - start)
        return fast

    # LLM path
    raw_code = ask_question(user_question, chat_history)
    code = clean_code(raw_code)
    print("Cleaned code to execute:", code)

    try:
        plan = code_to_plan(code)
        print("Parsed plan:", plan)
        result = execute_plan(plan, df)
        print("Safe exec result:", type(result).__name__)

        # Validate: if planner returned a raw DataFrame, it likely missed
        # the aggregation. Fall through to exec for:
        #   - empty DataFrame (0 rows = bad filter value)
        #   - large DataFrame (>1000 rows = missed aggregation)
        if isinstance(result, pd.DataFrame) and (len(result) == 0 or len(result) > 1000):
            print(f"Planner returned DataFrame with {len(result)} rows, likely incomplete parse")
            raise ValueError("Incomplete plan")

    except (ValueError, KeyError, TypeError) as parse_err:
        print(f"Planner failed ({parse_err}), falling back to guarded exec()")
        try:
            local_vars = {"df": df.copy(), "pd": pd}
            exec(code, {"__builtins__": {"len": len, "str": str, "int": int, "float": float, "round": round, "sorted": sorted, "list": list, "dict": dict, "tuple": tuple, "range": range, "enumerate": enumerate, "zip": zip, "min": min, "max": max, "sum": sum, "abs": abs, "True": True, "False": False, "None": None}}, local_vars)
            result = local_vars.get("result", None)
            if result is None:
                raise ValueError("Code ran but 'result' variable was not set")
            print("Exec fallback result:", type(result).__name__)
        except Exception as exec_err:
            print(f"Exec fallback also failed: {exec_err}")
            error_response = {
                "answer": "Unable to interpret this query. Please try rephrasing.",
                "headline": "Clarification required",
                "stats": [],
                "data": None,
                "code": code,
                "chart": None,
                "followups": ["Break this down by state", "Compare by device type", "Trend over time"],
                "verification": {"rows_used": len(df), "status": "error", "error": str(exec_err)}
            }
            print("Query time:", time.time() - start)
            return error_response
    except Exception as e:
        print(f"Code execution error: {e}")
        print("Query time:", time.time() - start)
        return {
            "answer": "Execution error: " + str(e) + " | Code was: " + code,
            "headline": "QUERY ERROR",
            "stats": [],
            "data": None,
            "code": code,
            "chart": None,
            "followups": []
        }

    verification = verify_result(df, result)

    # Sort Series descending so chart, headline, and stats all agree
    if isinstance(result, pd.Series) and len(result) > 0:
        result = result.sort_values(ascending=False)

    chart = generate_chart(result, user_question)
    followups = generate_followup_questions(user_question, result)
    headline, insight = generate_insight(user_question, result)

    stats = []
    if isinstance(result, pd.Series) and len(result) > 0:
        stats = [
            {"label": "HIGHEST", "value": str(result.index[0]) + ": " + str(round(float(result.iloc[0]), 2))},
            {"label": "AVERAGE", "value": str(round(float(result.mean()), 2))},
            {"label": "LOWEST", "value": str(result.index[-1]) + ": " + str(round(float(result.iloc[-1]), 2))}
        ]
    elif result is not None and not isinstance(result, pd.Series):
        stats = [{"label": "RESULT", "value": str(result)}]

    response = {
        "answer": insight,
        "headline": headline,
        "stats": stats,
        "data": str(result),
        "code": code,
        "chart": chart,
        "followups": followups,
        "verification": verification
    }
    set_cached(user_question, response)
    print("Query time:", time.time() - start)
    return response
