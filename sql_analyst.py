"""
sql_analyst.py - SQL-Based Data Analyst Pipeline

A 4-step pipeline for answering plain-English questions about UPI
transactions using PostgreSQL:

    1. UNDERSTAND  – Parse the user's question
    2. GENERATE SQL – Prompt DeepSeek to write a PostgreSQL SELECT
    3. EXECUTE     – Run the query safely (read-only)
    4. SUMMARIZE   – Produce a plain-English answer

Usage:
    from sql_analyst import ask_sql
    result = ask_sql("Which bank has the highest transaction volume?")
    print(result["sql"])
    print(result["summary"])
"""

import os
import re
import json
import time
import pandas as pd
import plotly.express as px
import plotly.utils
from openai import OpenAI
from dotenv import load_dotenv

# NOTE: `from db import get_connection` is imported lazily inside
# execute_sql() to avoid a hard dependency on psycopg2 at import time.

load_dotenv()

# Own AI client — avoids importing engine.py (which loads 250K rows)
_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    timeout=30.0
)


def _call_ai(prompt: str) -> str:
    """Call DeepSeek via OpenRouter (standalone, no engine.py dependency)."""
    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"[SQL Analyst] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[SQL Analyst] AI call error: {e}")
                return f"Error: {e}"
    return "Error: Too many retries"

# ---------------------------------------------------------------------------
# Table schema description (used in AI prompts)
# ---------------------------------------------------------------------------

TABLE_SCHEMA = """
Table: upi_transactions_2024
Columns:
  - transaction_id        TEXT      (e.g. TXN0000000001)
  - timestamp             TEXT      (format DD-MM-YYYY HH.MM, e.g. '08-10-2024 15.17')
  - transaction_type      TEXT      (P2P, P2M, Bill Payment, Recharge)
  - merchant_category     TEXT      (Food, Grocery, Shopping, Utilities, Entertainment, Healthcare, Transport, Fuel, Education, Other)
  - amount_inr            INTEGER   (transaction amount in INR, range ~10-20000)
  - transaction_status    TEXT      (SUCCESS, FAILED)
  - sender_age_group      TEXT      (18-25, 26-35, 36-45, 46-55, 56+)
  - receiver_age_group    TEXT      (18-25, 26-35, 36-45, 46-55, 56+)
  - sender_state          TEXT      (Maharashtra, Delhi, Karnataka, Tamil Nadu, Gujarat, Uttar Pradesh, Telangana, Rajasthan, West Bengal, Andhra Pradesh)
  - sender_bank           TEXT      (SBI, HDFC, ICICI, Axis, PNB, Kotak, Yes Bank, IndusInd)
  - receiver_bank         TEXT      (SBI, HDFC, ICICI, Axis, PNB, Kotak, Yes Bank, IndusInd)
  - device_type           TEXT      (Android, iOS, Web)
  - network_type          TEXT      (4G, 5G, WiFi, 3G)
  - fraud_flag            INTEGER   (0 = not fraud, 1 = fraud)
  - hour_of_day           INTEGER   (0-23)
  - day_of_week           TEXT      (Monday, Tuesday, ..., Sunday)
  - is_weekend            INTEGER   (0 = weekday, 1 = weekend)

Total rows: ~250,000
"""


# ---------------------------------------------------------------------------
# Step 2: Generate SQL
# ---------------------------------------------------------------------------

def generate_sql(question: str, chat_history: list = None) -> str:
    """
    Prompt DeepSeek to generate a PostgreSQL SELECT query for the question.
    Returns the raw SQL string.
    """
    history_context = ""
    if chat_history:
        recent = chat_history[-4:]
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_context += f"{role}: {content}\n"
        history_context = f"Recent conversation:\n{history_context}\n"

    prompt = f"""You are a PostgreSQL data analyst. You have access to the following table:

{TABLE_SCHEMA}

{history_context}Rules:
- Write ONLY a single valid PostgreSQL SELECT query
- Table name is upi_transactions_2024
- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or any DDL/DML
- Always LIMIT results to 100 rows unless the user explicitly asks for all
- Use readable column aliases (e.g. total_amount, transaction_count)
- Use ROUND for decimal values
- For percentages, multiply by 100 and round to 2 decimal places
- Use proper PostgreSQL syntax (e.g. ::numeric for casting)
- Return ONLY the SQL query, no explanation, no markdown, no code fences

Question: {question}

SQL:"""

    raw = _call_ai(prompt)
    return _clean_sql(raw)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences, think tags, and whitespace from AI output."""
    sql = raw.strip()

    # Remove <think>...</think> blocks (DeepSeek R1)
    sql = re.sub(r"<think>.*?</think>", "", sql, flags=re.DOTALL)

    # Remove markdown code fences
    sql = sql.replace("```sql", "").replace("```", "")

    # Remove leading/trailing whitespace and semicolons cleanup
    sql = sql.strip()

    # Ensure it ends with a semicolon
    if sql and not sql.endswith(";"):
        sql += ";"

    return sql


# ---------------------------------------------------------------------------
# Step 3: Execute SQL
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "COPY",
    "SET ROLE", "SET SESSION",
]


def validate_sql(sql: str) -> tuple:
    """
    Validate that the SQL is a safe SELECT query.
    Returns (is_valid, error_message).
    """
    normalized = sql.strip().upper()

    # Must start with SELECT or WITH (for CTEs)
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return False, "Only SELECT queries are allowed."

    # Check for forbidden keywords
    for kw in _FORBIDDEN_KEYWORDS:
        # Use word boundary check to avoid false positives
        pattern = r'\b' + kw + r'\b'
        if re.search(pattern, normalized):
            return False, f"Forbidden keyword detected: {kw}"

    return True, ""


def execute_sql(sql: str) -> pd.DataFrame:
    """
    Execute a validated SELECT query against PostgreSQL.
    Uses a read-only transaction for safety.
    Returns results as a pandas DataFrame.
    """
    is_valid, error = validate_sql(sql)
    if not is_valid:
        raise ValueError(f"SQL validation failed: {error}")

    # Lazy import to avoid crashing at module load if psycopg2 is missing
    from db import get_connection

    conn = get_connection()
    try:
        # Set transaction to read-only for extra safety
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY;")
        cur.close()

        df = pd.read_sql(sql.rstrip(";"), conn)
    finally:
        conn.close()

    return df


# ---------------------------------------------------------------------------
# Step 4: Summarize
# ---------------------------------------------------------------------------

def summarize_result(question: str, sql: str, df_result: pd.DataFrame) -> str:
    """
    Generate a plain-English summary of the query results.
    Uses LLM for rich summaries, falls back to template if LLM fails.
    """
    if df_result is None or len(df_result) == 0:
        return "The query returned no results. This might mean the filter conditions are too restrictive, or the data doesn't match the criteria. Try broadening your question."

    # Build a compact text representation of the result
    if len(df_result) <= 20:
        data_text = df_result.to_string(index=False)
    else:
        data_text = df_result.head(15).to_string(index=False)
        data_text += f"\n... ({len(df_result)} total rows)"

    prompt = f"""You are a financial data analyst. You ran this SQL query to answer a user's question:

Question: {question}

SQL: {sql}

Results:
{data_text}

Write a clear, concise summary in 3-5 sentences:
- Lead with the direct answer / key insight
- Mention specific numbers from the results
- If it's a trend, describe the direction
- Do NOT repeat raw data row by row
- Use Indian Rupee symbol (₹) for money values
- Be conversational but professional

Summary:"""

    raw = _call_ai(prompt)
    summary = _clean_summary(raw)

    # Fallback if LLM returns garbage
    if not summary or len(summary) < 20:
        summary = _template_summary(df_result)

    return summary


def _clean_summary(raw: str) -> str:
    """Clean LLM output for the summary."""
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def _template_summary(df_result: pd.DataFrame) -> str:
    """Fallback template-based summary."""
    rows = len(df_result)
    cols = ", ".join(df_result.columns.tolist())
    summary = f"The query returned {rows} row(s) with columns: {cols}."

    # Try to extract top value from first numeric column
    numeric_cols = df_result.select_dtypes(include="number").columns
    if len(numeric_cols) > 0 and rows > 0:
        col = numeric_cols[0]
        top_val = df_result[col].max()
        summary += f" The highest {col} is {top_val:,.2f}."

    return summary


# ---------------------------------------------------------------------------
# Chart generation for SQL results
# ---------------------------------------------------------------------------

def generate_sql_chart(df_result: pd.DataFrame, question: str):
    """Generate a Plotly chart from SQL query results."""
    try:
        if df_result is None or len(df_result) == 0:
            return None

        numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df_result.select_dtypes(exclude="number").columns.tolist()

        fig = None

        # One categorical + one numeric → bar chart
        if len(non_numeric_cols) >= 1 and len(numeric_cols) >= 1:
            x_col = non_numeric_cols[0]
            y_col = numeric_cols[0]

            # Decide orientation based on number of categories
            if len(df_result) > 8:
                fig = px.bar(
                    df_result, x=y_col, y=x_col,
                    title=question, orientation="h",
                    color_discrete_sequence=["#b1b2ff"]
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
            else:
                fig = px.bar(
                    df_result, x=x_col, y=y_col,
                    title=question,
                    color_discrete_sequence=["#b1b2ff"]
                )

        # Two numeric columns → scatter
        elif len(numeric_cols) >= 2:
            fig = px.scatter(
                df_result, x=numeric_cols[0], y=numeric_cols[1],
                title=question,
                color_discrete_sequence=["#b1b2ff"]
            )

        # Only numeric → histogram
        elif len(numeric_cols) == 1:
            fig = px.histogram(
                df_result, x=numeric_cols[0],
                title=question,
                color_discrete_sequence=["#b1b2ff"]
            )

        if fig is None:
            return None

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#3730a3", family="JetBrains Mono"),
            title_font_size=12,
            margin=dict(t=30, r=20, b=40, l=120)
        )

        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

    except Exception as e:
        print(f"SQL chart generation error: {e}")
        return None


# ---------------------------------------------------------------------------
# Orchestrator: ask_sql
# ---------------------------------------------------------------------------

def ask_sql(question: str, chat_history: list = None) -> dict:
    """
    Full SQL analyst pipeline:
      1. Generate SQL from the question
      2. Execute the SQL safely
      3. Summarize the results
      4. Generate a chart if applicable

    Returns a dict with: sql, result, summary, chart, error
    """
    import time
    start = time.time()

    # Validate question minimally
    q = question.strip()
    if len(q) < 5:
        return {
            "sql": None,
            "result": None,
            "summary": "Please ask a more specific question about UPI transaction data.",
            "chart": None,
            "error": "Question too short",
        }

    try:
        # Step 2: Generate SQL
        sql = generate_sql(question, chat_history)
        print(f"[SQL Analyst] Generated SQL:\n{sql}")

        # Step 3: Execute
        df_result = execute_sql(sql)
        print(f"[SQL Analyst] Query returned {len(df_result)} rows")

        # Step 4: Summarize
        summary = summarize_result(question, sql, df_result)

        # Chart
        chart = generate_sql_chart(df_result, question)

        # Format result for display
        if len(df_result) <= 50:
            result_str = df_result.to_string(index=False)
        else:
            result_str = df_result.head(50).to_string(index=False)
            result_str += f"\n\n... showing 50 of {len(df_result)} rows"

        elapsed = round(time.time() - start, 2)
        print(f"[SQL Analyst] Total time: {elapsed}s")

        return {
            "sql": sql,
            "result": result_str,
            "summary": summary,
            "chart": chart,
            "error": None,
            "rows_returned": len(df_result),
            "time_seconds": elapsed,
        }

    except ValueError as ve:
        # SQL validation error
        print(f"[SQL Analyst] Validation error: {ve}")
        return {
            "sql": None,
            "result": None,
            "summary": f"Query rejected for safety: {ve}",
            "chart": None,
            "error": str(ve),
        }

    except Exception as e:
        print(f"[SQL Analyst] Error: {e}")
        return {
            "sql": sql if 'sql' in dir() else None,
            "result": None,
            "summary": f"An error occurred while processing your query. Please try rephrasing. Error: {str(e)}",
            "chart": None,
            "error": str(e),
        }
