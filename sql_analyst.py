import os
import re
import json
import time
import pandas as pd
import plotly.express as px
import plotly.utils

from llm_client import llm_client
from shared_utils import BaseAnalyst, validate_question, format_clarification_response, generate_followup_questions

# NOTE: `from db import get_connection` is imported lazily inside
# execute_sql() to avoid a hard dependency on psycopg2 at import time.

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

def _clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"<think>.*?</think>", "", sql, flags=re.DOTALL)
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.strip()
    if sql and not sql.endswith(";"):
        sql += ";"
    return sql

_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "COPY",
    "SET ROLE", "SET SESSION",
]

def validate_sql(sql: str) -> tuple:
    normalized = sql.strip().upper()
    if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
        return False, "Only SELECT queries are allowed."
    for kw in _FORBIDDEN_KEYWORDS:
        pattern = r'\b' + kw + r'\b'
        if re.search(pattern, normalized):
            return False, f"Forbidden keyword detected: {kw}"
    return True, ""

def execute_sql(sql: str) -> pd.DataFrame:
    is_valid, error_msg = validate_sql(sql)
    if not is_valid:
        raise ValueError(error_msg)

    from db import get_connection
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY;")
        cur.close()
        df = pd.read_sql(sql.rstrip(";"), conn)
    finally:
        conn.close()
    return df

def generate_sql_chart(df_result: pd.DataFrame, question: str):
    try:
        if df_result is None or len(df_result) == 0:
            return None

        numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df_result.select_dtypes(exclude="number").columns.tolist()
        fig = None

        if len(non_numeric_cols) >= 1 and len(numeric_cols) >= 1:
            x_col = non_numeric_cols[0]
            y_col = numeric_cols[0]
            if len(df_result) > 8:
                fig = px.bar(df_result, x=y_col, y=x_col, title=question, orientation="h", color_discrete_sequence=["#b1b2ff"])
                fig.update_layout(yaxis=dict(autorange="reversed"))
            else:
                fig = px.bar(df_result, x=x_col, y=y_col, title=question, color_discrete_sequence=["#b1b2ff"])
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df_result, x=numeric_cols[0], y=numeric_cols[1], title=question, color_discrete_sequence=["#b1b2ff"])
        elif len(numeric_cols) == 1:
            fig = px.histogram(df_result, x=numeric_cols[0], title=question, color_discrete_sequence=["#b1b2ff"])

        if fig is None:
            return None

        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#3730a3", family="JetBrains Mono"), title_font_size=12, margin=dict(t=30, r=20, b=40, l=120))
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    except Exception as e:
        print(f"SQL chart generation error: {e}")
        return None

class SQLAnalyst(BaseAnalyst):
    def _generate_sql(self, question: str, chat_history: list = None, use_fallback: bool = False) -> dict:
        history_context = ""
        if chat_history:
            recent = chat_history[-4:]
            for msg in recent:
                history_context += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            history_context = f"Recent conversation:\n{history_context}\n"

        prompt = f"""You are a PostgreSQL data analyst. Table schema:
{TABLE_SCHEMA}

{history_context}Rules:
- Write ONLY a single valid PostgreSQL SELECT query
- Table name is upi_transactions_2024
- NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or any DDL/DML
- Always LIMIT results to 100 rows unless explicitly asked
- Use readable column aliases
- Result ONLY contains raw SQL, no markdown fences.

Question: {question}
SQL:"""
        return llm_client.call_model(prompt, use_fallback=use_fallback)

    def _summarize_result(self, question: str, sql: str, df_result: pd.DataFrame, use_fallback: bool = False) -> str:
        if df_result is None or len(df_result) == 0:
            return "The query returned no results."
            
        data_text = df_result.to_string(index=False) if len(df_result) <= 20 else df_result.head(15).to_string(index=False) + f"\n... ({len(df_result)} total rows)"
        
        prompt = f"""You are a financial analyst summarizing SQL results.
Question: {question}
SQL: {sql}
Results:\n{data_text}\n
Write a concise 3-5 sentence summary. Do not repeat raw data line-by-line."""

        try:
            resp = llm_client.call_model(prompt, use_fallback=use_fallback)
            text = resp["content"].strip()
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        except Exception:
            rows = len(df_result)
            cols = ", ".join(df_result.columns.tolist())
            return f"The query returned {rows} row(s) with columns: {cols}."

    def analyze(self, query: str, chat_history: list = None) -> dict:
        is_valid, err_msg = validate_question(query)
        if not is_valid:
            return format_clarification_response(err_msg)

        try:
            # Step 1: LLM for SQL
            resp = self._generate_sql(query, chat_history)
            sql = _clean_sql(resp["content"])
            
            # Step 2: Query DB
            df_result = execute_sql(sql)
            
            # Step 3: LLM for Summary
            summary = self._summarize_result(query, sql, df_result)
            
            # Step 4: Chart
            chart = generate_sql_chart(df_result, query)
            
            # Format Data
            if len(df_result) <= 50:
                result_str = df_result.to_string(index=False)
            else:
                result_str = df_result.head(50).to_string(index=False) + f"\n\n... showing 50 of {len(df_result)} rows"

            stats = [{"label": "ROWS", "value": str(len(df_result))}]
            if len(df_result) > 0 and len(df_result.select_dtypes(include="number").columns) > 0:
                col = df_result.select_dtypes(include="number").columns[0]
                stats.insert(0, {"label": f"MAX {col.upper()}", "value": f"{df_result[col].max():.2f}"})

            return {
                "answer": summary,
                "headline": "SQL Database Insight",
                "stats": stats,
                "data": result_str,
                "code": sql,
                "chart": chart,
                "followups": generate_followup_questions(query),
                "verification": {
                    "valid": True,
                    "warnings": [],
                    "errors": [],
                    "model_used": resp["model_used"],
                    "fallback_used": resp["fallback_used"]
                }
            }

        except ValueError as ve:
            print(f"[SQL Analyst] Validation/Execution Failed: {ve}")
            return format_clarification_response(str(ve))
        except Exception as e:
            print(f"[SQL Analyst] Request Failed: {e}")
            return format_clarification_response(str(e))
