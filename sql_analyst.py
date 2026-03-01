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
# execute_sql() to avoid a hard dependency on sqlite3 at import time.

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
        df = pd.read_sql(sql.rstrip(";"), conn)
    finally:
        conn.close()
    return df

def _humanize_col(col_name: str) -> str:
    """Convert column_name or 'Column Name' to readable 'Column Name'."""
    return col_name.replace("_", " ").strip().title()


def _detect_chart_type(df: pd.DataFrame, question: str) -> str:
    """Pick the best chart type based on the question and data shape."""
    q = question.lower()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    n_rows = len(df)

    # Pie chart: good for percentages, proportions, shares, distribution across few categories
    pie_keywords = ["percentage", "percent", "proportion", "share", "distribution", "breakdown", "split", "ratio", "pie"]
    if any(kw in q for kw in pie_keywords) and len(cat_cols) >= 1 and len(num_cols) >= 1 and 2 <= n_rows <= 12:
        return "pie"

    # Scatter: correlation between two numeric columns
    if len(num_cols) >= 2 and len(cat_cols) == 0:
        return "scatter"

    # Bar chart: the default for category + metric
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        return "hbar" if n_rows > 8 else "bar"

    # Histogram: single numeric column, many rows
    if len(num_cols) >= 1 and n_rows > 2:
        return "histogram"

    return "none"


def generate_sql_chart(df_result: pd.DataFrame, question: str):
    """Generate a Plotly chart with proper value display and ordering."""
    try:
        if df_result is None or len(df_result) == 0:
            return None
        # Single scalar result — no chart needed
        if len(df_result) == 1 and len(df_result.columns) <= 2:
            return None

        # Work on a copy; force all potentially numeric columns to numeric
        df = df_result.copy()
        for col in df.columns:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0 and converted.notna().sum() >= len(df) * 0.5:
                    df[col] = converted
            except Exception:
                pass

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        if len(num_cols) == 0:
            return None

        chart_type = _detect_chart_type(df, question)
        if chart_type == "none":
            return None

        # Color palette
        COLORS = ["#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6",
                   "#fb923c", "#facc15", "#4ade80", "#22d3ee", "#60a5fa"]
        fig = None
        
        # Detect if values are percentages
        q = question.lower()
        is_percentage = any(kw in q for kw in ["rate", "percent", "percentage", "ratio", "proportion"])

        if chart_type == "pie":
            cat_col = cat_cols[0]
            val_col = num_cols[0]
            # Sort by value descending for consistent legend order
            df = df.sort_values(val_col, ascending=False).reset_index(drop=True)
            
            # Use column names with the dataframe
            fig = px.pie(
                df, 
                names=cat_col,  # Column name
                values=val_col,  # Column name
                title=question.capitalize(),
                color_discrete_sequence=COLORS,
                hole=0.35
            )
            fig.update_traces(
                textposition="inside",
                textinfo="label+percent+value",
                textfont_size=10,
                marker=dict(line=dict(color='#1a1b41', width=2))
            )

        elif chart_type == "bar":
            cat_col = cat_cols[0]
            val_col = num_cols[0]
            # Sort descending - highest first
            df = df.sort_values(val_col, ascending=False).reset_index(drop=True)
            
            # Use column names with the dataframe
            fig = px.bar(
                df, 
                x=cat_col,  # Column name
                y=val_col,  # Column name
                title=question.capitalize(),
                color_discrete_sequence=["#818cf8"],
                labels={cat_col: _humanize_col(cat_col), val_col: _humanize_col(val_col)},
                text=val_col,  # Column name for text
                category_orders={cat_col: df[cat_col].tolist()}  # Lock the sort order
            )
            
            # Format text based on data type
            if is_percentage:
                fig.update_traces(texttemplate='%{y:.2f}%', textposition="outside", textfont_size=10)
                fig.update_layout(
                    yaxis=dict(title=_humanize_col(val_col), tickformat=".1f", ticksuffix="%", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    xaxis=dict(title=_humanize_col(cat_col), showgrid=False, categoryorder='array', categoryarray=df[cat_col].tolist())
                )
            else:
                fig.update_traces(texttemplate='%{y:,.0f}', textposition="outside", textfont_size=10)
                fig.update_layout(
                    yaxis=dict(title=_humanize_col(val_col), tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    xaxis=dict(title=_humanize_col(cat_col), showgrid=False, categoryorder='array', categoryarray=df[cat_col].tolist())
                )

        elif chart_type == "hbar":
            cat_col = cat_cols[0]
            val_col = num_cols[0]
            # Sort descending first, then reverse for display (highest at top)
            df = df.sort_values(val_col, ascending=False).reset_index(drop=True)
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse for top-to-bottom
            
            # Use column names with the dataframe
            fig = px.bar(
                df, 
                x=val_col,  # Column name
                y=cat_col,  # Column name
                title=question.capitalize(),
                orientation="h",
                color_discrete_sequence=["#818cf8"],
                labels={val_col: _humanize_col(val_col), cat_col: _humanize_col(cat_col)},
                text=val_col  # Column name for text
            )
            
            # Format text based on data type
            if is_percentage:
                fig.update_traces(texttemplate='%{x:.2f}%', textposition="outside", textfont_size=10)
                fig.update_layout(
                    xaxis=dict(title=_humanize_col(val_col), tickformat=".1f", ticksuffix="%", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    yaxis=dict(title=_humanize_col(cat_col), showgrid=False, categoryorder='array', categoryarray=df[cat_col].tolist())
                )
            else:
                fig.update_traces(texttemplate='%{x:,.0f}', textposition="outside", textfont_size=10)
                fig.update_layout(
                    xaxis=dict(title=_humanize_col(val_col), tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    yaxis=dict(title=_humanize_col(cat_col), showgrid=False, categoryorder='array', categoryarray=df[cat_col].tolist())
                )

        elif chart_type == "scatter":
            # Use column names with the dataframe
            fig = px.scatter(
                df, 
                x=num_cols[0],  # Column name
                y=num_cols[1],  # Column name
                title=question.capitalize(),
                color_discrete_sequence=["#818cf8"],
                labels={num_cols[0]: _humanize_col(num_cols[0]), num_cols[1]: _humanize_col(num_cols[1])},
            )
            fig.update_layout(
                xaxis=dict(title=_humanize_col(num_cols[0]), tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                yaxis=dict(title=_humanize_col(num_cols[1]), tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
            )

        elif chart_type == "histogram":
            # Use column name with the dataframe
            fig = px.histogram(
                df, 
                x=num_cols[0],  # Column name
                title=question.capitalize(),
                color_discrete_sequence=["#818cf8"],
                labels={num_cols[0]: _humanize_col(num_cols[0])},
            )
            fig.update_layout(
                yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                xaxis=dict(title=_humanize_col(num_cols[0]), tickformat=",", showgrid=False),
            )

        if fig is None:
            return None

        # Global styling
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#3730a3", family="JetBrains Mono", size=11),
            title=dict(font=dict(size=13), x=0.01),
            margin=dict(t=45, r=25, b=55, l=100),
            showlegend=(chart_type == "pie"),
        )

        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    except Exception as e:
        print(f"[CHART] Generation error: {e}")
        return None

class SQLAnalyst(BaseAnalyst):
    def _generate_sql(self, question: str, chat_history: list = None, use_fallback: bool = False) -> dict:
        history_context = ""
        if chat_history:
            recent = chat_history[-2:]  # Keep last 2 for context
            for msg in recent:
                history_context += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            history_context = f"Recent context:\n{history_context}\n"

        prompt = f"""You are an expert SQLite data analyst working with UPI transaction data.

Database Schema:
{TABLE_SCHEMA}

{history_context}Query Requirements:
- Write a single, optimized SELECT query for SQLite
- Table name: upi_transactions_2024
- Security: NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE
- Performance: LIMIT results to 100 rows maximum
- Comparison queries: When asked "which X has the most/least Y", return ALL groups sorted by metric (not just LIMIT 1) for full comparison
- Sorting: ORDER BY the primary metric DESC unless specifically asked for ascending
- Readability: Use clear column aliases (e.g., AS "Total Amount", AS "Failure Rate %")
- Output: Return ONLY the raw SQL query, no markdown code blocks, no explanations

User Question: {question}

SQL Query:"""
        return llm_client.call_model(prompt, use_fallback=use_fallback)

    def _summarize_result(self, question: str, sql: str, df_result: pd.DataFrame, use_fallback: bool = False) -> str:
        if df_result is None or len(df_result) == 0:
            return "The query returned no results."
            
        # Prepare data summary
        data_text = df_result.to_string(index=False) if len(df_result) <= 15 else df_result.head(10).to_string(index=False) + f"\n... ({len(df_result)} total)"
        
        # Calculate key statistics for numeric columns
        stats_context = ""
        num_cols = df_result.select_dtypes(include="number").columns.tolist()
        if len(num_cols) > 0:
            col = num_cols[0]
            stats_context = f"\nKey Stats: Max={df_result[col].max():.2f}, Min={df_result[col].min():.2f}, Avg={df_result[col].mean():.2f}"
        
        prompt = f"""You are a senior financial data analyst providing insights on UPI transaction data.

Question: {question}
SQL Query: {sql}

Results:
{data_text}{stats_context}

Instructions:
1. Write a comprehensive 4-6 sentence analysis in a professional, analyst tone
2. Start with the key finding that directly answers the question
3. Highlight the most significant patterns, trends, or outliers in the data
4. Compare top performers vs bottom performers if applicable
5. Provide business context or implications where relevant
6. Use specific numbers and percentages from the data
7. Write in present tense, as if presenting to stakeholders

Analysis:"""

        try:
            resp = llm_client.call_model(prompt, use_fallback=use_fallback)
            text = resp["content"].strip()
            # Remove any thinking tags if present
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Remove any markdown formatting
            text = text.replace("**", "").replace("##", "").strip()
            return text
        except Exception as e:
            print(f"[SQL Analyst] Summary generation failed: {e}")
            # Fallback to basic summary
            rows = len(df_result)
            cols = ", ".join(df_result.columns.tolist())
            
            # Try to create a basic insight from the data
            if rows > 0 and len(num_cols) > 0:
                col = num_cols[0]
                cat_cols = df_result.select_dtypes(exclude="number").columns.tolist()
                if len(cat_cols) > 0:
                    top_row = df_result.iloc[0]
                    return f"Analysis shows {rows} results. {top_row[cat_cols[0]]} leads with {top_row[col]:.2f}, followed by other entries. The data reveals significant variation across categories, with values ranging from {df_result[col].min():.2f} to {df_result[col].max():.2f}."
            
            return f"The query returned {rows} row(s) with columns: {cols}."

    def analyze(self, query: str, chat_history: list = None) -> dict:
        is_valid, err_msg = validate_question(query)
        if not is_valid:
            return format_clarification_response(err_msg)

        try:
            # Step 1: LLM for SQL
            resp = self._generate_sql(query, chat_history)
            sql = _clean_sql(resp["content"])
            print(f"[SQL] {sql}")  # Log SQL to terminal only
            
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
                "summary": summary,
                "answer": summary,
                "headline": "SQL Database Insight",
                "stats": stats,
                "result": result_str,
                "data": result_str,
                "sql": sql,
                "code": sql,
                "chart": chart,
                "rows_returned": len(df_result),
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
