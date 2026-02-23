import pandas as pd
from openai import OpenAI
import time
import plotly.express as px
import plotly.utils
import json
from planner import code_to_plan
from safe_exec import execute_plan
from verify import verify_result

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

df = pd.read_csv("upi_transactions_2024.csv")

df.columns = (df.columns
              .str.strip()
              .str.replace(' ', '_')
              .str.replace('(', '')
              .str.replace(')', '')
              .str.lower())

print("Dataset loaded. Columns:", df.columns.tolist())
print("Shape:", df.shape)


def call_ai(prompt):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600
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


def ask_question(user_question):
    prompt = (
        "You are a data analyst. You have a pandas dataframe called 'df' with 250,000 UPI transactions.\n\n"
        "DataFrame df has columns:\n"
        "transaction_type, amount_inr, transaction_status, sender_age_group,\n"
        "sender_state, sender_bank, receiver_bank, device_type,\n"
        "network_type, fraud_flag, hour_of_day, day_of_week,\n"
        "is_weekend, merchant_category\n\n"
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
    code = code.replace("<think>", "")
    if "</think>" in code:
        code = code.split("</think>")[-1]
    code = code.strip()
    return code


def generate_chart(result, user_question):
    try:
        if result is None:
            return None

        # --- Handle pd.Series (most common from groupby) ---
        if isinstance(result, pd.Series):
            chart_df = result.reset_index()
            chart_df.columns = ["Category", "Value"]
            x_label = result.index.name or "Category"
            y_label = result.name or "Value"

            # Auto-detect ratio values (0–1) and convert to percentage
            if chart_df["Value"].max() <= 1 and chart_df["Value"].min() >= 0:
                chart_df["Value"] = chart_df["Value"] * 100
                y_label = "Percentage (%)"

            chart_title = f"{y_label} by {x_label}"

            # 1. Datetime or hour/day index → line chart
            if (pd.api.types.is_datetime64_any_dtype(result.index)
                    or str(x_label).lower() in ("hour_of_day", "day_of_week", "month", "date", "year")):
                chart_df["Category"] = chart_df["Category"].astype(str)
                fig = px.line(
                    chart_df, x="Category", y="Value",
                    title=chart_title, markers=True,
                    labels={"Category": x_label, "Value": y_label},
                    color_discrete_sequence=["#b1b2ff"]
                )

            # 2. <=6 categories and values sum ≈ 100% → pie chart
            elif len(chart_df) <= 6 and 95 <= chart_df["Value"].sum() <= 105:
                chart_df["Category"] = chart_df["Category"].astype(str)
                fig = px.pie(
                    chart_df, names="Category", values="Value",
                    title=chart_title,
                    color_discrete_sequence=["#b1b2ff", "#aac4ff", "#d2daff", "#3730a3"]
                )

            # 3. Categorical index + numeric values → horizontal bar
            else:
                chart_df = chart_df.sort_values("Value", ascending=False)
                chart_df["Category"] = chart_df["Category"].astype(str)
                fig = px.bar(
                    chart_df, x="Value", y="Category",
                    title=chart_title, orientation="h",
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
    """Rule-based followup suggestions — no LLM call."""
    return [
        "Break this down by state",
        "Compare by device type",
        "Trend over time",
    ]


def generate_insight(user_question, result):
    """Deterministic template-based insight — no LLM call."""
    rows = len(df)

    if isinstance(result, pd.Series) and len(result) > 0:
        metric = str(result.index[0])
    elif isinstance(result, pd.DataFrame):
        metric = f"{len(result)} rows"
    elif isinstance(result, (int, float)):
        metric = f"₹{float(result):,.2f}"
    else:
        metric = str(result)[:50]

    headline = f"{metric} computed"
    insight = f"Based on {rows:,} transactions."

    return headline, insight


CODE_CACHE = {}


def get_full_answer(user_question, chat_history=None):

    if user_question in CODE_CACHE:
        code = CODE_CACHE[user_question]
        print("Cache hit for:", user_question)
    else:
        raw_code = ask_question(user_question)
        code = clean_code(raw_code)
        CODE_CACHE[user_question] = code
    print("Cleaned code to execute:", code)

    try:
        plan = code_to_plan(code)
        print("Parsed plan:", plan)
        result = execute_plan(plan, df)
        print("Result:", result)
    except (ValueError, KeyError) as parse_err:
        print(f"Safe exec failed ({parse_err}), falling back to exec()")
        try:
            local_vars = {"df": df, "pd": pd}
            exec(code, local_vars)
            result = local_vars.get("result", None)
            if result is None:
                raise ValueError("Code ran but 'result' variable was not set")
            print("Fallback result:", result)
        except Exception as e:
            raise e
    except Exception as e:
        print(f"Code execution error: {e}")
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

    return {
        "answer": insight,
        "headline": headline,
        "stats": stats,
        "data": str(result),
        "code": code,
        "chart": chart,
        "followups": followups,
        "verification": verification
    }
