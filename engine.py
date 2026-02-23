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
                messages=[{"role": "user", "content": prompt}]
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
        "Columns available:\n"
        "- transaction_type: P2P, P2M, Bill Payment, Recharge\n"
        "- amount_inr: amount in rupees\n"
        "- transaction_status: SUCCESS or FAILED\n"
        "- sender_age_group: 18-25, 26-35, 36-45, 46-55, 56+\n"
        "- sender_state: Indian state\n"
        "- sender_bank: SBI, HDFC, ICICI, Axis, PNB, Kotak, IndusInd, Yes Bank\n"
        "- receiver_bank: same as sender_bank\n"
        "- device_type: Android, iOS, Web\n"
        "- network_type: 4G, 5G, WiFi, 3G\n"
        "- fraud_flag: 1 = flagged, 0 = normal\n"
        "- hour_of_day: 0 to 23\n"
        "- day_of_week: Monday to Sunday\n"
        "- is_weekend: 1 = weekend, 0 = weekday\n"
        "- merchant_category: Food, Grocery, Fuel etc (only for P2M)\n\n"
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
        if not isinstance(result, pd.Series):
            return None

        chart_df = result.reset_index()
        chart_df.columns = ["Category", "Value"]
        chart_df["Category"] = chart_df["Category"].astype(str)

        question_lower = user_question.lower()

        if any(word in question_lower for word in ["hour", "time", "trend", "over", "day", "week"]):
            fig = px.line(
                chart_df, x="Category", y="Value",
                title=user_question, markers=True,
                color_discrete_sequence=["#b1b2ff"]
            )
        elif any(word in question_lower for word in ["percentage", "share", "distribution", "proportion"]):
            fig = px.pie(
                chart_df, names="Category", values="Value",
                title=user_question,
                color_discrete_sequence=["#b1b2ff", "#aac4ff", "#d2daff", "#3730a3"]
            )
        else:
            fig = px.bar(
                chart_df, x="Value", y="Category",
                title=user_question, orientation="h",
                color_discrete_sequence=["#b1b2ff"]
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))

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
        top_label = str(result.index[0])
        top_value = round(float(result.iloc[0]), 2)
        headline = f"{top_label} LEADS".upper()
        insight = (
            f"{top_label} ranks highest at {top_value:,.2f}. "
            f"Analysis based on {rows:,} transactions in the dataset. "
            f"The bottom entry is {result.index[-1]} at {round(float(result.iloc[-1]), 2):,.2f}."
        )
    elif isinstance(result, (int, float)):
        headline = "METRIC COMPUTED"
        insight = (
            f"The computed value is ₹{float(result):,.2f} "
            f"across {rows:,} transactions."
        )
    elif isinstance(result, pd.DataFrame):
        headline = "DATA RETRIEVED"
        insight = (
            f"Query returned {len(result):,} rows "
            f"from a total of {rows:,} transactions."
        )
    else:
        headline = "ANALYSIS COMPLETE"
        insight = f"Result: {str(result)[:200]}. Based on {rows:,} transactions."

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
