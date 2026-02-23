import pandas as pd
from openai import OpenAI
import time
import plotly.express as px
import plotly.utils
import json

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
    prompt = (
        "A user asked this question about UPI payment data:\n"
        "\"" + user_question + "\"\n\n"
        "The result was:\n"
        + str(result)[:500] + "\n\n"
        "Suggest exactly 3 short follow-up questions.\n"
        "Return ONLY a JSON array of 3 strings, nothing else.\n"
        "Example: [\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
    )
    response = call_ai(prompt)
    try:
        response = response.replace("```json", "").replace("```", "").strip()
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        questions = json.loads(response)
        if isinstance(questions, list) and len(questions) == 3:
            return questions
    except Exception as e:
        print(f"Followup generation error: {e}")
    return [
        "Which state has the highest failure rate?",
        "How does this compare on weekends?",
        "Which age group is most affected?"
    ]


def get_full_answer(user_question, chat_history=None):

    raw_code = ask_question(user_question)
    code = clean_code(raw_code)
    print("Cleaned code to execute:", code)

    try:
        local_vars = {"df": df, "pd": pd}
        exec(code, local_vars)
        result = local_vars.get("result", None)
        if result is None:
            raise ValueError("Code ran but 'result' variable was not set")
        print("Result:", result)
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

    chart = generate_chart(result, user_question)
    followups = generate_followup_questions(user_question, result)

    history_text = ""
    if chat_history:
        history_text = "Previous conversation:\n"
        for msg in chat_history[-4:]:
            history_text += msg["role"] + ": " + msg["content"] + "\n"
        history_text += "\n"

    explanation_prompt = (
        history_text
        + "Someone asked this question about UPI payment data:\n"
        + "\"" + user_question + "\"\n\n"
        + "The data analysis returned this result:\n"
        + str(result)[:500] + "\n\n"
        + "Respond like a sharp financial analyst giving a briefing.\n"
        + "Return ONLY a JSON object with exactly two keys:\n"
        + "\"headline\": a bold finding in ALL CAPS, max 6 words\n"
        + "\"insight\": 2-3 sentences with actual numbers and one business reason\n"
        + "Return ONLY the JSON, no explanation, no markdown, no think tags."
    )

    explanation_raw = call_ai(explanation_prompt)
    print("Explanation raw:", explanation_raw)

    try:
        clean = explanation_raw.replace("```json", "").replace("```", "").strip()
        if "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        parsed = json.loads(clean)
        headline = parsed.get("headline", "ANALYSIS COMPLETE")
        insight = parsed.get("insight", explanation_raw)
    except Exception as e:
        print(f"Explanation parse error: {e}")
        headline = "ANALYSIS COMPLETE"
        insight = explanation_raw

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
        "followups": followups
    }
