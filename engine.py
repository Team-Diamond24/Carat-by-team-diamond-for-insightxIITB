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


def call_ai(prompt):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="google/gemma-3-12b-it:free",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 30 * (attempt + 1)
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
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
        "Write ONLY pandas code to answer this question.\n"
        "Store answer in variable called 'result'.\n"
        "If result is a comparison or ranking, store as a pandas Series with named index.\n"
        "No explanation, no print statements, no markdown.\n\n"
        "Question: " + user_question
    )
    return call_ai(prompt)


def generate_chart(result, user_question):
    try:
        if not isinstance(result, pd.Series):
            return None

        chart_df = result.reset_index()
        chart_df.columns = ["Category", "Value"]

        question_lower = user_question.lower()

        if any(word in question_lower for word in ["hour", "time", "trend", "over", "day", "week"]):
            fig = px.line(
                chart_df, x="Category", y="Value",
                title=user_question, markers=True,
                color_discrete_sequence=["#00d4aa"]
            )
        elif any(word in question_lower for word in ["percentage", "share", "distribution", "proportion"]):
            fig = px.pie(
                chart_df, names="Category", values="Value",
                title=user_question,
                color_discrete_sequence=["#00d4aa", "#3b82f6", "#b1b2ff", "#aac4ff"]
            )
        else:
            fig = px.bar(
                chart_df, x="Value", y="Category",
                title=user_question, orientation="h",
                color_discrete_sequence=["#00d4aa"]
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#3730a3", family="JetBrains Mono"),
            title_font_size=12,
            margin=dict(t=30, r=20, b=40, l=100),
            yaxis=dict(autorange="reversed")
        )

        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))

    except Exception:
        return None


def generate_followup_questions(user_question, result):
    prompt = (
        "A user asked this question about UPI payment data:\n"
        "\"" + user_question + "\"\n\n"
        "The result was:\n"
        + str(result) + "\n\n"
        "Suggest exactly 3 short follow-up questions the user might want to ask next.\n"
        "Make them specific and directly related to the result.\n"
        "Return ONLY a JSON array of 3 strings, nothing else.\n"
        "Example: [\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
    )
    response = call_ai(prompt)
    try:
        response = response.replace("```json", "").replace("```", "").strip()
        questions = json.loads(response)
        if isinstance(questions, list) and len(questions) == 3:
            return questions
    except Exception:
        pass
    return [
        "Which state has the highest failure rate?",
        "How does this compare on weekends?",
        "Which age group is most affected?"
    ]


def get_full_answer(user_question, chat_history=None):
    code = ask_question(user_question)
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        local_vars = {"df": df}
        exec(code, local_vars)
        result = local_vars["result"]
    except Exception as e:
        return {
            "answer": "Sorry, could not compute that. Try rephrasing your question.",
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
        + str(result) + "\n\n"
        + "Respond like a sharp financial analyst giving a briefing.\n"
        + "First line: a bold headline in ALL CAPS (max 6 words) summarizing the finding.\n"
        + "Second part: 2-3 sentences of insight. Include the actual numbers.\n"
        + "Give one business reason for the pattern.\n"
        + "Format your response as JSON like this:\n"
        + "{\"headline\": \"YOUR HEADLINE HERE\", \"insight\": \"Your 2-3 sentence insight here.\"}\n"
        + "Return ONLY the JSON, nothing else."
    )

    explanation_raw = call_ai(explanation_prompt)

    try:
        explanation_raw = explanation_raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(explanation_raw)
        headline = parsed.get("headline", "ANALYSIS COMPLETE")
        insight = parsed.get("insight", explanation_raw)
    except Exception:
        headline = "ANALYSIS COMPLETE"
        insight = explanation_raw

    stats = []
    if isinstance(result, pd.Series) and len(result) > 0:
        stats = [
            {"label": "HIGHEST", "value": str(result.index[0]) + " " + str(round(result.iloc[0], 2))},
            {"label": "AVERAGE", "value": str(round(result.mean(), 2))},
            {"label": "LOWEST", "value": str(result.index[-1]) + " " + str(round(result.iloc[-1], 2))}
        ]
    elif not isinstance(result, pd.Series):
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
