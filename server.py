from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from engine import get_full_answer, generate_chart, df
import os
import json

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


@app.route("/")
def landing():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "No question provided"}), 400

    history = history[-2:]
    result = get_full_answer(question, history)

    # Stage 1: headline + stats immediately (no chart)
    stage1 = dict(result, chart=None, stage=1)

    # Stage 2: full payload with chart
    stage2 = dict(result, stage=2)

    def generate():
        yield json.dumps(stage1) + "\n"
        yield json.dumps(stage2) + "\n"

    return Response(stream_with_context(generate()), content_type="application/x-ndjson")


@app.route("/stats")
def stats():
    try:
        total = len(df)
        success_rate = round((df["transaction_status"] == "SUCCESS").sum() / total * 100, 1)
        avg_amount = round(df["amount_inr"].mean())
        failure_rate = round(100 - success_rate, 1)
        peak_hour = int(df["hour_of_day"].value_counts().index[0])
        top_state = df["sender_state"].value_counts().index[0]

        return jsonify({
            "total_transactions": f"{total:,}",
            "success_rate": f"{success_rate}%",
            "avg_amount": f"Rs {avg_amount:,}",
            "failure_rate": f"{failure_rate}%",
            "peak_hour": f"{peak_hour}:00",
            "top_state": top_state
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Carat by Team Diamond starting...")
    print("Landing page: http://localhost:5050")
    print("Dashboard:    http://localhost:5050/dashboard")
    app.run(debug=True, port=5050)
