import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_INSTALLED = True
except ImportError:
    print("\n[WARNING] 'flask-limiter' module not found. Rate limiting will be DISABLED.")
    print("          Run `pip install flask-limiter` to enable rate limits.\n")
    LIMITER_INSTALLED = False
    class Limiter:
        def __init__(self, *args, **kwargs): pass
        def limit(self, *args, **kwargs):
            def decorator(f): return f
            return decorator
    def get_remote_address(): return "127.0.0.1"

from llm_client import llm_client
from failure_logger import failure_logger
from sql_analyst import SQLAnalyst

app = Flask(__name__, static_folder="static")
CORS(app)

# --- Rate Limiting ---
rate_limit_str = os.environ.get("RATE_LIMIT_PER_MINUTE", "20")
limit_query = f"{rate_limit_str}/minute"
limit_admin = "10/minute"

if LIMITER_INSTALLED:
    limiter = Limiter(get_remote_address, app=app, default_limits=[])
else:
    limiter = Limiter()

@app.errorhandler(429)
def handle_rate_limit(e):
    return jsonify({"error": "Too many requests. Please wait a moment."}), 429

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- Single SQL Analyst ---
analyst = SQLAnalyst()
print("[SERVER] Active Analyst Core: SQLAnalyst (SQLite)")


@app.route("/")
def landing():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.route("/ask", methods=["POST"])
@app.route("/query", methods=["POST"])
@app.route("/sql-ask", methods=["POST"])  # Legacy alias
@limiter.limit(limit_query)
def ask():
    data = request.get_json()
    question = data.get("question", "")
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "No question provided"}), 400

    history = history[-2:]
    result = analyst.analyze(question, history)
    return jsonify(result)


# --- Admin AST Failures Endpoint ---
@app.route("/admin/failures", methods=["GET"])
@limiter.limit(limit_admin)
def get_ast_failures():
    admin_key = os.environ.get("ADMIN_KEY")
    req_key = request.headers.get("X-Admin-Key")
    
    if not admin_key or req_key != admin_key:
        return jsonify({"error": "Unauthorized endpoint"}), 403
        
    failures = failure_logger.get_recent_failures(limit=50)
    return jsonify({"ast_failures": failures, "count": len(failures)})


@app.route("/stats")
def stats():
    """Query SQLite directly for dashboard KPI cards."""
    import pandas as _pd
    from db import get_connection
    conn = get_connection()
    try:
        row = _pd.read_sql(
            "SELECT COUNT(*) as total,"
            " ROUND(SUM(CASE WHEN transaction_status='SUCCESS' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) as success_rate,"
            " ROUND(AVG(amount_inr)) as avg_amount,"
            " ROUND(SUM(CASE WHEN transaction_status='FAILED' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) as failure_rate"
            " FROM upi_transactions_2024", conn
        ).iloc[0]
        peak_hour_row = _pd.read_sql(
            "SELECT hour_of_day, COUNT(*) as cnt FROM upi_transactions_2024"
            " GROUP BY hour_of_day ORDER BY cnt DESC LIMIT 1", conn
        ).iloc[0]
        top_state_row = _pd.read_sql(
            "SELECT sender_state, COUNT(*) as cnt FROM upi_transactions_2024"
            " GROUP BY sender_state ORDER BY cnt DESC LIMIT 1", conn
        ).iloc[0]
        return jsonify({
            "total_transactions": f"{int(row['total']):,}",
            "success_rate": f"{row['success_rate']}%",
            "avg_amount": f"Rs {int(row['avg_amount']):,}",
            "failure_rate": f"{row['failure_rate']}%",
            "peak_hour": f"{int(peak_hour_row['hour_of_day'])}:00",
            "top_state": top_state_row['sender_state']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


if __name__ == "__main__":
    print("Carat by Team Diamond starting...")

    # Pre-cache quick query responses for instant results
    from warmup_cache import warmup_cache
    warmup_cache()

    print("Landing page: http://localhost:5050")
    print("Dashboard:    http://localhost:5050/dashboard")
    app.run(port=5050)
