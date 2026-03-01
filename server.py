import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_INSTALLED = True
except ImportError:
    print("\n[WARNING] 'slowapi' module not found. Rate limiting will be DISABLED.")
    print("          If you wish to enable rate limits, fix your python environment and run `pip install slowapi`.\n")
    SLOWAPI_INSTALLED = False
    class Limiter:
        def __init__(self, *args, **kwargs): pass
        def init_app(self, app): pass
        def limit(self, *args, **kwargs):
            def decorator(f): return f
            return decorator
    def get_remote_address(): return "127.0.0.1"
    class RateLimitExceeded(Exception): pass
    def _rate_limit_exceeded_handler(*args): pass

from llm_client import llm_client
from failure_logger import failure_logger
from engine import PandasAnalyst
from sql_analyst import SQLAnalyst

app = Flask(__name__, static_folder="static")
CORS(app)

# --- Improvement 4: Rate Limiting ---
rate_limit_str = os.environ.get("RATE_LIMIT_PER_MINUTE", "20")
limit_query = f"{rate_limit_str}/minute"
limit_admin = "10/minute"

limiter = Limiter(key_func=get_remote_address)
if SLOWAPI_INSTALLED:
    limiter.init_app(app)
    app.register_error_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.errorhandler(RateLimitExceeded)
def handle_rate_limit(e):
    return jsonify({"error": "Too many requests. Please wait a moment."}), 429

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- Improvement 1: Unified Base Analyst Route ---
USE_CSV = os.environ.get("USE_CSV", "false").lower() == "true"
analyst = PandasAnalyst() if USE_CSV else SQLAnalyst()
print(f"[SERVER] Active Analyst Core: {type(analyst).__name__}")


@app.route("/")
def landing():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/dashboard")
def dashboard():
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.route("/ask", methods=["POST"])
@app.route("/query", methods=["POST"])  # Alias
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

# Backwards compatible override routing for explicit SQL path if UI still uses it
@app.route("/sql-ask", methods=["POST"])
@limiter.limit(limit_query)
def sql_ask():
    data = request.get_json()
    question = data.get("question", "")
    history = data.get("history", [])
    if not question:
        return jsonify({"error": "No question provided"}), 400
    history = history[-2:]
    
    # Force SQL analyst regardless of USE_CSV for legacy compat
    forced_analyst = analyst if isinstance(analyst, SQLAnalyst) else SQLAnalyst()
    result = forced_analyst.analyze(question, history)
    return jsonify(result)


# --- Improvement 3: Admin AST Failures Endpoint ---
@app.route("/admin/failures", methods=["GET"])
@limiter.limit(limit_admin)
def get_ast_failures():
    admin_key = os.environ.get("ADMIN_KEY")
    req_key = request.headers.get("X-Admin-Key")
    
    # Very basic auth
    if not admin_key or req_key != admin_key:
        return jsonify({"error": "Unauthorized endpoint"}), 403
        
    failures = failure_logger.get_recent_failures(limit=50)
    return jsonify({"ast_failures": failures, "count": len(failures)})


@app.route("/stats")
def stats():
    # Only available securely if using Pandas directly; DB isn't loaded uniformly here
    # This acts as a basic fallback
    try:
        if isinstance(analyst, PandasAnalyst):
            from engine import df
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
        else:
             return jsonify({"error": "Stats are not explicitly configured for SQLAnalyst."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("InsightX by Team Diamond starting...")
    print("Landing page: http://localhost:5050")
    print("Dashboard:    http://localhost:5050/dashboard")
    app.run(port=5050)
