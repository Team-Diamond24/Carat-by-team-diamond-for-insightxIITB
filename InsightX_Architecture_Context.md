# InsightX: Project Architecture & Context Summary

**Project Name:** InsightX
**Description:** InsightX is an AI-powered conversational data analytics platform designed to analyze UPI (Unified Payments Interface) transaction data. It allows users to ask natural language questions (e.g., "Which bank has the highest transaction failure rate?") and receives structured answers, data tables, Plotly charts, and statistical summaries. The system is designed with a strict focus on security, actively avoiding arbitrary code execution (`exec()` or `eval()`) when parsing AI outputs.

## Technology Stack
- **Backend:** Python, FastAPI (`server.py`)
- **Frontend:** Vanilla HTML/CSS/JS with Fetch API (`index.html`, `dashboard.html`)
- **Data Processing:** Pandas, PostgreSQL (`psycopg2`)
- **Charting Engine:** Plotly Express
- **AI/LLM Provider:** OpenRouter (DeepSeek/GPT models)
- **Code Parsing:** Python Built-in AST (`ast` module)

## Core Architecture & Execution Flow

InsightX implements a robust 5-step pipeline for translating natural language into safe, deterministic data frames:

1. **Entity Resolution & Caching (`engine.py`)** 
   - Uses `difflib` and hardcoded alias maps with strict regex word boundaries (`\b`) to dynamically catch user typos (e.g., auto-correcting "orrisa" to "Odisha").
   - Maintains a context-aware cache that hashes both the query string and the recent `chat_history` (last 4 turns) via MD5 to prevent cross-talk collisions between different chat threads.
   - Includes a "fast path" deterministic lookup for 16 common query patterns to bypass the LLM entirely, saving latency and credits.

2. **LLM Code Generation (`engine.py`)**
   - If the fast path fails, a structured prompt is sent to the LLM (OpenRouter) alongside the actual exact values of the categorical columns (to prevent hallucinated categories).
   - The LLM outputs simple Pandas code snippets (e.g., `result = df.groupby('sender_bank')['amount_inr'].mean()`).

3. **AST Parsing (`planner.py`)**
   - **Crucial Security Layer:** To avoid arbitrary code execution, `planner.py` uses an `ast.NodeVisitor` to parse the Python syntax tree of the LLM's output.
   - It extracts operations recursively (like `filter`, `groupby`, single/multi-column aggregations, `value_counts`, and multi-conditional boolean masks `df[(cond1) & (cond2)]`).
   - Translates the AST into a rigorously structured JSON/Dictionary execution plan.

4. **Safe Execution Layer (`safe_exec.py`)**
   - Receives the plan dictionary from `planner.py`.
   - Dispatch registry strictly maps the plan dictionary to deterministic Pandas handler methods (e.g., `_handle_groupby`, `_handle_filter_multi`).
   - Supports multi-column groupby operations, safely flattening pandas `MultiIndex` returns to standard strings so downstream charting won't break.
   - **Never uses `exec()` or `eval()`**. If the structure is unsupported, it throws an error back to the engine.

5. **SQL Alternative Pipeline (`sql_analyst.py` & `db.py`)**
   - An alternative analysis pipeline that interfaces directly with a PostgreSQL database instead of loading everything into a Pandas DataFrame via CSV.
   - Generates read-only `SELECT` SQL queries via LLM, connects via `psycopg2`, safely executes the query, and summarizes the output in plain English. Controlled by the `USE_CSV=false` environment variable tag.

## Key Files & Layout
- `engine.py` - Core coordinating pipeline (Fuzzy matching, caching, LLM orchestration, result charting).
- `planner.py` - Syntactic AST-based code-to-plan converter.
- `safe_exec.py` - Deterministic pandas script runner using the plan dictionary.
- `verify.py` - Post-analysis data validation checks.
- `server.py` - FastAPI application entrypoint.
- `db.py` - Postgres database connection handler.
- `sql_analyst.py` - SQL Database equivalent to `engine.py`.
- `dashboard.html` / `index.html` - The user interfaces.
- `.env` / `.env.example` - Environment configuration, keeping `OPENROUTER_API_KEY` scoped and secure out of source control.

## Security Posture
- **No Prompt Injection Code Execution:** The `exec()` fallback is completely non-existent in this architecture. Any generated code that cannot be parsed by the AST `planner.py` securely defaults to a "Clarification required" UI response.
- **Git Hardening:** Development keys and `.env` files are fully excluded from source control (`.gitignore`), relying entirely on local environment variables via `os.environ.get()`.
