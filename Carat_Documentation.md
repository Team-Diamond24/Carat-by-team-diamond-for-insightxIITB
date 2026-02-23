# Carat by Team Diamond - Product Documentation

## 1. Product Overview
**Carat** is a conversational analytics dashboard designed for querying and analyzing UPI transaction data seamlessly using natural language. Built for "Techfest IIT Bombay x NPCI", it empowers non-technical users to access analyst-grade insights without writing SQL or Pandas code. 
Users input questions in plain English, and the AI backend processes the queries into actionable code, executes it against a 250,000-row UPI dataset (`upi_transactions_2024.csv`), and pipes the response back into a beautifully designed UI with auto-generated charts, insights, and continuous chat memory.

## 2. Architecture & Tech Stack
The project is built on a lightweight, deeply integrated stack separated into a frontend dashboard and a Python-powered intelligent pipeline.

- **Frontend:** HTML5, Tailwind CSS (via CDN), Vanilla JavaScript, Plotly.js
- **Backend Server:** Flask, Flask-CORS
- **AI/Data Engine:** Python, Pandas, OpenAI Python SDK (routing to DeepSeek R1 via OpenRouter), Plotly Express 
- **Database:** Static CSV dataset loaded into Pandas memory at startup.

## 3. Project Structure
- `engine.py`: The Core AI pipeline. Handles CSV loading, AI API communication, code sanitization, python runtime execution, chart generation, and formatting.
- `server.py`: The Flask server. Exposes the static HTML files and provides REST endpoints for the dashboard to hit (`/ask`, `/stats`).
- `dashboard.html`: The main terminal UI. Features a left sidebar with live KPIs and quick queries, and a main chat/feed view where results are streamed and rendered.
- `index.html`: The landing page with branding, an automated ticker tape of stats, and a brief feature overview.
- `upi_transactions_2024.csv`: The dataset containing 250,000 mock UPI transactions for the year 2024.

---

## 4. Detailed Component Breakdown

### A. The Engine (`engine.py`)
This is the heart of the application. It bypasses traditional text-to-SQL by doing **text-to-pandas**, executing the code locally, and using the exact output to generate an analyst-like response.

**1. Data Loading:**
- Reads `upi_transactions_2024.csv` into a global Pandas DataFrame (`df`).
- Normalizes column names (lowercase, strips spaces, removes parentheses, replaces spaces with underscores).
- *Columns available include:* `transaction_type`, `amount_inr`, `transaction_status`, `sender_age_group`, `sender_state`, `sender_bank`, `receiver_bank`, `device_type`, `network_type`, `fraud_flag`, `hour_of_day`, `day_of_week`, `is_weekend`, `merchant_category`.

**2. AI Client Setup:**
- Uses the `openai` Python package but points the `base_url` to OpenRouter (`https://openrouter.ai/api/v1`).
- Currently hardcoded to use the model: `deepseek/deepseek-r1`.
- Built-in retry mechanism (3 attempts) with a backoff strategy for rate limits (HTTP 429).

**3. The Execution Pipeline (`get_full_answer`):**
When a query is received, the pipeline triggers sequentially:
1. **ask_question()**: Prompts DeepSeek to generate **executable Pandas code**. The AI is strictly instructed to return *only* Python code that assigns a variable named `result`.
2. **clean_code()**: A sanitizer that strips ````python`, markdown, and thoroughly extracts code from DeepSeek's `<think>...</think>` tags to ensure Python's `exec()` doesn't crash.
3. **exec() Sandbox**: The code string runs via `exec(code, local_vars)` where `local_vars` passes in the `df` environment. Extracts the `result` variable.
4. **generate_chart()**: If `result` is a `pd.Series` (e.g. grouped data), Plotly Express automatically assesses the topic based on keywords in the question (e.g. "trend", "share", "percentage"). It selects between a Line chart (`px.line`), Pie chart (`px.pie`), or Bar chart (`px.bar`) and outputs the Plotly JSON payload.
5. **generate_followup_questions()**: Prompts DeepSeek to suggest 3 contextual questions based on the result.
6. **Insight Generation**: Sends the original question and the raw data result to DeepSeek again to act like a "financial analyst", returning a 6-word `headline` and a 2-3 sentence `insight` in JSON. 
7. **Return Payload**: Packages the insight text, headline, stats summaries (highest/lowest/average), raw data string, executed code, chart JSON, and followups to the dashboard.

### B. The Server (`server.py`)
A lightweight Flask implementation to serve the application locally.
- **Port config:** Runs on `http://localhost:5050` (Specifically changed from 5000/5001 to avoid Windows internal port conflicts and Chrome HTTPS auto-upgrade 400 errors).
- **CORS:** Enabled globally via `flask_cors`.

*Routes:*
- `GET /`: Serves `index.html` from the absolute directory path using `os.path.abspath`.
- `GET /dashboard`: Serves `dashboard.html`.
- `POST /ask`: The primary websocket/fetch endpoint. Accepts JSON `{"question": "...", "history": [...]}`. Returns the dictionary constructed by `engine.py`.
- `GET /stats`: Generates live KPIs from the pandas `df` dynamically (total transactions, success/failure rate, avg amount, peak hour, top state) to feed the UI.

### C. The Frontend (`index.html` & `dashboard.html`)
The frontend is strictly client-side vanilla JavaScript built with extremely sharp Tailwind aesthetics (dark brutalism / terminal theme). Color palette relies on `#1a1b41` (primary text), `#b1b2ff` (accent lines), and `#eef1ff` (background).

**Landing Page (`index.html`)**
- High layout typography using 'Inter' and 'JetBrains Mono'.
- Contains an infinite CSS ticker (`@keyframes ticker`) that hits `/stats` on load to scroll live application metrics.

**Dashboard Terminal (`dashboard.html`)**
- **Sidebar:** Contains predefined "Quick Queries" that instantly trigger the chat bar. Features a "Live" status dot.
- **Top KPI Strip:** Fetches data from `/stats` on load and populates three metric cards statically.
- **Chat Feed:** The main chat UI uses a simulated "Typing Indicator" (`typing-dot` animation) while waiting for the LLM. 
- **Card Rendering Scheme:** When the server responds, it renders a complex HTML card (`addResponseCard()`. It injects the Headline, Insight, and small analytical stat chips. If Plotly JSON is present in the response, it executes `Plotly.newPlot` directly into the DOM node to render an interactive chart.
- **Followups:** Each response card optionally generates clickable chips at the bottom that inject the string into the main chat logic. History array is tracked locally in `chatHistory` and sent back to the server.

---

## 5. Recent Fixes & Current State Context
*If continuing development with an AI, keep these states in mind:*

1. **Port Assignments:** The server is bound to port `5050`. Do not use port 5000 (Docker mapping conflict `404 page not found` plain-text issue) or port 5001 (Google Chrome HSTS auto-HTTPS upgrade issue resulting in `ERR_CONNECTION_RESET / 400 Bad Request`).
2. **Path Resolving:** `send_from_directory` uses absolute path logic `os.path.abspath(os.path.dirname(__file__))` instead of relative dots (`.`) to prevent standard Flask 404s.
3. **DeepSeek R1 Integration:** OpenRouter's free-tier `deepseek/deepseek-r1:free` endpoint returns a 404. Development is currently using the paid standard tag `deepseek/deepseek-r1`.
4. **DeepSeek Thinking Tags:** DeepSeek R1 generates aggressive `<think>` tags in its response. `clean_code()` and the JSON parser in `get_full_answer()` explicitly split out the `</think>` tags before parsing JSON or `exec()` code otherwise it will crash the server.
