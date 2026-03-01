import pandas as pd
import time
import math
import plotly.express as px
import plotly.utils
import json
import difflib
import re
import os

from planner import code_to_plan
from safe_exec import execute_plan
from verify import verify_result

from llm_client import llm_client
from failure_logger import failure_logger
from shared_utils import (
    BaseAnalyst, get_cached, set_cached, validate_question,
    generate_followup_questions, format_clarification_response
)

USE_CSV = os.environ.get("USE_CSV", "false").lower() == "true"
csv_path = os.path.join(os.path.dirname(__file__), "upi_transactions_2024.csv")

if USE_CSV:
    print("[ENGINE] Loading CSV data...", end="", flush=True)
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                "transaction id": "transaction_id",
                "transaction type": "transaction_type",
                "amount (INR)": "amount_inr"
            })
            print(f" Done! ({len(df):,} rows)")
        else:
            df = pd.DataFrame()
            print(f" Warning: {csv_path} not found. Empty DataFrame created.")
    except Exception as e:
        df = pd.DataFrame()
        print(f" Failed: {e}")
else:
    df = None
    print("[ENGINE] SQL mode — skipping CSV load.")

def resolve_entities(question, df):
    """
    Auto-correct typos in the user's question by matching against known data.
    """
    corrections = []
    suggestions = []
    
    ALIASES = {
        "orissa": "Odisha", "orrisa": "Odisha",
        "bangalore": "Karnataka", "bengaluru": "Karnataka",
        "himachal": "Himachal Pradesh",
        "himachalpradesh": "Himachal Pradesh",
        "bombay": "Maharashtra", "mumbai": "Maharashtra",
        "chennai": "Tamil Nadu", "hyderabad": "Telangana",
        "kolkata": "West Bengal", "pune": "Maharashtra",
        "sbi": "SBI", "hdfc": "HDFC", "icici": "ICICI",
        "paytm": "Paytm", "gpay": "GPay", "phonepe": "PhonePe",
        "broadband": "WiFi", "wifi": "WiFi", "wi-fi": "WiFi",
        "4g": "4G", "5g": "5G", "3g": "3G",
    }

    q_lower = question.lower()
    q_words = q_lower.split()
    for alias, real_name in ALIASES.items():
        if alias in q_words or alias in q_lower:
            exists_in_data = False
            for col in df.columns:
                if df[col].dtype == "object" and real_name in df[col].values:
                    exists_in_data = True
                    break
            if exists_in_data and real_name.lower() not in q_lower:
                pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                new_question = pattern.sub(real_name, question)
                if new_question != question:
                    corrections.append((alias, real_name))
                    question = new_question
                    q_lower = question.lower()

    MATCH_COLUMNS = [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() <= 30]
    all_known = {}
    for col in MATCH_COLUMNS:
        for val in df[col].dropna().unique():
            vl = str(val).lower()
            if len(vl) >= 3:
                all_known[vl] = (str(val), col)

    words = question.split()
    candidates = [w.lower().strip('?,.!') for w in words if len(w) >= 3]
    for i in range(len(words) - 1):
        bigram = words[i].lower().strip('?,.!') + " " + words[i+1].lower().strip('?,.!')
        candidates.append(bigram)

    q_lower = question.lower()
    known_keys = list(all_known.keys())

    for candidate in candidates:
        if candidate in all_known:
            continue
        close = difflib.get_close_matches(candidate, known_keys, n=1, cutoff=0.5)
        if close:
            matched_key = close[0]
            real_value, col_name = all_known[matched_key]
            ratio = difflib.SequenceMatcher(None, candidate, matched_key).ratio()

            if real_value.lower() in q_lower:
                continue

            if ratio >= 0.6:
                pattern = re.compile(re.escape(candidate), re.IGNORECASE)
                new_question = pattern.sub(real_value, question)
                if new_question != question:
                    corrections.append((candidate, real_value))
                    question = new_question
                    q_lower = question.lower()
            elif ratio >= 0.5:
                suggestions.append(f"Did you mean '{real_value}'?")

    suggestions = list(dict.fromkeys(suggestions))
    return question, corrections, suggestions


def clean_code(raw_code):
    code = raw_code
    code = code.replace("```python", "")
    code = code.replace("```", "")
    code = code.strip()
    return code


def generate_chart(result, user_question):
    try:
        if result is None: return None
        if isinstance(result, dict): result = pd.Series(dict)
        
        if isinstance(result, pd.Series):
            if len(result) == 0: return None
            
            # Detect if this is percentage data
            q = user_question.lower()
            is_percentage = any(kw in q for kw in ["rate", "percent", "percentage", "ratio", "proportion"])
            
            # Keep original order - highest to lowest
            sorted_result = result.sort_values(ascending=False)
            categories = [str(x) for x in sorted_result.index.tolist()]
            values = sorted_result.values.tolist()
            
            # Create chart dataframe
            chart_df = pd.DataFrame({"Category": categories, "Value": [float(v) for v in values]})
            x_label = sorted_result.index.name or "Category"
            y_label = sorted_result.name or "Value"
            
            # Decide chart type based on data characteristics
            n_items = len(chart_df)
            
            # Pie chart for percentages with few categories
            if is_percentage and n_items <= 8:
                fig = px.pie(
                    chart_df, 
                    names="Category", 
                    values="Value",
                    title=f"{y_label} by {x_label}",
                    color_discrete_sequence=["#818cf8", "#a78bfa", "#c084fc", "#e879f9", "#f472b6", "#fb923c", "#facc15", "#4ade80"],
                    hole=0.35
                )
                fig.update_traces(
                    textposition="inside",
                    textinfo="label+percent",
                    textfont_size=11,
                    marker=dict(line=dict(color='#1a1b41', width=2))
                )
            # Horizontal bar for many items (easier to read labels)
            elif n_items > 6:
                # For horizontal bars, reverse order so highest is at top
                chart_df = chart_df.iloc[::-1].reset_index(drop=True)
                fig = px.bar(
                    chart_df, 
                    x="Value", 
                    y="Category",
                    title=f"{y_label} by {x_label}",
                    orientation="h",
                    labels={"Category": x_label, "Value": y_label},
                    color_discrete_sequence=["#818cf8"],
                    text="Value"
                )
                # Format text on bars
                if is_percentage:
                    fig.update_traces(texttemplate='%{x:.2f}%', textposition="outside")
                else:
                    fig.update_traces(texttemplate='%{x:,.0f}', textposition="outside")
                    
                fig.update_layout(
                    xaxis=dict(tickformat=",.0f" if not is_percentage else ".1f", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    yaxis=dict(showgrid=False, categoryorder='total ascending')
                )
            # Vertical bar for few items
            else:
                fig = px.bar(
                    chart_df, 
                    x="Category", 
                    y="Value",
                    title=f"{y_label} by {x_label}",
                    labels={"Category": x_label, "Value": y_label},
                    color_discrete_sequence=["#818cf8"],
                    text="Value"
                )
                if is_percentage:
                    fig.update_traces(texttemplate='%{y:.2f}%', textposition="outside")
                else:
                    fig.update_traces(texttemplate='%{y:,.0f}', textposition="outside")
                    
                fig.update_layout(
                    yaxis=dict(tickformat=",.0f" if not is_percentage else ".1f", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    xaxis=dict(showgrid=False)
                )
                
        elif isinstance(result, pd.DataFrame) and len(result) > 0:
            numeric_cols = result.select_dtypes(include="number").columns.tolist()
            non_numeric_cols = result.select_dtypes(exclude="number").columns.tolist()
            
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(
                    result, 
                    x=x_col, 
                    y=y_col, 
                    title=user_question,
                    labels={x_col: x_col, y_col: y_col},
                    color_discrete_sequence=["#818cf8"]
                )
                fig.update_layout(
                    xaxis=dict(tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    yaxis=dict(tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)")
                )
            elif len(numeric_cols) == 1 and len(non_numeric_cols) == 0:
                col = numeric_cols[0]
                fig = px.histogram(
                    result, 
                    x=col, 
                    title=user_question,
                    labels={col: col},
                    color_discrete_sequence=["#818cf8"]
                )
                fig.update_layout(
                    yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                    xaxis=dict(tickformat=",", showgrid=False)
                )
            elif len(non_numeric_cols) >= 1 and len(numeric_cols) >= 1:
                cat_col = non_numeric_cols[0]
                val_col = numeric_cols[0]
                # Sort by value descending
                sorted_df = result.sort_values(val_col, ascending=False).reset_index(drop=True)
                
                if len(sorted_df) > 6:
                    # Horizontal bar - reverse for top-to-bottom display
                    sorted_df = sorted_df.iloc[::-1].reset_index(drop=True)
                    fig = px.bar(
                        sorted_df, 
                        x=val_col, 
                        y=cat_col,
                        title=user_question,
                        orientation="h",
                        labels={cat_col: cat_col, val_col: val_col},
                        color_discrete_sequence=["#818cf8"],
                        text=val_col
                    )
                    fig.update_traces(texttemplate='%{x:,.0f}', textposition="outside")
                    fig.update_layout(
                        xaxis=dict(tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                        yaxis=dict(showgrid=False, categoryorder='total ascending')
                    )
                else:
                    # Vertical bar
                    fig = px.bar(
                        sorted_df, 
                        x=cat_col, 
                        y=val_col,
                        title=user_question,
                        labels={cat_col: cat_col, val_col: val_col},
                        color_discrete_sequence=["#818cf8"],
                        text=val_col
                    )
                    fig.update_traces(texttemplate='%{y:,.0f}', textposition="outside")
                    fig.update_layout(
                        yaxis=dict(tickformat=",", showgrid=True, gridcolor="rgba(129,140,248,0.15)"),
                        xaxis=dict(showgrid=False)
                    )
            else: 
                return None
        else: 
            return None

        # Global styling
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#3730a3", family="JetBrains Mono", size=11),
            title=dict(font=dict(size=13), x=0.01),
            margin=dict(t=45, r=25, b=55, l=120),
            showlegend=False
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    except Exception as e:
        print(f"[CHART] Generation error: {e}")
        return None


def generate_insight(user_question, result):
    rows = len(df)
    q = user_question.lower()
    is_pct = any(w in q for w in ["rate", "percent", "percentage", "ratio", "proportion", "fraud", "success", "failure"])
    is_count = any(w in q for w in ["count", "total", "how many", "number of", "volume"])

    if isinstance(result, pd.Series) and len(result) > 0:
        # Sort descending to get proper highest/lowest
        sorted_result = result.sort_values(ascending=False)
        top_idx, top_val = sorted_result.index[0], sorted_result.iloc[0]
        bot_idx, bot_val = sorted_result.index[-1], sorted_result.iloc[-1]
        
        if is_pct:
            headline = f"{top_idx}: {float(top_val):.2f}%"
            insight = f"{top_idx} leads at {float(top_val):.2f}%, while {bot_idx} is lowest at {float(bot_val):.2f}%. Based on {rows:,} transactions."
        elif is_count:
            headline = f"{top_idx}: {int(top_val):,}"
            insight = f"{top_idx} has the highest count with {int(top_val):,}. {bot_idx} has the least with {int(bot_val):,}. Based on {rows:,} transactions."
        else:
            headline = f"{top_idx}: ₹{float(top_val):,.2f}"
            insight = f"{top_idx} leads with ₹{float(top_val):,.2f}. {bot_idx} is lowest at ₹{float(bot_val):,.2f}. Based on {rows:,} transactions."
    elif isinstance(result, pd.DataFrame):
        headline = f"{len(result):,} rows computed"
        insight = f"Returned {len(result):,} rows from {rows:,} total transactions."
    elif isinstance(result, (int, float)):
        val = float(result)
        if is_pct:
            headline = f"{val:.2f}%"
            insight = f"The result is {val:.2f}%. Based on {rows:,} transactions."
        elif is_count:
            headline = f"{int(val):,}"
            insight = f"The count is {int(val):,}. Based on {rows:,} transactions."
        else:
            headline = f"₹{val:,.2f}"
            insight = f"The value is ₹{val:,.2f}. Based on {rows:,} transactions."
    else:
        headline = str(result)[:60]
        insight = f"Based on {rows:,} transactions."

    return headline, insight

def _build_fast_response(result, question, headline, insight):
    chart = generate_chart(result, question)
    verification = verify_result(df, None, result)
    followups = generate_followup_questions(question, result)
    stats = []
    if isinstance(result, pd.Series) and len(result) > 0:
        # Sort descending for proper highest/lowest
        sorted_result = result.sort_values(ascending=False)
        stats = [
            {"label": "HIGHEST", "value": str(sorted_result.index[0]) + ": " + str(round(float(sorted_result.iloc[0]), 2))},
            {"label": "AVERAGE", "value": str(round(float(sorted_result.mean()), 2))},
            {"label": "LOWEST", "value": str(sorted_result.index[-1]) + ": " + str(round(float(sorted_result.iloc[-1]), 2))}
        ]
    elif result is not None and not isinstance(result, pd.Series):
        stats = [{"label": "RESULT", "value": str(result)}]
    
    return {
        "answer": insight,
        "headline": headline,
        "stats": stats,
        "data": str(result),
        "code": "# fast path — no LLM",
        "chart": chart,
        "followups": followups,
        "verification": verification,
        "skip_llm": True
    }

def try_fast_query(question, df):
    q = question.lower()
    
    # a) weekend traffic %
    if "weekend" in q and any(w in q for w in ["percent", "percentage", "traffic", "rate"]):
        result = round(df['is_weekend'].mean() * 100, 2)
        headline = f"{result}% weekend traffic"
        insight = f"{result}% of all transactions occur on weekends. Based on {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # b) overall fraud rate (no filters)
    if "fraud" in q and "rate" in q and "state" not in q and "bank" not in q and "5000" not in q and "high" not in q:
        result = round(df['fraud_flag'].mean() * 100, 2)
        headline = f"{result}% fraud rate"
        insight = f"The overall fraud rate is {result}% across all {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # c) high-value fraud rate
    if "fraud" in q and ("5000" in q or "high" in q or "above" in q):
        high_val = df[df['amount_inr'] > 5000]
        result = round(high_val['fraud_flag'].mean() * 100, 2)
        headline = f"{result}% high-value fraud rate"
        insight = f"{result}% of transactions above ₹5,000 are flagged for fraud. Based on {len(high_val):,} high-value transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # d) failure rate by device type
    if "failure rate" in q and "device" in q:
        total = df.groupby('device_type')['transaction_id'].count()
        failed = df[df['transaction_status']=='FAILED'].groupby('device_type')['transaction_id'].count()
        result = (failed / total * 100).round(2).sort_values(ascending=False)
        headline = f"{result.index[0]} highest failure: {result.iloc[0]}%"
        insight = f"{result.index[0]} has the highest failure rate at {result.iloc[0]}%. Based on {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # e) failure rate 5G vs WiFi
    if "failure" in q and any(w in q for w in ["5g", "wifi", "wi-fi"]) and "device" not in q:
        networks = ['5G', 'WiFi']
        total = df[df['network_type'].isin(networks)].groupby('network_type')['transaction_id'].count()
        failed = df[(df['transaction_status']=='FAILED') & (df['network_type'].isin(networks))].groupby('network_type')['transaction_id'].count()
        result = (failed / total * 100).round(2)
        headline = f"5G vs WiFi failure rates"
        insight = f"5G failure rate: {result.get('5G',0):.2f}%, WiFi failure rate: {result.get('WiFi',0):.2f}%. Based on {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # f) success rate iOS vs Android
    if "success" in q and any(w in q for w in ["ios", "android"]):
        devices = ['iOS', 'Android']
        total = df[df['device_type'].isin(devices)].groupby('device_type')['transaction_id'].count()
        success = df[(df['transaction_status']=='SUCCESS') & (df['device_type'].isin(devices))].groupby('device_type')['transaction_id'].count()
        result = (success / total * 100).round(2)
        headline = f"iOS: {result.get('iOS',0):.2f}% | Android: {result.get('Android',0):.2f}%"
        insight = f"iOS success rate: {result.get('iOS',0):.2f}%, Android: {result.get('Android',0):.2f}%. Based on {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # g) 4G vs broadband (WiFi) volume
    if "volume" in q and ("4g" in q or "broadband" in q or "wifi" in q) and "failure" not in q:
        networks = ['4G', 'WiFi']
        result = df[df['network_type'].isin(networks)].groupby('network_type')['transaction_id'].count().sort_values(ascending=False)
        headline = f"4G vs WiFi transaction volume"
        insight = f"4G: {result.get('4G',0):,} transactions, WiFi: {result.get('WiFi',0):,} transactions. Based on {len(df):,} total."
        return _build_fast_response(result, question, headline, insight)
    
    # h) state with most transactions for 18-25
    if "state" in q and ("18-25" in q or "18 to 25" in q):
        filtered = df[df['sender_age_group'] == '18-25']
        result = filtered.groupby('sender_state')['transaction_id'].count().sort_values(ascending=False)
        headline = f"{result.index[0]} leads: {result.iloc[0]:,} total"
        insight = f"{result.index[0]} leads with {result.iloc[0]:,} transactions from the 18-25 age group. Based on {len(filtered):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # i) P2P by age group
    if "p2p" in q and "age" in q:
        p2p = df[df['transaction_type'] == 'P2P']
        result = p2p.groupby('sender_age_group')['transaction_id'].count().sort_values(ascending=False)
        headline = f"P2P by age group"
        insight = f"The {result.index[0]} age group leads P2P transactions with {result.iloc[0]:,}. Based on {len(p2p):,} P2P transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # j) bank with most failed transactions
    if "bank" in q and any(w in q for w in ["fail", "failed", "failure"]) and "rate" not in q:
        failed = df[df['transaction_status'] == 'FAILED']
        result = failed.groupby('sender_bank')['transaction_id'].count().sort_values(ascending=False)
        headline = f"{result.index[0]} has most failures: {result.iloc[0]:,}"
        insight = f"{result.index[0]} has the highest number of failed transactions at {result.iloc[0]:,}. Based on {len(failed):,} failed transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # k) average amount by category
    if "average" in q and "amount" in q and ("category" in q or "food" in q or "entertainment" in q):
        result = df.groupby('merchant_category')['amount_inr'].mean().round(2).sort_values(ascending=False)
        headline = f"Avg amount by category"
        insight = f"Average transaction amounts range from ₹{result.min():,.2f} to ₹{result.max():,.2f}. Based on {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    # l) peak hour by specific category (e.g. Entertainment)
    if "peak hour" in q and "category" not in q:
        known_categories = ['Food', 'Grocery', 'Shopping', 'Utilities', 'Entertainment', 'Healthcare', 'Transport', 'Fuel', 'Education', 'Other']
        extracted_category = None
        for cat in known_categories:
            if cat.lower() in q:
                extracted_category = cat
                break
        if extracted_category:
            filtered = df[df['merchant_category'] == extracted_category]
            if len(filtered) > 0:
                peak_hour = int(filtered['hour_of_day'].value_counts().index[0])
                result = peak_hour
                headline = f"Peak hour: {peak_hour}:00"
                insight = f"The {extracted_category} category sees peak transaction volume at {peak_hour}:00. Based on {len(filtered):,} transactions."
                return _build_fast_response(result, question, headline, insight)
    
    # m) peak hours for ALL categories
    if "peak hour" in q and "category" in q:
        result = df.groupby('merchant_category')['hour_of_day'].agg(lambda x: x.value_counts().index[0])
        result.name = "peak_hour"
        headline = "Peak hours by merchant category"
        insight = f"Peak transaction hours vary by merchant category. Based on {len(df):,} transactions."
        return _build_fast_response(result, question, headline, insight)
    
    return None


class PandasAnalyst(BaseAnalyst):
    def _ask_llm(self, user_question, chat_history, use_fallback=False):
        history_context = ""
        if chat_history:
            recent = chat_history[-4:]
            for msg in recent:
                history_context += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            history_context = f"Recent conversation:\n{history_context}\n"

        col_list = ", ".join(df.columns.tolist())
        row_count = len(df)
        value_hints = []
        for col in df.columns:
            if df[col].dtype == "object":
                uniques = df[col].dropna().unique().tolist()
                if len(uniques) < 20: value_hints.append(f"{col}: {uniques}")
                else: value_hints.append(f"{col}: (too many, e.g. {uniques[:3]})")
            else: value_hints.append(f"{col}: numeric")
        col_detail = "\n".join(value_hints)

        prompt = (
            f"You are a pandas data analyst. DataFrame 'df' has {row_count:,} UPI transactions.\n\n"
            f"Allowed values:\n{col_detail}\n\n{history_context}"
            "RULES:\n1. WRITE ONLY EXECUTABLE PANDAS CODE. STORE IN 'result'.\n"
            "2. No markdown, no print, no comments. Just raw python code like `result = ...`.\n"
            "3. NEVER use .size(), .idxmax(), .sort_values(). For grouping, use .count() or .sum() on a specific column. (e.g. df.groupby('hour_of_day')['transaction_id'].count())\n"
            "4. Supported patterns: df.groupby('col')['agg_col'].func(), df['col'].func(), df[df['col']==val], df['col'].value_counts()\n\n"
            f"Question: {user_question}"
        )
        return llm_client.call_model(prompt, use_fallback=use_fallback)

    def analyze(self, query: str, chat_history: list = None) -> dict:
        start = time.time()
        
        is_valid, err_msg = validate_question(query)
        if not is_valid:
            return format_clarification_response(err_msg)

        resolved_question, corrections, suggestions = resolve_entities(query, df)
        
        cached = get_cached(query, chat_history)
        if cached: return cached

        fast = try_fast_query(resolved_question, df)
        if fast:
            set_cached(query, fast, chat_history)
            return fast

        # Primary LLM Query
        llm_resp = self._ask_llm(resolved_question, chat_history, use_fallback=False)
        code = clean_code(llm_resp["content"])
        
        plan = None
        result = None
        
        try:
            plan = code_to_plan(code)
            result = execute_plan(plan, df)
        except Exception as primary_err:
            print(f"[ENGINE] Primary model AST failed: {primary_err}")
            failure_logger.log_failure(llm_resp["content"], query, str(primary_err))
            
            # Improvement 5: Fallback retry loop
            print("[ENGINE] Falling back to secondary model...")
            try:
                llm_resp = self._ask_llm(resolved_question, chat_history, use_fallback=True)
                code = clean_code(llm_resp["content"])
                plan = code_to_plan(code)
                result = execute_plan(plan, df)
            except Exception as fallback_err:
                print(f"[ENGINE] Fallback also failed: {fallback_err}")
                failure_logger.log_failure(llm_resp["content"], query, str(fallback_err))
                return format_clarification_response(str(fallback_err), code)

        if isinstance(result, str):
            return {"answer": result, "headline": "RESULT", "stats": [], "data": result, "code": code, "chart": None, "followups": [], "verification": {"valid": True, "warnings": [], "errors": []}}
            
        if result is None or (isinstance(result, float) and math.isnan(result)):
            return format_clarification_response("Result was empty or NaN.", code)

        verification = verify_result(df, plan, result)

        if isinstance(result, pd.Series) and len(result) > 0:
            result = result.sort_values(ascending=False)

        chart = generate_chart(result, query)
        followups = generate_followup_questions(query, result)
        headline, insight = generate_insight(query, result)

        stats = []
        if isinstance(result, pd.Series) and len(result) > 0:
            # Result is already sorted descending
            stats = [
                {"label": "HIGHEST", "value": f"{result.index[0]}: {result.iloc[0]:.2f}"},
                {"label": "AVERAGE", "value": f"{result.mean():.2f}"},
                {"label": "LOWEST", "value": f"{result.index[-1]}: {result.iloc[-1]:.2f}"}
            ]

        if corrections:
            insight += " (Auto-corrected: " + ", ".join(f"'{o}' -> '{f}'" for o, f in corrections) + ")"
            
        verification["model_used"] = llm_resp["model_used"]
        verification["fallback_used"] = llm_resp["fallback_used"]

        response = {
            "answer": insight,
            "headline": headline,
            "stats": stats,
            "data": str(result),
            "code": code,
            "chart": chart,
            "followups": followups,
            "verification": verification
        }
        
        set_cached(query, response, chat_history)
        return response
