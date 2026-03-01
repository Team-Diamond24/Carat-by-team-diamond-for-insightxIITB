import hashlib
import re
from collections import OrderedDict

# --- Universal Result Cache (LRU) ---
FAST_CACHE = OrderedDict()
_CACHE_MAX = 200

def canonicalize(question: str, chat_history=None):
    """Normalize question and mix with history for cache key."""
    norm_q = question.strip().lower()
    if not chat_history:
        return norm_q
    
    recent = chat_history[-4:]
    history_str = "|".join([f"{msg.get('role', '')}:{msg.get('content', '')}" for msg in recent])
    hist_hash = hashlib.md5(history_str.encode()).hexdigest()
    return f"{norm_q}::{hist_hash}"


def get_cached(question: str, chat_history=None):
    """Return cached result or None. Marks the entry as recently used (LRU)."""
    key = canonicalize(question, chat_history)
    if key in FAST_CACHE:
        FAST_CACHE.move_to_end(key)  # Mark as recently used
        return FAST_CACHE[key]
    return None


def set_cached(question: str, value: dict, chat_history=None):
    """Store result in cache, evict least-recently-used if full."""
    key = canonicalize(question, chat_history)
    if key in FAST_CACHE:
        FAST_CACHE.move_to_end(key)
    FAST_CACHE[key] = value
    if len(FAST_CACHE) > _CACHE_MAX:
        FAST_CACHE.popitem(last=False)  # Evict least recently used


# --- Shared Question Validator ---
_DATA_KEYWORDS = {
    "transaction", "amount", "bank", "state", "fraud", "device", "network",
    "age", "group", "p2p", "p2m", "upi", "payment", "recharge", "bill",
    "success", "fail", "rate", "average", "mean", "total", "count",
    "compare", "top", "highest", "lowest", "most", "least", "peak",
    "hour", "day", "weekend", "merchant", "category", "trend", "distribution",
    "breakdown", "percentage", "how", "what", "which", "show", "list",
    "many", "much", "volume", "status", "sender", "receiver", "flagged",
    "android", "ios", "web", "wifi", "scatter", "chart", "plot",
    "income", "spending", "transfer", "value", "number", "ratio",
    "expensive", "cheap", "rich", "poor", "high", "low", "over", "under",
    "between", "across", "per", "each", "every", "monthly", "daily",
    "weekly", "type", "kind", "sort", "rank", "order", "sum", "median",
    "max", "min", "quartile", "analyze", "analyse", "insight", "pattern",
    "correlation", "split", "segment", "filter", "anomaly", "detect",
}

def validate_question(question: str) -> tuple[bool, str]:
    """Validate that a question is worth sending to the LLM."""
    q = question.strip().lower()
    
    if len(q) < 10:
        return False, "This question is too short. Please ask a specific data question."
        
    words = set(re.findall(r'\w+', q))
    if not any(kw in words for kw in _DATA_KEYWORDS):
        return False, "This doesn't seem to be a question about UPI transactions, banks, states, or fraud. Please ask a data question."
        
    return True, ""


# --- Shared Followup Synthesizer ---
def generate_followup_questions(question: str, result_summary: str = "") -> list:
    """Generate generic followups based on keywords in the question."""
    q = question.lower()
    followups = []
    
    if "bank" in q:
        followups = ["Which state has the most bank errors?", "Compare HDFC and SBI", "What is the average amount sent from Axis?"]
    elif "state" in q or "where" in q:
        followups = ["What is the highest spending state?", "Show fraud by state", "Are there any states with >5% fraud?"]
    elif "fraud" in q or "flag" in q:
        followups = ["Which bank flags the most fraud?", "Is fraud higher on weekends?", "What is the average fraudulent amount?"]
    elif "device" in q or "ios" in q or "android" in q:
        followups = ["Compare transaction failure rates between iOS and Android", "What is the median amount on Android?"]
    else:
        followups = ["What is the overall average transaction amount?", "Show me the top 3 states by volume", "Which bank has the most failures?"]
        
    return followups


class BaseAnalyst:
    """
    Abstract interface for handling natural language queries.
    Implemented by PandasAnalyst (engine.py) and SQLAnalyst (sql_analyst.py).
    """
    def analyze(self, query: str, chat_history: list) -> dict:
        """
        Takes a natural language query and context, returns structured dict:
        {
            "answer": str,
            "headline": str,
            "stats": list[dict],
            "data": str or dict,
            "code": str,
            "chart": str or None,
            "followups": list[str],
            "verification": dict
        }
        """
        raise NotImplementedError("analyze() must be implemented by subclasses.")

def format_clarification_response(error_str: str, code: str = None) -> dict:
    return {
        "answer": "Unable to interpret or safely execute this query. Please try rephrasing.",
        "headline": "Clarification required",
        "stats": [],
        "data": None,
        "code": code,
        "chart": None,
        "followups": ["Break this down by state", "Compare by device type", "Trend over time"],
        "verification": {"valid": False, "warnings": [], "errors": [error_str]}
    }
