import os
import json
from datetime import datetime, timezone

class FailureLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "ast_failures.jsonl")

    def log_failure(self, raw_llm_output: str, query: str, error: str, unsupported_node_type: str = None):
        """Log an AST parser failure."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_llm_output": raw_llm_output,
            "query": query,
            "error": str(error),
            "unsupported_node_type": unsupported_node_type or "Unknown"
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[FailureLogger] Failed to write log: {e}")

    def get_recent_failures(self, limit: int = 50) -> list:
        """Retrieve recent parser failures for admin review."""
        if not os.path.exists(self.log_file):
            return []
            
        failures = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        failures.append(json.loads(line))
        except Exception as e:
            print(f"[FailureLogger] Failed to read logs: {e}")
            
        return failures[-limit:]

failure_logger = FailureLogger()
