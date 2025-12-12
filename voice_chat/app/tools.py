import json
import time
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def save_interaction(user_text, raw_llm_output, bot_text, func_output=None):
    """Save the interaction log to a file"""
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_text": user_text,
        "raw_llm_output": raw_llm_output,
        "function_output": func_output,
        "bot_response": bot_text
    }
    timestamp = int(time.time() * 1000)
    log_file = LOG_DIR / f"log_{timestamp}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)
    print(f"[LOG] Saved to {log_file}")

def calculate(expression: str):
    """
    Evaluate a mathematical expression and return the numeric result.
    Returns int/float for simple expressions, str for symbolic ones.
    """
    try:
        from sympy import sympify, N
        print(f"[TOOL] Calculator input: '{expression}'")
        
        # Parse expression
        result = sympify(expression)
        
        # Try to convert to numeric
        try:
            numeric_result = float(N(result))
            # Return int if whole number
            if numeric_result.is_integer():
                final_result = int(numeric_result)
            else:
                final_result = round(numeric_result, 6) 
            
            print(f"[TOOL] Calculator output: {final_result}")
            return final_result
            
        except (TypeError, AttributeError):
            # Symbolic expression (like 'x + 1')
            str_result = str(result)
            print(f"[TOOL] Calculator output (symbolic): {str_result}")
            return str_result
            
    except Exception as e:
        error_msg = f"Calculation error: {str(e)}"
        print(f"[TOOL ERROR] {error_msg}")
        return error_msg

def search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search (placeholder implementation).
    """
    print(f"[TOOL] ArXiv search: '{query}'")
    return f"[arXiv snippet related to '{query}']"

# ---------------- FUNCTION_MAP ----------------
FUNCTION_MAP = {
    "calculate": calculate,
    "search_arxiv": search_arxiv,   
}