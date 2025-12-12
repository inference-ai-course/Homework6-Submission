from huggingface_hub import InferenceClient
import os
import json
from dotenv import load_dotenv
from app.tools import search_arxiv, calculate

# Load environment variables from .env file
load_dotenv()

# Load token from environment variable for security
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in .env file or as environment variable.")

client = InferenceClient(token=HF_TOKEN)

# ----------- SYSTEM PROMPT ------------
SYSTEM_PROMPT = """
You are a helpful and careful assistant with tool-calling abilities.

You can call ONLY the following tools:
1. search_arxiv(query: str)
2. calculate(expression: str)

When a tool is required, respond ONLY with a strict JSON object:
{
  "function": "function_name",
  "arguments": {
      "key": "value"
  }
}

Rules:
- Never include explanations or additional text outside the JSON.
- Never use backticks.
- Never include comments.
- Never hallucinate tools that do not exist.
- If no tool is needed, respond normally in plain text.
- Your JSON must be strictly valid.
"""
# ----------- JSON REPAIR ------------
def try_parse_json(text: str):
    """
    Try loading JSON. If fails, attempt simple repairs.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 🔧 Basic repairs: strip whitespace, remove trailing commas
    repaired = (
        text.replace("\n", " ")
            .replace("\t", " ")
            .strip()
            .rstrip(",")
    )

    # Try again
    try:
        return json.loads(repaired)
    except Exception:
        return None

# ----------- LLM CALL ------------
def generate_response(text: str) -> str:
    try:
        # chat completion API
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            max_tokens=300
        )
        # response is ChatCompletionOutput object
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message["content"]
        else:
            return str(response)
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
# ----------- TOOL ROUTER ------------
def route_llm_output(llm_output: str) -> str:

    parsed = try_parse_json(llm_output)

    if not parsed or not isinstance(parsed, dict):
        return llm_output.strip()

    func_name = parsed.get("function")

    if func_name == "calculate":
        return "Let me calculate that for you..."
    elif func_name == "search_arxiv":
        return "Searching academic papers for you..."
    else:
        return "Processing your request..."
