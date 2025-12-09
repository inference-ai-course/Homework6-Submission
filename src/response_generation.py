import torch
from transformers import pipeline
import json

conversation_history = []

llm = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.1-8B",
    model_kwargs={"dtype": torch.bfloat16}, 
    device_map="auto"
)

def generate_response(user_text):
    # Add user message to history (chat template expects "content" not "text")
    conversation_history.append({"role": "user", "content": user_text})
    
    # Generate response with temperature=0 for deterministic output
    outputs = llm(user_text, max_new_tokens=100, return_full_text=False, temperature=1)
    bot_response = outputs[0]["generated_text"].strip()
    
    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": bot_response})
    return bot_response

def search_arxiv(query):
    return "I'm sorry, I can't search the web for you."

def calculate(expression):
    """
    Evaluate a mathematical expression and return the result as a string.
    """
    try:
        from sympy import sympify
        result = sympify(expression)  # use sympy for safe evaluation
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def route_llm_output(llm_output: str) -> str:
    """
    Route LLM response to the correct tool if it's a function call, else return the text.
    Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
    """
    try:
        output = json.loads(llm_output)
        func_name = output.get("function")
        args = output.get("arguments", {})
    except (json.JSONDecodeError, TypeError):
        # Not a JSON function call; return the text directly
        return llm_output

    if func_name == "search_arxiv":
        query = args.get("query", "")
        return search_arxiv(query)
    elif func_name == "calculate":
        expr = args.get("expression", "")
        return calculate(expr)
    else:
        return f"Error: Unknown function '{func_name}'"