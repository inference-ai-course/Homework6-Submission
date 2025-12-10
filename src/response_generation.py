import torch
from transformers import pipeline, AutoTokenizer
import json

conversation_history = []

llm = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.1-8B",
    model_kwargs={"dtype": torch.bfloat16}, 
    device_map="auto"
)

# Get the tokenizer for chat template formatting
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

def generate_response(user_text):
    # System prompt instructing the model about function calling
    system_prompt = """You are a helpful AI assistant. You can respond to users in two ways:

1. **Function Calls**: If the user's question can be answered by:
   - Performing a mathematical calculation (e.g., "what is 2+2?", "calculate 15*7")
   - Searching arXiv for research papers (e.g., "search for papers on quantum entanglement", "find arXiv papers about machine learning")

   Then you MUST output ONLY a JSON object in this exact format:
   {"function": "calculate", "arguments": {"expression": "2+2"}}
   or
   {"function": "search_arxiv", "arguments": {"query": "quantum entanglement"}}

2. **Normal Text Response**: For all other questions, respond normally with helpful text.

Important: 
- Output ONLY the JSON object for function calls, no additional text
- For normal responses, output plain text as usual
- Be concise and direct"""
    
    # Add user message to history (chat template expects "content" not "text")
    conversation_history.append({"role": "user", "content": user_text})
    
    # Build messages with system prompt
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    
    # Apply chat template to format messages correctly
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response using the formatted prompt
    outputs = llm(
        formatted_prompt,
        max_new_tokens=200,
        return_full_text=False,
        temperature=0.7,
        do_sample=True
    )
    bot_response = outputs[0]["generated_text"].strip()
    
    # Route the output to handle function calls if needed
    final_response = route_llm_output(bot_response)
    
    # Add assistant response to history (store the raw response, not the routed one)
    conversation_history.append({"role": "assistant", "content": bot_response})
    
    return final_response

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