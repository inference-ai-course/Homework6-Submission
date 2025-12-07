from ollama import Client
import json
from server.src.tools.tools import calculate, search_arxiv

client = Client(host='http://localhost:11434')

model_name = "gpt-oss:20b-cloud"

prompts = [
    {"role": "system", "content": "You are a helpful assistant. Your responses will be used for TTS as a live conversation, so keep your responses short. The user will not be able to see any visuals or read any latex/math. Respond in one sentence."},
    {"role": "system", "content": """You have access to 2 tools. When appropriate, invoke these tools in a JSON format. 

The first tool is calculate, and it is used to calculate mathematical expressions. Use this when any math is required, even if it is trivial, like 1 + 1. The format is: {"function": "calculate", "arguments": { "expression": string }}

Your second tool is searching arXiv for relevant passages from scientific papers. Use this whenever the user asks a question related to science. The format is:

{"function": "search_arxiv", "arguments": { "query": string }}"""}
]


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

    print("Function call made to: ", func_name)
    if func_name == "search_arxiv":
        query = args.get("query", "")
        output =  search_arxiv(query)
    elif func_name == "calculate":
        expr = args.get("expression", "")
        output =  calculate(expr)
    else:
        return f"Error: Unknown function '{func_name}'"
    print("Function output: ", output)
    return output

def generate_response(user_text):
    conversation = [*prompts, {"role": "user", "content": user_text}]

    response = client.chat(model=model_name, messages=conversation, options={
    })

    tool_calls = response.message.tool_calls
    if tool_calls:
        call = tool_calls[0]
        print(call)
        name = call.function.name
        print("Function call made to: ", name)
        if name == "calculate":
            output = calculate(call.function.arguments["arguments"]["expression"])
        elif name == "search_arxiv":
            output = search_arxiv(call.function.arguments["arguments"]["query"])
        else:
            return f"Error: Unknown function '{name}'"
        print("Function output:", output)
        return output

    generated_text = response["message"]["content"]
    print("LLM output:", generated_text)

    return generated_text