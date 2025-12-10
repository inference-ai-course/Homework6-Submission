from transformers import AutoTokenizer, pipeline
from sympy import sympify, SympifyError
import requests
import xml.etree.ElementTree as ET
import logging
import json

logger = logging.getLogger(__name__)

conversation_history = []

model_path = "meta-llama/Llama-3.2-1B-Instruct"
llm = pipeline("text-generation", model=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def generate_response(user_text):
    messages = []
    instruction1 = '''
        If the user question contains "arxiv" or "ARXIV", return the answer in this json format:
        {"function": "search_arxiv", "arguments": {"query": "query content"}}

        For example, if the user question is "Do ARXIV search on quantum entanglement", then you should return 
        {"function": "search_arxiv", "arguments": {"query": "quantum entanglement"}}
    '''

    instruction2 = '''
        If the user question contains "calculate" or "Calculate", return the answer in this json format:
        {"function": "calculate", "arguments": {"expression": "math expression"}}

        For example, if the user question is "Calculate 3 plus 3", then you should return 
        {"function": "calculate", "arguments": {"query": "3 + 3"}}
    '''        
    # Add system prompt
    messages.append({"role": "system", "content": instruction1})
    messages.append({"role": "system", "content": instruction2})
    
    conversation_history.append({"role": "user", "content": user_text})
    for msg in conversation_history[-5:]:
        messages.append({"role": msg["role"],"content": msg["content"]})
    
    # Format messages for LLaMA3 chat format
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate response
    outputs = llm(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    bot_response = outputs[0]["generated_text"]
    bot_response = bot_response[len(prompt):].strip()
    conversation_history.append({"role": "assistant", "content": bot_response})
    return bot_response

def is_function(bot_text):
    if bot_text.startswith('{"function": "'):
        return True, bot_text
    else:
        signal = '```json'
        start = bot_text.find(signal)
        if start < 0:
            return False, bot_text
        else:
            bot_text = bot_text[start + len(signal):].strip()
            function_text = bot_text[0: bot_text.find('"}}')+3]
            return function_text.startswith('{"function": "'), function_text


def call_function(function_text):
    '''
    function_text format:
    {"function": "search_arxiv", "arguments": {"query": "Liar's Bench"}}
    {"function": "calculate", "arguments": {"expression": "2 ** 3"}}
    '''
    try:
        output = json.loads(function_text)
        func_name = output.get("function")
        args = output.get("arguments", {})
    except (json.JSONDecodeError, TypeError):
        return function_text

    if func_name == "search_arxiv":
        query = args.get("query", "")
        return search_arxiv(query)
    elif func_name == "calculate":
        expr = args.get("expression", "")
        return calculate(expr)
    else:
        return f"Error: Unknown function '{func_name}'"


def search_arxiv(query):
    logger.info(f'***search_arxiv() called: query={query}')
    api = f'https://export.arxiv.org/api/query?search_query={query}'
    response = requests.get(api)
    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    # Find all entries
    entries = root.findall('atom:entry', ns)
    papers = []    
    for entry in entries:
        paper_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text
        papers.append({
            'id': paper_id,
            'title': title,
            'summary': summary
        })
    top_n = 3
    result =  f'Here are the top {top_n} ARXIV papers found on "{query}":\n'
    for paper in papers[:top_n]:
        result += f"Title: {paper['title']}; Summary: {paper['summary']}\n"
    logger.info(f'***search_arxiv() returns: {result}')
    return result


def calculate(expression):
    """
    Calculate using SymPy (symbolic mathematics library).
    Safer than eval and supports more complex expressions.
    """
    logger.info(f'***calculate() called: expression={expression}')
    try:
        result = sympify(expression)
        ret = f'The result of "{expression}" is {result}'
        logger.info(f'***calculate() returns: {ret}')
        return ret
    except SympifyError as e:
        err =  f"Error: {e}"
        logger.error(err)
        return err
    
