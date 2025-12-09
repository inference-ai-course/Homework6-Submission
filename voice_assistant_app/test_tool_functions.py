#!/usr/bin/env python3
"""
Test script for tool functions implementation
"""
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.tool_functions import ToolFunctions
from services.function_call_handler import FunctionCallHandler


def test_tool_functions():
    """Test the tool functions directly"""
    print("=== Testing Tool Functions ===")
    
    tool_functions = ToolFunctions()
    
    # Test search_arxiv
    print("\n1. Testing search_arxiv...")
    try:
        result = tool_functions.search_arxiv("quantum entanglement")
        print(f"Result: {result[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test calculate
    print("\n2. Testing calculate...")
    try:
        result = tool_functions.calculate("2+2")
        print(f"Result: {result}")
        
        result = tool_functions.calculate("sin(pi/2)")
        print(f"Result: {result}")
        
        result = tool_functions.calculate("sqrt(16)")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test function call parsing
    print("\n3. Testing function call parsing...")
    test_cases = [
        '{"function": "calculate", "arguments": {"expression": "2+2"}}',
        '{"function": "search_arxiv", "arguments": {"query": "machine learning"}}',
        'This is a normal response without function calls.',
        'I think {"function": "calculate", "arguments": {"expression": "5*3"}} is the answer.'
    ]
    
    for test_case in test_cases:
        parsed = tool_functions.parse_function_call(test_case)
        print(f"Input: {test_case}")
        print(f"Parsed: {parsed}")
        print()


def test_function_call_handler():
    """Test the function call handler"""
    print("=== Testing Function Call Handler ===")
    
    handler = FunctionCallHandler()
    
    # Test processing LLM responses
    test_responses = [
        '{"function": "calculate", "arguments": {"expression": "2+2"}}',
        '{"function": "search_arxiv", "arguments": {"query": "artificial intelligence"}}',
        'Hello! How can I help you today?',
        'The weather is nice today.'
    ]
    
    for response in test_responses:
        print(f"\nProcessing: {response}")
        result, was_function_call = handler.process_llm_response(response)
        print(f"Result: {result}")
        print(f"Was function call: {was_function_call}")


if __name__ == "__main__":
    print("Testing Tool Functions Implementation")
    print("=" * 50)
    
    try:
        test_tool_functions()
        print("\n" + "=" * 50)
        test_function_call_handler()
        print("\n" + "=" * 50)
        print("Testing completed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
