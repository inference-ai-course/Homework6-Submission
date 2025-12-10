# Tool Functions Implementation Guide

## Overview

This voice assistant now supports tool functions that allow it to perform mathematical calculations and search arXiv for academic papers. The implementation includes:

1. **Tool Functions**: `search_arxiv` and `calculate` functions
2. **Function Calling**: LLM can generate JSON function calls when appropriate
3. **Detection & Execution**: Automatic parsing and execution of function calls
4. **Fallback Behavior**: Graceful handling of errors and normal responses

## Available Functions

### 1. search_arxiv(query: str) -> str
Searches arXiv for academic papers and returns a summary.

**Usage Examples:**
- "Search for papers on quantum entanglement"
- "What research exists about machine learning?"
- "Find academic papers on artificial intelligence"

**Expected LLM Output:**
```json
{"function": "search_arxiv", "arguments": {"query": "quantum entanglement"}}
```

### 2. calculate(expression: str) -> str
Evaluates mathematical expressions using SymPy.

**Usage Examples:**
- "What is 2+2?"
- "Calculate sin(pi/2)"
- "What's the square root of 16?"
- "Evaluate 5 * 3 + 2"

**Expected LLM Output:**
```json
{"function": "calculate", "arguments": {"expression": "2+2"}}
```

## Implementation Details

### Files Added/Modified

1. **services/tool_functions.py**: Core tool functions implementation
2. **services/function_call_handler.py**: Function call detection and execution
3. **services/llama_service.py**: Updated system prompt for function calling
4. **main.py**: Integrated function call handler into voice chat endpoints
5. **requirements.txt**: Added arxiv and sympy dependencies

### Function Call Flow

1. User speaks a query
2. Whisper transcribes to text
3. LLaMA generates response (may include function call JSON)
4. FunctionCallHandler detects and parses function calls
5. If function call detected:
   - Execute the function
   - Use function result as assistant response
6. If no function call:
   - Use original LLM response
7. SpeechT5 synthesizes final response to audio

### Error Handling

- Invalid function names return error messages
- Parse errors fall back to original LLM response
- Network errors in arxiv search return helpful error messages
- Invalid math expressions return error explanations

## Testing

Run the test script to verify functionality:

```bash
python test_tool_functions.py
```

This tests:
- Direct function calls
- JSON parsing
- Function call detection
- Error handling

## Usage in Production

The system automatically detects when function calls are needed based on the LLM's response. No additional configuration is required - the enhanced system prompt instructs the LLM to use function calls when appropriate.

## Supported Mathematical Operations

The `calculate` function supports:
- Basic arithmetic: +, -, *, /
- Powers: ^ or **
- Trigonometric: sin, cos, tan
- Logarithms: log, ln
- Square root: sqrt
- Constants: pi, e
- Parentheses for grouping

## arXiv Search Features

The `search_arxiv` function:
- Returns top 3 most relevant papers
- Includes title, authors, publication date
- Provides summary snippet (first 300 characters)
- Handles no results gracefully
