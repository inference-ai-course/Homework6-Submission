def search_arxiv(query: str) -> str:
    """
    Simulate an arXiv search or return a dummy passage for the given query.
    In a real system, this might query the arXiv API and extract a summary.
    """
    # returning dummy text to avoid rate limits
    return f"[arXiv snippet related to '{query}']"

def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    """
    try:
        from sympy import sympify
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"