"""
Tool Functions Service
Provides helper functions for arXiv search and mathematical calculations
"""
import json
import logging
import re
import arxiv
import sympy
from sympy.parsing.sympy_parser import parse_expr
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ToolFunctions:
    """Service for tool functions that can be called by the LLM"""
    
    def __init__(self):
        """Initialize the tool functions service"""
        self.available_functions = {
            "search_arxiv": self.search_arxiv,
            "calculate": self.calculate
        }
    
    def search_arxiv(self, query: str) -> str:
        """
        Search arXiv for relevant papers and return a summary
        
        Args:
            query: Search query for arXiv papers
            
        Returns:
            Summary of relevant paper(s) found
        """
        try:
            logger.info(f"Searching arXiv for query: {query}")
            
            # Search arXiv
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                # Get summary from abstract
                summary = f"Title: {paper.title}\n"
                summary += f"Authors: {', '.join(author.name for author in paper.authors[:3])}"
                if len(paper.authors) > 3:
                    summary += " et al."
                summary += f"\nPublished: {paper.published.strftime('%Y-%m-%d')}\n"
                summary += f"Summary: {paper.summary[:300]}..."
                
                results.append(summary)
            
            if not results:
                return f"No relevant papers found for query: {query}"
            
            # Return the most relevant result
            return results[0]
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return f"Sorry, I encountered an error searching arXiv: {str(e)}"
    
    def calculate(self, expression: str) -> str:
        """
        Evaluate a mathematical expression using SymPy
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation as a string
        """
        try:
            logger.info(f"Calculating expression: {expression}")
            
            # Clean the expression
            expr = expression.strip()
            
            # Handle common mathematical functions and constants
            replacements = {
                'pi': 'pi',
                'e': 'E',
                'sqrt': 'sqrt',
                'sin': 'sin',
                'cos': 'cos',
                'tan': 'tan',
                'log': 'log',
                'ln': 'log',
                '^': '**'
            }
            
            for old, new in replacements.items():
                expr = expr.replace(old, new)
            
            # Parse and evaluate the expression
            parsed_expr = parse_expr(expr, evaluate=True)
            result = parsed_expr.evalf()
            
            # Format the result
            if result.is_Integer:
                result_str = str(int(result))
            elif result.is_Rational and result.q == 1:
                result_str = str(int(result))
            elif abs(result) < 1e-10:
                result_str = "0"
            else:
                # Round to reasonable precision
                result_str = f"{float(result):.6g}"
            
            return f"The result of {expression} is {result_str}"
            
        except Exception as e:
            logger.error(f"Error calculating expression: {e}")
            return f"Sorry, I couldn't calculate {expression}. Error: {str(e)}"
    
    def parse_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a function call from LLM output
        
        Args:
            text: LLM output text that might contain a function call
            
        Returns:
            Parsed function call as dict, or None if not a function call
        """
        try:
            # Look for JSON function call pattern
            json_pattern = r'\{["\']function["\']:\s*["\']([^"\']+)["\'],\s*["\']arguments["\']:\s*\{([^}]*)\}\}'
            match = re.search(json_pattern, text, re.IGNORECASE)
            
            if match:
                function_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                # Parse arguments
                arguments = {}
                arg_pattern = r'["\']([^"\']+)["\']:\s*["\']([^"\']+)["\']'
                for arg_match in re.finditer(arg_pattern, arguments_str):
                    arg_name = arg_match.group(1).strip()
                    arg_value = arg_match.group(2).strip()
                    arguments[arg_name] = arg_value
                
                return {
                    "function": function_name,
                    "arguments": arguments
                }
            
            # Alternative pattern: more flexible JSON parsing
            try:
                # Try to extract JSON from the text
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    if 'function' in parsed and 'arguments' in parsed:
                        return parsed
            except json.JSONDecodeError:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing function call: {e}")
            return None
    
    def execute_function_call(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a function call
        
        Args:
            function_name: Name of the function to call
            arguments: Arguments to pass to the function
            
        Returns:
            Result of the function call
        """
        try:
            if function_name not in self.available_functions:
                return f"Unknown function: {function_name}"
            
            function = self.available_functions[function_name]
            
            if function_name == "search_arxiv":
                return function(arguments.get("query", ""))
            elif function_name == "calculate":
                return function(arguments.get("expression", ""))
            else:
                return f"Function {function_name} not implemented"
                
        except Exception as e:
            logger.error(f"Error executing function call: {e}")
            return f"Error executing {function_name}: {str(e)}"
