"""
Function Call Handler Service
Handles detection and execution of function calls from LLM responses
"""
import logging
from typing import Dict, Any, Optional, Tuple
from .tool_functions import ToolFunctions

logger = logging.getLogger(__name__)


class FunctionCallHandler:
    """Service for handling function calls from LLM responses"""
    
    def __init__(self):
        """Initialize the function call handler"""
        self.tool_functions = ToolFunctions()
    
    def process_llm_response(self, llm_response: str) -> Tuple[str, bool]:
        """
        Process LLM response to detect and execute function calls
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Tuple of (final_response, was_function_call)
            - final_response: Either the function result or original LLM response
            - was_function_call: Boolean indicating if a function was called
        """
        try:
            # Try to parse function call from LLM response
            function_call = self.tool_functions.parse_function_call(llm_response)
            
            if function_call:
                logger.info(f"Detected function call: {function_call}")
                
                # Execute the function call
                function_name = function_call.get("function")
                arguments = function_call.get("arguments", {})
                
                result = self.tool_functions.execute_function_call(function_name, arguments)
                
                logger.info(f"Function result: {result}")
                return result, True
            
            # No function call detected, return original response
            logger.info("No function call detected, using original LLM response")
            return llm_response, False
            
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            # Fallback to original response on error
            return llm_response, False
    
    def is_function_call(self, text: str) -> bool:
        """
        Check if text contains a function call
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains a function call, False otherwise
        """
        return self.tool_functions.parse_function_call(text) is not None
