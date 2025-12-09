"""
LLM Service
Handles conversation using various LLM models
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLaMAService:
    """Service for conversational AI using various LLM models"""

    def __init__(self, model_path: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize LLM service

        Args:
            model_path: Path or HuggingFace model ID for LLM
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"LLM will use device: {self.device}")
        
    def load_model(self):
        """Load the LLM model and tokenizer"""
        try:
            logger.info(f"Loading LLM model: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                device_map="auto" if self.device == "mps" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "mps" else -1
            )
            
            logger.info(f"{self.model_path} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    def format_conversation(self, conversation_history: List[Dict]) -> str:
        """
        Format conversation history for LLM
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Formatted conversation string
        """
        formatted = []
        
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def generate_response(
        self, 
        user_message: str, 
        conversation_history: List[Dict] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response to user message
        
        Args:
            user_message: User's input message
            conversation_history: Previous conversation messages
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        if self.pipeline is None:
            self.load_model()
        
        try:
            # Build the conversation context
            messages = []
            
            # Add system prompt with function calling instructions
            system_prompt = """You are a helpful voice assistant. Provide concise, natural, and conversational responses suitable for speech synthesis.

You have access to the following tools:
1. search_arxiv: Search arXiv for academic papers. Use this when the user asks about research, scientific topics, or wants academic information.
2. calculate: Calculate mathematical expressions. Use this when the user asks for calculations, math problems, or numerical computations.

When the user's question can be answered using these tools, output a JSON function call in this exact format:
{"function": "function_name", "arguments": {"parameter_name": "parameter_value"}}

Examples:
- For math: {"function": "calculate", "arguments": {"expression": "2+2"}}
- For research: {"function": "search_arxiv", "arguments": {"query": "quantum entanglement"}}

If the question doesn't require these tools, respond normally in text without any JSON."""
            
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-10:]:  # Keep last 10 messages
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Format messages for LLaMA3 chat format
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.info(f"Generating response for: {user_message[:50]}...")
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the generated text
            response = generated_text[len(prompt):].strip()
            
            # Clean up the response
            if response.startswith("Assistant:"):
                response = response[10:].strip()
            
            logger.info(f"Generated response: {response[:50]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def generate_streaming_response(
        self,
        user_message: str,
        conversation_history: List[Dict] = None
    ):
        """
        Generate streaming response (for future implementation)
        
        Args:
            user_message: User's input message
            conversation_history: Previous conversation messages
            
        Yields:
            Generated text chunks
        """
        # This is a placeholder for streaming implementation
        # Actual streaming requires TextIteratorStreamer from transformers
        response = self.generate_response(user_message, conversation_history)
        yield response
