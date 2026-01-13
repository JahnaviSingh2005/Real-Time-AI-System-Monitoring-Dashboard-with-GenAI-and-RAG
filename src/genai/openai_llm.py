"""
OpenAI LLM Integration
----------------------
This module handles communication with OpenAI's API to provide
high-quality AI-generated answers for the system monitoring assistant.
"""

import os
from typing import Optional, List, Dict

# System instruction for the AI assistant
SYSTEM_INSTRUCTION = """You are an expert AI System Monitoring Assistant. Your role is to:
1. Answer questions about the system's current performance using the live metrics provided
2. Help troubleshoot CPU, memory, disk, and network issues
3. Provide actionable recommendations based on the current system state
4. Explain what different metrics mean and when they're concerning
5. Help users understand and manage running processes

Always use the CURRENT LIVE SYSTEM METRICS provided in the context to give accurate, real-time answers.
Be concise but informative. Use bullet points and formatting for clarity.
If you don't have enough information, say so and suggest what to check."""


class OpenAILLM:
    """
    Wrapper for OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API Key. If None, looks for OPENAI_API_KEY env var.
            model_name: Name of the OpenAI model to use.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.client = None
        self.is_configured = False
        
        if self.api_key:
            self.configure(self.api_key)
            
    def configure(self, api_key: str, model_name: Optional[str] = None):
        """Configure the OpenAI API with a key and optional model change"""
        try:
            if model_name:
                self.model_name = model_name
            
            # Import openai here to avoid import errors if not installed
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.api_key = api_key
                self.is_configured = True
                return True
            except ImportError:
                print("OpenAI package not installed. Run: pip install openai")
                self.is_configured = False
                return False
                
        except Exception as e:
            print(f"Error configuring OpenAI: {e}")
            self.is_configured = False
            return False
            
    def get_available_models(self) -> List[str]:
        """Return available OpenAI models"""
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ]
            
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using OpenAI.
        """
        if not self.is_configured or not self.client:
            return "OpenAI API is not configured. Please provide a valid API key."
            
        full_prompt = ""
        if context:
            full_prompt = f"Context:\n{context}\n\n"
        full_prompt += f"User Question: {prompt}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from OpenAI: {str(e)}")
            return None

    def generate_chat_response(self, history: List[Dict], user_message: str, context: Optional[str] = None) -> str:
        """
        Generate a response in a chat session.
        """
        if not self.is_configured or not self.client:
            return "OpenAI API is not configured."
            
        # Build messages array
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        
        # Add history
        for msg in history[-10:]:
            role = "user" if msg['role'] == 'user' else "assistant"
            messages.append({"role": role, "content": msg['message']})
        
        # Add current message with context
        current_content = ""
        if context:
            current_content = f"CURRENT SYSTEM DATA:\n{context}\n\nUSER QUESTION: {user_message}"
        else:
            current_content = user_message
            
        messages.append({"role": "user", "content": current_content})
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in OpenAI chat: {error_msg}")
            # Return None to allow fallback to local analysis
            return None
