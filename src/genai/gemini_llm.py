"""
Gemini LLM Integration
----------------------
This module handles communication with Google's Gemini API to provide
high-quality AI-generated answers for the system monitoring assistant.
"""

import google.generativeai as genai
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


class GeminiLLM:
    """
    Wrapper for Google Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Google AI API Key. If None, looks for GEMINI_API_KEY env var.
            model_name: Name of the Gemini model to use.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        self.is_configured = False
        
        if self.api_key:
            self.configure(self.api_key)
            
    def configure(self, api_key: str, model_name: Optional[str] = None):
        """Configure the Gemini API with a key and optional model change"""
        try:
            if model_name:
                self.model_name = model_name
            genai.configure(api_key=api_key)
            
            # Create model with system instruction for supported models
            try:
                self.model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=SYSTEM_INSTRUCTION
                )
            except Exception:
                # Fallback for models that don't support system_instruction
                self.model = genai.GenerativeModel(self.model_name)
                
            self.api_key = api_key
            self.is_configured = True
            return True
        except Exception as e:
            print(f"Error configuring Gemini: {e}")
            self.is_configured = False
            return False
            
    def get_available_models(self) -> List[str]:
        """Fetch available models from the API"""
        if not self.is_configured:
            return []
        try:
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # Remove 'models/' prefix if present for display
                    name = m.name.replace('models/', '')
                    models.append(name)
            return sorted(models)
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
            
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: User question
            context: Optional context from RAG or system metrics
            
        Returns:
            AI-generated response
        """
        if not self.is_configured or not self.model:
            return "Gemini API is not configured. Please provide a valid API key."
            
        full_prompt = ""
        if context:
            full_prompt = f"Context:\n{context}\n\n"
            
        full_prompt += f"User Question: {prompt}"
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response from Gemini: {str(e)}")
            return None

    def generate_chat_response(self, history: List[Dict], user_message: str, context: Optional[str] = None) -> str:
        """
        Generate a response in a chat session.
        """
        if not self.is_configured or not self.model:
            return "Gemini API is not configured."
            
        # Convert history format to Gemini format
        gemini_history = []
        for msg in history[-10:]:  # Last 10 messages for context
            role = "user" if msg['role'] == 'user' else "model"
            gemini_history.append({"role": role, "parts": [msg['message']]})
            
        try:
            # Start chat with history
            chat = self.model.start_chat(history=gemini_history)
            
            # Build the prompt with context
            prompt = ""
            if context:
                prompt = f"CURRENT SYSTEM DATA:\n{context}\n\nUSER QUESTION: {user_message}"
            else:
                prompt = user_message
                
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in Gemini chat: {error_msg}")
            
            # For ALL errors, return None to allow fallback to local analysis
            # This ensures the app works even when Gemini is unavailable
            # The local analysis will handle: storage, processes, cleanup, status queries
            return None

