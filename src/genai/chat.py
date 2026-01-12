"""
Chat Interface for System Monitoring
-------------------------------------
This module provides a chat interface for users to ask questions
about the system using RAG for context retrieval.
"""

from typing import List, Dict, Optional
from datetime import datetime
from .rag_system import RAGSystem
from .gemini_llm import GeminiLLM


class ChatInterface:
    """
    Chat interface for asking questions about system monitoring.
    
    Uses RAG to provide context-aware responses and Gemini for 
    high-quality AI generation.
    """
    
    def __init__(self, rag_system: RAGSystem, llm: Optional[GeminiLLM] = None):
        """
        Initialize chat interface.
        
        Args:
            rag_system: RAG system for retrieving relevant context
            llm: Gemini LLM for generating responses
        """
        self.rag_system = rag_system
        self.llm = llm
        self.chat_history: List[Dict] = []
        
        # Predefined responses for common questions
        self.knowledge_base = {
            'cpu': {
                'keywords': ['cpu', 'processor', 'processing'],
                'response': """**About CPU Usage:**
                
CPU (Central Processing Unit) usage indicates how much of your processor's capacity is being used. 

**Normal Range:** 0-70%
**Warning:** 70-85%
**Critical:** >85%

**Common Causes of High CPU:**
- Background applications
- Antivirus scans
- System updates
- Runaway processes
- Insufficient cooling (thermal throttling)

**Quick Fixes:**
1. Open Task Manager (Ctrl+Shift+Esc)
2. Sort processes by CPU usage
3. End non-essential high-CPU processes
4. Check for malware
"""
            },
            'memory': {
                'keywords': ['memory', 'ram'],
                'response': """**About Memory Usage:**
                
Memory (RAM) usage shows how much of your system's random access memory is in use.

**Normal Range:** 0-75%
**Warning:** 75-85%
**Critical:** >85%

**Common Causes of High Memory:**
- Too many open applications
- Memory leaks in applications
- Browser with many tabs
- Large file processing
- Insufficient RAM for workload

**Quick Fixes:**
1. Close unnecessary applications
2. Restart memory-intensive programs
3. Clear browser cache and close unused tabs
4. Use Task Manager to identify memory hogs
5. Consider adding more RAM if consistently high
"""
            },
            'disk': {
                'keywords': ['disk', 'storage', 'space'],
                'response': """**About Disk Usage:**
                
Disk usage indicates how much of your storage space is filled.

**Normal Range:** 0-80%
**Warning:** 80-90%
**Critical:** >90%

**Common Causes of High Disk Usage:**
- Large media files
- System logs
- Temporary files
- Application caches
- Old backups

**Quick Fixes:**
1. Run Disk Cleanup utility
2. Delete temporary files
3. Empty Recycle Bin
4. Uninstall unused programs
5. Move large files to external storage
6. Use Storage Sense (Windows)
"""
            },
            'anomaly': {
                'keywords': ['anomaly', 'unusual', 'abnormal', 'strange'],
                'response': """**About Anomaly Detection:**
                
This system uses two methods to detect anomalies:

**1. Rule-Based Detection:**
- Checks if metrics exceed predefined thresholds
- Fast and simple
- Example: Alert if CPU > 85%

**2. ML-Based Detection (Isolation Forest):**
- Learns normal system behavior patterns
- Detects unusual combinations of metrics
- Can find subtle issues that rules miss
- Example: Detecting unusual correlations

**When Both Trigger:**
If both systems detect an anomaly, it's likely a significant issue requiring immediate attention.
"""
            },
            'rag': {
                'keywords': ['rag', 'past', 'history', 'similar', 'before'],
                'response': """**About RAG (Retrieval-Augmented Generation):**
                
RAG helps find similar past incidents to provide context for current issues.

**How It Works:**
1. Stores past incidents in a vector database
2. When an issue occurs, searches for similar past cases
3. Retrieves relevant resolutions and context
4. Helps you benefit from past experience

**Benefits:**
- Learn from history
- Faster troubleshooting
- Consistent solutions
- Knowledge preservation

You can ask: "Show me similar past incidents" or "What happened before with high CPU?"
"""
            }
        }
    
    def process_message(self, user_message: str, current_metrics: Optional[Dict] = None) -> str:
        """
        Process user message and generate response.
        
        Args:
            user_message: User's question or message
            current_metrics: Current system metrics (optional)
            
        Returns:
            Response text
        """
        # Add to chat history
        self.chat_history.append({
            'timestamp': datetime.now(),
            'role': 'user',
            'message': user_message
        })
        
        response = None
        user_message_lower = user_message.lower()
        
        # If Gemini is configured, use it with RAG context
        if self.llm and self.llm.is_configured:
            # 1. Get context from RAG
            rag_context = ""
            if any(word in user_message_lower for word in ['similar', 'past', 'before', 'history']):
                if current_metrics:
                    rag_context = self.rag_system.get_context_for_anomaly(current_metrics)
                else:
                    incidents = self.rag_system.search_similar_incidents(user_message, n_results=3)
                    if incidents:
                        rag_context = "Found similar past incidents:\n" + "\n".join([f"- {inc['description']}" for inc in incidents])
            
            # 2. Add current metrics context if available
            metrics_context = ""
            if current_metrics:
                metrics_context = f"Current System Stats: CPU {current_metrics.get('cpu_percent')}% , Mem {current_metrics.get('memory_percent')}% , Disk {current_metrics.get('disk_percent')}%"
            
            full_context = f"{metrics_context}\n{rag_context}"
            
            # 3. Generate response using Gemini
            response = self.llm.generate_chat_response(self.chat_history[:-1], user_message, full_context)
        
        # Fallback to internal knowledge base if Gemini failed or is not available
        if not response:
            # Check for knowledge base matches
            response = self._check_knowledge_base(user_message_lower)
            
            # If no knowledge base match, try RAG search
            if not response:
                response = self._search_with_rag(user_message, current_metrics)
        
        # Add response to history
        self.chat_history.append({
            'timestamp': datetime.now(),
            'role': 'assistant',
            'message': response
        })
        
        return response
    
    def _check_knowledge_base(self, message: str) -> Optional[str]:
        """Check if message matches knowledge base entries"""
        for topic, data in self.knowledge_base.items():
            if any(keyword in message for keyword in data['keywords']):
                return data['response']
        return None
    
    def _search_with_rag(self, message: str, current_metrics: Optional[Dict]) -> str:
        """Search RAG system for relevant information"""
        
        # Check for specific queries
        if any(word in message.lower() for word in ['similar', 'past', 'before', 'history']):
            # User wants to see past incidents
            if current_metrics:
                context = self.rag_system.get_context_for_anomaly(current_metrics)
                return context
            else:
                # Search based on message
                incidents = self.rag_system.search_similar_incidents(message, n_results=3)
                if incidents:
                    response_parts = ["Here are some relevant past incidents:\n"]
                    for i, inc in enumerate(incidents, 1):
                        metadata = inc['metadata']
                        response_parts.append(f"\n**{i}. {metadata.get('metric', 'Unknown')} - {metadata.get('severity', 'unknown')} severity**")
                        response_parts.append(f"Time: {metadata.get('timestamp', 'Unknown')}")
                        response_parts.append(f"Details: {inc['description'][:200]}...")
                    return "\n".join(response_parts)
                else:
                    return "I couldn't find any relevant past incidents for your query."
        
        # General help
        elif any(word in message.lower() for word in ['help', 'what can you do', 'how to use']):
            return """**How I Can Help:**
            
I'm your AI system monitoring assistant. I can help you with:

🔍 **Understanding Metrics:**
- Ask about CPU, memory, or disk usage
- "What is normal CPU usage?"
- "Why is my memory high?"

📚 **Historical Context:**
- "Show me similar past incidents"
- "What happened before with high CPU?"
- "Find past memory issues"

🛠️ **Troubleshooting:**
- "How to fix high CPU?"
- "What causes memory spikes?"
- "Disk cleanup tips"

💡 **Anomaly Detection:**
- "What is anomaly detection?"
- "How does RAG work?"

Just ask your question naturally!
"""
        
        # Status query
        elif any(word in message.lower() for word in ['status', 'health', 'how is', 'current']):
            if current_metrics:
                cpu = current_metrics.get('cpu_percent', 0)
                memory = current_metrics.get('memory_percent', 0)
                disk = current_metrics.get('disk_percent', 0)
                
                status = "**Current System Status:**\n\n"
                status += f"🖥️ CPU: {cpu:.1f}% - "
                status += "✅ Normal\n" if cpu < 70 else "⚠️ High\n" if cpu < 85 else "🔴 Critical\n"
                
                status += f"💾 Memory: {memory:.1f}% - "
                status += "✅ Normal\n" if memory < 75 else "⚠️ High\n" if memory < 85 else "🔴 Critical\n"
                
                status += f"💿 Disk: {disk:.1f}% - "
                status += "✅ Normal" if disk < 80 else "⚠️ High" if disk < 90 else "🔴 Critical"
                
                return status
            else:
                return "No current metrics available. Please check the main dashboard."
        
        # Default response
        return """I'm not sure how to answer that. Try asking about:
- CPU, memory, or disk usage
- Past similar incidents
- System status
- Troubleshooting tips

Or type 'help' to see what I can do!"""
    
    def get_chat_history(self, last_n: int = 10) -> List[Dict]:
        """
        Get recent chat history.
        
        Args:
            last_n: Number of recent messages to return
            
        Returns:
            List of recent chat messages
        """
        return self.chat_history[-last_n:] if self.chat_history else []
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []


if __name__ == "__main__":
    # Test the chat interface
    print("Testing Chat Interface...")
    
    from rag_system import RAGSystem
    
    # Initialize RAG and chat
    rag = RAGSystem(persist_directory="./test_chroma_db")
    chat = ChatInterface(rag)
    
    # Test questions
    test_questions = [
        "What is CPU usage?",
        "Help me understand memory",
        "What can you do?",
        "Show me past incidents"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = chat.process_message(question)
        print(f"A: {response[:200]}...")
