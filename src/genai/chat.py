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
            },
            'cleanup': {
                'keywords': ['cleanup', 'clean', 'remove', 'delete', 'free space', 'clear'],
                'response': """**Storage Cleanup Guide:**

🧹 **Safe Files to Delete:**
1. **Temp Files**: `%TEMP%` folder contents
2. **Browser Cache**: Clear from browser settings
3. **Windows Update Cleanup**: Run Disk Cleanup as Admin
4. **Recycle Bin**: Empty it regularly
5. **Old Downloads**: Review Downloads folder

⚠️ **Check Before Deleting:**
- Large video/image files you don't need
- Old installers (.exe, .msi files)
- Duplicate files
- Old backup files (.bak, .old)

🛠️ **Quick Cleanup Commands (Run as Admin):**
```
cleanmgr /d C:
```

💡 **Use the 'Analyze Storage' quick button for personalized recommendations!**
"""
            },
            'processes': {
                'keywords': ['process', 'task', 'running', 'app', 'application', 'program'],
                'response': """**About Running Processes:**

🔍 **Understanding Processes:**
- Each running program is a process
- Processes consume CPU and Memory
- Some run in background (services)

**Common High-Resource Processes:**
- `chrome.exe` - Web browser (often high memory)
- `MsMpEng.exe` - Windows Defender
- `SearchIndexer.exe` - Windows Search
- `svchost.exe` - Windows services

**Safe to End (Usually):**
- Multiple browser tabs
- Unused applications
- Background updaters

**Never End:**
- `System` process
- `csrss.exe`
- `winlogon.exe`
- `services.exe`

💡 **Use 'Show Top Processes' button for live analysis!**
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
        
        # If Gemini is configured, use it with FULL context
        if self.llm and self.llm.is_configured:
            try:
                # 1. ALWAYS build comprehensive metrics context
                metrics_context = ""
                if current_metrics:
                    metrics_context = f"""CURRENT LIVE SYSTEM METRICS:
- CPU Usage: {current_metrics.get('cpu_percent', 'N/A')}%
- Memory Usage: {current_metrics.get('memory_percent', 'N/A')}%
- Disk Usage: {current_metrics.get('disk_percent', 'N/A')}%
- Memory Used: {current_metrics.get('memory_used_gb', 'N/A'):.2f} GB
- Memory Available: {current_metrics.get('memory_available_gb', 'N/A'):.2f} GB
- Disk Used: {current_metrics.get('disk_used_gb', 'N/A'):.2f} GB
- Disk Total: {current_metrics.get('disk_total_gb', 'N/A'):.2f} GB
- CPU Cores: {current_metrics.get('cpu_count', 'N/A')}
- Network Sent: {current_metrics.get('network_bytes_sent_mb', 'N/A'):.2f} MB
- Network Received: {current_metrics.get('network_bytes_recv_mb', 'N/A'):.2f} MB
"""
                    # Add top processes if available
                    top_cpu_procs = current_metrics.get('top_cpu_processes', [])
                    top_mem_procs = current_metrics.get('top_memory_processes', [])
                    if top_cpu_procs:
                        metrics_context += "\nTOP CPU PROCESSES:\n"
                        for p in top_cpu_procs[:5]:
                            metrics_context += f"  - {p.get('name', 'unknown')} (PID: {p.get('pid', 'N/A')}): {p.get('cpu_percent', 0):.1f}% CPU\n"
                    if top_mem_procs:
                        metrics_context += "\nTOP MEMORY PROCESSES:\n"
                        for p in top_mem_procs[:5]:
                            metrics_context += f"  - {p.get('name', 'unknown')} (PID: {p.get('pid', 'N/A')}): {p.get('memory_percent', 0):.1f}% RAM\n"
                
                # 2. Get RAG context for relevant queries
                rag_context = ""
                if any(word in user_message_lower for word in ['similar', 'past', 'before', 'history', 'incident']):
                    if current_metrics:
                        rag_context = self.rag_system.get_context_for_anomaly(current_metrics)
                    else:
                        incidents = self.rag_system.search_similar_incidents(user_message, n_results=3)
                        if incidents:
                            rag_context = "\nRELEVANT PAST INCIDENTS:\n" + "\n".join([f"- {inc['description']}" for inc in incidents])
                
                # 3. Build full context
                full_context = f"{metrics_context}\n{rag_context}".strip()
                
                # 4. Generate response using Gemini
                response = self.llm.generate_chat_response(self.chat_history[:-1], user_message, full_context)
            except Exception as e:
                print(f"Gemini API error in chat: {e}")
                response = None
        
        # Fallback to internal handlers if Gemini failed or is not available
        if not response:
            # PRIORITY 1: Check for LIVE metrics analysis requests first
            # These should take precedence over static knowledge base
            response = self._handle_live_analysis(user_message_lower, current_metrics)
            
            # PRIORITY 2: Check static knowledge base for general questions
            if not response:
                response = self._check_knowledge_base(user_message_lower)
            
            # PRIORITY 3: Try RAG search for past incidents
            if not response:
                response = self._search_with_rag(user_message, current_metrics)
        
        # Add response to history
        self.chat_history.append({
            'timestamp': datetime.now(),
            'role': 'assistant',
            'message': response
        })
        
        return response
    
    def _handle_live_analysis(self, message: str, current_metrics: Optional[Dict]) -> Optional[str]:
        """Handle requests that need live metrics analysis - takes priority over knowledge base"""
        
        # Storage/disk analysis - prioritize live data
        if any(word in message for word in ['analyze', 'storage', 'space', 'drive', 'disk size', 'how much']):
            return self._get_storage_analysis(current_metrics)
        
        # Process analysis - prioritize live data
        if any(word in message for word in ['top', 'processes', 'consuming', 'using', 'heavy', 'running']):
            return self._get_process_analysis(current_metrics)
        
        # Cleanup suggestions - prioritize personalized recommendations
        if any(word in message for word in ['cleanup', 'clean', 'suggestion', 'recommend', 'free']):
            return self._get_cleanup_suggestions(current_metrics)
        
        # System status - prioritize live data
        if any(word in message for word in ['status', 'health', 'current']):
            return self._get_system_status(current_metrics)
        
        return None
    
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
            return self._get_system_status(current_metrics)
        
        # Storage analysis
        elif any(word in message.lower() for word in ['storage', 'analyze', 'space', 'disk size', 'drive']):
            return self._get_storage_analysis(current_metrics)
        
        # Top processes / CPU analysis
        elif any(word in message.lower() for word in ['top', 'processes', 'consuming', 'using', 'heavy']):
            return self._get_process_analysis(current_metrics)
        
        # Cleanup suggestions
        elif any(word in message.lower() for word in ['cleanup', 'clean', 'suggestion', 'recommend', 'remove', 'free']):
            return self._get_cleanup_suggestions(current_metrics)
        
        # Default response
        return """I'm not sure how to answer that. Try asking about:
- CPU, memory, or disk usage
- Storage analysis and cleanup
- Top processes consuming resources
- Past similar incidents
- System status

Or type 'help' to see what I can do!"""
    
    def _get_system_status(self, current_metrics: Optional[Dict]) -> str:
        """Get comprehensive system status"""
        if not current_metrics:
            return "No current metrics available. Please check the main dashboard."
            
        cpu = current_metrics.get('cpu_percent', 0)
        memory = current_metrics.get('memory_percent', 0)
        disk = current_metrics.get('disk_percent', 0)
        mem_used = current_metrics.get('memory_used_gb', 0)
        mem_avail = current_metrics.get('memory_available_gb', 0)
        disk_used = current_metrics.get('disk_used_gb', 0)
        disk_total = current_metrics.get('disk_total_gb', 0)
        
        status = "**📊 Current System Status:**\n\n"
        
        # CPU
        cpu_icon = "✅" if cpu < 70 else "⚠️" if cpu < 85 else "🔴"
        status += f"🖥️ **CPU:** {cpu:.1f}% {cpu_icon}\n"
        
        # Memory
        mem_icon = "✅" if memory < 75 else "⚠️" if memory < 85 else "🔴"
        status += f"💾 **Memory:** {memory:.1f}% ({mem_used:.1f} GB used / {mem_avail:.1f} GB free) {mem_icon}\n"
        
        # Disk
        disk_icon = "✅" if disk < 80 else "⚠️" if disk < 90 else "🔴"
        disk_free = disk_total - disk_used
        status += f"💿 **Disk:** {disk:.1f}% ({disk_used:.1f} GB used / {disk_free:.1f} GB free) {disk_icon}\n"
        
        # Network
        net_sent = current_metrics.get('network_bytes_sent_mb', 0)
        net_recv = current_metrics.get('network_bytes_recv_mb', 0)
        status += f"🌐 **Network:** ↑{net_sent:.1f} MB sent / ↓{net_recv:.1f} MB received\n"
        
        return status
    
    def _get_storage_analysis(self, current_metrics: Optional[Dict]) -> str:
        """Get detailed storage analysis"""
        if not current_metrics:
            return "No metrics available for storage analysis."
        
        disk = current_metrics.get('disk_percent', 0)
        disk_used = current_metrics.get('disk_used_gb', 0)
        disk_total = current_metrics.get('disk_total_gb', 0)
        disk_free = disk_total - disk_used
        
        analysis = "**💿 Storage Analysis:**\n\n"
        analysis += f"📊 **Usage:** {disk:.1f}% full\n"
        analysis += f"📁 **Used Space:** {disk_used:.1f} GB\n"
        analysis += f"📂 **Free Space:** {disk_free:.1f} GB\n"
        analysis += f"💽 **Total Capacity:** {disk_total:.1f} GB\n\n"
        
        # Health assessment
        if disk < 70:
            analysis += "✅ **Health:** Excellent - Plenty of space available\n"
        elif disk < 80:
            analysis += "✅ **Health:** Good - Consider cleaning up soon\n"
        elif disk < 90:
            analysis += "⚠️ **Health:** Warning - Running low on space!\n"
            analysis += "\n**Recommended Action:** Run cleanup to free at least 10% space\n"
        else:
            analysis += "🔴 **Health:** Critical - Immediate cleanup needed!\n"
            analysis += "\n**Urgent:** Free up space immediately to prevent system issues\n"
        
        analysis += "\n💡 *Click 'Cleanup Suggestions' for detailed recommendations*"
        return analysis
    
    def _get_process_analysis(self, current_metrics: Optional[Dict]) -> str:
        """Get top process analysis"""
        if not current_metrics:
            return "No process data available."
        
        cpu = current_metrics.get('cpu_percent', 0)
        memory = current_metrics.get('memory_percent', 0)
        top_cpu = current_metrics.get('top_cpu_processes', [])
        top_mem = current_metrics.get('top_memory_processes', [])
        
        analysis = "**🔝 Top Resource-Consuming Processes:**\n\n"
        
        # CPU processes
        analysis += f"🖥️ **CPU Intensive** (Total: {cpu:.1f}%):\n"
        if top_cpu:
            for i, p in enumerate(top_cpu[:5], 1):
                name = p.get('name', 'Unknown')
                pid = p.get('pid', 'N/A')
                cpu_pct = p.get('cpu_percent', 0)
                icon = "🔴" if cpu_pct > 30 else "⚠️" if cpu_pct > 15 else "✅"
                analysis += f"  {i}. {icon} **{name}** (PID: {pid}): {cpu_pct:.1f}%\n"
        else:
            analysis += "  No high CPU processes detected\n"
        
        analysis += f"\n💾 **Memory Intensive** (Total: {memory:.1f}%):\n"
        if top_mem:
            for i, p in enumerate(top_mem[:5], 1):
                name = p.get('name', 'Unknown')
                pid = p.get('pid', 'N/A')
                mem_pct = p.get('memory_percent', 0)
                icon = "🔴" if mem_pct > 20 else "⚠️" if mem_pct > 10 else "✅"
                analysis += f"  {i}. {icon} **{name}** (PID: {pid}): {mem_pct:.1f}%\n"
        else:
            analysis += "  No high memory processes detected\n"
        
        # Suggestions
        analysis += "\n**💡 Tips:**\n"
        if cpu > 80:
            analysis += "- Consider closing high CPU processes\n"
        if memory > 80:
            analysis += "- Close unused applications to free memory\n"
        if cpu < 50 and memory < 50:
            analysis += "- System is running smoothly! ✅\n"
        
        return analysis
    
    def _get_cleanup_suggestions(self, current_metrics: Optional[Dict]) -> str:
        """Get personalized cleanup suggestions"""
        suggestions = "**🧹 Storage Cleanup Suggestions:**\n\n"
        
        # Check disk usage
        disk = 0
        if current_metrics:
            disk = current_metrics.get('disk_percent', 0)
            disk_free = current_metrics.get('disk_total_gb', 0) - current_metrics.get('disk_used_gb', 0)
            suggestions += f"📊 Current: {disk:.1f}% used ({disk_free:.1f} GB free)\n\n"
        
        suggestions += "**🗑️ Safe to Delete (Typically 5-20 GB):**\n"
        suggestions += "1. **Temp Files** - Run: `%TEMP%` in Explorer, delete all\n"
        suggestions += "2. **Windows Temp** - `C:\\Windows\\Temp` (Run as Admin)\n"
        suggestions += "3. **Recycle Bin** - Right-click → Empty\n"
        suggestions += "4. **Browser Cache** - Clear in browser settings\n"
        suggestions += "5. **Windows Update Cleanup** - Disk Cleanup (Run as Admin)\n\n"
        
        suggestions += "**📁 Review These Folders:**\n"
        suggestions += "- `Downloads` - Old installers, files you don't need\n"
        suggestions += "- `Videos` - Large video files\n"
        suggestions += "- `Desktop` - Unused files accumulate here\n\n"
        
        suggestions += "**🔧 Automated Cleanup:**\n"
        suggestions += "```\n"
        suggestions += "# Run Disk Cleanup (as Administrator)\n"
        suggestions += "cleanmgr /d C: /VERYLOWDISK\n"
        suggestions += "```\n\n"
        
        # Priority based on disk usage
        if disk > 90:
            suggestions += "🔴 **URGENT:** Disk nearly full! Start with Recycle Bin and Temp files immediately.\n"
        elif disk > 80:
            suggestions += "⚠️ **Priority:** Clean up this week to maintain performance.\n"
        else:
            suggestions += "✅ **Status:** Healthy storage. Monthly cleanup recommended.\n"
        
        return suggestions
    
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
