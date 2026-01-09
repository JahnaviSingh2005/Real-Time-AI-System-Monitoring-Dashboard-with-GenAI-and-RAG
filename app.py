"""
Real-Time AI System Monitoring Dashboard
-----------------------------------------
Main Streamlit application that brings everything together.

This dashboard provides:
- Real-time system metrics monitoring
- Rule-based and ML-based anomaly detection
- GenAI-powered explanations
- RAG-based incident retrieval
- Interactive chat interface
"""

import streamlit as st
import time
import pandas as pd
from datetime import datetime
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.metrics.collector import SystemMetricsCollector
from src.anomaly.rule_based import RuleBasedAlertSystem
from src.anomaly.ml_detector import MLAnomalyDetector
from src.genai.explainer import AnomalyExplainer
from src.genai.rag_system import RAGSystem
from src.genai.chat import ChatInterface
from src.utils.helpers import (
    create_gauge_chart, create_time_series_chart, create_multi_metric_chart,
    format_alert_message, get_health_status, calculate_trend, get_trend_emoji
)
from src.utils.storage import IncidentStorage

# Page configuration
st.set_page_config(
    page_title="AI System Monitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fffde7;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .health-excellent {
        color: #4caf50;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .health-warning {
        color: #ff9800;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .health-critical {
        color: #f44336;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'metrics_collector' not in st.session_state:
        st.session_state.metrics_collector = SystemMetricsCollector(history_size=100)
    
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = RuleBasedAlertSystem()
    
    if 'ml_detector' not in st.session_state:
        st.session_state.ml_detector = MLAnomalyDetector()
    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = AnomalyExplainer()
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(persist_directory="./data/chroma_db")
        # Load incidents on first run
        if not hasattr(st.session_state, 'incidents_loaded'):
            st.session_state.rag_system.load_incidents_from_json("./data/incidents.json")
            st.session_state.incidents_loaded = True
    
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = ChatInterface(st.session_state.rag_system)
    
    if 'storage' not in st.session_state:
        st.session_state.storage = IncidentStorage()
        
    if 'ml_trained' not in st.session_state:
        st.session_state.ml_trained = False
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Incident Tracking
    if 'active_incident' not in st.session_state:
        st.session_state.active_incident = None # {start, alerts, metrics}


# Main app
def main():
    """Main application function"""
    
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🖥️ AI System Monitoring Dashboard EXT</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Global Settings")
        
        # Refresh interval
        refresh_interval = st.slider(
            "Refresh (sec)", 1, 10, 3
        )

        st.divider()
        st.header("🔔 Alert Thresholds")
        st.session_state.cpu_threshold = st.slider("CPU Threshold (%)", 50, 95, 85)
        st.session_state.mem_threshold = st.slider("Memory Threshold (%)", 50, 95, 85)
        st.session_state.top_n = st.number_input("Top Processes Count", 3, 15, 5)

        # Update alert system thresholds
        st.session_state.alert_system.set_threshold('cpu_percent', 'high', float(st.session_state.cpu_threshold))
        st.session_state.alert_system.set_threshold('memory_percent', 'high', float(st.session_state.mem_threshold))
        
        # Auto-train ML model
        if st.button("🤖 Train ML Model"):
            df = st.session_state.metrics_collector.get_metrics_dataframe()
            if len(df) >= 10:
                with st.spinner("Training ML model..."):
                    success = st.session_state.ml_detector.train(df)
                    if success:
                        st.session_state.ml_trained = True
                        st.success("Model trained successfully!")
                    else:
                        st.error("Training failed. Need more data.")
            else:
                st.warning(f"Need at least 10 samples. Currently have {len(df)}.")
        
        # ML Model status
        if st.session_state.ml_trained:
            st.success("✅ ML Model: Trained")
        else:
            st.info("ℹ️ ML Model: Not trained yet")
        
        # System info
        st.header("💻 System Info")
        sys_info = st.session_state.metrics_collector.get_system_info()
        st.text(f"OS: {sys_info.get('platform', 'Unknown')}")
        st.text(f"CPUs: {sys_info.get('cpu_count', 0)}")
        st.text(f"RAM: {sys_info.get('total_memory_gb', 0):.1f} GB")
        
        # RAG stats
        st.header("📚 Knowledge Base")
        stats = st.session_state.rag_system.get_statistics()
        st.text(f"Incidents: {stats['total_incidents']}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Monitoring", 
        "🔍 Anomaly Detection", 
        "💬 Chat Assistant",
        "� Incident Replay",
        "�📖 About"
    ])
    
    # Tab 1: Live Monitoring
    with tab1:
        show_live_monitoring()
    
    # Tab 2: Anomaly Detection
    with tab2:
        show_anomaly_detection()
    
    # Tab 3: Chat Assistant
    with tab3:
        show_chat_interface()
    
    # Tab 4: Incident Replay
    with tab4:
        show_incident_replay()
    
    # Tab 5: About
    with tab5:
        show_about()
    
    # Auto-refresh
    time.sleep(refresh_interval)
    st.rerun()


def show_live_monitoring():
    """Display live monitoring tab"""
    
    # Collect current metrics with user-configured top N
    top_n = st.session_state.get('top_n', 5)
    current_metrics = st.session_state.metrics_collector.collect_current_metrics(top_n_procs=top_n)
    
    # Persist to SQLite
    st.session_state.storage.save_metrics(current_metrics)

    # Calculate health status with user thresholds
    health = get_health_status(
        current_metrics['cpu_percent'],
        current_metrics['memory_percent'],
        current_metrics['disk_percent'],
        cpu_threshold=st.session_state.get('cpu_threshold', 85.0),
        mem_threshold=st.session_state.get('mem_threshold', 85.0)
    )
    
    # Display health status
    st.markdown(f"## {health['emoji']} System Health: {health['status']}")
    st.markdown(f"**Score:** {health['score']}/100 - {health['message']}")
    
    # FEATURE 1: Top Resource-Consuming Processes
    st.subheader("🔝 Top Resource-Consuming Processes")
    proc_col1, proc_col2 = st.columns(2)
    
    with proc_col1:
        st.markdown("**CPU Intensive**")
        cpu_procs = current_metrics.get('top_cpu_processes', [])
        if cpu_procs:
            df_cpu = pd.DataFrame(cpu_procs)
            st.table(df_cpu[['name', 'pid', 'cpu_percent']])
            
    with proc_col2:
        st.markdown("**Memory Intensive**")
        mem_procs = current_metrics.get('top_memory_processes', [])
        if mem_procs:
            df_mem = pd.DataFrame(mem_procs)
            st.table(df_mem[['name', 'pid', 'memory_percent']])

    st.divider()
    
    # Gauges for current metrics
    cpu_t = st.session_state.get('cpu_threshold', 85.0)
    mem_t = st.session_state.get('mem_threshold', 85.0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_cpu = create_gauge_chart(
            current_metrics['cpu_percent'],
            "CPU Usage",
            threshold_yellow=cpu_t * 0.8,
            threshold_red=cpu_t
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
        
        # Trend
        df = st.session_state.metrics_collector.get_metrics_dataframe()
        if not df.empty and len(df) > 5:
            trend = calculate_trend(df['cpu_percent'].tolist())
            st.markdown(f"**Trend:** {get_trend_emoji(trend)} {trend.capitalize()}")
    
    with col2:
        fig_mem = create_gauge_chart(
            current_metrics['memory_percent'],
            "Memory Usage",
            threshold_yellow=mem_t * 0.8,
            threshold_red=mem_t
        )
        st.plotly_chart(fig_mem, use_container_width=True)
        
        if not df.empty and len(df) > 5:
            trend = calculate_trend(df['memory_percent'].tolist())
            st.markdown(f"**Trend:** {get_trend_emoji(trend)} {trend.capitalize()}")
    
    with col3:
        fig_disk = create_gauge_chart(
            current_metrics['disk_percent'],
            "Disk Usage",
            threshold_yellow=80,
            threshold_red=90
        )
        st.plotly_chart(fig_disk, use_container_width=True)
        
        if not df.empty and len(df) > 5:
            trend = calculate_trend(df['disk_percent'].tolist())
            st.markdown(f"**Trend:** {get_trend_emoji(trend)} {trend.capitalize()}")
    
    st.divider()
    
    # Historical charts
    st.subheader("📈 Historical Trends")
    
    df = st.session_state.metrics_collector.get_metrics_dataframe()
    
    if not df.empty:
        # Multi-metric chart
        fig_multi = create_multi_metric_chart(df)
        st.plotly_chart(fig_multi, use_container_width=True)
        
        # Individual charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cpu_ts = create_time_series_chart(df, 'cpu_percent', 'CPU Over Time', '#FF6B6B')
            st.plotly_chart(fig_cpu_ts, use_container_width=True)
        
        with col2:
            fig_mem_ts = create_time_series_chart(df, 'memory_percent', 'Memory Over Time', '#4ECDC4')
            st.plotly_chart(fig_mem_ts, use_container_width=True)
    else:
        st.info("Collecting data... Please wait for metrics to accumulate.")
    
    # Detailed metrics
    with st.expander("🔍 Detailed Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CPU Count", current_metrics['cpu_count'])
            st.metric("CPU Frequency (MHz)", f"{current_metrics['cpu_freq_current']:.0f}")
            st.metric("Memory Used (GB)", f"{current_metrics['memory_used_gb']:.2f}")
            st.metric("Memory Available (GB)", f"{current_metrics['memory_available_gb']:.2f}")
        
        with col2:
            st.metric("Disk Used (GB)", f"{current_metrics['disk_used_gb']:.2f}")
            st.metric("Disk Total (GB)", f"{current_metrics['disk_total_gb']:.2f}")
            st.metric("Network Sent (MB)", f"{current_metrics['network_bytes_sent_mb']:.2f}")
            st.metric("Network Received (MB)", f"{current_metrics['network_bytes_recv_mb']:.2f}")


def show_anomaly_detection():
    """Display anomaly detection tab with incident tracking"""
    
    st.header("🔍 Anomaly Detection")
    
    current_metrics = st.session_state.metrics_collector.get_latest_metrics()
    if not current_metrics:
        st.warning("No metrics available yet. Please wait...")
        return
    
    # ML condition
    is_ml_anomaly = False
    score = 0
    if st.session_state.ml_trained:
        is_ml_anomaly, score = st.session_state.ml_detector.predict(current_metrics)

    # Rule-based alerts
    st.subheader("📋 Rule-Based Alerts")
    alerts = st.session_state.alert_system.check_all_metrics(current_metrics)
    
    # Persistence & Incident Tracking
    if alerts or is_ml_anomaly:
        # Save each alert to permanent storage
        for alert in alerts:
            st.session_state.storage.save_alert(alert.to_dict())
            
        # Start or continue incident
        if st.session_state.active_incident is None:
            st.session_state.active_incident = {
                'start_time': datetime.now(),
                'alerts': [a.to_dict() for a in alerts],
                'metrics_history': [current_metrics]
            }
        else:
            st.session_state.active_incident['alerts'].extend([a.to_dict() for a in alerts])
            st.session_state.active_incident['metrics_history'].append(current_metrics)
    else:
        # Check if an incident just ended
        if st.session_state.active_incident is not None:
            inc = st.session_state.active_incident
            inc['end_time'] = datetime.now()
            summary = f"Incident with {len(inc['alerts'])} alerts."
            
            # Save to SQLite
            st.session_state.storage.create_incident(
                inc['start_time'], inc['end_time'], summary, inc['alerts']
            )
            
            # Auto-generate report (internal storage)
            report = st.session_state.explainer.generate_incident_report(inc)
            st.session_state.last_report = report
            
            # Reset active incident
            st.session_state.active_incident = None
            st.success("🏁 Incident ended. Report generated!")

    if alerts:
        for alert in alerts:
            alert_dict = alert.to_dict()
            severity = alert_dict['severity']
            st.markdown(
                f'<div class="alert-{severity}">{format_alert_message(alert_dict)}</div>',
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No alerts - All metrics within normal range")
    
    st.divider()
    
    # ML-based detection UI
    st.subheader("🤖 ML-Based Anomaly Detection")
    if st.session_state.ml_trained:
        col1, col2 = st.columns(2)
        with col1:
            if is_ml_anomaly: st.error("🚨 **Anomaly Detected!**")
            else: st.success("✅ **Normal Behavior**")
        with col2:
            st.metric("Anomaly Score", f"{score:.3f}")
    else:
        st.info("ℹ️ ML model not trained yet.")
    
    st.divider()
    
    # GenAI Explanation
    st.subheader("🤖 AI-Generated Explanation")
    if alerts or (st.session_state.ml_trained and is_ml_anomaly):
        explanation = st.session_state.explainer.explain_anomaly(
            current_metrics,
            is_ml_anomaly=is_ml_anomaly if st.session_state.ml_trained else False,
            anomaly_score=score if st.session_state.ml_trained else 0,
            alerts=[a.to_dict() for a in alerts]
        )
        st.markdown(explanation)
    else:
        st.info("No anomalies detected. System is operating normally.")


def show_chat_interface():
    """Display chat interface tab"""
    
    st.header("💬 AI Chat Assistant")
    st.markdown("Ask questions about your system, past incidents, or troubleshooting tips.")
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about system monitoring..."):
        # Add user message
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get current metrics for context
        current_metrics = st.session_state.metrics_collector.get_latest_metrics()
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_interface.process_message(
                    prompt,
                    current_metrics
                )
                st.markdown(response)
        
        # Add assistant message
        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': response
        })
    
    # Quick action buttons
    st.divider()
    st.subheader("Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💡 How can you help?"):
            prompt = "What can you do?"
            st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
            current_metrics = st.session_state.metrics_collector.get_latest_metrics()
            response = st.session_state.chat_interface.process_message(prompt, current_metrics)
            st.session_state.chat_messages.append({'role': 'assistant', 'content': response})
            st.rerun()
    
    with col2:
        if st.button("📊 Show system status"):
            prompt = "What is the current system status?"
            st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
            current_metrics = st.session_state.metrics_collector.get_latest_metrics()
            response = st.session_state.chat_interface.process_message(prompt, current_metrics)
            st.session_state.chat_messages.append({'role': 'assistant', 'content': response})
            st.rerun()
    
    with col3:
        if st.button("📚 Similar incidents"):
            prompt = "Show me similar past incidents"
            st.session_state.chat_messages.append({'role': 'user', 'content': prompt})
            current_metrics = st.session_state.metrics_collector.get_latest_metrics()
            response = st.session_state.chat_interface.process_message(prompt, current_metrics)
            st.session_state.chat_messages.append({'role': 'assistant', 'content': response})
            st.rerun()


def show_about():
    """Display about tab"""
    
    st.header("📖 About This Dashboard")
    
    st.markdown("""
    ## Real-Time AI System Monitoring Dashboard
    
    This is a comprehensive system monitoring solution that combines traditional monitoring 
    with cutting-edge AI technologies.
    
    ### 🎯 Features
    
    **1. Real-Time Monitoring**
    - Live CPU, memory, and disk usage tracking
    - Beautiful visualizations and gauges
    - Historical trend analysis
    - System health scoring
    
    **2. Multi-Level Anomaly Detection**
    - **Rule-Based**: Threshold-based alerts for immediate issues
    - **ML-Based**: Isolation Forest algorithm detects unusual patterns
    - Combines both approaches for comprehensive coverage
    
    **3. AI-Powered Explanations**
    - Natural language explanations of anomalies
    - Intelligent recommendations based on context
    - Root cause analysis suggestions
    
    **4. RAG (Retrieval-Augmented Generation)**
    - Stores and retrieves similar past incidents
    - Learns from historical data
    - Provides context-aware solutions
    - Uses ChromaDB vector database
    
    **5. Interactive Chat Assistant**
    - Ask questions in natural language
    - Get help with troubleshooting
    - Learn about system metrics
    - Query past incidents
    
    ### 🛠️ Technology Stack
    
    - **Dashboard**: Streamlit
    - **Metrics Collection**: psutil
    - **Visualization**: Plotly
    - **ML Detection**: scikit-learn (Isolation Forest)
    - **Vector DB**: ChromaDB
    - **Embeddings**: sentence-transformers
    - **GenAI**: Transformers (CPU-optimized)
    
    ### 📊 How It Works
    
    1. **Data Collection**: psutil continuously gathers system metrics
    2. **Storage**: Metrics are stored in memory with configurable history
    3. **Analysis**: Both rule-based and ML algorithms analyze the data
    4. **Detection**: Anomalies are detected and flagged
    5. **Explanation**: AI generates natural language explanations
    6. **RAG**: Similar past incidents are retrieved for context
    7. **Response**: User gets comprehensive insights and recommendations
    
    ### 🚀 Getting Started
    
    1. The dashboard auto-refreshes to show real-time data
    2. Wait for some data to accumulate (at least 10 samples)
    3. Train the ML model using the sidebar button
    4. Explore different tabs for various features
    5. Ask questions in the chat interface
    
    ### 💡 Tips
    
    - Let the system run for a few minutes to collect baseline data
    - Train the ML model during normal system operation
    - Use the chat to understand what different metrics mean
    - Check "Similar Past Incidents" when anomalies occur
    
    ### 📝 Architecture
    
    ```
    ┌─────────────────────────────────────────────────┐
    │           Streamlit Dashboard (UI)              │
    └───────────┬─────────────────────────────────────┘
                │
    ┌───────────┴─────────────────────────────────────┐
    │                                                  │
    │  ┌──────────────┐  ┌──────────────┐            │
    │  │   Metrics    │  │   Anomaly    │            │
    │  │  Collector  │  │  Detection   │            │
    │  │  (psutil)   │  │  (Rules+ML)  │            │
    │  └──────────────┘  └──────────────┘            │
    │                                                  │
    │  ┌──────────────┐  ┌──────────────┐            │
    │  │   GenAI      │  │     RAG      │            │
    │  │  Explainer   │  │   System     │            │
    │  │              │  │  (ChromaDB)  │            │
    │  └──────────────┘  └──────────────┘            │
    │                                                  │
    └──────────────────────────────────────────────────┘
    ```
    
    ### 👨‍💻 Built With
    
    This project demonstrates best practices in:
    - Modular Python architecture
    - Real-time data processing
    - Machine Learning for anomaly detection
    - RAG implementation
    - Modern UI/UX with Streamlit
    
    ---
    
    **Version**: 1.0.1 (Extended)
    **License**: MIT  
    **CPU-Only**: Yes, runs without GPU  
    """)


def show_incident_replay():
    """Display incident replay tab"""
    st.header("🕒 Incident Timeline & Replay")
    
    incidents = st.session_state.storage.get_incidents()
    
    if not incidents:
        st.info("No recorded incidents yet. Data is recorded when anomalies are detected.")
        return

    # Select incident
    incident_options = {f"{i['start_time']} - {i['summary']}": i for i in incidents}
    selected_label = st.selectbox("Select Incident to Replay", list(incident_options.keys()))
    
    if selected_label:
        incident = incident_options[selected_label]
        st.markdown(f"### Incident Details")
        st.json(json.loads(incident['json_data']))
        
        # Load historical metrics for this period
        history = st.session_state.storage.get_metrics_for_period(incident['start_time'], incident['end_time'])
        
        if history:
            st.subheader("📈 Metric Replay")
            h_df = pd.DataFrame(history)
            
            # Re-parse JSON data if needed or use columns
            st.line_chart(h_df[['cpu_percent', 'memory_percent', 'disk_percent']])
            
            # Show top processes at peak
            peak_row = h_df.iloc[h_df['cpu_percent'].idxmax()]
            st.markdown(f"**Peak Resource Usage observed at {peak_row['timestamp']}**")
            try:
                peak_data = json.loads(peak_row['json_data'])
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Top CPU processes at peak:")
                    st.table(pd.DataFrame(peak_data.get('top_cpu_processes', [])))
                with col2:
                    st.write("Top Memory processes at peak:")
                    st.table(pd.DataFrame(peak_data.get('top_memory_processes', [])))
            except:
                st.write("Process data unavailable for this record.")
                
        # Generate Report Button for past incident
        if st.button("📄 Generate Report for this Incident"):
            inc_data = {
                'start_time': incident['start_time'],
                'end_time': incident['end_time'],
                'alerts': json.loads(incident['json_data']),
                'metrics_history': [json.loads(m['json_data']) for m in history]
            }
            report = st.session_state.explainer.generate_incident_report(inc_data)
            st.markdown("---")
            st.markdown(report)
            st.download_button("📥 Download Report", report, file_name=f"past_incident_{incident['id']}.md")


if __name__ == "__main__":
    main()
