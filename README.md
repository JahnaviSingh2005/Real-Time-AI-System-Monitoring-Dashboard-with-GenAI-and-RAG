# Real-Time AI System Monitoring Dashboard with GenAI and RAG

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive, production-ready system monitoring dashboard that combines real-time metrics collection with AI-powered anomaly detection, natural language explanations, and RAG-based incident retrieval.

## 🌟 Features

### 📊 Real-Time Monitoring
- **Live Metrics**: CPU, Memory, Disk, and Network usage
- **Beautiful Visualizations**: Interactive gauges and time-series charts
- **Trend Analysis**: Automatic trend detection (increasing/decreasing/stable)
- **Health Scoring**: Overall system health assessment

### 🔍 Dual Anomaly Detection
- **Rule-Based Alerts**: Configurable threshold-based alerting
- **ML-Based Detection**: Isolation Forest algorithm for pattern recognition
- **Severity Levels**: Critical, High, Medium, Low classifications
- **Real-Time Alerts**: Instant notification of issues

### 🤖 AI-Powered Insights
- **Multi-Provider Support**: Choose between Gemini, OpenAI, or Local-only mode
- **GenAI Explanations**: Natural language explanations of anomalies
- **Root Cause Analysis**: Intelligent suggestions for possible causes
- **Actionable Recommendations**: Step-by-step troubleshooting guidance
- **Context-Aware**: Considers multiple factors for accurate analysis

### 📚 RAG System
- **Vector Database**: ChromaDB for efficient similarity search
- **Incident Retrieval**: Find similar past incidents automatically
- **Learn from History**: Benefit from previous resolutions
- **Knowledge Base**: Expandable incident repository

### 💬 Interactive Chat Assistant
- **Natural Language**: Ask questions in plain English
- **Live Metrics Analysis**: Real-time storage, CPU, memory analysis
- **Process Monitoring**: View top resource-consuming processes
- **Cleanup Suggestions**: Personalized disk cleanup recommendations
- **Works Offline**: Core features work without API keys
- **Quick Action Buttons**: One-click access to common queries

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Streamlit Web Interface                    │
│  (Live Monitoring | Anomaly Detection | Chat | About)   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴───────────┐
        │                        │
┌───────▼────────┐    ┌─────────▼──────────┐
│     Metrics    │    │      Anomaly       │
│   Collection   │    │     Detection      │
│    (psutil)    │    │  (Rules + ML)      │
└───────┬────────┘    └─────────┬──────────┘
        │                       │
        │         ┌─────────────┴────────────┐
        │         │                          │
┌───────▼─────────▼──────┐      ┌───────────▼──────┐
│   GenAI Explainer      │      │   RAG System     │
│  (transformers)        │      │   (ChromaDB)     │
└────────────────────────┘      └──────────────────┘
                │                        │
                └────────┬───────────────┘
                         │
        ┌────────────────▼────────────────┐
        │         Chat Interface          │
        │  ┌─────────┐  ┌─────────────┐   │
        │  │ Gemini  │  │   OpenAI    │   │
        │  │   LLM   │  │    LLM      │   │
        │  └─────────┘  └─────────────┘   │
        │         Local Analysis          │
        └─────────────────────────────────┘
```

## 📁 Project Structure

```
AI System Monitoring/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── README.md                  # This file
├── QUICKSTART.md              # Quick start guide
├── ARCHITECTURE.md            # Detailed architecture documentation
├── USAGE_GUIDE.md             # Comprehensive usage guide
│
├── src/                       # Source code
│   ├── __init__.py
│   │
│   ├── metrics/               # Metrics collection
│   │   ├── __init__.py
│   │   └── collector.py       # System metrics collector (psutil)
│   │
│   ├── anomaly/               # Anomaly detection
│   │   ├── __init__.py
│   │   ├── rule_based.py      # Threshold-based alerts
│   │   └── ml_detector.py     # ML-based detection (Isolation Forest)
│   │
│   ├── genai/                 # AI components
│   │   ├── __init__.py
│   │   ├── explainer.py       # GenAI explanations & incident reports
│   │   ├── rag_system.py      # RAG implementation (ChromaDB)
│   │   ├── gemini_llm.py      # Google Gemini integration
│   │   ├── openai_llm.py      # OpenAI ChatGPT integration
│   │   └── chat.py            # Chat interface with live analysis
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── helpers.py         # Visualization and formatting helpers
│       └── storage.py         # SQLite storage for metrics & incidents
│
└── data/                      # Data storage
    ├── incidents.json         # Sample incident dataset
    ├── metrics.db             # SQLite database (auto-created)
    └── chroma_db/             # Vector database (auto-created)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Windows, macOS, or Linux

### Installation

1. **Clone or download this project**
   ```bash
   cd "AI System Monitoring"
   ```

2. **Create and activate virtual environment**
   
   Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install:
   - Streamlit (dashboard)
   - psutil (system metrics)
   - scikit-learn (ML)
   - ChromaDB (vector database)
   - sentence-transformers (embeddings)
   - transformers (GenAI)
   - plotly (visualizations)
   - And more...

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If not, manually navigate to that URL

## 📖 Usage Guide

### First-Time Setup

1. **Let it run**: Allow the dashboard to collect data for 2-3 minutes
2. **Train ML model**: Click "Train ML Model" in the sidebar after ~10 samples
3. **Explore features**: Navigate through different tabs

### Tab Overview

#### 📊 Live Monitoring
- View real-time system metrics
- See health status and scores
- Analyze historical trends
- Check detailed system information

#### 🔍 Anomaly Detection
- Monitor rule-based alerts
- Check ML anomaly detection
- Read AI-generated explanations
- View similar past incidents

#### 💬 Chat Assistant
- Ask questions about the system
- Get explanations of metrics
- Query historical incidents
- Receive troubleshooting help

#### 📖 About
- Learn about features
- Understand the architecture
- View technology stack
- Get usage tips

### Example Questions for Chat

**Live Analysis (works without API):**
- "Show system status"
- "Analyze my storage"
- "Show top processes"
- "Give me cleanup suggestions"

**AI-Powered (requires API key):**
- "Write a poem about system monitoring"
- "Explain why my CPU is high"
- "What could cause memory leaks?"
- "Show me similar past incidents"

## 🔧 Configuration

### Adjusting Thresholds

Edit thresholds in the sidebar or programmatically:

```python
# In rule_based.py
alert_system.set_threshold('cpu_percent', 'high', 80.0)
```

### Customizing Refresh Rate

Use the slider in the sidebar (1-10 seconds)

### Adding Custom Incidents

Edit `data/incidents.json` to add your own historical incidents for RAG:

```json
{
  "timestamp": "2026-01-08 10:00:00",
  "metric": "cpu_usage",
  "value": 95.0,
  "description": "Your incident description",
  "resolution": "How you fixed it",
  "severity": "high"
}
```

## 🧪 Testing Individual Modules

Each module can be tested independently:

```bash
# Test metrics collector
python src/metrics/collector.py

# Test rule-based alerts
python src/anomaly/rule_based.py

# Test ML detector
python src/anomaly/ml_detector.py

# Test explainer
python src/genai/explainer.py

# Test RAG system
python src/genai/rag_system.py

# Test utilities
python src/utils/helpers.py
```

## 📊 How It Works

### 1. Metrics Collection
- `psutil` library gathers CPU, memory, disk, and network stats
- Data stored in memory with configurable history size
- Continuous collection at specified intervals

### 2. Rule-Based Detection
- Checks metrics against predefined thresholds
- Immediate alerts for threshold violations
- Configurable severity levels

### 3. ML-Based Detection
- Isolation Forest algorithm (unsupervised learning)
- Learns normal system behavior patterns
- Detects unusual combinations and outliers
- Requires training on baseline data

### 4. GenAI Explanations
- Template-based natural language generation
- Context-aware recommendations
- Severity-based prioritization
- Actionable troubleshooting steps

### 5. RAG System
- ChromaDB stores incident vectors
- Sentence transformers create embeddings
- Similarity search finds relevant past cases
- Provides historical context

### 6. Chat Interface
- Combines knowledge base with RAG
- Context-aware responses
- Real-time system state integration
- Natural language understanding

## 🎯 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dashboard | Streamlit | Web interface |
| Metrics | psutil | System monitoring |
| ML Detection | Isolation Forest | Anomaly detection |
| Vector DB | ChromaDB | Incident storage |
| Embeddings | sentence-transformers | Text vectorization |
| AI (Option 1) | Google Gemini | Advanced AI responses |
| AI (Option 2) | OpenAI GPT | Alternative AI provider |
| Visualization | Plotly | Interactive charts |
| Data Processing | pandas, numpy | Data manipulation |
| Storage | SQLite | Metrics persistence |

## 💡 Best Practices

### For Accurate ML Detection
- Run system under normal conditions initially
- Train model with at least 10-20 samples
- Retrain periodically to adapt to new patterns
- Monitor during various workloads

### For Effective RAG
- Add detailed incident descriptions
- Include resolution steps
- Categorize by severity
- Update regularly with new incidents

### For System Performance
- Adjust refresh interval based on needs
- Limit history size if memory constrained
- Clean up old ChromaDB data periodically

## 🐛 Troubleshooting

### Issue: ML model not detecting anomalies
**Solution**: Ensure model is trained and has enough baseline data

### Issue: Chat not finding past incidents
**Solution**: Check that `incidents.json` loaded successfully

### Issue: High memory usage
**Solution**: Reduce `history_size` in SystemMetricsCollector

### Issue: Slow refresh
**Solution**: Increase refresh interval or optimize data collection

## 🚀 Future Enhancements

- [ ] Export reports to PDF
- [ ] Email/SMS alerts
- [ ] Multi-system monitoring
- [ ] Custom metric plugins
- [ ] Advanced ML models
- [ ] Integration with logging systems
- [ ] Cloud deployment guide
- [ ] API endpoints

## 📖 General Documentation & Learning Resources

### 🖥️ Dashboard & UI
- **[Streamlit Documentation](https://docs.streamlit.io/)**: Official guide for the dashboard framework.
- **[Plotly Python Documentation](https://plotly.com/python/)**: Guide for creating interactive gauges and time-series charts.

### ⚙️ System Monitoring
- **[psutil Documentation](https://psutil.readthedocs.io/)**: Learn about the cross-platform library used for system metrics.
- **[System Monitoring Best Practices](https://www.datadoghq.com/blog/monitoring-101-collecting-data/)**: General concepts of system observability.

### 🧠 Artificial Intelligence & ML
- **[Isolation Forest (Scikit-Learn)](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest)**: Understanding the algorithm used for anomaly detection.
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**: Resources for running local, CPU-friendly language models.

### 📚 RAG & Vector Databases
- **[ChromaDB Documentation](https://docs.trychroma.com/)**: Deep dive into the vector database used for incident storage.
- **[Retrieval-Augmented Generation (RAG) Explained](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)**: Comprehensive overview of combining LLMs with vector search.
- **[Sentence Transformers](https://www.sbert.net/)**: How text data is converted into high-dimensional embeddings.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional anomaly detection algorithms
- More comprehensive incident database
- Enhanced visualizations
- Mobile-responsive design
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - feel free to use it for learning or production!

## 👨‍💻 Author

Built as a comprehensive demonstration of:
- Real-time system monitoring
- AI/ML integration
- RAG implementation
- Modern Python best practices
- Production-ready architecture

## 🙏 Acknowledgments

- **Streamlit** - Amazing dashboard framework
- **scikit-learn** - Powerful ML library
- **ChromaDB** - Efficient vector database
- **HuggingFace** - Transformers and models
- **psutil** - Comprehensive system monitoring

---

**⭐ Star this project if you find it useful!**

**📧 Questions?** Check the About tab in the dashboard or open an issue.

**🎓 Learning Project**: This is perfect for understanding how to build production-ready AI systems!
