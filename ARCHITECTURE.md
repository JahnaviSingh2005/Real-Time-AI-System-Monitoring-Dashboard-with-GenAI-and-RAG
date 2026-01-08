# Architecture Deep Dive

## System Overview

This document provides a detailed technical explanation of the system architecture, design decisions, and implementation details.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                         │
│                      (Streamlit App)                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │     Live     │ │   Anomaly    │ │     Chat     │           │
│  │  Monitoring  │ │  Detection   │ │  Interface   │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Session State Management                    │   │
│  │  (Singleton instances, history, cached data)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                     CORE MODULES LAYER                          │
│                                                                  │
│  ┌─────────────────┐                ┌──────────────────┐       │
│  │ Metrics Module  │                │  Anomaly Module  │       │
│  │                 │                │                  │       │
│  │ ┌─────────────┐ │                │ ┌──────────────┐ │       │
│  │ │  Collector  │ │                │ │ Rule-Based   │ │       │
│  │ │  (psutil)   │ │                │ │   Alerts     │ │       │
│  │ └─────────────┘ │                │ └──────────────┘ │       │
│  │                 │                │ ┌──────────────┐ │       │
│  │ - CPU metrics   │                │ │ ML Detector  │ │       │
│  │ - Memory stats  │                │ │(IsolationFor)│ │       │
│  │ - Disk usage    │                │ └──────────────┘ │       │
│  │ - Network I/O   │                │                  │       │
│  └─────────────────┘                └──────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    GenAI Module                          │  │
│  │                                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │  Explainer   │  │ RAG System   │  │     Chat     │  │  │
│  │  │              │  │              │  │  Interface   │  │  │
│  │  │ - Template   │  │ - ChromaDB   │  │  - KB match  │  │  │
│  │  │ - Insights   │  │ - Embeddings │  │  - RAG query │  │  │
│  │  │ - Recommend  │  │ - Similarity │  │  - Response  │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Utils Module                          │  │
│  │  - Visualization (Plotly charts)                        │  │
│  │  - Data formatting                                       │  │
│  │  - Health calculations                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                      DATA LAYER                                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   In-Memory  │  │  ChromaDB    │  │  JSON Incidents      │ │
│  │   History    │  │ Vector Store │  │    (data/)           │ │
│  │  (deque/list)│  │              │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                   SYSTEM LAYER                                  │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │  psutil    │  │ scikit-    │  │ sentence-  │              │
│  │  (System   │  │  learn     │  │transformers│              │
│  │   APIs)    │  │   (ML)     │  │(Embeddings)│              │
│  └────────────┘  └────────────┘  └────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Metrics Collection Module

**File**: `src/metrics/collector.py`

**Purpose**: Gather real-time system metrics

**Key Classes**:
- `SystemMetricsCollector`: Main collector class

**Design Decisions**:
- Uses **psutil** for cross-platform compatibility
- Stores metrics in **memory** for fast access
- Maintains **fixed-size history** to prevent memory bloat
- Collects metrics at **configurable intervals**

**Data Flow**:
```
psutil → collect_current_metrics() → metrics_history → DataFrame
```

**Metrics Collected**:
- CPU: Percentage, count, frequency
- Memory: Used, available, total (bytes → GB)
- Disk: Used, total, percentage
- Network: Bytes sent/received (bytes → MB)

### 2. Anomaly Detection Module

#### 2.1 Rule-Based Detection

**File**: `src/anomaly/rule_based.py`

**Purpose**: Fast, threshold-based alerting

**Key Classes**:
- `RuleBasedAlertSystem`: Threshold checker
- `Alert`: Alert data structure
- `AlertSeverity`: Enum for severity levels

**Algorithm**:
```python
for each metric:
    if value >= critical_threshold:
        return CRITICAL alert
    elif value >= high_threshold:
        return HIGH alert
    elif value >= medium_threshold:
        return MEDIUM alert
```

**Advantages**:
- ✅ Fast (O(1) complexity)
- ✅ Predictable
- ✅ Easy to configure
- ✅ No training required

**Limitations**:
- ❌ Can't detect complex patterns
- ❌ Fixed thresholds may not suit all systems
- ❌ Doesn't learn from data

#### 2.2 ML-Based Detection

**File**: `src/anomaly/ml_detector.py`

**Purpose**: Pattern-based anomaly detection

**Algorithm**: Isolation Forest

**Why Isolation Forest?**
1. **Unsupervised**: No labels needed
2. **Fast**: O(n log n) complexity
3. **Effective**: Great for outlier detection
4. **CPU-friendly**: Lightweight trees
5. **Few parameters**: Easy to tune

**How It Works**:
```
1. Build random trees by randomly selecting:
   - A feature
   - A split value
2. Anomalies are isolated in fewer splits
3. Calculate anomaly score based on path length
4. Score < threshold → Anomaly
```

**Features Used**:
- CPU percentage
- Memory percentage
- Disk percentage

**Training Process**:
```python
1. Collect baseline data (normal behavior)
2. Extract features (cpu, memory, disk)
3. Fit Isolation Forest model
4. Model learns "normal" patterns
```

**Prediction**:
```python
1. New metrics arrive
2. Extract same features
3. Model scores the point
4. return (is_anomaly, score)
```

**Advantages**:
- ✅ Detects complex patterns
- ✅ Adapts to your system
- ✅ Finds subtle anomalies
- ✅ No threshold tuning

**Limitations**:
- ❌ Needs training data
- ❌ Requires baseline period
- ❌ More computational overhead

### 3. GenAI Module

#### 3.1 Explanation Generator

**File**: `src/genai/explainer.py`

**Purpose**: Generate human-readable explanations

**Design Choice**: Template-based (not pure LLM)

**Why Template-Based?**
1. **Reliable**: Consistent output quality
2. **Fast**: No LLM inference overhead
3. **Accurate**: No hallucinations
4. **CPU-friendly**: Minimal computation
5. **Controllable**: Predictable responses

**Structure**:
```python
explanation = [
    header,
    current_state,
    ml_detection_info,
    rule_alerts,
    insights (based on values),
    recommendations (based on context)
]
```

**Insight Generation Logic**:
```python
if cpu > 90:
    "Extremely high CPU - possible runaway process"
elif cpu > 70:
    "Elevated CPU - heavy computational load"

if cpu > 80 AND memory > 80:
    "Resource contention - both CPU and memory stressed"
```

**Recommendation Logic**:
Severity-based escalation:
```python
if cpu > 95:
    "URGENT: Terminate problematic processes"
elif cpu > 85:
    "Check Task Manager, close unnecessary apps"
```

#### 3.2 RAG System

**File**: `src/genai/rag_system.py`

**Purpose**: Retrieve similar past incidents

**Components**:
1. **Vector Database**: ChromaDB
2. **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
3. **Storage**: Persistent on disk

**Why ChromaDB?**
- ✅ Lightweight
- ✅ Easy to use
- ✅ Persistent storage
- ✅ Built-in similarity search
- ✅ No server required

**Why all-MiniLM-L6-v2?**
- ✅ Small (80MB)
- ✅ Fast on CPU
- ✅ Good quality embeddings
- ✅ 384 dimensions (manageable)

**RAG Flow**:
```
1. Incident → text description
2. Text → embedding (384-dim vector)
3. Vector → ChromaDB storage
4. Query → embedding
5. Similarity search → top-k incidents
6. Format → return results
```

**Similarity Metric**: Cosine similarity (default in ChromaDB)

**Indexing**:
```python
incident_text = f"{description} Resolution: {resolution}"
embedding = model.encode(incident_text)
chroma.add(embedding, metadata, id)
```

**Retrieval**:
```python
query = "high CPU usage causing slowdown"
query_embedding = model.encode(query)
results = chroma.query(query_embedding, n_results=3)
```

#### 3.3 Chat Interface

**File**: `src/genai/chat.py`

**Purpose**: Interactive Q&A about system

**Architecture**:
```
User Question
    │
    ├──> Knowledge Base Match? ──Yes──> Return KB answer
    │                         └─No
    │
    └──> RAG Search ──> Format ──> Return contextual answer
```

**Knowledge Base**: Predefined Q&A for common topics
- CPU usage
- Memory usage
- Disk usage
- Anomaly detection
- RAG system

**RAG Integration**: For historical queries
- "Show similar past incidents"
- "What happened before with high CPU?"

**Context Awareness**: Uses current metrics when available

### 4. Visualization Module

**File**: `src/utils/helpers.py`

**Purpose**: Create charts and format data

**Chart Types**:

1. **Gauge Charts**: Current metric value
   - Color-coded by threshold
   - Shows delta from reference
   - Visual threshold zones

2. **Time Series**: Historical trends
   - Line + markers
   - Hover tooltips
   - Time on X-axis

3. **Multi-Metric**: All metrics together
   - Different colors per metric
   - Unified hover mode
   - Legend

**Health Score Calculation**:
```python
cpu_score = 100 - cpu_percent
memory_score = 100 - memory_percent
disk_score = 100 - disk_percent

overall_score = (cpu_score + memory_score + disk_score) / 3

# 70-100: Healthy
# 50-70: Warning
# 0-50: Critical
```

## Data Flow

### Monitoring Cycle

```
1. Timer triggers (every N seconds)
   ↓
2. collect_current_metrics()
   ↓
3. Update session state
   ↓
4. Rule-based check
   ↓
5. ML prediction (if trained)
   ↓
6. Generate explanation (if anomaly)
   ↓
7. RAG retrieval (if anomaly)
   ↓
8. Render UI
   ↓
9. st.rerun() → back to step 1
```

### Chat Flow

```
1. User enters message
   ↓
2. Add to chat history
   ↓
3. Check knowledge base
   ├──Match found? → Return KB answer
   └──No match
      ↓
4. Analyze intent
   ├──Status query? → Get current metrics
   ├──History query? → RAG search
   └──Help query? → Return help text
      ↓
5. Format response
   ↓
6. Add to chat history
   ↓
7. Display
```

## Design Patterns Used

### 1. Singleton Pattern
**Where**: Session state management
**Why**: Single instance of each service across refreshes

### 2. Strategy Pattern
**Where**: Anomaly detection (Rule-based vs ML-based)
**Why**: Multiple interchangeable detection strategies

### 3. Template Pattern
**Where**: Explanation generation
**Why**: Consistent structure with customizable parts

### 4. Repository Pattern
**Where**: ChromaDB interaction
**Why**: Abstract data storage details

### 5. Observer Pattern
**Where**: Metrics collection and alerting
**Why**: Decouple detection from notification

## Performance Considerations

### Memory Management
- **Fixed-size history**: Prevents unbounded growth
- **DataFrame conversion**: Only when needed
- **Lazy loading**: Models loaded on demand

### CPU Optimization
- **Batch operations**: Process multiple metrics together
- **Efficient algorithms**: O(n log n) or better
- **CPU-only models**: No GPU requirement
- **Vectorized operations**: NumPy/Pandas

### Disk I/O
- **Persistent ChromaDB**: Saves to disk incrementally
- **JSON caching**: Incidents loaded once
- **Minimal writes**: Only when adding incidents

### Network
- **No external API calls**: Fully offline after setup
- **Local models**: No inference API needed

## Security Considerations

### Data Privacy
- ✅ All data stays local
- ✅ No telemetry or tracking
- ✅ No external API calls

### Input Validation
- ✅ Type checking on user inputs
- ✅ Bounds checking on thresholds
- ✅ Safe file path handling

### Error Handling
- ✅ Try-catch blocks around critical operations
- ✅ Graceful degradation
- ✅ User-friendly error messages

## Scalability

### Current Limitations
- Single machine monitoring
- In-memory storage
- UI refresh overhead

### Future Enhancements
1. **Multi-machine**: Agent-server architecture
2. **Time-series DB**: InfluxDB/Prometheus
3. **Distributed processing**: Celery/Ray
4. **Web API**: REST endpoints
5. **Caching**: Redis for hot data

## Testing Strategy

### Unit Tests
- Individual module functions
- Mock external dependencies
- Edge cases

### Integration Tests
- Module interactions
- End-to-end flows
- Data pipeline

### Manual Testing
- Run each module standalone
- Verify dashboard functionality
- Stress test with synthetic data

## Deployment Options

### Local Development
```
streamlit run app.py
```

### Production Server
```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Future)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

### Cloud Platforms
- **Streamlit Cloud**: Direct deployment
- **Heroku**: With Procfile
- **AWS EC2**: Manual setup
- **Google Cloud Run**: Containerized

## Conclusion

This architecture balances:
- **Performance**: Fast, CPU-efficient
- **Accuracy**: Multiple detection methods
- **Usability**: Intuitive interface
- **Maintainability**: Modular design
- **Extensibility**: Easy to enhance

The system demonstrates production-ready patterns while remaining accessible to beginners.
