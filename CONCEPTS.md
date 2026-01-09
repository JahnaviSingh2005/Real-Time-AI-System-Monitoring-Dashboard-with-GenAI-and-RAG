# 🧠 Core Concepts & Technologies

This document provides a deep dive into the theoretical and technical concepts used in the **AI System Monitoring Dashboard**.

---

## 1. System Metrics Collection (`psutil`)
At its core, the dashboard relies on [psutil](https://github.com/giampaolo/psutil) (process and system utilities), a cross-platform library for retrieving information on running processes and system utilization.

- **CPU Utilization**: Obtained via `psutil.cpu_percent()`. We use an interval to calculate the mean usage over a specific window.
- **Memory Management**: Tracks Virtual Memory (RAM) using `psutil.virtual_memory()`, monitoring used vs. available gigabytes.
- **Process Correlation**: The dashboard iterates through `psutil.process_iter()` to find "resource hogs"—processes with the highest CPU or Memory footprint—providing context when system-wide anomalies occur.

---

## 2. Anomaly Detection: Two-Tiered Approach

### Tier A: Rule-Based Alerts (Deterministic)
Rule-based detection uses user-configurable thresholds (e.g., "Alert if CPU > 85%"). 
- **Pros**: Predictable, easy to understand, zero "training" time.
- **Cons**: Cannot detect subtle patterns or "soft" failures where metrics are within range but behaving strangely.

### Tier B: Machine Learning (Isolation Forest)
We use the [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) algorithm from Scikit-Learn for ML-based detection.
- **How it works**: Isolation Forest works on the principle that anomalies are few and different. It "isolates" observations by randomly selecting a feature and then randomly selecting a split value. Outliers are typically isolated in fewer splits than normal points.
- **Contamination**: A parameter that defines the expected proportion of outliers in the dataset.
- **Anomaly Score**: A measure of how "isolated" a point is. Negative scores indicate anomalies.

---

## 3. Generative AI (GenAI) Explaners
The project uses a lightweight Language Model (LLM)—specifically `distilgpt2`—to provide natural language insights.

- **Semantic Translation**: Instead of just showing "CPU: 99%", the GenAI explains *why* this matters (e.g., "The system is experiencing extreme computational stress which may lead to application timeouts").
- **CPU Optimization**: We use the HuggingFace `transformers` pipeline configured for CPU execution (`device=-1`), ensuring that the monitoring tool itself doesn't become the bottleneck.

---

## 4. Retrieval-Augmented Generation (RAG)
RAG combines the power of LLMs with specialized internal knowledge (in this case, your historical incident logs).

- **Vector Database (ChromaDB)**: Unlike traditional databases that search for exact text matches, ChromaDB stores "embeddings" (mathematical representations of text). 
- **Semantic Search**: When an incident occurs, the system converts the current metrics into a query, finds the most similar *past* incident in the Vector DB, and provides the "past solution" to the user.
- **Context Window**: This ensures the AI model stays "grounded" in your specific system's history rather than giving generic advice.

---

## 5. Persistence & Incident Lifecycle
To enable "Incident Replay," the system manages data across three layers:

1.  **Real-time Buffer**: Small window of metrics kept in memory for live charts.
2.  **Long-term Storage (SQLite)**: Every collected metric and alert is persisted to a local SQLite database (`monitoring_history.db`).
3.  **Incident Grouping**: When an anomaly is detected, the system starts an "Active Incident." When metrics return to normal, it wraps these logs into a single "Incident Entry" for future post-mortem analysis.

---

## 6. Resources for Further Learning

### Official Documentation
- [Streamlit Docs](https://docs.streamlit.io/) - For UI components and Session State.
- [Plotly Python](https://plotly.com/python/) - For the interactive gauge and time-series charts.
- [Scikit-Learn Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html) - For deep dives into Isolation Forest.
- [ChromaDB Docs](https://docs.trychroma.com/) - For vector storage concepts.

### Concepts
- [What is RAG?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [Understanding Outlier Detection](https://towardsdatascience.com/anomaly-detection-with-isolation-forest-e41f7f559d09)
