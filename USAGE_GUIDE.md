# 🎯 Complete Usage Guide

## Table of Contents
1. [First Launch](#first-launch)
2. [Dashboard Overview](#dashboard-overview)
3. [Live Monitoring Tab](#live-monitoring-tab)
4. [Anomaly Detection Tab](#anomaly-detection-tab)
5. [Chat Assistant Tab](#chat-assistant-tab)
6. [Advanced Features](#advanced-features)
7. [Tips & Tricks](#tips--tricks)
8. [Common Scenarios](#common-scenarios)

---

## First Launch

### Step 1: Start the Dashboard
```bash
streamlit run app.py
```

### Step 2: Wait for Browser
- Dashboard opens automatically in your default browser
- URL: `http://localhost:8501`
- If it doesn't open, manually navigate to the URL

### Step 3: Initial Data Collection
- **Wait 2-3 minutes** for the system to collect baseline data
- Watch the gauges update every few seconds
- See the history charts populate

### Step 4: Train ML Model
- After collecting 10+ data points
- Click **"🤖 Train ML Model"** in the sidebar
- Wait for "Model trained successfully!" message

---

## Dashboard Overview

### Layout

```
┌─────────────────────────────────────────────────┐
│          🖥️ AI System Monitoring Dashboard      │
├──────────────┬──────────────────────────────────┤
│              │                                   │
│   SIDEBAR    │         MAIN CONTENT             │
│              │                                   │
│  ⚙️ Settings  │   📊 Live Monitoring             │
│  🤖 Train ML  │   🔍 Anomaly Detection            │
│  💻 Sys Info  │   💬 Chat Assistant               │
│  📚 RAG Stats │   📖 About                        │
│              │                                   │
└──────────────┴──────────────────────────────────┘
```

### Sidebar Controls

**Settings**:
- **Refresh Interval**: How often to update (1-10 seconds)
  - Lower = More responsive, higher CPU usage
  - Higher = Less responsive, lower CPU usage
  - Recommended: 3-5 seconds

**ML Model**:
- **Train Button**: Trains the Isolation Forest model
- **Status Indicator**: Shows if model is trained
- **Requirement**: Need 10+ data samples

**System Info**:
- Operating System
- CPU count
- Total RAM
- (Read-only information)

**Knowledge Base**:
- Number of historical incidents loaded
- RAG system status

---

## Live Monitoring Tab

### Health Status (Top)

```
✅ System Health: Healthy
Score: 85.2/100 - System is running smoothly
```

**What it means**:
- **Score 70-100**: ✅ Healthy (Green)
- **Score 50-70**: ⚠️ Warning (Orange)
- **Score 0-50**: 🔴 Critical (Red)

### Gauge Charts

Three real-time gauges show:

1. **CPU Usage**
   - Green zone: 0-70% (Normal)
   - Yellow zone: 70-85% (Warning)
   - Red zone: 85-100% (Critical)

2. **Memory Usage**
   - Green zone: 0-75% (Normal)
   - Yellow zone: 75-85% (Warning)
   - Red zone: 85-100% (Critical)

3. **Disk Usage**
   - Green zone: 0-80% (Normal)
   - Yellow zone: 80-90% (Warning)
   - Red zone: 90-100% (Critical)

### Trends

Below each gauge, you'll see trend indicators:
- 📈 **Increasing**: Usage going up
- 📉 **Decreasing**: Usage going down
- ➡️ **Stable**: No significant change

**Example**:
```
CPU Usage: 65.2%
Trend: 📈 Increasing
```

### Historical Charts

**Multi-Metric Chart**:
- Shows all three metrics on one chart
- Different colors for each metric
- Hover to see exact values
- Unified timeline

**Individual Charts**:
- Detailed view of each metric
- Easier to spot patterns
- Line + markers for clarity

### Detailed Metrics (Expandable)

Click "🔍 Detailed Metrics" to see:
- CPU count and frequency
- Memory used/available/total
- Disk used/total
- Network sent/received
- All values in appropriate units (GB, MB, MHz)

---

## Anomaly Detection Tab

### Rule-Based Alerts

**How it works**:
- Checks metrics against thresholds
- Instant alerts when limits exceeded
- Four severity levels

**Example Alert**:
```
🔴 CRITICAL [14:30:25]: CPU is at 95.2% (threshold: 95%)
```

**No Alerts**:
```
✅ No alerts - All metrics within normal range
```

### ML-Based Detection

**What you see**:
```
🚨 Anomaly Detected!
Anomaly Score: -0.345
```

**or**

```
✅ Normal Behavior
Anomaly Score: 0.123
```

**Understanding Scores**:
- **Negative scores**: Anomaly (more negative = more unusual)
- **Positive scores**: Normal
- **Around zero**: Borderline

**Recent Anomalies Table**:
- Shows last 5 ML-detected anomalies
- Includes timestamp, metrics, and score
- Helps identify patterns

### AI-Generated Explanation

When anomaly detected, you'll see:

```
🔍 Anomaly Analysis

 Current System State:
- CPU Usage: 92.5%
- Memory Usage: 78.3%
- Disk Usage: 65.0%

ML Detection: Anomaly detected (score: -0.250)
The ML model identified unusual patterns in system behavior.

Active Alerts (2):
- HIGH: CPU is at 92.5% (threshold: 85%)
- MEDIUM: Memory is at 78.3% (threshold: 75%)

Possible Causes:
- Elevated CPU usage. System may be under heavy computational load.
- High memory consumption. Applications may be using excessive RAM.

Recommended Actions:
- Check Task Manager for processes consuming excessive CPU
- Consider closing unnecessary applications
- Review running applications and close memory-intensive ones
```

### Similar Past Incidents (RAG)

```
📚 Similar Past Incidents:

Incident 1 (2026-01-01 10:30:00)
- Severity: high
- Metric: cpu_usage at 95.5%
- Issue: CPU usage spiked to 95.5% due to a runaway background process...
- Resolution: Terminated background process ID 1234...

Incident 2 (2026-01-03 08:45:00)
- Severity: medium
- Metric: cpu_usage at 88.7%
- Issue: High CPU usage during backup process...
- Resolution: Adjusted alert threshold for backup windows...
```

---

## Chat Assistant Tab

### How to Use

**Ask Questions Naturally**:
```
You: "What is high CPU usage?"
```

```
Assistant: About CPU Usage:

CPU (Central Processing Unit) usage indicates how much of your 
processor's capacity is being used.

Normal Range: 0-70%
Warning: 70-85%
Critical: >85%

Common Causes of High CPU:
- Background applications
- Antivirus scans
- System updates
- Runaway processes
...
```

### Example Questions

**About Metrics**:
- "What is CPU usage?"
- "Explain memory usage"
- "Why is my disk full?"
- "What is normal RAM usage?"

**System Status**:
- "What is the current system status?"
- "How is my system doing?"
- "Show me system health"

**Historical**:
- "Show me similar past incidents"
- "What happened before with high CPU?"
- "Find past memory issues"

**Help**:
- "What can you do?"
- "How can you help?"
- "Help me"

### Quick Action Buttons

Three preset buttons for common queries:

1. **💡 How can you help?**
   - Shows all capabilities
   - Lists topics you can ask about

2. **📊 Show system status**
   - Current CPU, Memory, Disk
   - Status indicators (✅/⚠️/🔴)

3. **📚 Similar incidents**
   - Finds related past cases
   - Based on current metrics

### Chat History

- All conversations are saved during session
- Scroll up to see previous exchanges
- Cleared when you refresh the page

---

## Advanced Features

### Custom Thresholds

**To modify alert thresholds**, edit the code:

```python
# In src/anomaly/rule_based.py
alert_system.set_threshold('cpu_percent', 'critical', 90.0)
```

**Or use environment variables** (`.env` file):
```
CPU_THRESHOLD_CRITICAL=90
MEMORY_THRESHOLD_CRITICAL=90
```

### Add Custom Incidents

Edit `data/incidents.json`:

```json
{
  "timestamp": "2026-01-08 10:00:00",
  "metric": "memory_usage",
  "value": 95.0,
  "description": "Detailed description of what happened",
  "resolution": "How you fixed it",
  "severity": "high"
}
```

Then restart the app to load new incidents.

### Export Data

Currently manual. To get metrics history:

```python
# Add this to your code to save data
df = st.session_state.metrics_collector.get_metrics_dataframe()
df.to_csv('metrics_export.csv', index=False)
```

---

## Tips & Tricks

### Optimization

1. **Adjust Refresh Rate**:
   - 1-2 seconds: Very responsive, but CPU-intensive
   - 3-5 seconds: Good balance (recommended)
   - 5-10 seconds: Lower overhead

2. **Clean Up History**:
   - History is capped at 100 data points
   - Older data automatically discarded
   - No manual cleanup needed

3. **Train ML Regularly**:
   - Retrain weekly or when workload changes
   - ML adapts to new patterns
   - Button in sidebar

### Best Practices

1. **Baseline Collection**:
   - Let run for 5-10 minutes during normal use
   - Train ML model
   - This becomes your "normal" baseline

2. **Incident Documentation**:
   - When you fix an issue, add it to `incidents.json`
   - Include what you did
   - RAG will find it next time

3. **Regular Monitoring**:
   - Check dashboard once or twice daily
   - Look for unusual patterns
   - Train ML if you see "not trained" message

### Keyboard Shortcuts

(Standard Streamlit shortcuts)
- **Ctrl+Shift+R**: Rerun app
- **Ctrl+K**: Clear cache
- **Esc**: Close sidebar (if open)

---

## Common Scenarios

### Scenario 1: High CPU Alert

**What you see**:
```
🔴 CRITICAL: CPU is at 96.5% (threshold: 95%)
```

**What to do**:
1. Go to "Anomaly Detection" tab
2. Read the AI explanation
3. Check "Similar Past Incidents"
4. Follow recommended actions
5. Open Task Manager (Ctrl+Shift+Esc)
6. Sort by CPU usage
7. Identify the culprit
8. End task or investigate further

### Scenario 2: Memory Leak Suspected

**Symptoms**:
- Memory usage keeps increasing
- Trend shows 📈 Increasing
- Eventually hits warning/critical

**What to do**:
1. Check "Historical Trends" for pattern
2. Ask chat: "What causes memory leaks?"
3. Monitor over time (10-20 minutes)
4. If confirmed, restart application
5. Add incident to `incidents.json` for future reference

### Scenario 3: Everything Looks Normal

**What you see**:
```
✅ System Health: Healthy
Score: 92.5/100
```

**What to do**:
- Great! System is fine
- Check occasionally for changes
- Use this time to train ML model
- Explore the chat feature

### Scenario 4: ML Model Says Anomaly, but No Alerts

**What it means**:
- ML detected unusual *pattern*
- But no single metric exceeded threshold
- Could be combination of factors

**Example**:
- CPU: 75% (not critical alone)
- Memory: 80% (not critical alone)
- Together: Unusual for your system

**What to do**:
- Read AI explanation
- Check if this is expected (e.g., during backup)
- If unexpected, investigate further
- Consider it early warning

### Scenario 5: Need to Understand a Metric

**Use the chat**:
```
You: "What is CPU usage?"
Assistant: [Detailed explanation]

You: "What is normal for my system?"
Assistant: [Context-aware answer based on your baselines]
```

---

## Keyboard Shortcuts Summary

| Shortcut | Action |
|----------|--------|
| Ctrl+Shift+Esc | Open Task Manager (Windows) |
| Ctrl+C | Stop Streamlit server (in terminal) |
| F5 | Refresh browser (re-renders but loses session) |
| Ctrl+K | Clear Streamlit cache |

---

## Need Help?

### In the App
- Click "About" tab for overview
- Use chat: "help"
- Check sidebar for system info

### Documentation
- `README.md`: Full documentation
- `QUICKSTART.md`: Quick reference
- `ARCHITECTURE.md`: Technical details
- `INSTALL.md`: Installation help

### Debugging
- Run `python test_modules.py`
- Check terminal for error messages
- Look at Streamlit logs

---

## Enjoy Monitoring! 🚀

Remember:
- ✅ Let it collect data before training ML
- ✅ Check regularly for anomalies
- ✅ Add your own incidents to learn over time
- ✅ Use the chat for questions
- ✅ Adjust settings to your preference

**Happy monitoring!** 🖥️📊🔍
