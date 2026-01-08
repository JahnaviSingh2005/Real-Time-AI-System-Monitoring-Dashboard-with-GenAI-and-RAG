# Quick Start Guide

## Installation (5 minutes)

### Step 1: Setup Virtual Environment
```bash
# Navigate to project directory
cd "AI System Monitoring"

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will take 2-3 minutes as it downloads all required packages.

### Step 3: Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## First Time Use

### 1. Observe Live Metrics (1 minute)
- Watch the gauges update in real-time
- Notice CPU, Memory, and Disk usage
- See the health score indicator

### 2. Collect Baseline Data (2-3 minutes)
- Let the system run for a few minutes
- This collects normal operating patterns
- Watch the history charts populate

### 3. Train ML Model (30 seconds)
- Once you have 10+ data points
- Click "🤖 Train ML Model" in the sidebar
- Wait for confirmation message

### 4. Explore Features (5 minutes)

**Live Monitoring Tab:**
- Real-time gauges
- Historical trends
- Detailed metrics

**Anomaly Detection Tab:**
- Rule-based alerts
- ML anomaly detection
- AI explanations
- Similar past incidents

**Chat Assistant Tab:**
- Ask "What can you do?"
- Try "Show system status"
- Query "What is CPU usage?"

## Example Usage Scenarios

### Scenario 1: Check System Health
1. Open "Live Monitoring" tab
2. Look at health status at top
3. Check if any metrics are in red/orange zones
4. View trends to see if issues are getting worse

### Scenario 2: Investigate an Anomaly
1. Go to "Anomaly Detection" tab
2. See if alerts are triggered
3. Read the AI explanation
4. Check similar past incidents
5. Follow recommended actions

### Scenario 3: Learn About Metrics
1. Go to "Chat Assistant" tab
2. Ask "What is high CPU usage?"
3. Ask "How do I fix memory issues?"
4. Ask "Show me similar past incidents"

## Quick Tips

✅ **DO:**
- Let it run for a few minutes before training ML
- Train ML during normal system operation
- Ask questions in the chat to learn
- Check past incidents when issues occur

❌ **DON'T:**
- Train ML immediately (need baseline data)
- Ignore the health status indicator
- Set refresh interval too low (causes overhead)
- Close without stopping the server properly

## Common Questions

**Q: How often should I retrain the ML model?**
A: Once a week or when system workload changes significantly.

**Q: What if I see an anomaly but no alerts?**
A: ML detected unusual pattern that didn't breach thresholds. Still worth investigating.

**Q: Can I monitor multiple machines?**
A: Current version monitors the local machine. Multi-system support is planned.

**Q: Is internet required?**
A: Only for initial package installation. Dashboard runs entirely offline after setup.

**Q: How much RAM does it use?**
A: Typically 200-500MB. Adjust history_size if needed.

## Troubleshooting

**Problem: "Model not trained" message**
→ Wait for more data points, then click Train ML Model button

**Problem: Dashboard won't start**
→ Make sure virtual environment is activated
→ Check if port 8501 is available

**Problem: High CPU usage from the dashboard itself**
→ Increase refresh interval in sidebar
→ Reduce history size

**Problem: No past incidents showing**
→ Check if `data/incidents.json` exists
→ Look at "Knowledge Base" count in sidebar

## Next Steps

After getting familiar with basics:

1. **Customize thresholds** in `src/anomaly/rule_based.py`
2. **Add your own incidents** to `data/incidents.json`
3. **Explore individual modules** by running them standalone
4. **Read the full README.md** for advanced features
5. **Check the About tab** in the dashboard for architecture

## Need Help?

- Check the **About tab** in the dashboard
- Read the **full README.md**
- Use the **chat assistant** in the app
- Review **code comments** in source files

---

**Enjoy monitoring! 🚀**

Total setup time: ~10 minutes
