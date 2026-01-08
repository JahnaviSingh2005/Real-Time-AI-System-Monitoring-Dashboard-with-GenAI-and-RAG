"""
System Test Script
------------------
Test all modules independently to ensure they work correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("AI System Monitoring Dashboard - Module Tests")
print("=" * 60)

# Test 1: Metrics Collector
print("\n[1/6] Testing Metrics Collector...")
try:
    from src.metrics.collector import SystemMetricsCollector
    collector = SystemMetricsCollector()
    metrics = collector.collect_current_metrics()
    print(f"✅ Metrics Collector: OK")
    print(f"    CPU: {metrics['cpu_percent']:.1f}%")
    print(f"    Memory: {metrics['memory_percent']:.1f}%")
    print(f"    Disk: {metrics['disk_percent']:.1f}%")
except Exception as e:
    print(f"❌ Metrics Collector: FAILED - {e}")

# Test 2: Rule-Based Alerts
print("\n[2/6] Testing Rule-Based Alerts...")
try:
    from src.anomaly.rule_based import RuleBasedAlertSystem
    alert_system = RuleBasedAlertSystem()
    test_metrics = {
        'cpu_percent': 92.0,
        'memory_percent': 78.0,
        'disk_percent': 65.0
    }
    alerts = alert_system.check_all_metrics(test_metrics)
    print(f"✅ Rule-Based Alerts: OK")
    print(f"    Generated {len(alerts)} alert(s) for test data")
except Exception as e:
    print(f"❌ Rule-Based Alerts: FAILED - {e}")

# Test 3: ML Detector
print("\n[3/6] Testing ML Detector...")
try:
    from src.anomaly.ml_detector import MLAnomalyDetector
    import numpy as np
    import pandas as pd
    
    detector = MLAnomalyDetector()
    
    # Create synthetic training data
    np.random.seed(42)
    train_data = pd.DataFrame({
        'cpu_percent': np.random.normal(50, 10, 30),
        'memory_percent': np.random.normal(60, 8, 30),
        'disk_percent': np.random.normal(70, 5, 30)
    })
    
    success = detector.train(train_data)
    if success:
        test_sample = {
            'cpu_percent': 95.0,
            'memory_percent': 98.0,
            'disk_percent': 92.0
        }
        is_anomaly, score = detector.predict(test_sample)
        print(f"✅ ML Detector: OK")
        print(f"    Model trained on {len(train_data)} samples")
        print(f"    Test sample: Anomaly={is_anomaly}, Score={score:.3f}")
    else:
        print(f"⚠️ ML Detector: Training failed (needs more data)")
except Exception as e:
    print(f"❌ ML Detector: FAILED - {e}")

# Test 4: Explainer
print("\n[4/6] Testing GenAI Explainer...")
try:
    from src.genai.explainer import AnomalyExplainer
    
    explainer = AnomalyExplainer()
    test_metrics = {
        'cpu_percent': 92.5,
        'memory_percent': 78.3,
        'disk_percent': 65.0
    }
    
    explanation = explainer.explain_anomaly(
        test_metrics,
        is_ml_anomaly=True,
        anomaly_score=-0.25
    )
    
    print(f"✅ GenAI Explainer: OK")
    print(f"    Generated {len(explanation)} character explanation")
except Exception as e:
    print(f"❌ GenAI Explainer: FAILED - {e}")

# Test 5: RAG System
print("\n[5/6] Testing RAG System...")
try:
    from src.genai.rag_system import RAGSystem
    
    # Create test directory
    test_db_path = "./test_chroma_db_temp"
    rag = RAGSystem(persist_directory=test_db_path)
    
    # Try to load incidents
    incidents_loaded = rag.load_incidents_from_json("./data/incidents.json")
    
    if incidents_loaded:
        stats = rag.get_statistics()
        print(f"✅ RAG System: OK")
        print(f"    Loaded {stats['total_incidents']} incidents")
        
        # Test search
        results = rag.search_similar_incidents("high CPU usage", n_results=2)
        print(f"    Search test: Found {len(results)} similar incidents")
    else:
        print(f"⚠️ RAG System: Running but incidents not loaded")
    
    # Cleanup
    import shutil
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
        
except Exception as e:
    print(f"❌ RAG System: FAILED - {e}")

# Test 6: Utilities
print("\n[6/6] Testing Utilities...")
try:
    from src.utils.helpers import get_health_status, calculate_trend
    
    health = get_health_status(cpu=65.0, memory=70.0, disk=55.0)
    print(f"✅ Utilities: OK")
    print(f"    Health status: {health['emoji']} {health['status']} ({health['score']}/100)")
    
    trend = calculate_trend([50, 55, 60, 65, 70])
    print(f"    Trend detection: {trend}")
except Exception as e:
    print(f"❌ Utilities: FAILED - {e}")

# Summary
print("\n" + "=" * 60)
print("Module Testing Complete!")
print("=" * 60)
print("\nIf all tests passed ✅, you're ready to run the dashboard!")
print("Run: streamlit run app.py")
print("=" * 60)
