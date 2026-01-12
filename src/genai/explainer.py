"""
GenAI Explainer
---------------
This module uses a lightweight language model to generate natural language
explanations for system anomalies. Uses CPU-friendly models.
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


class AnomalyExplainer:
    """
    Generates natural language explanations for system anomalies using GenAI.
    
    Uses a small, CPU-friendly language model to provide human-readable
    explanations of what might be causing system issues.
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the explainer with a language model.
        
        Args:
            model_name: Name of the HuggingFace model to use (default: distilgpt2)
                       distilgpt2 is small (~300MB) and runs well on CPU
        """
        self.model_name = model_name
        self.generator = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the language model (lazy loading)"""
        if not self.is_loaded:
            try:
                print(f"Loading GenAI model: {self.model_name}...")
                # Use text-generation pipeline with CPU
                self.generator = pipeline(
                    'text-generation',
                    model=self.model_name,
                    device=-1  # -1 means CPU
                )
                self.is_loaded = True
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.is_loaded = False
    
    def explain_anomaly(self, metrics: Dict, is_ml_anomaly: bool = False, 
                       anomaly_score: float = 0.0, alerts: list = None) -> str:
        """
        Generate explanation for detected anomaly.
        
        Args:
            metrics: Current system metrics
            is_ml_anomaly: Whether ML model detected anomaly
            anomaly_score: Anomaly score from ML model
            alerts: List of rule-based alerts
            
        Returns:
            Natural language explanation of the anomaly
        """
        # Build explanation using template-based approach (more reliable than pure generation)
        explanation_parts = []
        
        # Header
        explanation_parts.append("🔍 **Anomaly Analysis**\n")
        
        # Current state
        cpu = metrics.get('cpu_percent', 0)
        memory = metrics.get('memory_percent', 0)
        disk = metrics.get('disk_percent', 0)
        
        explanation_parts.append(f"**Current System State:**")
        explanation_parts.append(f"- CPU Usage: {cpu:.1f}%")
        explanation_parts.append(f"- Memory Usage: {memory:.1f}%")
        explanation_parts.append(f"- Disk Usage: {disk:.1f}%\n")
        
        # ML Detection
        if is_ml_anomaly:
            explanation_parts.append(f"**ML Detection:** Anomaly detected (score: {anomaly_score:.3f})")
            explanation_parts.append("The ML model identified unusual patterns in system behavior.\n")
        
        # Rule-based alerts
        if alerts:
            explanation_parts.append(f"**Active Alerts ({len(alerts)}):**")
            for alert in alerts:
                explanation_parts.append(f"- {alert.get('message', 'Unknown alert')}")
            explanation_parts.append("")
        
        # Top Processes Correlation
        top_cpu = metrics.get('top_cpu_processes', [])
        top_mem = metrics.get('top_memory_processes', [])
        
        if top_cpu or top_mem:
            explanation_parts.append("**Top Resource-Consuming Processes:**")
            if cpu > 70 and top_cpu:
                explanation_parts.append("*Top CPU Consumers:*")
                for p in top_cpu[:3]:
                    explanation_parts.append(f"  - {p['name']} (PID: {p['pid']}): {p['cpu_percent']:.1f}% CPU")
            
            if memory > 75 and top_mem:
                explanation_parts.append("*Top Memory Consumers:*")
                for p in top_mem[:3]:
                    explanation_parts.append(f"  - {p['name']} (PID: {p['pid']}): {p['memory_percent']:.1f}% RAM")
            explanation_parts.append("")

        # Generate insights based on metrics
        insights = self._generate_insights(cpu, memory, disk, top_cpu, top_mem)
        if insights:
            explanation_parts.append("**Possible Causes:**")
            for insight in insights:
                explanation_parts.append(f"- {insight}")
            explanation_parts.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(cpu, memory, disk, alerts)
        if recommendations:
            explanation_parts.append("**Recommended Actions:**")
            for rec in recommendations:
                explanation_parts.append(f"- {rec}")
        
        return "\n".join(explanation_parts)
    
    def _generate_insights(self, cpu: float, memory: float, disk: float, 
                           top_cpu: list = None, top_mem: list = None) -> list:
        """Generate insights based on metric values and top processes"""
        insights = []
        
        if cpu > 90:
            msg = "Extremely high CPU usage detected."
            if top_cpu and top_cpu[0]['cpu_percent'] > 50:
                msg += f" Process '{top_cpu[0]['name']}' is consuming {top_cpu[0]['cpu_percent']:.1f}% CPU."
            insights.append(msg)
        elif cpu > 70:
            insights.append("Elevated CPU usage. System may be under heavy computational load.")
        
        if memory > 90:
            msg = "Critical memory usage."
            if top_mem and top_mem[0]['memory_percent'] > 30:
                msg += f" Process '{top_mem[0]['name']}' is using {top_mem[0]['memory_percent']:.1f}% of RAM."
            insights.append(msg)
        elif memory > 75:
            insights.append("High memory consumption. Applications may be using excessive RAM.")
        
        if disk > 90:
            insights.append("Disk space critically low. Immediate cleanup recommended.")
        elif disk > 80:
            insights.append("Disk usage is high. Consider removing unnecessary files.")
        
        # Combined conditions
        if cpu > 80 and memory > 80:
            insights.append("Both CPU and memory are stressed. System may be experiencing resource contention.")
        
        return insights
    
    def _generate_recommendations(self, cpu: float, memory: float, 
                                 disk: float, alerts: list) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        if cpu > 85:
            recommendations.append("Check Task Manager for processes consuming excessive CPU")
            recommendations.append("Consider closing unnecessary applications")
            if cpu > 95:
                recommendations.append("URGENT: Identify and terminate problematic processes immediately")
        
        if memory > 85:
            recommendations.append("Review running applications and close memory-intensive ones")
            recommendations.append("Clear browser caches and temporary files")
            if memory > 95:
                recommendations.append("URGENT: Restart memory-heavy applications or reboot system")
        
        if disk > 85:
            recommendations.append("Run disk cleanup utility")
            recommendations.append("Remove old logs and temporary files")
            recommendations.append("Consider moving large files to external storage")
        
        # General recommendations
        if alerts and len(alerts) > 2:
            recommendations.append("Multiple alerts active - prioritize by severity")
        
        if not recommendations:
            recommendations.append("Monitor system for continued unusual behavior")
            recommendations.append("Check system logs for additional details")
        
        return recommendations
    
    def generate_summary(self, metrics_history: list) -> str:
        """
        Generate a summary of system health over time.
        
        Args:
            metrics_history: List of historical metric readings
            
        Returns:
            Summary text of system health trends
        """
        if not metrics_history:
            return "No historical data available for analysis."
        
        df = pd.DataFrame(metrics_history)
        
        summary_parts = []
        summary_parts.append("📊 **System Health Summary**\n")
        
        # Calculate statistics
        avg_cpu = df['cpu_percent'].mean()
        max_cpu = df['cpu_percent'].max()
        avg_memory = df['memory_percent'].mean()
        max_memory = df['memory_percent'].max()
        
        summary_parts.append(f"**Analysis Period:** Last {len(metrics_history)} readings\n")
        
        summary_parts.append("**CPU Statistics:**")
        summary_parts.append(f"- Average: {avg_cpu:.1f}%")
        summary_parts.append(f"- Peak: {max_cpu:.1f}%")
        
        if avg_cpu > 70:
            summary_parts.append(f"- ⚠️ High average CPU usage - system under sustained load\n")
        else:
            summary_parts.append(f"- ✅ CPU usage within normal range\n")
        
        summary_parts.append("**Memory Statistics:**")
        summary_parts.append(f"- Average: {avg_memory:.1f}%")
        summary_parts.append(f"- Peak: {max_memory:.1f}%")
        
        if avg_memory > 75:
            summary_parts.append(f"- ⚠️ High average memory usage - consider optimization")
        else:
            summary_parts.append(f"- ✅ Memory usage within normal range")
        
        return "\n".join(summary_parts)


    def generate_incident_report(self, incident_data: Dict) -> str:
        """
        Generate a detailed incident report from gathered data.
        
        Args:
            incident_data: Dictionary containing:
                - start_time, end_time
                - alerts (list)
                - metrics_history (list)
                - top_processes (list)
                
        Returns:
            Formatted Markdown report
        """
        start = incident_data.get('start_time')
        end = incident_data.get('end_time')
        alerts = incident_data.get('alerts', [])
        metrics = incident_data.get('metrics_history', [])
        
        report = []
        report.append(f"# 🛡️ Incident Report: {start} to {end}")
        report.append(f"**Status:** Resolved (Generated at {datetime.now()})\n")
        
        report.append("## 📝 Executive Summary")
        if alerts:
            report.append(f"During this period, {len(alerts)} alerts were triggered. ")
            metrics_df = pd.DataFrame(metrics)
            if not metrics_df.empty:
                max_cpu = metrics_df['cpu_percent'].max()
                max_mem = metrics_df['memory_percent'].max()
                report.append(f"System experienced peak CPU usage of {max_cpu:.1f}% and peak Memory usage of {max_mem:.1f}%.")
        else:
            report.append("Detailed analysis of system behavior during the selected timeframe.")
        report.append("\n")
        
        report.append("## 🕒 Timeline of Events")
        for alert in alerts[:10]: # Limit to 10 for brevity
            report.append(f"- **{alert.get('timestamp')}**: {alert.get('message')}")
        report.append("\n")
        
        report.append("## 🔍 Root Cause Analysis")
        # Extract process correlation if available
        culprits = set()
        for m in metrics:
            for p in m.get('top_cpu_processes', []):
                if p['cpu_percent'] > 50: culprits.add(p['name'])
        
        if culprits:
            report.append(f"Possible application-level causes identified: **{', '.join(culprits)}**.")
        else:
            report.append("No specific process-level culprit identified. Possible system-wide resource contention.")
        report.append("\n")
        
        report.append("## 🛠️ Remediation Steps")
        report.append("1. **Verify Services**: Ensure all critical services are healthy.")
        report.append("2. **Resource Scaling**: If this occurs frequently, consider increasing CPU/RAM allocation.")
        report.append("3. **Log Review**: Check application logs for the identified processes.")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test the explainer
    print("Testing Anomaly Explainer...")
    
    explainer = AnomalyExplainer()
    
    # Test metrics
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
    
    print(explanation)
