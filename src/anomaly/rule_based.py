"""
Rule-Based Alert System
-----------------------
This module implements simple threshold-based alerts for system metrics.
It checks if metrics exceed predefined thresholds and generates alerts.
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Alert:
    """Represents a system alert"""
    
    def __init__(self, metric: str, value: float, threshold: float, 
                 severity: AlertSeverity, message: str):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.severity = severity
        self.message = message
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'metric': self.metric,
            'value': self.value,
            'threshold': self.threshold,
            'severity': self.severity.value,
            'message': self.message
        }


class RuleBasedAlertSystem:
    """
    Rule-based alert system that checks metrics against thresholds.
    
    This system allows configuring custom thresholds for different metrics
    and generates alerts when those thresholds are exceeded.
    """
    
    def __init__(self):
        """Initialize with default thresholds"""
        # Default thresholds for different metrics
        self.thresholds = {
            'cpu_percent': {
                'medium': 70.0,
                'high': 85.0,
                'critical': 95.0
            },
            'memory_percent': {
                'medium': 75.0,
                'high': 85.0,
                'critical': 95.0
            },
            'disk_percent': {
                'medium': 80.0,
                'high': 90.0,
                'critical': 95.0
            }
        }
        
        self.alert_history: List[Alert] = []
    
    def set_threshold(self, metric: str, severity: str, value: float):
        """
        Set a custom threshold for a metric.
        
        Args:
            metric: Name of the metric (e.g., 'cpu_percent')
            severity: Severity level ('low', 'medium', 'high', 'critical')
            value: Threshold value
        """
        if metric not in self.thresholds:
            self.thresholds[metric] = {}
        self.thresholds[metric][severity] = value
    
    def check_metric(self, metric_name: str, value: float) -> Optional[Alert]:
        """
        Check a single metric against thresholds.
        
        Args:
            metric_name: Name of the metric to check
            value: Current value of the metric
            
        Returns:
            Alert object if threshold exceeded, None otherwise
        """
        if metric_name not in self.thresholds:
            return None
        
        thresholds = self.thresholds[metric_name]
        
        # Check from highest to lowest severity
        if 'critical' in thresholds and value >= thresholds['critical']:
            alert = Alert(
                metric=metric_name,
                value=value,
                threshold=thresholds['critical'],
                severity=AlertSeverity.CRITICAL,
                message=f"CRITICAL: {metric_name} is at {value:.1f}% (threshold: {thresholds['critical']}%)"
            )
        elif 'high' in thresholds and value >= thresholds['high']:
            alert = Alert(
                metric=metric_name,
                value=value,
                threshold=thresholds['high'],
                severity=AlertSeverity.HIGH,
                message=f"HIGH: {metric_name} is at {value:.1f}% (threshold: {thresholds['high']}%)"
            )
        elif 'medium' in thresholds and value >= thresholds['medium']:
            alert = Alert(
                metric=metric_name,
                value=value,
                threshold=thresholds['medium'],
                severity=AlertSeverity.MEDIUM,
                message=f"MEDIUM: {metric_name} is at {value:.1f}% (threshold: {thresholds['medium']}%)"
            )
        else:
            return None
        
        self.alert_history.append(alert)
        return alert
    
    def check_all_metrics(self, metrics: Dict) -> List[Alert]:
        """
        Check all metrics and return list of alerts.
        
        Args:
            metrics: Dictionary of metric names and values
            
        Returns:
            List of Alert objects for metrics exceeding thresholds
        """
        alerts = []
        
        # Check CPU
        if 'cpu_percent' in metrics:
            alert = self.check_metric('cpu_percent', metrics['cpu_percent'])
            if alert:
                alerts.append(alert)
        
        # Check Memory
        if 'memory_percent' in metrics:
            alert = self.check_metric('memory_percent', metrics['memory_percent'])
            if alert:
                alerts.append(alert)
        
        # Check Disk
        if 'disk_percent' in metrics:
            alert = self.check_metric('disk_percent', metrics['disk_percent'])
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """
        Get the most recent alerts.
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of alert dictionaries
        """
        recent = self.alert_history[-count:] if self.alert_history else []
        return [alert.to_dict() for alert in recent]
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history = []


if __name__ == "__main__":
    # Test the alert system
    print("Testing Rule-Based Alert System...")
    
    alert_system = RuleBasedAlertSystem()
    
    # Test with sample metrics
    test_metrics = {
        'cpu_percent': 92.5,
        'memory_percent': 78.3,
        'disk_percent': 65.0
    }
    
    alerts = alert_system.check_all_metrics(test_metrics)
    
    print(f"\nGenerated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"- {alert.message}")
