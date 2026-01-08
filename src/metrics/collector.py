"""
System Metrics Collector
-------------------------
This module collects real-time system metrics using psutil.
It tracks CPU usage, memory usage, disk usage, and network statistics.
"""

import psutil
import time
from datetime import datetime
from typing import Dict, List
import pandas as pd


class SystemMetricsCollector:
    """
    Collects and manages system metrics including CPU, memory, disk, and network.
    
    This class provides methods to gather real-time system information and 
    maintains a history of metrics for analysis and visualization.
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize the metrics collector.
        
        Args:
            history_size: Maximum number of historical data points to keep
        """
        self.history_size = history_size
        self.metrics_history: List[Dict] = []
        
    def collect_current_metrics(self) -> Dict:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary containing current system metrics including CPU, memory, 
            disk, and network statistics with timestamp
        """
        # Get CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Get memory metrics
        memory = psutil.virtual_memory()
        
        # Get disk metrics
        disk = psutil.disk_usage('/')
        
        # Get network metrics
        network = psutil.net_io_counters()
        
        # Compile all metrics
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),  # Convert to GB
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'network_bytes_sent_mb': network.bytes_sent / (1024**2),  # Convert to MB
            'network_bytes_recv_mb': network.bytes_recv / (1024**2),
        }
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
            
        return metrics
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get historical metrics as a pandas DataFrame.
        
        Returns:
            DataFrame containing all historical metrics
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def get_latest_metrics(self) -> Dict:
        """
        Get the most recent metrics.
        
        Returns:
            Dictionary of latest metrics or empty dict if no data
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}
    
    def get_average_metrics(self, last_n: int = 10) -> Dict:
        """
        Calculate average metrics over the last N readings.
        
        Args:
            last_n: Number of recent readings to average
            
        Returns:
            Dictionary containing average values for key metrics
        """
        if not self.metrics_history:
            return {}
        
        recent_data = self.metrics_history[-last_n:]
        df = pd.DataFrame(recent_data)
        
        averages = {
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'avg_memory_percent': df['memory_percent'].mean(),
            'avg_disk_percent': df['disk_percent'].mean(),
        }
        
        return averages
    
    def get_system_info(self) -> Dict:
        """
        Get static system information.
        
        Returns:
            Dictionary containing system hardware and OS information
        """
        import platform
        
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        return info


if __name__ == "__main__":
    # Test the metrics collector
    print("Testing System Metrics Collector...")
    collector = SystemMetricsCollector()
    
    # Collect metrics a few times
    for i in range(3):
        metrics = collector.collect_current_metrics()
        print(f"\nIteration {i+1}:")
        print(f"CPU: {metrics['cpu_percent']:.1f}%")
        print(f"Memory: {metrics['memory_percent']:.1f}%")
        print(f"Disk: {metrics['disk_percent']:.1f}%")
        time.sleep(2)
    
    # Show system info
    print("\nSystem Information:")
    for key, value in collector.get_system_info().items():
        print(f"{key}: {value}")
