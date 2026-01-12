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
        
    def get_top_processes(self, n: int = 5, sort_by: str = 'cpu_percent') -> List[Dict]:
        """
        Get the top N processes by resource usage.
        
        Args:
            n: Number of processes to return
            sort_by: Metric to sort by ('cpu_percent' or 'memory_percent')
            
        Returns:
            List of dictionaries containing process information
        """
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                # Need to call cpu_percent() at least once to get a non-zero value
                # but we'll use the one from proc.info if available
                pinfo = proc.info
                processes.append({
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'cpu_percent': pinfo['cpu_percent'],
                    'memory_percent': pinfo['memory_percent']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sort and return top N
        sorted_procs = sorted(processes, key=lambda x: x[sort_by], reverse=True)
        return sorted_procs[:n]

    def collect_current_metrics(self, top_n_procs: int = 5) -> Dict:
        """
        Collect current system metrics including top processes.
        
        Args:
            top_n_procs: Number of top resource-consuming processes to collect
            
        Returns:
            Dictionary containing current system metrics and top processes
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
        
        # Get top processes
        top_cpu_procs = self.get_top_processes(n=top_n_procs, sort_by='cpu_percent')
        top_mem_procs = self.get_top_processes(n=top_n_procs, sort_by='memory_percent')
        
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
            'top_cpu_processes': top_cpu_procs,
            'top_memory_processes': top_mem_procs
        }
        
        # Add to history (remove heavy process data from history to save memory if needed)
        # For now, keep it for replay features
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
    
    def get_storage_analysis(self, path: str = ".", n_files: int = 10, depth: int = 1) -> Dict:
        """
        Analyze storage usage in the specified path.
        
        Args:
            path: Directory path to analyze
            n_files: Number of top large files to return
            depth: How many levels deep to scan
            
        Returns:
            Dictionary containing disk partitions and largest files
        """
        import os
        
        # 1. Get partition info
        partitions = []
        for part in psutil.disk_partitions():
            try:
                # Check if drive is ready (important for Windows)
                if 'cdrom' in part.opts or part.fstype == '':
                    continue
                usage = psutil.disk_usage(part.mountpoint)
                partitions.append({
                    'device': part.device,
                    'mountpoint': part.mountpoint,
                    'fstype': part.fstype,
                    'total_gb': usage.total / (1024**3),
                    'used_gb': usage.used / (1024**3),
                    'free_gb': usage.free / (1024**3),
                    'percent': usage.percent
                })
            except (PermissionError, OSError):
                continue
                
        # 2. Get largest files in specified path
        large_files = []
        try:
            target_path = os.path.abspath(path)
            # Use a faster scan approach
            for root, dirs, files in os.walk(target_path):
                # Calculate current depth
                rel_path = os.path.relpath(root, target_path)
                current_depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
                
                if current_depth > depth:
                    del dirs[:] # Don't go deeper
                    continue
                    
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fp)
                        large_files.append({
                            'name': f,
                            'path': fp,
                            'size_mb': size / (1024**2)
                        })
                    except (OSError, PermissionError):
                        continue
        except Exception as e:
            print(f"Storage scan error: {e}")
            
        # Sort and take top N
        large_files = sorted(large_files, key=lambda x: x['size_mb'], reverse=True)[:n_files]
        
        return {
            'partitions': partitions,
            'large_files': large_files
        }

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
