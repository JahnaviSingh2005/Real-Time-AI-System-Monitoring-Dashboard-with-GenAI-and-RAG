"""
Incident Persistence & Storage
-------------------------------
This module handles saving and retrieving system metrics and alerts 
to/from a local SQLite database for incident replay and reporting.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class IncidentStorage:
    """
    Handles SQLite persistence for system monitoring data.
    """
    
    def __init__(self, db_path: str = "data/monitoring_history.db"):
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table for system metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_percent REAL,
                    json_data TEXT
                )
            ''')
            
            # Table for alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric TEXT,
                    value REAL,
                    severity TEXT,
                    message TEXT
                )
            ''')
            
            # Table for incidents
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    summary TEXT,
                    report_path TEXT,
                    json_data TEXT
                )
            ''')
            
            conn.commit()
            
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save a snapshot of metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO metrics (timestamp, cpu_percent, memory_percent, disk_percent, json_data) VALUES (?, ?, ?, ?, ?)",
                (
                    metrics.get('timestamp', datetime.now()).isoformat(),
                    metrics.get('cpu_percent'),
                    metrics.get('memory_percent'),
                    metrics.get('disk_percent'),
                    json.dumps(metrics, default=str)
                )
            )
            conn.commit()
            
    def save_alert(self, alert: Dict[str, Any]):
        """Save a single alert"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO alerts (timestamp, metric, value, severity, message) VALUES (?, ?, ?, ?, ?)",
                (
                    alert.get('timestamp', datetime.now()).isoformat(),
                    alert.get('metric'),
                    alert.get('value'),
                    alert.get('severity'),
                    alert.get('message')
                )
            )
            conn.commit()

    def get_recent_metrics(self, limit: int = 100) -> List[Dict]:
        """Retrieve recent metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def create_incident(self, start_time: datetime, end_time: datetime, summary: str, alerts: List[Dict]):
        """Group data into an incident record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO incidents (start_time, end_time, summary, json_data) VALUES (?, ?, ?, ?)",
                (
                    start_time.isoformat(),
                    end_time.isoformat(),
                    summary,
                    json.dumps(alerts, default=str)
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_incidents(self) -> List[Dict]:
        """Get all stored incidents"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents ORDER BY start_time DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_metrics_for_period(self, start_time: str, end_time: str) -> List[Dict]:
        """Get metrics within a time range for replay"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM metrics WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC",
                (start_time, end_time)
            )
            return [dict(row) for row in cursor.fetchall()]
