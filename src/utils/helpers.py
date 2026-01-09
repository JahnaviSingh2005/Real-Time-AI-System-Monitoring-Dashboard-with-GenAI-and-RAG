"""
Utility Functions
-----------------
Helper functions for visualization, data formatting, and common operations.
"""

import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List


def create_gauge_chart(value: float, title: str, max_value: float = 100,
                       threshold_yellow: float = 70, threshold_red: float = 85) -> go.Figure:
    """
    Create a gauge chart for displaying metric values.
    
    Args:
        value: Current value to display
        title: Chart title
        max_value: Maximum value for the gauge
        threshold_yellow: Warning threshold
        threshold_red: Critical threshold
        
    Returns:
        Plotly figure object
    """
    # Determine color based on thresholds
    if value >= threshold_red:
        color = "red"
    elif value >= threshold_yellow:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': threshold_yellow},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold_yellow], 'color': 'lightgray'},
                {'range': [threshold_yellow, threshold_red], 'color': 'lightyellow'},
                {'range': [threshold_red, max_value], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_red
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_time_series_chart(df: pd.DataFrame, metric: str, 
                             title: str, color: str = "blue") -> go.Figure:
    """
    Create a time series line chart for metric history.
    
    Args:
        df: DataFrame with 'timestamp' and metric columns
        metric: Name of metric column to plot
        title: Chart title
        color: Line color
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if not df.empty and metric in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Usage (%)",
        hovermode='x unified',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_multi_metric_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a multi-line chart showing all metrics together.
    
    Args:
        df: DataFrame with timestamp and metric columns
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if not df.empty:
        # Add CPU line
        if 'cpu_percent' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cpu_percent'],
                mode='lines',
                name='CPU',
                line=dict(color='#FF6B6B', width=2)
            ))
        
        # Add Memory line
        if 'memory_percent' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['memory_percent'],
                mode='lines',
                name='Memory',
                line=dict(color='#4ECDC4', width=2)
            ))
        
        # Add Disk line
        if 'disk_percent' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['disk_percent'],
                mode='lines',
                name='Disk',
                line=dict(color='#95E1D3', width=2)
            ))
    
    fig.update_layout(
        title='All Metrics Over Time',
        xaxis_title='Time',
        yaxis_title='Usage (%)',
        hovermode='x unified',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def format_alert_message(alert: Dict) -> str:
    """
    Format alert dictionary into human-readable message with emoji.
    
    Args:
        alert: Alert dictionary
        
    Returns:
        Formatted string
    """
    severity = alert.get('severity', 'unknown')
    
    emoji_map = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    }
    
    emoji = emoji_map.get(severity, '⚪')
    message = alert.get('message', 'No message')
    timestamp = alert.get('timestamp', datetime.now())
    
    if isinstance(timestamp, datetime):
        time_str = timestamp.strftime('%H:%M:%S')
    else:
        time_str = str(timestamp)
    
    return f"{emoji} **{severity.upper()}** [{time_str}]: {message}"


def get_health_status(cpu: float, memory: float, disk: float, 
                      cpu_threshold: float = 85.0, mem_threshold: float = 85.0) -> Dict:
    """
    Calculate overall system health status.
    
    Args:
        cpu: CPU usage percentage
        memory: Memory usage percentage
        disk: Disk usage percentage
        cpu_threshold: Custom CPU threshold for warning
        mem_threshold: Custom Memory threshold for warning
        
    Returns:
        Dictionary with status, score, and message
    """
    # Calculate health score (0-100, where 100 is perfect)
    # Penalize more if above threshold
    cpu_penalty = cpu if cpu < cpu_threshold else cpu + (cpu - cpu_threshold) * 2
    mem_penalty = memory if memory < mem_threshold else memory + (memory - mem_threshold) * 2
    
    cpu_score = max(0, 100 - (cpu_penalty / 100 * 100))
    memory_score = max(0, 100 - (mem_penalty / 100 * 100))
    disk_score = max(0, 100 - disk)
    
    overall_score = (cpu_score + memory_score + disk_score) / 3
    
    # Determine status
    if cpu >= cpu_threshold or memory >= mem_threshold:
        status = "Critical"
        emoji = "🔴"
        color = "red"
        message = "System resource limits exceeded!"
    elif overall_score >= 70:
        status = "Healthy"
        emoji = "✅"
        color = "green"
        message = "System is running smoothly"
    elif overall_score >= 50:
        status = "Warning"
        emoji = "⚠️"
        color = "orange"
        message = "System is under moderate load"
    else:
        status = "Degraded"
        emoji = "🟠"
        color = "orange"
        message = "System performance is degraded"
    
    return {
        'status': status,
        'score': round(overall_score, 1),
        'emoji': emoji,
        'color': color,
        'message': message
    }


def format_bytes(bytes_value: float) -> str:
    """
    Format bytes into human-readable format.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime for display.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted string
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def calculate_trend(values: List[float], window: int = 5) -> str:
    """
    Calculate trend direction from recent values.
    
    Args:
        values: List of numeric values
        window: Number of recent values to consider
        
    Returns:
        Trend string: "increasing", "decreasing", or "stable"
    """
    if len(values) < 2:
        return "stable"
    
    recent = values[-window:]
    
    if len(recent) < 2:
        return "stable"
    
    # Calculate simple trend
    first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
    second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
    
    diff = second_half - first_half
    
    if abs(diff) < 2:  # Threshold for "stable"
        return "stable"
    elif diff > 0:
        return "increasing"
    else:
        return "decreasing"


def get_trend_emoji(trend: str) -> str:
    """Get emoji for trend direction"""
    trend_map = {
        'increasing': '📈',
        'decreasing': '📉',
        'stable': '➡️'
    }
    return trend_map.get(trend, '➡️')


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test health status
    health = get_health_status(cpu=65.0, memory=70.0, disk=55.0)
    print(f"\nHealth Status: {health['emoji']} {health['status']}")
    print(f"Score: {health['score']}/100")
    print(f"Message: {health['message']}")
    
    # Test trend
    test_values = [50, 52, 55, 60, 65, 68]
    trend = calculate_trend(test_values)
    print(f"\nTrend: {get_trend_emoji(trend)} {trend}")
    
    # Test byte formatting
    print(f"\nFormatted: {format_bytes(1536000000)}")
