"""
ML-Based Anomaly Detection
---------------------------
This module implements anomaly detection using Isolation Forest algorithm.
It learns normal system behavior and identifies unusual patterns.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple
import pickle
import os


class MLAnomalyDetector:
    """
    Machine Learning-based anomaly detector using Isolation Forest.
    
    Isolation Forest is an unsupervised learning algorithm that identifies
    anomalies by isolating outliers in the feature space. It's particularly
    effective for detecting unusual system behavior.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the ML anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (default 0.1 = 10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_trained = False
        self.feature_columns = ['cpu_percent', 'memory_percent', 'disk_percent']
        self.anomaly_history: List[Dict] = []
    
    def prepare_features(self, metrics_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from metrics data.
        
        Args:
            metrics_data: DataFrame containing system metrics
            
        Returns:
            Numpy array of features for the model
        """
        # Select relevant features
        features = metrics_data[self.feature_columns].values
        return features
    
    def train(self, metrics_data: pd.DataFrame) -> bool:
        """
        Train the anomaly detection model.
        
        Args:
            metrics_data: Historical metrics data as DataFrame
            
        Returns:
            True if training successful, False otherwise
        """
        if len(metrics_data) < 10:
            print("Not enough data to train. Need at least 10 samples.")
            return False
        
        try:
            # Prepare features
            X = self.prepare_features(metrics_data)
            
            # Train the model
            self.model.fit(X)
            self.is_trained = True
            
            print(f"Model trained on {len(X)} samples")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict(self, metrics: Dict) -> Tuple[bool, float]:
        """
        Predict if current metrics are anomalous.
        
        Args:
            metrics: Dictionary containing current system metrics
            
        Returns:
            Tuple of (is_anomaly: bool, anomaly_score: float)
            is_anomaly is True if anomaly detected
            anomaly_score is negative for anomalies (more negative = more anomalous)
        """
        if not self.is_trained:
            return False, 0.0
        
        try:
            # Prepare single sample
            sample = np.array([[
                metrics.get('cpu_percent', 0),
                metrics.get('memory_percent', 0),
                metrics.get('disk_percent', 0)
            ]])
            
            # Predict (-1 for anomaly, 1 for normal)
            prediction = self.model.predict(sample)[0]
            
            # Get anomaly score (lower = more anomalous)
            score = self.model.score_samples(sample)[0]
            
            is_anomaly = prediction == -1
            
            # Record anomaly
            if is_anomaly:
                anomaly_record = {
                    'timestamp': metrics.get('timestamp'),
                    'cpu_percent': metrics.get('cpu_percent'),
                    'memory_percent': metrics.get('memory_percent'),
                    'disk_percent': metrics.get('disk_percent'),
                    'anomaly_score': score
                }
                self.anomaly_history.append(anomaly_record)
            
            return is_anomaly, score
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return False, 0.0
    
    def get_anomaly_probability(self, metrics: Dict) -> float:
        """
        Get probability/confidence that metrics are anomalous.
        
        Args:
            metrics: Dictionary containing current system metrics
            
        Returns:
            Probability between 0 and 1 (1 = definitely anomalous)
        """
        if not self.is_trained:
            return 0.0
        
        is_anomaly, score = self.predict(metrics)
        
        # Convert score to probability (score ranges roughly from -0.5 to 0.5)
        # More negative = more anomalous
        probability = max(0, min(1, -score))
        
        return probability
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            print("Cannot save untrained model")
            return
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'contamination': self.contamination
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if loading successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.contamination = model_data['contamination']
            self.is_trained = True
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_recent_anomalies(self, count: int = 10) -> List[Dict]:
        """
        Get recent detected anomalies.
        
        Args:
            count: Number of recent anomalies to return
            
        Returns:
            List of anomaly records
        """
        return self.anomaly_history[-count:] if self.anomaly_history else []


if __name__ == "__main__":
    # Test the ML anomaly detector
    print("Testing ML Anomaly Detector...")
    
    # Create synthetic training data (normal behavior)
    np.random.seed(42)
    normal_data = {
        'cpu_percent': np.random.normal(50, 10, 50),
        'memory_percent': np.random.normal(60, 8, 50),
        'disk_percent': np.random.normal(70, 5, 50),
    }
    df_train = pd.DataFrame(normal_data)
    
    # Initialize and train
    detector = MLAnomalyDetector()
    detector.train(df_train)
    
    # Test with normal sample
    normal_sample = {
        'cpu_percent': 55.0,
        'memory_percent': 62.0,
        'disk_percent': 68.0
    }
    is_anomaly, score = detector.predict(normal_sample)
    print(f"\nNormal sample: Anomaly={is_anomaly}, Score={score:.3f}")
    
    # Test with anomalous sample
    anomalous_sample = {
        'cpu_percent': 95.0,
        'memory_percent': 98.0,
        'disk_percent': 92.0
    }
    is_anomaly, score = detector.predict(anomalous_sample)
    print(f"Anomalous sample: Anomaly={is_anomaly}, Score={score:.3f}")
