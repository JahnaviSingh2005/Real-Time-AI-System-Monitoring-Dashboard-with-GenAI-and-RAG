"""
RAG (Retrieval-Augmented Generation) System
--------------------------------------------
This module implements RAG using ChromaDB for vector storage and retrieval.
It stores past incidents and retrieves similar ones for context.
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class RAGSystem:
    """
    RAG system for retrieving similar past incidents.
    
    Uses ChromaDB as vector database and sentence transformers for embeddings.
    Helps provide context by finding similar historical incidents.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize RAG system.
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory
        self.collection_name = "system_incidents"
        
        # Initialize ChromaDB client (persistent storage)
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "System monitoring incidents"}
            )
            print(f"Created new collection: {self.collection_name}")
        
        # Initialize embedding model (this is small and CPU-friendly)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
        
        self.is_initialized = True
    
    def load_incidents_from_json(self, json_path: str) -> bool:
        """
        Load incidents from JSON file into the vector database.
        
        Args:
            json_path: Path to JSON file containing incidents
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(json_path):
            print(f"JSON file not found: {json_path}")
            return False
        
        try:
            with open(json_path, 'r') as f:
                incidents = json.load(f)
            
            # Add incidents to collection
            for idx, incident in enumerate(incidents):
                self.add_incident(
                    description=incident['description'],
                    resolution=incident['resolution'],
                    metadata={
                        'timestamp': incident['timestamp'],
                        'metric': incident['metric'],
                        'value': incident['value'],
                        'severity': incident['severity']
                    },
                    incident_id=f"incident_{idx}"
                )
            
            print(f"Loaded {len(incidents)} incidents into vector database")
            return True
            
        except Exception as e:
            print(f"Error loading incidents: {e}")
            return False
    
    def add_incident(self, description: str, resolution: str, 
                     metadata: Dict, incident_id: Optional[str] = None):
        """
        Add a new incident to the vector database.
        
        Args:
            description: Description of the incident
            resolution: How the incident was resolved
            metadata: Additional metadata (timestamp, metric, value, etc.)
            incident_id: Unique ID for the incident (auto-generated if not provided)
        """
        # Create incident ID if not provided
        if incident_id is None:
            incident_id = f"incident_{datetime.now().timestamp()}"
        
        # Combine description and resolution for embedding
        full_text = f"{description} Resolution: {resolution}"
        
        # Add to collection
        self.collection.add(
            documents=[full_text],
            metadatas=[metadata],
            ids=[incident_id]
        )
    
    def search_similar_incidents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search for similar past incidents.
        
        Args:
            query: Query text describing the current issue
            n_results: Number of similar incidents to return
            
        Returns:
            List of similar incidents with their details
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            similar_incidents = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    incident = {
                        'description': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    similar_incidents.append(incident)
            
            return similar_incidents
            
        except Exception as e:
            print(f"Error searching incidents: {e}")
            return []
    
    def find_incidents_by_metric(self, metric: str, n_results: int = 5) -> List[Dict]:
        """
        Find incidents related to a specific metric.
        
        Args:
            metric: Metric name (e.g., 'cpu_usage', 'memory_usage')
            n_results: Number of incidents to return
            
        Returns:
            List of incidents for that metric
        """
        try:
            results = self.collection.get(
                where={"metric": metric},
                limit=n_results
            )
            
            incidents = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    incident = {
                        'description': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    incidents.append(incident)
            
            return incidents
            
        except Exception as e:
            print(f"Error finding incidents by metric: {e}")
            return []
    
    def get_context_for_anomaly(self, current_metrics: Dict, 
                               alerts: List = None) -> str:
        """
        Get relevant context from past incidents for current anomaly.
        
        Args:
            current_metrics: Current system metrics
            alerts: List of current alerts
            
        Returns:
            Formatted context string with similar past incidents
        """
        # Build query from current situation
        query_parts = []
        
        cpu = current_metrics.get('cpu_percent', 0)
        memory = current_metrics.get('memory_percent', 0)
        disk = current_metrics.get('disk_percent', 0)
        
        if cpu > 70:
            query_parts.append(f"high CPU usage {cpu}%")
        if memory > 70:
            query_parts.append(f"high memory usage {memory}%")
        if disk > 80:
            query_parts.append(f"high disk usage {disk}%")
        
        if not query_parts:
            query_parts.append("system performance issue")
        
        query = " ".join(query_parts)
        
        # Search for similar incidents
        similar = self.search_similar_incidents(query, n_results=3)
        
        if not similar:
            return "No similar past incidents found."
        
        # Format context
        context_parts = []
        context_parts.append("📚 **Similar Past Incidents:**\n")
        
        for i, incident in enumerate(similar, 1):
            metadata = incident['metadata']
            context_parts.append(f"**Incident {i}** ({metadata.get('timestamp', 'Unknown time')})")
            context_parts.append(f"- Severity: {metadata.get('severity', 'unknown')}")
            context_parts.append(f"- Metric: {metadata.get('metric', 'unknown')} at {metadata.get('value', 0)}%")
            
            # Extract description and resolution
            full_text = incident['description']
            if 'Resolution:' in full_text:
                desc, res = full_text.split('Resolution:', 1)
                context_parts.append(f"- Issue: {desc.strip()[:150]}...")
                context_parts.append(f"- Resolution: {res.strip()[:150]}...")
            else:
                context_parts.append(f"- Details: {full_text[:200]}...")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the incident database"""
        try:
            count = self.collection.count()
            return {
                'total_incidents': count,
                'collection_name': self.collection_name
            }
        except:
            return {'total_incidents': 0, 'collection_name': self.collection_name}


if __name__ == "__main__":
    # Test the RAG system
    print("Testing RAG System...")
    
    # Initialize
    rag = RAGSystem(persist_directory="./test_chroma_db")
    
    # Load incidents from JSON
    rag.load_incidents_from_json("../data/incidents.json")
    
    # Test search
    query = "high CPU usage causing system slowdown"
    print(f"\nSearching for: {query}")
    
    results = rag.search_similar_incidents(query, n_results=2)
    print(f"\nFound {len(results)} similar incidents")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['description'][:100]}...")
        print(f"   Metadata: {result['metadata']}")
