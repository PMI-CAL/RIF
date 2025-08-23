"""
Utility functions for RIF Agent LightRAG integration.
Provides common patterns and helpers for agent knowledge management.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

from agent_integration import create_agent_rag, RIFAgentRAG


class AgentSession:
    """
    Manages a knowledge session for an agent, tracking all knowledge operations.
    """
    
    def __init__(self, agent_type: str, session_id: str = None):
        """
        Initialize agent session.
        
        Args:
            agent_type: Type of RIF agent
            session_id: Optional session identifier
        """
        self.agent_type = agent_type
        self.session_id = session_id or f"{agent_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.agent_rag = create_agent_rag(agent_type)
        self.agent_rag.session_id = self.session_id
        
        self.start_time = datetime.utcnow()
        self.knowledge_captured = []
        self.queries_performed = []
        
        self.logger = logging.getLogger(f"rif.session.{agent_type}")
        self.logger.info(f"Started session {self.session_id}")
    
    def query_knowledge(self, query: str, collection: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query knowledge and track the operation.
        
        Args:
            query: Search query
            collection: Specific collection to search
            limit: Maximum results
            
        Returns:
            Query results
        """
        start_time = datetime.utcnow()
        
        if collection:
            results = self.agent_rag.rag.retrieve_knowledge(query, collection, limit)
        else:
            results = self.agent_rag.query_similar_work(query, limit)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Track query
        self.queries_performed.append({
            "timestamp": start_time.isoformat(),
            "query": query,
            "collection": collection,
            "results_count": len(results),
            "duration_seconds": duration
        })
        
        self.logger.info(f"Query completed: {len(results)} results in {duration:.2f}s")
        return results
    
    def capture_knowledge(self, content: str, knowledge_type: str, metadata: Dict[str, Any]) -> str:
        """
        Capture knowledge and track the operation.
        
        Args:
            content: Knowledge content
            knowledge_type: Type of knowledge
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = self.agent_rag.capture_knowledge(content, knowledge_type, metadata)
        
        # Track capture
        self.knowledge_captured.append({
            "timestamp": datetime.utcnow().isoformat(),
            "doc_id": doc_id,
            "knowledge_type": knowledge_type,
            "content_length": len(content)
        })
        
        self.logger.info(f"Captured knowledge: {doc_id} ({knowledge_type})")
        return doc_id
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session activity."""
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "duration_seconds": duration,
            "knowledge_captured": len(self.knowledge_captured),
            "queries_performed": len(self.queries_performed),
            "start_time": self.start_time.isoformat(),
            "captured_items": self.knowledge_captured,
            "query_history": self.queries_performed
        }
    
    def end_session(self) -> Dict[str, Any]:
        """End session and return summary."""
        summary = self.get_session_summary()
        self.logger.info(f"Ended session {self.session_id} - Duration: {summary['duration_seconds']:.1f}s")
        return summary


@contextmanager
def agent_knowledge_session(agent_type: str, session_id: str = None):
    """
    Context manager for agent knowledge sessions.
    
    Usage:
        with agent_knowledge_session('analyst') as session:
            results = session.query_knowledge("similar issues")
            session.capture_knowledge("pattern found", "pattern", {...})
    """
    session = AgentSession(agent_type, session_id)
    try:
        yield session
    finally:
        session.end_session()


class KnowledgePatternMatcher:
    """
    Advanced pattern matching for knowledge discovery.
    """
    
    def __init__(self, agent_rag: RIFAgentRAG):
        self.agent_rag = agent_rag
    
    def find_evolution_patterns(self, entity: str, time_window_days: int = 90) -> List[Dict[str, Any]]:
        """
        Find how knowledge about an entity has evolved over time.
        
        Args:
            entity: Entity to track (component, pattern, etc.)
            time_window_days: Days to look back
            
        Returns:
            Chronological list of knowledge evolution
        """
        # Calculate time filter
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        
        # Query for entity across collections
        results = self.agent_rag.query_similar_work(entity)
        
        # Filter and sort by timestamp
        filtered_results = []
        for result in results:
            metadata = result.get('metadata', {})
            timestamp_str = metadata.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp >= cutoff_date:
                        result['parsed_timestamp'] = timestamp
                        filtered_results.append(result)
                except ValueError:
                    continue  # Skip items with invalid timestamps
        
        # Sort by timestamp
        filtered_results.sort(key=lambda x: x['parsed_timestamp'])
        
        return filtered_results
    
    def find_success_patterns(self, domain: str, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find patterns associated with successful outcomes.
        
        Args:
            domain: Domain to search (testing, architecture, etc.)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of successful patterns
        """
        results = self.agent_rag.query_similar_work(domain)
        
        successful_patterns = []
        for result in results:
            metadata = result.get('metadata', {})
            confidence = 1.0 - result.get('distance', 1.0)  # Convert distance to confidence
            
            # Check for success indicators
            success_indicators = [
                metadata.get('success', False),
                metadata.get('resolution_success', False),
                metadata.get('status') == 'completed',
                'success' in result.get('content', '').lower()
            ]
            
            if any(success_indicators) and confidence >= min_confidence:
                result['computed_confidence'] = confidence
                successful_patterns.append(result)
        
        # Sort by confidence
        successful_patterns.sort(key=lambda x: x['computed_confidence'], reverse=True)
        
        return successful_patterns
    
    def find_anti_patterns(self, domain: str) -> List[Dict[str, Any]]:
        """
        Find anti-patterns (things to avoid) in a domain.
        
        Args:
            domain: Domain to search
            
        Returns:
            List of anti-patterns
        """
        results = self.agent_rag.query_similar_work(f"{domain} failure problems issues")
        
        anti_patterns = []
        for result in results:
            metadata = result.get('metadata', {})
            content = result.get('content', '').lower()
            
            # Check for failure indicators
            failure_indicators = [
                metadata.get('failure', False),
                metadata.get('status') == 'failed',
                'failed' in content,
                'error' in content,
                'problem' in content,
                'issue' in content
            ]
            
            if any(failure_indicators):
                anti_patterns.append(result)
        
        return anti_patterns


class AgentKnowledgeAnalyzer:
    """
    Analyzes knowledge usage and effectiveness across agents.
    """
    
    def __init__(self):
        self.agents = {
            name: create_agent_rag(name) 
            for name in ['analyst', 'architect', 'implementer', 'validator', 'planner']
        }
    
    def analyze_knowledge_coverage(self) -> Dict[str, Any]:
        """Analyze knowledge coverage across all agents."""
        coverage = {
            "total_collections": 0,
            "agent_coverage": {},
            "knowledge_gaps": [],
            "overlap_analysis": {}
        }
        
        # Get collection stats
        sample_agent = list(self.agents.values())[0]
        collection_stats = sample_agent.get_agent_stats()
        coverage["total_collections"] = len(collection_stats)
        
        # Analyze each agent's coverage
        for agent_name, agent_rag in self.agents.items():
            stats = agent_rag.get_agent_stats()
            coverage["agent_coverage"][agent_name] = {
                "collections_used": len([c for c, s in stats.items() if s.get('total_items', 0) > 0]),
                "total_contributions": sum(s.get('total_items', 0) for s in stats.values()),
                "collection_breakdown": stats
            }
        
        return coverage
    
    def suggest_knowledge_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements to knowledge base usage."""
        suggestions = []
        
        coverage = self.analyze_knowledge_coverage()
        
        # Identify agents with low knowledge usage
        for agent_name, stats in coverage["agent_coverage"].items():
            if stats["total_contributions"] == 0:
                suggestions.append({
                    "type": "low_usage",
                    "agent": agent_name,
                    "suggestion": f"Agent {agent_name} is not capturing knowledge - implement capture hooks",
                    "priority": "high"
                })
            elif stats["collections_used"] < 2:
                suggestions.append({
                    "type": "limited_scope",
                    "agent": agent_name,
                    "suggestion": f"Agent {agent_name} only uses {stats['collections_used']} collections - expand scope",
                    "priority": "medium"
                })
        
        return suggestions


# Utility functions for common agent knowledge operations
def bulk_migrate_knowledge(source_files: List[str], agent_type: str, knowledge_type: str) -> List[str]:
    """
    Bulk migrate knowledge from files into agent's knowledge base.
    
    Args:
        source_files: List of file paths to migrate
        agent_type: Target agent type
        knowledge_type: Type of knowledge being migrated
        
    Returns:
        List of document IDs created
    """
    agent_rag = create_agent_rag(agent_type)
    doc_ids = []
    
    for file_path in source_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            metadata = {
                "source_file": file_path,
                "migrated_from": "bulk_migration",
                "file_size": len(content)
            }
            
            doc_id = agent_rag.capture_knowledge(content, knowledge_type, metadata)
            doc_ids.append(doc_id)
            
        except Exception as e:
            logging.error(f"Failed to migrate {file_path}: {e}")
            continue
    
    return doc_ids


def cross_agent_knowledge_search(query: str, agent_types: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search knowledge across multiple agent types.
    
    Args:
        query: Search query
        agent_types: List of agent types to search (None for all)
        
    Returns:
        Dictionary mapping agent types to their results
    """
    if agent_types is None:
        agent_types = ['analyst', 'architect', 'implementer', 'validator', 'planner']
    
    results = {}
    
    for agent_type in agent_types:
        try:
            agent_rag = create_agent_rag(agent_type)
            agent_results = agent_rag.query_similar_work(query)
            results[agent_type] = agent_results
        except Exception as e:
            logging.error(f"Failed to search {agent_type}: {e}")
            results[agent_type] = []
    
    return results


def export_agent_knowledge(agent_type: str, output_dir: str) -> Dict[str, str]:
    """
    Export all knowledge for an agent type.
    
    Args:
        agent_type: Type of agent
        output_dir: Directory to export to
        
    Returns:
        Dictionary mapping collection names to export file paths
    """
    import os
    
    agent_rag = create_agent_rag(agent_type)
    export_paths = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for collection_name in agent_rag.rag.collections.keys():
        output_file = os.path.join(output_dir, f"{agent_type}_{collection_name}.json")
        
        if agent_rag.rag.export_collection(collection_name, output_file):
            export_paths[collection_name] = output_file
    
    return export_paths