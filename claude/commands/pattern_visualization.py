"""
Pattern Visualization System for RIF Framework

This module implements comprehensive pattern visualization capabilities that provide
graphical insights into pattern relationships, success metrics, and clustering analysis.
Designed to work with the Pattern Reinforcement System (issue #78) to provide
visual analytics for pattern performance and interconnections.

Key Features:
- Interactive pattern relationship graphs with node sizing by usage frequency
- Success rate visualization with color-coded metrics
- Automatic pattern clustering based on similarity and usage patterns
- Export capabilities for reports and documentation
- Integration with existing RIF knowledge system

Implementation follows the design from Issue #79 requirements.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Visualization and analysis libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Try to import plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import pattern reinforcement system
try:
    from .pattern_reinforcement_system import PatternReinforcementSystem, PatternMetrics, OutcomeType
    PATTERN_SYSTEM_AVAILABLE = True
except ImportError:
    PATTERN_SYSTEM_AVAILABLE = False

# Import knowledge interface
try:
    from knowledge.interface import get_knowledge_system, KnowledgeInterface
    KNOWLEDGE_INTERFACE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_INTERFACE_AVAILABLE = False


class VisualizationType(Enum):
    """Types of pattern visualizations available."""
    NETWORK_GRAPH = "network_graph"
    SUCCESS_HEATMAP = "success_heatmap"
    CLUSTER_ANALYSIS = "cluster_analysis"
    TIMELINE_ANALYSIS = "timeline_analysis"
    PERFORMANCE_DASHBOARD = "performance_dashboard"


class ClusteringMethod(Enum):
    """Clustering algorithms for pattern analysis."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    SIMILARITY_BASED = "similarity_based"


@dataclass
class PatternNode:
    """
    Represents a pattern in the visualization graph.
    
    Contains all the metadata needed for visualization including
    positioning, styling, and interactive information.
    """
    id: str
    name: str
    usage_count: int = 0
    success_rate: float = 0.0
    complexity: str = "medium"
    last_used: Optional[datetime] = None
    
    # Visual properties
    size: int = 10
    color: str = "#3498db"
    position: Optional[Tuple[float, float]] = None
    
    # Metadata for tooltips and interaction
    description: Optional[str] = None
    domain: Optional[str] = None
    source_issue: Optional[str] = None
    related_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternNode':
        """Create from dictionary."""
        data = data.copy()
        if data.get('last_used'):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class PatternEdge:
    """
    Represents a relationship between patterns in the graph.
    
    Captures the strength and nature of relationships between
    patterns for visualization purposes.
    """
    source: str
    target: str
    weight: float = 0.0
    relationship_type: str = "similarity"
    
    # Metadata
    confidence: float = 1.0
    evidence_count: int = 0
    last_reinforced: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.last_reinforced:
            data['last_reinforced'] = self.last_reinforced.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternEdge':
        """Create from dictionary."""
        data = data.copy()
        if data.get('last_reinforced'):
            data['last_reinforced'] = datetime.fromisoformat(data['last_reinforced'])
        return cls(**data)


@dataclass
class PatternCluster:
    """
    Represents a cluster of related patterns.
    
    Groups patterns that are similar in usage, success patterns,
    or domain characteristics.
    """
    cluster_id: int
    pattern_ids: List[str]
    centroid: Optional[List[float]] = None
    
    # Cluster characteristics
    dominant_domain: Optional[str] = None
    average_success_rate: float = 0.0
    total_usage: int = 0
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Visual properties
    color: str = "#95a5a6"
    label: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternCluster':
        """Create from dictionary."""
        return cls(**data)


class PatternVisualization:
    """
    Core pattern visualization system for analyzing and displaying pattern relationships.
    
    This system provides comprehensive visualization capabilities for the RIF pattern
    library including:
    1. Network graphs showing pattern relationships and dependencies
    2. Success rate analysis with color-coded metrics
    3. Clustering analysis to identify pattern groups
    4. Interactive dashboards for pattern exploration
    
    Integrates with the Pattern Reinforcement System to provide visual analytics
    for pattern performance and evolution over time.
    """
    
    def __init__(self, 
                 pattern_system: Optional[PatternReinforcementSystem] = None,
                 knowledge_system: Optional[KnowledgeInterface] = None,
                 visualization_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern visualization system.
        
        Args:
            pattern_system: Pattern reinforcement system for metrics
            knowledge_system: Knowledge system for pattern data
            visualization_config: Configuration for visualization options
        """
        self.logger = logging.getLogger(f"{__name__}.PatternVisualization")
        
        # Initialize pattern system
        if pattern_system:
            self.pattern_system = pattern_system
        elif PATTERN_SYSTEM_AVAILABLE:
            try:
                self.pattern_system = PatternReinforcementSystem()
            except Exception as e:
                self.logger.error(f"Failed to initialize pattern system: {e}")
                self.pattern_system = None
        else:
            self.pattern_system = None
            self.logger.warning("Pattern reinforcement system not available")
        
        # Initialize knowledge system
        if knowledge_system:
            self.knowledge = knowledge_system
        elif KNOWLEDGE_INTERFACE_AVAILABLE:
            try:
                self.knowledge = get_knowledge_system()
            except Exception as e:
                self.logger.error(f"Failed to initialize knowledge system: {e}")
                self.knowledge = None
        else:
            self.knowledge = None
            self.logger.warning("Knowledge interface not available - using in-memory storage")
        
        # Configuration
        self.config = {
            'max_patterns_display': 100,
            'similarity_threshold': 0.3,
            'cluster_min_size': 2,
            'default_layout': 'spring',
            'color_scheme': 'viridis',
            'node_size_range': (10, 50),
            'edge_width_range': (1, 8),
            'enable_interactivity': True,
            'export_formats': ['html', 'png', 'json'],
            'cache_visualizations': True,
            'async_processing': True,
        }
        
        if visualization_config:
            self.config.update(visualization_config)
        
        # Cache for computed visualizations
        self.visualization_cache: Dict[str, Dict[str, Any]] = {}
        self.pattern_cache: Dict[str, PatternNode] = {}
        self.relationship_cache: Dict[str, List[PatternEdge]] = {}
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_lock = threading.RLock()
        
        # Performance metrics
        self.performance_metrics = {
            'visualizations_generated': 0,
            'patterns_analyzed': 0,
            'relationships_computed': 0,
            'clusters_identified': 0,
            'cache_hits': 0,
            'average_generation_time': 0.0,
            'last_analysis_run': None
        }
        
        self.logger.info("Pattern visualization system initialized successfully")
    
    def generate_pattern_graph(self, 
                              filters: Optional[Dict[str, Any]] = None,
                              layout_algorithm: str = "spring",
                              include_metrics: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive pattern graph with nodes and edges.
        
        This is the main method that implements the requirements from issue #79,
        creating a graph structure with:
        - Nodes representing patterns (sized by usage, colored by success rate)
        - Edges representing relationships (weighted by correlation)
        - Clusters identified automatically
        
        Args:
            filters: Optional filters to apply to patterns
            layout_algorithm: Algorithm for node positioning
            include_metrics: Whether to include detailed metrics
            
        Returns:
            Dictionary containing nodes, edges, clusters, and metadata
        """
        start_time = time.time()
        
        try:
            with self.processing_lock:
                self.logger.info("Generating pattern relationship graph")
                
                # Get all patterns with their metrics
                patterns = self._load_patterns_with_metrics(filters)
                if not patterns:
                    self.logger.warning("No patterns found for visualization")
                    return {'nodes': [], 'edges': [], 'clusters': []}
                
                # Create nodes
                nodes = []
                for pattern in patterns:
                    node = self._create_pattern_node(pattern)
                    if node:
                        nodes.append(node)
                
                if not nodes:
                    return {'nodes': [], 'edges': [], 'clusters': []}
                
                # Create edges based on relationships
                edges = self._compute_pattern_relationships(nodes)
                
                # Identify clusters
                clusters = self._identify_pattern_clusters(nodes, edges)
                
                # Apply layout algorithm
                positioned_nodes = self._apply_layout_algorithm(nodes, edges, layout_algorithm)
                
                # Prepare result
                result = {
                    'nodes': [node.to_dict() for node in positioned_nodes],
                    'edges': [edge.to_dict() for edge in edges],
                    'clusters': [cluster.to_dict() for cluster in clusters],
                    'metadata': {
                        'total_patterns': len(positioned_nodes),
                        'total_relationships': len(edges),
                        'total_clusters': len(clusters),
                        'layout_algorithm': layout_algorithm,
                        'generation_time': time.time() - start_time,
                        'timestamp': datetime.now().isoformat(),
                        'filters_applied': filters or {}
                    }
                }
                
                # Include detailed metrics if requested
                if include_metrics:
                    result['metrics'] = self._compute_graph_metrics(positioned_nodes, edges, clusters)
                
                # Update performance metrics
                generation_time = time.time() - start_time
                self._update_performance_metrics(generation_time, len(positioned_nodes), len(edges))
                
                # Cache result if enabled
                if self.config['cache_visualizations']:
                    cache_key = self._generate_cache_key(filters, layout_algorithm, include_metrics)
                    self.visualization_cache[cache_key] = result
                
                self.logger.info(f"Generated pattern graph: {len(positioned_nodes)} nodes, "
                               f"{len(edges)} edges, {len(clusters)} clusters in {generation_time:.2f}s")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error generating pattern graph: {e}")
            return {'nodes': [], 'edges': [], 'clusters': [], 'error': str(e)}
    
    def _load_patterns_with_metrics(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Load patterns from knowledge system with their performance metrics."""
        patterns = []
        
        try:
            # Load patterns from knowledge system
            if self.knowledge:
                # Get pattern files
                pattern_results = self.knowledge.retrieve_knowledge(
                    query="pattern",
                    collection="patterns",
                    n_results=self.config['max_patterns_display'],
                    filters=filters or {}
                )
                
                for result in pattern_results:
                    try:
                        pattern_data = json.loads(result['content'])
                        
                        # Enrich with metrics from pattern system
                        if self.pattern_system and 'pattern_id' in pattern_data:
                            metrics = self.pattern_system.get_pattern_metrics(pattern_data['pattern_id'])
                            if metrics:
                                pattern_data['metrics'] = metrics
                        
                        patterns.append(pattern_data)
                        
                    except Exception as e:
                        self.logger.warning(f"Error parsing pattern data: {e}")
                        continue
            
            # Apply filters if specified
            if filters:
                patterns = self._apply_pattern_filters(patterns, filters)
            
            self.logger.debug(f"Loaded {len(patterns)} patterns with metrics")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return []
    
    def _create_pattern_node(self, pattern_data: Dict[str, Any]) -> Optional[PatternNode]:
        """Create a PatternNode from pattern data."""
        try:
            pattern_id = pattern_data.get('pattern_id', f"pattern_{int(time.time())}")
            name = pattern_data.get('pattern_description', pattern_id)[:50]
            
            # Extract metrics if available
            metrics = pattern_data.get('metrics', {})
            usage_count = metrics.get('total_applications', 0)
            success_rate = metrics.get('success_rate', 0.0)
            
            # Determine visual properties
            size = self._calculate_node_size(usage_count)
            color = self._get_success_color(success_rate)
            
            # Create node
            node = PatternNode(
                id=pattern_id,
                name=name,
                usage_count=usage_count,
                success_rate=success_rate,
                complexity=pattern_data.get('complexity', 'medium'),
                size=size,
                color=color,
                description=pattern_data.get('pattern_description', ''),
                domain=pattern_data.get('domain', ''),
                source_issue=pattern_data.get('source_issue', ''),
                related_patterns=pattern_data.get('related_patterns', [])
            )
            
            # Add to cache
            self.pattern_cache[pattern_id] = node
            
            return node
            
        except Exception as e:
            self.logger.error(f"Error creating pattern node: {e}")
            return None
    
    def _compute_pattern_relationships(self, nodes: List[PatternNode]) -> List[PatternEdge]:
        """Compute relationships between patterns based on various criteria."""
        edges = []
        
        try:
            # Create similarity matrix based on multiple factors
            similarity_matrix = self._compute_pattern_similarity_matrix(nodes)
            
            # Create edges for significant relationships
            threshold = self.config['similarity_threshold']
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > threshold:
                        edge = PatternEdge(
                            source=nodes[i].id,
                            target=nodes[j].id,
                            weight=similarity,
                            relationship_type="similarity",
                            confidence=min(similarity * 2, 1.0)  # Scale confidence
                        )
                        edges.append(edge)
            
            # Add explicit relationships from pattern data
            for node in nodes:
                if node.related_patterns:
                    for related_id in node.related_patterns:
                        # Check if related pattern exists in our node set
                        if any(n.id == related_id for n in nodes):
                            # Check if edge already exists
                            if not any(e.source == node.id and e.target == related_id for e in edges):
                                edge = PatternEdge(
                                    source=node.id,
                                    target=related_id,
                                    weight=1.0,  # Explicit relationships have high weight
                                    relationship_type="explicit",
                                    confidence=1.0
                                )
                                edges.append(edge)
            
            self.logger.debug(f"Computed {len(edges)} pattern relationships")
            return edges
            
        except Exception as e:
            self.logger.error(f"Error computing pattern relationships: {e}")
            return []
    
    def _compute_pattern_similarity_matrix(self, nodes: List[PatternNode]):
        """Compute similarity matrix between patterns using multiple factors."""
        n = len(nodes)
        
        if not NUMPY_AVAILABLE:
            # Fallback to basic Python implementation
            similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        else:
            similarity_matrix = np.zeros((n, n))
        
        try:
            # Factors for similarity calculation
            factors = {
                'domain_similarity': 0.3,
                'complexity_similarity': 0.2,
                'success_rate_similarity': 0.2,
                'usage_pattern_similarity': 0.2,
                'description_similarity': 0.1
            }
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        if NUMPY_AVAILABLE:
                            similarity_matrix[i][j] = 1.0
                        else:
                            similarity_matrix[i][j] = 1.0
                        continue
                    
                    node_i, node_j = nodes[i], nodes[j]
                    total_similarity = 0.0
                    
                    # Domain similarity
                    domain_sim = 1.0 if node_i.domain == node_j.domain else 0.0
                    total_similarity += domain_sim * factors['domain_similarity']
                    
                    # Complexity similarity
                    complexity_levels = {'low': 1, 'medium': 2, 'high': 3, 'very-high': 4}
                    comp_i = complexity_levels.get(node_i.complexity, 2)
                    comp_j = complexity_levels.get(node_j.complexity, 2)
                    comp_sim = 1.0 - abs(comp_i - comp_j) / 3.0
                    total_similarity += comp_sim * factors['complexity_similarity']
                    
                    # Success rate similarity
                    if node_i.success_rate > 0 and node_j.success_rate > 0:
                        success_sim = 1.0 - abs(node_i.success_rate - node_j.success_rate)
                        total_similarity += success_sim * factors['success_rate_similarity']
                    
                    # Usage pattern similarity
                    if node_i.usage_count > 0 and node_j.usage_count > 0:
                        max_usage = max(node_i.usage_count, node_j.usage_count)
                        min_usage = min(node_i.usage_count, node_j.usage_count)
                        usage_sim = min_usage / max_usage
                        total_similarity += usage_sim * factors['usage_pattern_similarity']
                    
                    # Simple description similarity (based on common words)
                    desc_sim = self._compute_text_similarity(node_i.description or "", 
                                                           node_j.description or "")
                    total_similarity += desc_sim * factors['description_similarity']
                    
                    if NUMPY_AVAILABLE:
                        similarity_matrix[i][j] = total_similarity
                    else:
                        similarity_matrix[i][j] = total_similarity
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing similarity matrix: {e}")
            # Return identity matrix as fallback
            if NUMPY_AVAILABLE:
                return np.eye(n)
            else:
                identity = [[0.0 for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    identity[i][i] = 1.0
                return identity
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_pattern_clusters(self, nodes: List[PatternNode], edges: List[PatternEdge]) -> List[PatternCluster]:
        """Identify clusters of related patterns using various clustering methods."""
        if len(nodes) < 2:
            return []
        
        try:
            # Create feature matrix for clustering
            features = self._create_pattern_feature_matrix(nodes)
            
            # Try different clustering methods
            clustering_results = {}
            
            if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
                # K-means clustering
                if len(nodes) >= 4:
                    optimal_k = min(len(nodes) // 3, 8)  # Reasonable cluster count
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                    kmeans_labels = kmeans.fit_predict(features)
                    clustering_results['kmeans'] = kmeans_labels
                
                # DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=max(2, len(nodes) // 10))
                dbscan_labels = dbscan.fit_predict(features)
                clustering_results['dbscan'] = dbscan_labels
            else:
                # Fallback to simple domain-based clustering
                self.logger.warning("Sklearn/numpy not available, using simple domain-based clustering")
                domain_clusters = {}
                for i, node in enumerate(nodes):
                    domain = node.domain or "unknown"
                    if domain not in domain_clusters:
                        domain_clusters[domain] = []
                    domain_clusters[domain].append(i)
                
                # Create labels array
                labels = [0] * len(nodes)
                cluster_id = 0
                for domain, indices in domain_clusters.items():
                    for idx in indices:
                        labels[idx] = cluster_id
                    cluster_id += 1
                
                clustering_results['domain_based'] = labels
            
            # Choose best clustering result
            best_labels = self._select_best_clustering(clustering_results, nodes)
            
            # Create cluster objects
            clusters = []
            unique_labels = list(set(best_labels)) if not NUMPY_AVAILABLE else np.unique(best_labels).tolist()
            
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                if NUMPY_AVAILABLE:
                    cluster_pattern_indices = np.where(np.array(best_labels) == cluster_id)[0]
                else:
                    cluster_pattern_indices = [i for i, label in enumerate(best_labels) if label == cluster_id]
                
                cluster_pattern_ids = [nodes[i].id for i in cluster_pattern_indices]
                
                if len(cluster_pattern_ids) >= self.config['cluster_min_size']:
                    cluster = self._create_cluster_object(cluster_id, cluster_pattern_ids, nodes)
                    clusters.append(cluster)
            
            self.logger.debug(f"Identified {len(clusters)} pattern clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error identifying clusters: {e}")
            return []
    
    def _create_pattern_feature_matrix(self, nodes: List[PatternNode]):
        """Create feature matrix for clustering analysis."""
        features = []
        
        # Define feature dimensions
        complexity_mapping = {'low': 1, 'medium': 2, 'high': 3, 'very-high': 4}
        
        # Calculate max values for normalization
        max_usage = max(n.usage_count for n in nodes) if nodes else 1
        max_connections = max(len(n.related_patterns) for n in nodes) if nodes else 1
        
        for node in nodes:
            feature_vector = [
                node.usage_count / max(1, max_usage),  # Normalized usage
                node.success_rate,  # Success rate
                complexity_mapping.get(node.complexity, 2) / 4.0,  # Normalized complexity
                len(node.related_patterns) / max(1, max_connections),  # Normalized connections
            ]
            
            # Add domain one-hot encoding
            domains = list(set(n.domain for n in nodes if n.domain))
            domain_vector = [1.0 if node.domain == domain else 0.0 for domain in domains]
            
            features.append(feature_vector + domain_vector)
        
        if NUMPY_AVAILABLE:
            return np.array(features)
        else:
            return features
    
    def _select_best_clustering(self, clustering_results: Dict[str, Any], 
                               nodes: List[PatternNode]):
        """Select the best clustering result based on quality metrics."""
        if not clustering_results:
            if NUMPY_AVAILABLE:
                return np.zeros(len(nodes))
            else:
                return [0] * len(nodes)
        
        # Simple heuristic: prefer clustering with reasonable number of clusters
        # and good separation
        best_score = -1
        best_labels = None
        
        for method, labels in clustering_results.items():
            if NUMPY_AVAILABLE:
                unique_labels = np.unique(labels)
            else:
                unique_labels = list(set(labels))
            
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            if n_clusters == 0:
                continue
            
            # Score based on cluster count and distribution
            ideal_clusters = max(2, min(len(nodes) // 4, 6))
            cluster_score = 1.0 - abs(n_clusters - ideal_clusters) / ideal_clusters
            
            # Bonus for balanced clusters
            if NUMPY_AVAILABLE:
                cluster_sizes = [np.sum(np.array(labels) == label) for label in unique_labels if label != -1]
            else:
                cluster_sizes = [sum(1 for l in labels if l == label) for label in unique_labels if label != -1]
            
            if cluster_sizes:
                mean_size = sum(cluster_sizes) / len(cluster_sizes)
                if NUMPY_AVAILABLE:
                    std_size = np.std(cluster_sizes)
                else:
                    variance = sum((s - mean_size) ** 2 for s in cluster_sizes) / len(cluster_sizes)
                    std_size = variance ** 0.5
                
                balance_score = 1.0 - (std_size / mean_size) if mean_size > 0 else 0
                total_score = cluster_score * 0.7 + balance_score * 0.3
            else:
                total_score = cluster_score
            
            if total_score > best_score:
                best_score = total_score
                best_labels = labels
        
        if best_labels is not None:
            return best_labels
        else:
            if NUMPY_AVAILABLE:
                return np.zeros(len(nodes))
            else:
                return [0] * len(nodes)
    
    def _create_cluster_object(self, cluster_id: int, pattern_ids: List[str], 
                              nodes: List[PatternNode]) -> PatternCluster:
        """Create a PatternCluster object from cluster analysis."""
        # Get nodes in this cluster
        cluster_nodes = [node for node in nodes if node.id in pattern_ids]
        
        # Calculate cluster characteristics
        total_usage = sum(node.usage_count for node in cluster_nodes)
        avg_success_rate = statistics.mean(node.success_rate for node in cluster_nodes)
        
        # Find dominant domain
        domains = [node.domain for node in cluster_nodes if node.domain]
        dominant_domain = max(set(domains), key=domains.count) if domains else None
        
        # Complexity distribution
        complexity_dist = {}
        for node in cluster_nodes:
            complexity_dist[node.complexity] = complexity_dist.get(node.complexity, 0) + 1
        
        # Generate cluster label
        label = f"Cluster {cluster_id}"
        if dominant_domain:
            label += f" ({dominant_domain})"
        
        return PatternCluster(
            cluster_id=cluster_id,
            pattern_ids=pattern_ids,
            dominant_domain=dominant_domain,
            average_success_rate=avg_success_rate,
            total_usage=total_usage,
            complexity_distribution=complexity_dist,
            label=label
        )
    
    def _apply_layout_algorithm(self, nodes: List[PatternNode], edges: List[PatternEdge], 
                               algorithm: str) -> List[PatternNode]:
        """Apply layout algorithm to position nodes in the graph."""
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node in nodes:
                G.add_node(node.id, **node.to_dict())
            
            # Add edges
            for edge in edges:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
            
            # Apply layout algorithm
            if algorithm == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif algorithm == "circular":
                pos = nx.circular_layout(G)
            elif algorithm == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            elif algorithm == "random":
                pos = nx.random_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Update node positions
            positioned_nodes = []
            for node in nodes:
                if node.id in pos:
                    node.position = pos[node.id]
                else:
                    node.position = (0.5, 0.5)  # Default center position
                positioned_nodes.append(node)
            
            return positioned_nodes
            
        except Exception as e:
            self.logger.error(f"Error applying layout algorithm: {e}")
            # Return nodes with default positions
            for node in nodes:
                node.position = (0.5, 0.5)
            return nodes
    
    def _calculate_node_size(self, usage_count: int) -> int:
        """Calculate node size based on usage count."""
        min_size, max_size = self.config['node_size_range']
        
        if usage_count == 0:
            return min_size
        
        # Logarithmic scaling for better visual distribution
        normalized = np.log(usage_count + 1) / np.log(101)  # Assume max usage around 100
        size = min_size + (max_size - min_size) * normalized
        
        return int(min(max_size, max(min_size, size)))
    
    def _get_success_color(self, success_rate: float) -> str:
        """Get color based on success rate."""
        # Red-Yellow-Green color scheme
        if success_rate >= 0.8:
            return "#27AE60"  # Green
        elif success_rate >= 0.6:
            return "#F39C12"  # Orange
        elif success_rate >= 0.4:
            return "#E67E22"  # Dark orange
        else:
            return "#E74C3C"  # Red
    
    def _compute_graph_metrics(self, nodes: List[PatternNode], edges: List[PatternEdge], 
                              clusters: List[PatternCluster]) -> Dict[str, Any]:
        """Compute comprehensive metrics for the generated graph."""
        try:
            # Basic metrics
            total_patterns = len(nodes)
            total_relationships = len(edges)
            total_clusters = len(clusters)
            
            # Usage metrics
            usage_counts = [node.usage_count for node in nodes]
            total_usage = sum(usage_counts)
            avg_usage = statistics.mean(usage_counts) if usage_counts else 0
            max_usage = max(usage_counts) if usage_counts else 0
            
            # Success metrics
            success_rates = [node.success_rate for node in nodes if node.success_rate > 0]
            avg_success_rate = statistics.mean(success_rates) if success_rates else 0
            
            # Relationship metrics
            edge_weights = [edge.weight for edge in edges]
            avg_relationship_strength = statistics.mean(edge_weights) if edge_weights else 0
            
            # Network metrics (if edges exist)
            network_density = 0.0
            avg_degree = 0.0
            
            if edges and total_patterns > 1:
                max_possible_edges = total_patterns * (total_patterns - 1) / 2
                network_density = total_relationships / max_possible_edges
                avg_degree = (2 * total_relationships) / total_patterns
            
            # Cluster metrics
            cluster_sizes = [len(cluster.pattern_ids) for cluster in clusters]
            avg_cluster_size = statistics.mean(cluster_sizes) if cluster_sizes else 0
            largest_cluster = max(cluster_sizes) if cluster_sizes else 0
            
            return {
                'graph_overview': {
                    'total_patterns': total_patterns,
                    'total_relationships': total_relationships,
                    'total_clusters': total_clusters,
                    'network_density': network_density,
                    'average_degree': avg_degree
                },
                'usage_analysis': {
                    'total_usage': total_usage,
                    'average_usage': avg_usage,
                    'max_usage': max_usage,
                    'patterns_with_usage': len([n for n in nodes if n.usage_count > 0])
                },
                'success_analysis': {
                    'average_success_rate': avg_success_rate,
                    'high_performing_patterns': len([n for n in nodes if n.success_rate >= 0.8]),
                    'low_performing_patterns': len([n for n in nodes if n.success_rate < 0.4]),
                    'patterns_with_metrics': len(success_rates)
                },
                'relationship_analysis': {
                    'average_relationship_strength': avg_relationship_strength,
                    'strong_relationships': len([e for e in edges if e.weight >= 0.7]),
                    'weak_relationships': len([e for e in edges if e.weight < 0.4])
                },
                'cluster_analysis': {
                    'average_cluster_size': avg_cluster_size,
                    'largest_cluster_size': largest_cluster,
                    'unclustered_patterns': total_patterns - sum(cluster_sizes)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error computing graph metrics: {e}")
            return {}
    
    def _apply_pattern_filters(self, patterns: List[Dict[str, Any]], 
                              filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to pattern list."""
        filtered_patterns = patterns
        
        # Domain filter
        if 'domain' in filters:
            domain = filters['domain']
            filtered_patterns = [p for p in filtered_patterns if p.get('domain') == domain]
        
        # Complexity filter
        if 'complexity' in filters:
            complexity = filters['complexity']
            filtered_patterns = [p for p in filtered_patterns if p.get('complexity') == complexity]
        
        # Success rate filter
        if 'min_success_rate' in filters:
            min_rate = filters['min_success_rate']
            filtered_patterns = [p for p in filtered_patterns 
                               if p.get('metrics', {}).get('success_rate', 0) >= min_rate]
        
        # Usage count filter
        if 'min_usage_count' in filters:
            min_usage = filters['min_usage_count']
            filtered_patterns = [p for p in filtered_patterns
                               if p.get('metrics', {}).get('total_applications', 0) >= min_usage]
        
        return filtered_patterns
    
    def _generate_cache_key(self, filters: Optional[Dict[str, Any]], 
                           layout: str, include_metrics: bool) -> str:
        """Generate cache key for visualization results."""
        import hashlib
        
        key_data = {
            'filters': filters or {},
            'layout': layout,
            'include_metrics': include_metrics,
            'config': self.config
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_performance_metrics(self, generation_time: float, nodes_count: int, edges_count: int):
        """Update system performance metrics."""
        self.performance_metrics['visualizations_generated'] += 1
        self.performance_metrics['patterns_analyzed'] += nodes_count
        self.performance_metrics['relationships_computed'] += edges_count
        
        # Update rolling average generation time
        current_avg = self.performance_metrics['average_generation_time']
        count = self.performance_metrics['visualizations_generated']
        
        self.performance_metrics['average_generation_time'] = (
            (current_avg * (count - 1) + generation_time) / count
        )
        
        self.performance_metrics['last_analysis_run'] = datetime.now().isoformat()
    
    def create_interactive_dashboard(self, graph_data: Dict[str, Any]) -> Optional[str]:
        """
        Create interactive HTML dashboard for pattern exploration.
        
        Args:
            graph_data: Graph data from generate_pattern_graph()
            
        Returns:
            HTML content for interactive dashboard or None if plotting unavailable
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available - cannot create interactive dashboard")
            return None
        
        try:
            # Create subplots for different visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Pattern Network", "Success Rate Distribution", 
                               "Usage Analysis", "Cluster Overview"),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Network visualization
            nodes = graph_data['nodes']
            edges = graph_data['edges']
            
            # Add network edges
            for edge in edges:
                source_node = next((n for n in nodes if n['id'] == edge['source']), None)
                target_node = next((n for n in nodes if n['id'] == edge['target']), None)
                
                if source_node and target_node and source_node.get('position') and target_node.get('position'):
                    x_coords = [source_node['position'][0], target_node['position'][0], None]
                    y_coords = [source_node['position'][1], target_node['position'][1], None]
                    
                    fig.add_trace(
                        go.Scatter(x=x_coords, y=y_coords, mode='lines',
                                 line=dict(width=1, color='gray'),
                                 showlegend=False, hoverinfo='none'),
                        row=1, col=1
                    )
            
            # Add network nodes
            if nodes:
                x_coords = [n.get('position', [0, 0])[0] for n in nodes]
                y_coords = [n.get('position', [0, 0])[1] for n in nodes]
                sizes = [n.get('size', 10) for n in nodes]
                colors = [n.get('success_rate', 0) for n in nodes]
                texts = [n.get('name', 'Unknown')[:20] for n in nodes]
                
                fig.add_trace(
                    go.Scatter(x=x_coords, y=y_coords, mode='markers+text',
                             marker=dict(size=sizes, color=colors, colorscale='RdYlGn',
                                       showscale=True, colorbar=dict(title="Success Rate")),
                             text=texts, textposition="middle center",
                             hovertemplate="<b>%{text}</b><br>Success Rate: %{marker.color:.2%}<extra></extra>",
                             showlegend=False),
                    row=1, col=1
                )
            
            # Success rate histogram
            success_rates = [n.get('success_rate', 0) for n in nodes if n.get('success_rate', 0) > 0]
            if success_rates:
                fig.add_trace(
                    go.Histogram(x=success_rates, nbinsx=20, showlegend=False,
                               marker=dict(color='lightblue')),
                    row=1, col=2
                )
            
            # Usage analysis bar chart
            usage_data = sorted([(n.get('name', 'Unknown')[:15], n.get('usage_count', 0)) 
                               for n in nodes], key=lambda x: x[1], reverse=True)[:10]
            
            if usage_data:
                names, usage_counts = zip(*usage_data)
                fig.add_trace(
                    go.Bar(x=list(names), y=list(usage_counts), showlegend=False,
                          marker=dict(color='lightcoral')),
                    row=2, col=1
                )
            
            # Cluster pie chart
            clusters = graph_data.get('clusters', [])
            if clusters:
                cluster_sizes = [len(c.get('pattern_ids', [])) for c in clusters]
                cluster_labels = [c.get('label', f"Cluster {i}") for i, c in enumerate(clusters)]
                
                fig.add_trace(
                    go.Pie(labels=cluster_labels, values=cluster_sizes, showlegend=True),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Pattern Analysis Dashboard",
                height=800,
                showlegend=True
            )
            
            # Generate HTML
            html_content = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            
            # Add metadata
            metadata = graph_data.get('metadata', {})
            html_header = f"""
            <div style="padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;">
                <h2>Pattern Visualization Dashboard</h2>
                <p><strong>Generated:</strong> {metadata.get('timestamp', 'Unknown')}</p>
                <p><strong>Patterns:</strong> {metadata.get('total_patterns', 0)} | 
                   <strong>Relationships:</strong> {metadata.get('total_relationships', 0)} | 
                   <strong>Clusters:</strong> {metadata.get('total_clusters', 0)}</p>
                <p><strong>Generation Time:</strong> {metadata.get('generation_time', 0):.2f} seconds</p>
            </div>
            """
            
            return html_header + html_content
            
        except Exception as e:
            self.logger.error(f"Error creating interactive dashboard: {e}")
            return None
    
    def export_visualization(self, graph_data: Dict[str, Any], 
                           output_path: str, format: str = "json") -> bool:
        """
        Export visualization data to various formats.
        
        Args:
            graph_data: Graph data to export
            output_path: Output file path
            format: Export format ('json', 'html', 'png')
            
        Returns:
            True if export succeeded, False otherwise
        """
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2, default=str)
                
            elif format.lower() == "html":
                html_content = self.create_interactive_dashboard(graph_data)
                if html_content:
                    with open(output_path, 'w') as f:
                        f.write(html_content)
                else:
                    return False
                    
            elif format.lower() == "png" and MATPLOTLIB_AVAILABLE:
                # Create static PNG visualization
                self._create_static_visualization(graph_data, output_path)
                
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Exported visualization to {output_path} ({format})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
            return False
    
    def _create_static_visualization(self, graph_data: Dict[str, Any], output_path: str):
        """Create static matplotlib visualization."""
        plt.figure(figsize=(12, 8))
        
        # Create network visualization
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Plot edges
        for edge in edges:
            source_node = next((n for n in nodes if n['id'] == edge['source']), None)
            target_node = next((n for n in nodes if n['id'] == edge['target']), None)
            
            if source_node and target_node and source_node.get('position') and target_node.get('position'):
                x_coords = [source_node['position'][0], target_node['position'][0]]
                y_coords = [source_node['position'][1], target_node['position'][1]]
                plt.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=edge.get('weight', 1))
        
        # Plot nodes
        if nodes:
            x_coords = [n.get('position', [0, 0])[0] for n in nodes]
            y_coords = [n.get('position', [0, 0])[1] for n in nodes]
            sizes = [n.get('size', 10) * 10 for n in nodes]  # Scale for matplotlib
            colors = [n.get('success_rate', 0) for n in nodes]
            
            scatter = plt.scatter(x_coords, y_coords, s=sizes, c=colors, 
                                cmap='RdYlGn', alpha=0.7)
            plt.colorbar(scatter, label='Success Rate')
            
            # Add labels for important nodes
            for node in nodes[:10]:  # Label top 10 nodes
                if node.get('position'):
                    plt.annotate(node.get('name', '')[:15], 
                               (node['position'][0], node['position'][1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        plt.title("Pattern Relationship Network")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            **self.performance_metrics,
            'cache_size': len(self.visualization_cache),
            'pattern_cache_size': len(self.pattern_cache),
            'relationship_cache_size': len(self.relationship_cache),
            'plotly_available': PLOTLY_AVAILABLE,
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'pattern_system_available': self.pattern_system is not None,
            'knowledge_system_available': self.knowledge is not None
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.visualization_cache.clear()
        self.pattern_cache.clear()
        self.relationship_cache.clear()
        self.logger.info("Pattern visualization system cleanup completed")


# CLI interface for pattern visualization
def main():
    """Command-line interface for pattern visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Pattern Visualization System")
    parser.add_argument("--output", "-o", default="pattern_graph.json", 
                       help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "html", "png"], 
                       default="json", help="Output format")
    parser.add_argument("--layout", "-l", choices=["spring", "circular", "kamada_kawai", "random"],
                       default="spring", help="Layout algorithm")
    parser.add_argument("--filter-domain", help="Filter patterns by domain")
    parser.add_argument("--filter-complexity", help="Filter patterns by complexity")
    parser.add_argument("--min-success-rate", type=float, help="Minimum success rate filter")
    parser.add_argument("--min-usage", type=int, help="Minimum usage count filter")
    parser.add_argument("--interactive", action="store_true", 
                       help="Generate interactive dashboard")
    
    args = parser.parse_args()
    
    # Create filters
    filters = {}
    if args.filter_domain:
        filters['domain'] = args.filter_domain
    if args.filter_complexity:
        filters['complexity'] = args.filter_complexity
    if args.min_success_rate:
        filters['min_success_rate'] = args.min_success_rate
    if args.min_usage:
        filters['min_usage_count'] = args.min_usage
    
    # Initialize visualization system
    viz = PatternVisualization()
    
    # Generate graph
    graph_data = viz.generate_pattern_graph(
        filters=filters if filters else None,
        layout_algorithm=args.layout,
        include_metrics=True
    )
    
    # Export results
    if args.interactive and args.format == "html":
        html_content = viz.create_interactive_dashboard(graph_data)
        if html_content:
            with open(args.output, 'w') as f:
                f.write(html_content)
            print(f"Interactive dashboard saved to {args.output}")
        else:
            print("Failed to create interactive dashboard")
    else:
        success = viz.export_visualization(graph_data, args.output, args.format)
        if success:
            print(f"Visualization exported to {args.output} ({args.format})")
        else:
            print("Export failed")
    
    # Print summary
    metadata = graph_data.get('metadata', {})
    print(f"\nSummary:")
    print(f"  Patterns: {metadata.get('total_patterns', 0)}")
    print(f"  Relationships: {metadata.get('total_relationships', 0)}")
    print(f"  Clusters: {metadata.get('total_clusters', 0)}")
    print(f"  Generation time: {metadata.get('generation_time', 0):.2f}s")


if __name__ == "__main__":
    main()