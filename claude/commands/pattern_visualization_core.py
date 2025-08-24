"""
Core Pattern Visualization Components

This module contains the core data structures and basic functionality
for pattern visualization without heavy dependencies. This ensures
the basic functionality works even when advanced libraries are not available.
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


class PatternVisualizationCore:
    """
    Core pattern visualization functionality without heavy dependencies.
    
    This class provides basic pattern analysis and graph generation
    capabilities that work without numpy, pandas, or sklearn.
    """
    
    def __init__(self, visualization_config: Optional[Dict[str, Any]] = None):
        """Initialize the core visualization system."""
        self.logger = logging.getLogger(f"{__name__}.PatternVisualizationCore")
        
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
            'export_formats': ['json'],
            'cache_visualizations': True,
        }
        
        if visualization_config:
            self.config.update(visualization_config)
        
        # Cache for computed visualizations
        self.visualization_cache: Dict[str, Dict[str, Any]] = {}
        self.pattern_cache: Dict[str, PatternNode] = {}
        
        # Performance metrics
        self.performance_metrics = {
            'visualizations_generated': 0,
            'patterns_analyzed': 0,
            'relationships_computed': 0,
            'clusters_identified': 0,
            'average_generation_time': 0.0,
            'last_analysis_run': None
        }
        
        self.logger.info("Core pattern visualization system initialized successfully")
    
    def generate_pattern_graph(self, 
                              patterns_data: List[Dict[str, Any]],
                              layout_algorithm: str = "spring",
                              include_metrics: bool = True) -> Dict[str, Any]:
        """
        Generate a pattern graph from provided pattern data.
        
        Args:
            patterns_data: List of pattern dictionaries
            layout_algorithm: Algorithm for node positioning
            include_metrics: Whether to include detailed metrics
            
        Returns:
            Dictionary containing nodes, edges, clusters, and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info("Generating pattern relationship graph")
            
            if not patterns_data:
                self.logger.warning("No patterns provided for visualization")
                return {'nodes': [], 'edges': [], 'clusters': []}
            
            # Create nodes
            nodes = []
            for pattern in patterns_data:
                node = self._create_pattern_node(pattern)
                if node:
                    nodes.append(node)
            
            if not nodes:
                return {'nodes': [], 'edges': [], 'clusters': []}
            
            # Create edges based on relationships
            edges = self._compute_pattern_relationships(nodes)
            
            # Identify clusters using simple domain-based clustering
            clusters = self._identify_pattern_clusters_simple(nodes)
            
            # Apply simple layout algorithm
            positioned_nodes = self._apply_simple_layout(nodes, edges, layout_algorithm)
            
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
                    'core_functionality': True
                }
            }
            
            # Include detailed metrics if requested
            if include_metrics:
                result['metrics'] = self._compute_graph_metrics(positioned_nodes, edges, clusters)
            
            # Update performance metrics
            generation_time = time.time() - start_time
            self._update_performance_metrics(generation_time, len(positioned_nodes), len(edges))
            
            self.logger.info(f"Generated pattern graph: {len(positioned_nodes)} nodes, "
                           f"{len(edges)} edges, {len(clusters)} clusters in {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating pattern graph: {e}")
            return {'nodes': [], 'edges': [], 'clusters': [], 'error': str(e)}
    
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
        """Compute relationships between patterns."""
        edges = []
        
        try:
            # Create similarity matrix using simple methods
            threshold = self.config['similarity_threshold']
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = self._compute_node_similarity(nodes[i], nodes[j])
                    
                    if similarity > threshold:
                        edge = PatternEdge(
                            source=nodes[i].id,
                            target=nodes[j].id,
                            weight=similarity,
                            relationship_type="similarity",
                            confidence=min(similarity * 2, 1.0)
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
    
    def _compute_node_similarity(self, node1: PatternNode, node2: PatternNode) -> float:
        """Compute similarity between two nodes using simple metrics."""
        similarity = 0.0
        
        # Domain similarity (30% weight)
        if node1.domain == node2.domain and node1.domain:
            similarity += 0.3
        
        # Complexity similarity (25% weight)
        complexity_levels = {'low': 1, 'medium': 2, 'high': 3, 'very-high': 4}
        comp1 = complexity_levels.get(node1.complexity, 2)
        comp2 = complexity_levels.get(node2.complexity, 2)
        comp_sim = 1.0 - abs(comp1 - comp2) / 3.0
        similarity += comp_sim * 0.25
        
        # Success rate similarity (25% weight)
        if node1.success_rate > 0 and node2.success_rate > 0:
            success_sim = 1.0 - abs(node1.success_rate - node2.success_rate)
            similarity += success_sim * 0.25
        
        # Usage similarity (20% weight)
        if node1.usage_count > 0 and node2.usage_count > 0:
            max_usage = max(node1.usage_count, node2.usage_count)
            min_usage = min(node1.usage_count, node2.usage_count)
            usage_sim = min_usage / max_usage
            similarity += usage_sim * 0.2
        
        return similarity
    
    def _identify_pattern_clusters_simple(self, nodes: List[PatternNode]) -> List[PatternCluster]:
        """Simple clustering based on domain and complexity."""
        if len(nodes) < 2:
            return []
        
        try:
            # Group by domain first, then by complexity
            domain_groups = {}
            for node in nodes:
                domain = node.domain or "unknown"
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(node)
            
            clusters = []
            cluster_id = 0
            
            for domain, domain_nodes in domain_groups.items():
                if len(domain_nodes) >= self.config['cluster_min_size']:
                    # Create cluster for this domain
                    pattern_ids = [node.id for node in domain_nodes]
                    cluster = self._create_cluster_object(cluster_id, pattern_ids, domain_nodes)
                    cluster.dominant_domain = domain
                    cluster.label = f"Cluster {cluster_id} ({domain})"
                    clusters.append(cluster)
                    cluster_id += 1
                else:
                    # Too small for own cluster, could add to "mixed" cluster
                    pass
            
            self.logger.debug(f"Identified {len(clusters)} pattern clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error identifying clusters: {e}")
            return []
    
    def _apply_simple_layout(self, nodes: List[PatternNode], edges: List[PatternEdge], 
                           algorithm: str) -> List[PatternNode]:
        """Apply simple layout algorithm without NetworkX."""
        try:
            import math
            import random
            
            if algorithm == "circular":
                # Circular layout
                n = len(nodes)
                for i, node in enumerate(nodes):
                    angle = 2 * math.pi * i / n
                    x = 0.5 + 0.4 * math.cos(angle)
                    y = 0.5 + 0.4 * math.sin(angle)
                    node.position = (x, y)
            
            elif algorithm == "grid":
                # Grid layout
                n = len(nodes)
                cols = int(math.ceil(math.sqrt(n)))
                for i, node in enumerate(nodes):
                    row = i // cols
                    col = i % cols
                    x = (col + 0.5) / cols
                    y = (row + 0.5) / math.ceil(n / cols)
                    node.position = (x, y)
            
            else:  # Default to random with slight clustering
                # Simple force-directed-like layout
                random.seed(42)  # For reproducible results
                
                # Initial random positions
                for node in nodes:
                    node.position = (random.random(), random.random())
                
                # Simple iterations to spread nodes
                for _ in range(10):
                    forces = {}
                    for node in nodes:
                        forces[node.id] = [0.0, 0.0]
                    
                    # Repulsion between all nodes
                    for i, node1 in enumerate(nodes):
                        for j, node2 in enumerate(nodes):
                            if i != j:
                                dx = node1.position[0] - node2.position[0]
                                dy = node1.position[1] - node2.position[1]
                                dist = max(0.01, math.sqrt(dx*dx + dy*dy))
                                force = 0.01 / (dist * dist)
                                forces[node1.id][0] += force * dx / dist
                                forces[node1.id][1] += force * dy / dist
                    
                    # Attraction for connected nodes
                    for edge in edges:
                        node1 = next(n for n in nodes if n.id == edge.source)
                        node2 = next(n for n in nodes if n.id == edge.target)
                        
                        dx = node2.position[0] - node1.position[0]
                        dy = node2.position[1] - node1.position[1]
                        dist = max(0.01, math.sqrt(dx*dx + dy*dy))
                        force = 0.001 * dist * edge.weight
                        
                        forces[node1.id][0] += force * dx / dist
                        forces[node1.id][1] += force * dy / dist
                        forces[node2.id][0] -= force * dx / dist
                        forces[node2.id][1] -= force * dy / dist
                    
                    # Apply forces
                    for node in nodes:
                        fx, fy = forces[node.id]
                        x, y = node.position
                        x += fx * 0.1
                        y += fy * 0.1
                        # Keep within bounds
                        x = max(0.1, min(0.9, x))
                        y = max(0.1, min(0.9, y))
                        node.position = (x, y)
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Error applying layout algorithm: {e}")
            # Fallback to simple positions
            for i, node in enumerate(nodes):
                node.position = (0.5, 0.5)  # Default center position
            return nodes
    
    def _create_cluster_object(self, cluster_id: int, pattern_ids: List[str], 
                              nodes: List[PatternNode]) -> PatternCluster:
        """Create a PatternCluster object."""
        # Calculate cluster characteristics
        total_usage = sum(node.usage_count for node in nodes)
        avg_success_rate = statistics.mean(node.success_rate for node in nodes) if nodes else 0.0
        
        # Find dominant domain
        domains = [node.domain for node in nodes if node.domain]
        dominant_domain = max(set(domains), key=domains.count) if domains else None
        
        # Complexity distribution
        complexity_dist = {}
        for node in nodes:
            complexity_dist[node.complexity] = complexity_dist.get(node.complexity, 0) + 1
        
        return PatternCluster(
            cluster_id=cluster_id,
            pattern_ids=pattern_ids,
            dominant_domain=dominant_domain,
            average_success_rate=avg_success_rate,
            total_usage=total_usage,
            complexity_distribution=complexity_dist,
            label=f"Cluster {cluster_id}"
        )
    
    def _calculate_node_size(self, usage_count: int) -> int:
        """Calculate node size based on usage count."""
        min_size, max_size = self.config['node_size_range']
        
        if usage_count == 0:
            return min_size
        
        # Simple logarithmic scaling
        import math
        normalized = math.log(usage_count + 1) / math.log(101)  # Assume max usage around 100
        size = min_size + (max_size - min_size) * normalized
        
        return int(min(max_size, max(min_size, size)))
    
    def _get_success_color(self, success_rate: float) -> str:
        """Get color based on success rate."""
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
            
            # Network metrics
            network_density = 0.0
            if total_patterns > 1:
                max_possible_edges = total_patterns * (total_patterns - 1) / 2
                network_density = total_relationships / max_possible_edges
            
            return {
                'graph_overview': {
                    'total_patterns': total_patterns,
                    'total_relationships': total_relationships,
                    'total_clusters': total_clusters,
                    'network_density': network_density
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
                    'low_performing_patterns': len([n for n in nodes if n.success_rate < 0.4])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error computing graph metrics: {e}")
            return {}
    
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
    
    def export_visualization(self, graph_data: Dict[str, Any], 
                           output_path: str, format: str = "json") -> bool:
        """Export visualization data to JSON format."""
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2, default=str)
                
                self.logger.info(f"Exported visualization to {output_path} (JSON)")
                return True
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            **self.performance_metrics,
            'cache_size': len(self.visualization_cache),
            'pattern_cache_size': len(self.pattern_cache),
            'core_functionality_only': True
        }