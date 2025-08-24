"""
Tests for Pattern Visualization System

Comprehensive test suite covering all aspects of the pattern visualization
system including graph generation, clustering, metrics computation,
and export functionality.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

# Import the module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from claude.commands.pattern_visualization import (
    PatternVisualization,
    PatternNode,
    PatternEdge,
    PatternCluster,
    VisualizationType,
    ClusteringMethod
)


class TestPatternNode(unittest.TestCase):
    """Test PatternNode data class."""
    
    def test_pattern_node_creation(self):
        """Test creating a PatternNode with all fields."""
        node = PatternNode(
            id="test_pattern",
            name="Test Pattern",
            usage_count=10,
            success_rate=0.85,
            complexity="high",
            size=25,
            color="#27AE60",
            description="A test pattern for validation",
            domain="testing",
            source_issue="79",
            related_patterns=["pattern1", "pattern2"]
        )
        
        self.assertEqual(node.id, "test_pattern")
        self.assertEqual(node.name, "Test Pattern")
        self.assertEqual(node.usage_count, 10)
        self.assertEqual(node.success_rate, 0.85)
        self.assertEqual(node.complexity, "high")
        self.assertEqual(len(node.related_patterns), 2)
    
    def test_pattern_node_serialization(self):
        """Test PatternNode to_dict and from_dict methods."""
        original_node = PatternNode(
            id="serialize_test",
            name="Serialization Test",
            usage_count=5,
            success_rate=0.75,
            last_used=datetime.now()
        )
        
        # Convert to dict
        node_dict = original_node.to_dict()
        self.assertIsInstance(node_dict, dict)
        self.assertEqual(node_dict['id'], "serialize_test")
        self.assertIsInstance(node_dict['last_used'], str)  # Should be ISO format
        
        # Convert back to object
        restored_node = PatternNode.from_dict(node_dict)
        self.assertEqual(restored_node.id, original_node.id)
        self.assertEqual(restored_node.name, original_node.name)
        self.assertIsInstance(restored_node.last_used, datetime)


class TestPatternEdge(unittest.TestCase):
    """Test PatternEdge data class."""
    
    def test_pattern_edge_creation(self):
        """Test creating a PatternEdge with all fields."""
        edge = PatternEdge(
            source="pattern1",
            target="pattern2",
            weight=0.75,
            relationship_type="similarity",
            confidence=0.9,
            evidence_count=5
        )
        
        self.assertEqual(edge.source, "pattern1")
        self.assertEqual(edge.target, "pattern2")
        self.assertEqual(edge.weight, 0.75)
        self.assertEqual(edge.confidence, 0.9)
    
    def test_pattern_edge_serialization(self):
        """Test PatternEdge serialization."""
        edge = PatternEdge(
            source="edge_test_1",
            target="edge_test_2",
            weight=0.5,
            last_reinforced=datetime.now()
        )
        
        edge_dict = edge.to_dict()
        restored_edge = PatternEdge.from_dict(edge_dict)
        
        self.assertEqual(restored_edge.source, edge.source)
        self.assertEqual(restored_edge.target, edge.target)
        self.assertEqual(restored_edge.weight, edge.weight)
        self.assertIsInstance(restored_edge.last_reinforced, datetime)


class TestPatternCluster(unittest.TestCase):
    """Test PatternCluster data class."""
    
    def test_pattern_cluster_creation(self):
        """Test creating a PatternCluster."""
        cluster = PatternCluster(
            cluster_id=1,
            pattern_ids=["pattern1", "pattern2", "pattern3"],
            dominant_domain="testing",
            average_success_rate=0.82,
            total_usage=25,
            complexity_distribution={"high": 2, "medium": 1}
        )
        
        self.assertEqual(cluster.cluster_id, 1)
        self.assertEqual(len(cluster.pattern_ids), 3)
        self.assertEqual(cluster.dominant_domain, "testing")
        self.assertEqual(cluster.average_success_rate, 0.82)
    
    def test_pattern_cluster_serialization(self):
        """Test PatternCluster serialization."""
        cluster = PatternCluster(
            cluster_id=2,
            pattern_ids=["p1", "p2"],
            centroid=[1.0, 2.0, 3.0]
        )
        
        cluster_dict = cluster.to_dict()
        restored_cluster = PatternCluster.from_dict(cluster_dict)
        
        self.assertEqual(restored_cluster.cluster_id, cluster.cluster_id)
        self.assertEqual(restored_cluster.pattern_ids, cluster.pattern_ids)
        self.assertEqual(restored_cluster.centroid, cluster.centroid)


class TestPatternVisualization(unittest.TestCase):
    """Test main PatternVisualization class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock knowledge system
        self.mock_knowledge = Mock()
        
        # Mock pattern system
        self.mock_pattern_system = Mock()
        
        # Create visualization instance
        self.viz = PatternVisualization(
            pattern_system=self.mock_pattern_system,
            knowledge_system=self.mock_knowledge,
            visualization_config={'max_patterns_display': 20}
        )
    
    def test_initialization(self):
        """Test PatternVisualization initialization."""
        self.assertIsNotNone(self.viz)
        self.assertEqual(self.viz.config['max_patterns_display'], 20)
        self.assertIsNotNone(self.viz.logger)
        self.assertIsInstance(self.viz.pattern_cache, dict)
        self.assertIsInstance(self.viz.visualization_cache, dict)
    
    def test_initialization_without_dependencies(self):
        """Test initialization when dependencies are not available."""
        viz = PatternVisualization()
        self.assertIsNotNone(viz)
        # Should not crash even without pattern system or knowledge system
    
    @patch('claude.commands.pattern_visualization.json.loads')
    def test_load_patterns_with_metrics(self, mock_json_loads):
        """Test loading patterns from knowledge system."""
        # Setup mock data
        mock_pattern_data = {
            'pattern_id': 'test_pattern_1',
            'pattern_description': 'Test Pattern Description',
            'domain': 'testing',
            'complexity': 'medium'
        }
        
        mock_json_loads.return_value = mock_pattern_data
        
        # Mock knowledge system response
        self.mock_knowledge.retrieve_knowledge.return_value = [
            {'content': json.dumps(mock_pattern_data)}
        ]
        
        # Mock pattern system metrics
        mock_metrics = {
            'total_applications': 5,
            'success_rate': 0.8,
            'last_used': datetime.now().isoformat()
        }
        self.mock_pattern_system.get_pattern_metrics.return_value = mock_metrics
        
        # Test method
        patterns = self.viz._load_patterns_with_metrics()
        
        self.assertGreater(len(patterns), 0)
        self.assertIn('metrics', patterns[0])
        self.assertEqual(patterns[0]['pattern_id'], 'test_pattern_1')
    
    def test_create_pattern_node(self):
        """Test creating PatternNode from pattern data."""
        pattern_data = {
            'pattern_id': 'node_test',
            'pattern_description': 'Node Test Pattern',
            'domain': 'testing',
            'complexity': 'high',
            'metrics': {
                'total_applications': 15,
                'success_rate': 0.9
            }
        }
        
        node = self.viz._create_pattern_node(pattern_data)
        
        self.assertIsNotNone(node)
        self.assertEqual(node.id, 'node_test')
        self.assertEqual(node.usage_count, 15)
        self.assertEqual(node.success_rate, 0.9)
        self.assertEqual(node.domain, 'testing')
    
    def test_create_pattern_node_invalid_data(self):
        """Test creating PatternNode with invalid data."""
        invalid_data = {'invalid': 'data'}
        
        node = self.viz._create_pattern_node(invalid_data)
        
        # Should handle gracefully and create node with defaults
        self.assertIsNotNone(node)
        self.assertIsNotNone(node.id)
    
    def test_calculate_node_size(self):
        """Test node size calculation based on usage count."""
        # Test with zero usage
        size_zero = self.viz._calculate_node_size(0)
        self.assertEqual(size_zero, self.viz.config['node_size_range'][0])
        
        # Test with high usage
        size_high = self.viz._calculate_node_size(100)
        self.assertGreaterEqual(size_high, self.viz.config['node_size_range'][0])
        self.assertLessEqual(size_high, self.viz.config['node_size_range'][1])
        
        # Test scaling
        size_low = self.viz._calculate_node_size(1)
        size_med = self.viz._calculate_node_size(10)
        self.assertLess(size_low, size_med)
    
    def test_get_success_color(self):
        """Test color assignment based on success rate."""
        # High success rate should be green
        color_high = self.viz._get_success_color(0.9)
        self.assertEqual(color_high, "#27AE60")  # Green
        
        # Medium success rate should be orange
        color_med = self.viz._get_success_color(0.7)
        self.assertEqual(color_med, "#F39C12")  # Orange
        
        # Low success rate should be red
        color_low = self.viz._get_success_color(0.2)
        self.assertEqual(color_low, "#E74C3C")  # Red
    
    def test_compute_text_similarity(self):
        """Test text similarity computation."""
        # Identical texts
        sim_identical = self.viz._compute_text_similarity("test text", "test text")
        self.assertEqual(sim_identical, 1.0)
        
        # Completely different texts
        sim_different = self.viz._compute_text_similarity("hello world", "foo bar")
        self.assertEqual(sim_different, 0.0)
        
        # Partially similar texts
        sim_partial = self.viz._compute_text_similarity("hello world", "hello universe")
        self.assertGreater(sim_partial, 0.0)
        self.assertLess(sim_partial, 1.0)
        
        # Empty texts
        sim_empty = self.viz._compute_text_similarity("", "test")
        self.assertEqual(sim_empty, 0.0)
    
    def test_compute_pattern_relationships(self):
        """Test computing relationships between patterns."""
        # Create test nodes
        nodes = [
            PatternNode(
                id="rel_test_1",
                name="Test Pattern 1",
                domain="testing",
                complexity="medium",
                success_rate=0.8,
                description="First test pattern"
            ),
            PatternNode(
                id="rel_test_2", 
                name="Test Pattern 2",
                domain="testing",
                complexity="medium", 
                success_rate=0.75,
                description="Second test pattern",
                related_patterns=["rel_test_1"]
            ),
            PatternNode(
                id="rel_test_3",
                name="Different Pattern",
                domain="production",
                complexity="high",
                success_rate=0.6,
                description="Completely different pattern"
            )
        ]
        
        edges = self.viz._compute_pattern_relationships(nodes)
        
        self.assertIsInstance(edges, list)
        # Should have at least one edge (explicit relationship from node 2 to node 1)
        self.assertGreater(len(edges), 0)
        
        # Check for explicit relationship
        explicit_edge = next((e for e in edges if e.relationship_type == "explicit"), None)
        self.assertIsNotNone(explicit_edge)
    
    def test_create_pattern_feature_matrix(self):
        """Test creation of feature matrix for clustering."""
        nodes = [
            PatternNode(id="feat1", name="F1", usage_count=10, success_rate=0.8, 
                       complexity="high", domain="test"),
            PatternNode(id="feat2", name="F2", usage_count=5, success_rate=0.6, 
                       complexity="medium", domain="prod"),
            PatternNode(id="feat3", name="F3", usage_count=15, success_rate=0.9, 
                       complexity="low", domain="test")
        ]
        
        features = self.viz._create_pattern_feature_matrix(nodes)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 3)  # Three patterns
        self.assertGreater(features.shape[1], 4)  # At least basic features + domain encoding
        
        # Check normalization
        usage_col = features[:, 0]  # First column should be normalized usage
        self.assertGreaterEqual(np.min(usage_col), 0.0)
        self.assertLessEqual(np.max(usage_col), 1.0)
    
    def test_identify_pattern_clusters(self):
        """Test pattern clustering functionality."""
        # Create enough nodes for meaningful clustering
        nodes = []
        for i in range(10):
            node = PatternNode(
                id=f"cluster_test_{i}",
                name=f"Pattern {i}",
                usage_count=i * 2,
                success_rate=0.5 + (i % 3) * 0.2,
                domain="test" if i < 5 else "prod",
                complexity="medium"
            )
            nodes.append(node)
        
        edges = []  # Empty edges for simplicity
        
        clusters = self.viz._identify_pattern_clusters(nodes, edges)
        
        self.assertIsInstance(clusters, list)
        # With 10 diverse nodes, should create some clusters
        if len(clusters) > 0:
            for cluster in clusters:
                self.assertIsInstance(cluster, PatternCluster)
                self.assertGreaterEqual(len(cluster.pattern_ids), self.viz.config['cluster_min_size'])
    
    def test_apply_layout_algorithm(self):
        """Test layout algorithm application."""
        nodes = [
            PatternNode(id="layout1", name="L1"),
            PatternNode(id="layout2", name="L2"),
            PatternNode(id="layout3", name="L3")
        ]
        
        edges = [
            PatternEdge(source="layout1", target="layout2", weight=0.8),
            PatternEdge(source="layout2", target="layout3", weight=0.6)
        ]
        
        positioned_nodes = self.viz._apply_layout_algorithm(nodes, edges, "spring")
        
        self.assertEqual(len(positioned_nodes), 3)
        for node in positioned_nodes:
            self.assertIsNotNone(node.position)
            self.assertEqual(len(node.position), 2)  # x, y coordinates
    
    def test_compute_graph_metrics(self):
        """Test graph metrics computation."""
        nodes = [
            PatternNode(id="metric1", name="M1", usage_count=10, success_rate=0.8),
            PatternNode(id="metric2", name="M2", usage_count=5, success_rate=0.6)
        ]
        
        edges = [
            PatternEdge(source="metric1", target="metric2", weight=0.7)
        ]
        
        clusters = [
            PatternCluster(cluster_id=1, pattern_ids=["metric1", "metric2"])
        ]
        
        metrics = self.viz._compute_graph_metrics(nodes, edges, clusters)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('graph_overview', metrics)
        self.assertIn('usage_analysis', metrics)
        self.assertIn('success_analysis', metrics)
        
        # Check specific values
        self.assertEqual(metrics['graph_overview']['total_patterns'], 2)
        self.assertEqual(metrics['graph_overview']['total_relationships'], 1)
        self.assertEqual(metrics['usage_analysis']['total_usage'], 15)
    
    def test_apply_pattern_filters(self):
        """Test pattern filtering functionality."""
        patterns = [
            {
                'pattern_id': 'filter1',
                'domain': 'testing',
                'complexity': 'high',
                'metrics': {'success_rate': 0.9, 'total_applications': 10}
            },
            {
                'pattern_id': 'filter2',
                'domain': 'production',
                'complexity': 'medium',
                'metrics': {'success_rate': 0.6, 'total_applications': 5}
            },
            {
                'pattern_id': 'filter3',
                'domain': 'testing',
                'complexity': 'low',
                'metrics': {'success_rate': 0.8, 'total_applications': 15}
            }
        ]
        
        # Test domain filter
        domain_filtered = self.viz._apply_pattern_filters(patterns, {'domain': 'testing'})
        self.assertEqual(len(domain_filtered), 2)
        
        # Test complexity filter
        complexity_filtered = self.viz._apply_pattern_filters(patterns, {'complexity': 'high'})
        self.assertEqual(len(complexity_filtered), 1)
        
        # Test success rate filter
        success_filtered = self.viz._apply_pattern_filters(patterns, {'min_success_rate': 0.7})
        self.assertEqual(len(success_filtered), 2)
        
        # Test usage filter
        usage_filtered = self.viz._apply_pattern_filters(patterns, {'min_usage_count': 8})
        self.assertEqual(len(usage_filtered), 2)
        
        # Test combined filters
        combined_filtered = self.viz._apply_pattern_filters(
            patterns, 
            {'domain': 'testing', 'min_success_rate': 0.75}
        )
        self.assertEqual(len(combined_filtered), 1)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        filters1 = {'domain': 'test'}
        filters2 = {'domain': 'prod'}
        
        key1 = self.viz._generate_cache_key(filters1, 'spring', True)
        key2 = self.viz._generate_cache_key(filters2, 'spring', True)
        key3 = self.viz._generate_cache_key(filters1, 'circular', True)
        
        # Different filters should produce different keys
        self.assertNotEqual(key1, key2)
        
        # Different layouts should produce different keys
        self.assertNotEqual(key1, key3)
        
        # Same parameters should produce same key
        key1_repeat = self.viz._generate_cache_key(filters1, 'spring', True)
        self.assertEqual(key1, key1_repeat)
    
    def test_update_performance_metrics(self):
        """Test performance metrics updates."""
        initial_count = self.viz.performance_metrics['visualizations_generated']
        
        self.viz._update_performance_metrics(2.5, 10, 15)
        
        self.assertEqual(
            self.viz.performance_metrics['visualizations_generated'], 
            initial_count + 1
        )
        self.assertEqual(self.viz.performance_metrics['patterns_analyzed'], 10)
        self.assertEqual(self.viz.performance_metrics['relationships_computed'], 15)
    
    @patch('claude.commands.pattern_visualization.json.loads')
    def test_generate_pattern_graph_complete(self, mock_json_loads):
        """Test complete pattern graph generation process."""
        # Setup mock pattern data
        mock_patterns = [
            {
                'pattern_id': 'complete_test_1',
                'pattern_description': 'Complete Test Pattern 1',
                'domain': 'testing',
                'complexity': 'medium'
            },
            {
                'pattern_id': 'complete_test_2',
                'pattern_description': 'Complete Test Pattern 2',
                'domain': 'testing',
                'complexity': 'high',
                'related_patterns': ['complete_test_1']
            }
        ]
        
        mock_json_loads.side_effect = mock_patterns
        
        # Mock knowledge system
        self.mock_knowledge.retrieve_knowledge.return_value = [
            {'content': json.dumps(pattern)} for pattern in mock_patterns
        ]
        
        # Mock pattern system metrics
        mock_metrics = {
            'total_applications': 8,
            'success_rate': 0.75
        }
        self.mock_pattern_system.get_pattern_metrics.return_value = mock_metrics
        
        # Generate graph
        result = self.viz.generate_pattern_graph(
            filters=None,
            layout_algorithm="spring",
            include_metrics=True
        )
        
        # Verify result structure
        self.assertIn('nodes', result)
        self.assertIn('edges', result)
        self.assertIn('clusters', result)
        self.assertIn('metadata', result)
        self.assertIn('metrics', result)
        
        # Verify metadata
        metadata = result['metadata']
        self.assertIn('total_patterns', metadata)
        self.assertIn('generation_time', metadata)
        self.assertIn('timestamp', metadata)
        
        # Should have created nodes
        self.assertGreater(len(result['nodes']), 0)
    
    def test_export_visualization_json(self):
        """Test JSON export functionality."""
        graph_data = {
            'nodes': [{'id': 'export_test', 'name': 'Export Test'}],
            'edges': [],
            'clusters': [],
            'metadata': {'total_patterns': 1}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            success = self.viz.export_visualization(graph_data, temp_path, "json")
            self.assertTrue(success)
            
            # Verify file contents
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertEqual(exported_data['nodes'][0]['id'], 'export_test')
            self.assertEqual(exported_data['metadata']['total_patterns'], 1)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('claude.commands.pattern_visualization.PLOTLY_AVAILABLE', True)
    @patch('claude.commands.pattern_visualization.pyo.plot')
    def test_create_interactive_dashboard(self, mock_plot):
        """Test interactive dashboard creation."""
        mock_plot.return_value = "<div>Mock plotly content</div>"
        
        graph_data = {
            'nodes': [
                {
                    'id': 'dashboard_test',
                    'name': 'Dashboard Test',
                    'position': [0.5, 0.5],
                    'size': 20,
                    'success_rate': 0.8,
                    'usage_count': 10
                }
            ],
            'edges': [],
            'clusters': [],
            'metadata': {
                'total_patterns': 1,
                'total_relationships': 0,
                'total_clusters': 0,
                'timestamp': datetime.now().isoformat(),
                'generation_time': 1.5
            }
        }
        
        html_content = self.viz.create_interactive_dashboard(graph_data)
        
        self.assertIsNotNone(html_content)
        self.assertIn('Pattern Visualization Dashboard', html_content)
        self.assertIn('Mock plotly content', html_content)
    
    @patch('claude.commands.pattern_visualization.PLOTLY_AVAILABLE', False)
    def test_create_interactive_dashboard_no_plotly(self):
        """Test dashboard creation when Plotly is not available."""
        graph_data = {'nodes': [], 'edges': [], 'clusters': []}
        
        html_content = self.viz.create_interactive_dashboard(graph_data)
        
        self.assertIsNone(html_content)
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        metrics = self.viz.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('visualizations_generated', metrics)
        self.assertIn('patterns_analyzed', metrics)
        self.assertIn('cache_size', metrics)
        self.assertIn('plotly_available', metrics)
        self.assertIn('pattern_system_available', metrics)
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        # Add some cache data
        self.viz.visualization_cache['test'] = {'data': 'test'}
        self.viz.pattern_cache['test'] = PatternNode(id='test', name='Test')
        
        # Cleanup
        self.viz.cleanup()
        
        # Verify caches are cleared
        self.assertEqual(len(self.viz.visualization_cache), 0)
        self.assertEqual(len(self.viz.pattern_cache), 0)


class TestPatternVisualizationIntegration(unittest.TestCase):
    """Integration tests for PatternVisualization system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.viz = PatternVisualization()
    
    def test_end_to_end_workflow_minimal(self):
        """Test minimal end-to-end workflow without external dependencies."""
        # Create test patterns manually to avoid dependency on knowledge system
        test_patterns = [
            {
                'pattern_id': 'integration_1',
                'pattern_description': 'Integration Test Pattern 1',
                'domain': 'testing',
                'complexity': 'medium',
                'metrics': {'total_applications': 5, 'success_rate': 0.8}
            },
            {
                'pattern_id': 'integration_2', 
                'pattern_description': 'Integration Test Pattern 2',
                'domain': 'testing',
                'complexity': 'high',
                'related_patterns': ['integration_1'],
                'metrics': {'total_applications': 3, 'success_rate': 0.6}
            }
        ]
        
        # Mock the pattern loading
        with patch.object(self.viz, '_load_patterns_with_metrics', return_value=test_patterns):
            # Generate graph
            result = self.viz.generate_pattern_graph(include_metrics=True)
            
            # Verify complete result
            self.assertIn('nodes', result)
            self.assertIn('edges', result)
            self.assertIn('clusters', result)
            self.assertIn('metadata', result)
            self.assertIn('metrics', result)
            
            # Verify nodes were created
            self.assertEqual(len(result['nodes']), 2)
            
            # Verify metadata is reasonable
            metadata = result['metadata']
            self.assertEqual(metadata['total_patterns'], 2)
            self.assertGreater(metadata['generation_time'], 0)
    
    def test_large_pattern_set_performance(self):
        """Test performance with a larger set of patterns."""
        # Create a larger set of test patterns
        test_patterns = []
        for i in range(50):
            pattern = {
                'pattern_id': f'perf_test_{i}',
                'pattern_description': f'Performance Test Pattern {i}',
                'domain': 'performance' if i % 2 == 0 else 'testing',
                'complexity': ['low', 'medium', 'high'][i % 3],
                'metrics': {
                    'total_applications': i + 1,
                    'success_rate': 0.5 + (i % 5) * 0.1
                }
            }
            if i > 0 and i % 5 == 0:
                pattern['related_patterns'] = [f'perf_test_{i-1}']
            test_patterns.append(pattern)
        
        with patch.object(self.viz, '_load_patterns_with_metrics', return_value=test_patterns):
            start_time = datetime.now()
            result = self.viz.generate_pattern_graph()
            end_time = datetime.now()
            
            # Should complete in reasonable time (less than 10 seconds)
            duration = (end_time - start_time).total_seconds()
            self.assertLess(duration, 10.0)
            
            # Should handle large pattern set
            self.assertEqual(result['metadata']['total_patterns'], 50)
            self.assertGreater(len(result['edges']), 0)  # Should find some relationships


class TestCLIInterface(unittest.TestCase):
    """Test command-line interface functionality."""
    
    @patch('sys.argv')
    @patch('claude.commands.pattern_visualization.PatternVisualization')
    def test_cli_basic_usage(self, mock_viz_class, mock_argv):
        """Test basic CLI usage."""
        mock_argv.__getitem__.side_effect = lambda i: [
            'pattern_visualization.py', 
            '--output', 'test.json',
            '--format', 'json'
        ][i]
        mock_argv.__len__.return_value = 5
        
        mock_viz_instance = Mock()
        mock_viz_class.return_value = mock_viz_instance
        
        mock_viz_instance.generate_pattern_graph.return_value = {
            'nodes': [], 'edges': [], 'clusters': [],
            'metadata': {'total_patterns': 0}
        }
        mock_viz_instance.export_visualization.return_value = True
        
        # Import and test main function
        from claude.commands.pattern_visualization import main
        
        # Should not crash
        try:
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.output = 'test.json'
                mock_args.format = 'json'
                mock_args.layout = 'spring'
                mock_args.filter_domain = None
                mock_args.filter_complexity = None
                mock_args.min_success_rate = None
                mock_args.min_usage = None
                mock_args.interactive = False
                mock_parse.return_value = mock_args
                
                # Should execute without errors
                main()
                
        except SystemExit:
            # CLI calls sys.exit, which is expected
            pass


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.ERROR)  # Reduce test noise
    
    # Run tests
    unittest.main(verbosity=2)