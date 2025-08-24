#!/usr/bin/env python3
"""
Comprehensive tests for Shadow Mode functionality.
Tests the parallel processing, comparison framework, and adapter integration.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'claude' / 'commands'))
sys.path.append(str(project_root / 'lightrag'))

try:
    from shadow_mode import (
        ShadowModeProcessor, LegacyKnowledgeSystem, OperationResult,
        ComparisonResult, get_shadow_processor
    )
    from shadow_comparison import (
        AdvancedComparator, ComparisonLogger, ContentDifference,
        ComparisonReport, create_comparison_framework
    )
    from knowledge_adapter import KnowledgeAdapter, get_knowledge_adapter
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestShadowModeProcessor(unittest.TestCase):
    """Test the main shadow mode processor."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.knowledge_path = Path(self.test_dir) / "knowledge"
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Create test config
        self.config = {
            'shadow_mode': {'enabled': True},
            'systems': {
                'legacy': {'enabled': True, 'paths': {}},
                'lightrag': {'enabled': False, 'config': {'knowledge_path': str(self.knowledge_path)}}
            },
            'parallel_processing': {
                'enabled': True,
                'timeout_ms': 5000,
                'max_concurrent_operations': 2,
                'primary_system': 'legacy',
                'shadow_system': 'lightrag'
            },
            'operations': {
                'store_knowledge': {'enabled': True, 'compare_results': True},
                'retrieve_knowledge': {'enabled': True, 'compare_results': True}
            },
            'logging': {'enabled': True, 'log_level': 'INFO'},
            'comparison': {
                'content_similarity_threshold': 0.9,
                'timing_variance_threshold': 2.0
            }
        }
        
        # Create test config file
        self.config_file = Path(self.test_dir) / "shadow-mode.yaml"
        with open(self.config_file, 'w') as f:
            import yaml
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        with patch('shadow_mode.ShadowModeProcessor._load_config', return_value=self.config):
            processor = ShadowModeProcessor()
            
            self.assertTrue(processor.is_enabled())
            self.assertIsNotNone(processor.legacy_system)
            self.assertEqual(processor.config['shadow_mode']['enabled'], True)
    
    def test_legacy_system_store(self):
        """Test legacy system store operation."""
        legacy_system = LegacyKnowledgeSystem(str(self.knowledge_path))
        
        result = legacy_system.store_knowledge(
            collection="patterns",
            content="Test pattern content",
            metadata={"test": True, "complexity": "low"}
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        self.assertGreater(result.duration_ms, 0)
        
        # Verify file was created
        patterns_dir = self.knowledge_path / "patterns"
        self.assertTrue(patterns_dir.exists())
        
        json_files = list(patterns_dir.glob("*.json"))
        self.assertGreater(len(json_files), 0)
    
    def test_legacy_system_retrieve(self):
        """Test legacy system retrieve operation."""
        legacy_system = LegacyKnowledgeSystem(str(self.knowledge_path))
        
        # First store some test data
        legacy_system.store_knowledge(
            "patterns",
            "Machine learning optimization pattern",
            {"complexity": "high", "tags": "ml,optimization"}
        )
        
        legacy_system.store_knowledge(
            "patterns", 
            "Database indexing pattern",
            {"complexity": "medium", "tags": "database,performance"}
        )
        
        # Test retrieval
        result = legacy_system.retrieve_knowledge("optimization")
        
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, list)
        self.assertGreater(len(result.data), 0)
        
        # Check first result contains our content
        first_result = result.data[0]
        self.assertIn("optimization", first_result["content"].lower())
        self.assertEqual(first_result["collection"], "patterns")
    
    def test_shadow_mode_disabled_fallback(self):
        """Test fallback behavior when shadow mode is disabled."""
        disabled_config = self.config.copy()
        disabled_config['shadow_mode']['enabled'] = False
        
        with patch('shadow_mode.ShadowModeProcessor._load_config', return_value=disabled_config):
            processor = ShadowModeProcessor()
            
            self.assertFalse(processor.is_enabled())
            
            # Should fall back to primary system only
            result = processor.store_knowledge(
                "patterns",
                "Test pattern",
                {"test": True}
            )
            
            # Should work even with shadow mode disabled
            self.assertIsNotNone(result)


class TestAdvancedComparator(unittest.TestCase):
    """Test the advanced comparison framework."""
    
    def setUp(self):
        """Set up test environment."""
        self.comparator = AdvancedComparator()
    
    def test_store_results_comparison(self):
        """Test comparison of store operation results."""
        primary_result = OperationResult(
            success=True,
            data="doc123",
            duration_ms=50.0,
            metadata={"system": "primary"}
        )
        
        shadow_result = OperationResult(
            success=True,
            data="doc456", 
            duration_ms=75.0,
            metadata={"system": "shadow"}
        )
        
        report = self.comparator.compare_store_results(primary_result, shadow_result)
        
        self.assertEqual(report.operation, "store_knowledge")
        self.assertIsInstance(report.differences, list)
        self.assertIsInstance(report.overall_similarity, float)
        self.assertGreaterEqual(report.overall_similarity, 0.0)
        self.assertLessEqual(report.overall_similarity, 1.0)
        
        # Should note the different document IDs
        id_differences = [d for d in report.differences if d.field == "document_id"]
        self.assertGreater(len(id_differences), 0)
    
    def test_retrieve_results_comparison(self):
        """Test comparison of retrieve operation results."""
        primary_results = [
            {
                "id": "doc1",
                "content": "Machine learning patterns for optimization",
                "metadata": {"complexity": "high"},
                "distance": 0.2
            },
            {
                "id": "doc2", 
                "content": "Database indexing strategies",
                "metadata": {"complexity": "medium"},
                "distance": 0.4
            }
        ]
        
        shadow_results = [
            {
                "id": "doc1",
                "content": "Machine learning patterns for optimization",
                "metadata": {"complexity": "high"},
                "distance": 0.15
            },
            {
                "id": "doc3",
                "content": "Performance optimization techniques", 
                "metadata": {"complexity": "medium"},
                "distance": 0.35
            }
        ]
        
        primary_result = OperationResult(
            success=True,
            data=primary_results,
            duration_ms=100.0
        )
        
        shadow_result = OperationResult(
            success=True,
            data=shadow_results,
            duration_ms=80.0
        )
        
        report = self.comparator.compare_retrieve_results(primary_result, shadow_result)
        
        self.assertEqual(report.operation, "retrieve_knowledge")
        self.assertIsInstance(report.differences, list)
        
        # Should detect the different second result
        unique_differences = [d for d in report.differences if "unique" in d.field]
        self.assertGreater(len(unique_differences), 0)
    
    def test_text_similarity_calculation(self):
        """Test text similarity calculation."""
        text1 = "This is a test of machine learning optimization patterns"
        text2 = "This is a test of machine learning optimization techniques"
        text3 = "Completely different content about databases"
        
        # Similar texts should have high similarity
        similarity1 = self.comparator._calculate_text_similarity(text1, text2)
        self.assertGreater(similarity1, 0.7)
        
        # Different texts should have low similarity
        similarity2 = self.comparator._calculate_text_similarity(text1, text3)
        self.assertLess(similarity2, 0.5)
        
        # Identical texts should have perfect similarity
        similarity3 = self.comparator._calculate_text_similarity(text1, text1)
        self.assertEqual(similarity3, 1.0)
    
    def test_metadata_similarity_calculation(self):
        """Test metadata similarity calculation."""
        meta1 = {"complexity": "high", "tags": "ml,optimization", "source": "test"}
        meta2 = {"complexity": "high", "tags": "ml,performance", "source": "test"}
        meta3 = {"type": "pattern", "status": "active"}
        
        # Similar metadata should have high similarity
        similarity1 = self.comparator._calculate_metadata_similarity(meta1, meta2)
        self.assertGreater(similarity1, 0.6)
        
        # Different metadata should have low similarity
        similarity2 = self.comparator._calculate_metadata_similarity(meta1, meta3)
        self.assertLess(similarity2, 0.5)
    
    def test_performance_analysis(self):
        """Test performance analysis."""
        fast_result = OperationResult(success=True, data="test", duration_ms=50.0)
        slow_result = OperationResult(success=True, data="test", duration_ms=200.0)
        
        analysis = self.comparator._analyze_performance(fast_result, slow_result)
        
        self.assertEqual(analysis['faster_system'], 'primary')
        self.assertEqual(analysis['ratio'], 4.0)  # 200/50
        self.assertTrue(analysis['significant_difference'])
        self.assertEqual(analysis['performance_category'], 'high')


class TestComparisonLogger(unittest.TestCase):
    """Test the comparison logging functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        config = {
            'logging': {
                'reports_dir': os.path.join(self.test_dir, 'reports')
            }
        }
        self.logger = ComparisonLogger(config)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_report_logging(self):
        """Test logging of comparison reports."""
        # Create a test report
        differences = [
            ContentDifference(
                type='content',
                field='test_field',
                primary_value='value1',
                shadow_value='value2',
                severity='medium',
                description='Test difference',
                impact_score=0.5
            )
        ]
        
        report = ComparisonReport(
            operation='test_operation',
            timestamp='2023-01-01T00:00:00Z',
            primary_system='primary',
            shadow_system='shadow',
            overall_similarity=0.8,
            differences=differences,
            performance_analysis={'ratio': 1.2, 'significant_difference': False},
            recommendations=['Test recommendation'],
            metadata={'test': True}
        )
        
        # Log the report
        self.logger.log_comparison(report)
        
        # Check that files were created
        reports_dir = Path(self.test_dir) / 'reports'
        self.assertTrue(reports_dir.exists())
        
        # Check for JSON report file
        json_files = list(reports_dir.glob('comparison_*.json'))
        self.assertGreater(len(json_files), 0)
        
        # Check for metrics file
        metrics_file = reports_dir / 'metrics.jsonl'
        self.assertTrue(metrics_file.exists())


class TestKnowledgeAdapter(unittest.TestCase):
    """Test the knowledge adapter integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.knowledge_path = Path(self.test_dir) / "knowledge"
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Mock the adapter to use our test directory
        with patch.object(KnowledgeAdapter, '__init__', lambda x, enable_shadow_mode=None: None):
            self.adapter = KnowledgeAdapter()
            self.adapter.logger = Mock()
            self.adapter.shadow_mode_enabled = False
            self.adapter.shadow_processor = None
            self.adapter.knowledge_path = self.knowledge_path
            self.adapter.collection_map = {
                'patterns': 'patterns',
                'decisions': 'decisions',
                'code_snippets': 'learning',
                'issue_resolutions': 'issues'
            }
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_store_pattern(self):
        """Test storing a pattern through the adapter."""
        pattern_data = {
            "title": "Test Pattern",
            "description": "A test pattern for unit testing",
            "complexity": "low",
            "source": "unit_test",
            "tags": ["test", "pattern"]
        }
        
        pattern_id = self.adapter.store_pattern(pattern_data)
        
        self.assertIsNotNone(pattern_id)
        
        # Verify file was created
        patterns_dir = self.knowledge_path / "patterns"
        self.assertTrue(patterns_dir.exists())
        
        json_files = list(patterns_dir.glob("*.json"))
        self.assertGreater(len(json_files), 0)
        
        # Verify content
        with open(json_files[0]) as f:
            stored_data = json.load(f)
        
        self.assertEqual(stored_data["collection"], "patterns")
        self.assertIn("Test Pattern", stored_data["content"])
    
    def test_search_patterns(self):
        """Test searching patterns through the adapter."""
        # Store some test patterns
        pattern1 = {
            "title": "Machine Learning Pattern",
            "description": "Pattern for ML optimization",
            "complexity": "high"
        }
        
        pattern2 = {
            "title": "Database Pattern", 
            "description": "Pattern for database indexing",
            "complexity": "medium"
        }
        
        self.adapter.store_pattern(pattern1)
        self.adapter.store_pattern(pattern2)
        
        # Search for patterns
        results = self.adapter.search_patterns("machine learning")
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Should find the ML pattern
        found_ml = any("machine learning" in result["content"].lower() for result in results)
        self.assertTrue(found_ml)
    
    def test_system_status(self):
        """Test getting system status."""
        status = self.adapter.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("adapter_initialized", status)
        self.assertIn("shadow_mode_enabled", status)
        self.assertIn("collections_available", status)
        
        self.assertTrue(status["adapter_initialized"])
        self.assertFalse(status["shadow_mode_enabled"])
    
    @patch('knowledge_adapter.get_shadow_processor')
    def test_shadow_mode_integration(self, mock_get_processor):
        """Test adapter integration with shadow mode."""
        # Mock shadow processor
        mock_processor = Mock()
        mock_processor.is_enabled.return_value = True
        mock_get_processor.return_value = mock_processor
        
        # Create adapter with shadow mode enabled
        with patch.object(KnowledgeAdapter, '__init__', lambda x, enable_shadow_mode=None: None):
            adapter = KnowledgeAdapter()
            adapter.shadow_mode_enabled = True
            adapter.shadow_processor = mock_processor
            adapter.logger = Mock()
        
        # Test store operation
        test_content = "Test content"
        test_metadata = {"test": True}
        
        with patch('knowledge_adapter.shadow_store_knowledge') as mock_shadow_store:
            mock_shadow_store.return_value = "shadow_doc_id"
            
            result = adapter.store_knowledge("patterns", test_content, test_metadata)
            
            # Should call shadow mode function
            mock_shadow_store.assert_called_once_with("patterns", test_content, test_metadata, None)
            self.assertEqual(result, "shadow_doc_id")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete shadow mode system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.knowledge_path = Path(self.test_dir) / "knowledge"
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end shadow mode workflow."""
        # Create test configuration
        config = {
            'shadow_mode': {'enabled': True},
            'systems': {
                'legacy': {'enabled': True},
                'lightrag': {'enabled': False}  # Disabled for this test
            },
            'parallel_processing': {
                'timeout_ms': 2000,
                'max_concurrent_operations': 2
            },
            'operations': {
                'store_knowledge': {'enabled': True},
                'retrieve_knowledge': {'enabled': True}
            },
            'logging': {'enabled': True}
        }
        
        # Test the workflow
        with patch('shadow_mode.ShadowModeProcessor._load_config', return_value=config):
            # Initialize components
            comparator, logger = create_comparison_framework(config)
            
            # Test comparison functionality
            primary_result = OperationResult(success=True, data="test", duration_ms=50.0)
            shadow_result = OperationResult(success=True, data="test", duration_ms=60.0)
            
            report = comparator.compare_store_results(primary_result, shadow_result)
            
            self.assertIsInstance(report, ComparisonReport)
            self.assertEqual(report.operation, "store_knowledge")
            self.assertGreaterEqual(report.overall_similarity, 0.8)  # Should be very similar
    
    def test_performance_under_load(self):
        """Test shadow mode performance under load."""
        legacy_system = LegacyKnowledgeSystem(str(self.knowledge_path))
        
        # Store multiple documents
        start_time = time.time()
        
        for i in range(10):
            result = legacy_system.store_knowledge(
                "patterns",
                f"Test pattern {i} with machine learning optimization techniques",
                {"test_id": i, "complexity": "medium"}
            )
            self.assertTrue(result.success)
        
        store_time = time.time() - start_time
        
        # Retrieve multiple queries
        start_time = time.time()
        
        for query in ["machine learning", "optimization", "test pattern"]:
            result = legacy_system.retrieve_knowledge(query, n_results=5)
            self.assertTrue(result.success)
            self.assertGreater(len(result.data), 0)
        
        retrieve_time = time.time() - start_time
        
        # Performance should be reasonable
        self.assertLess(store_time, 5.0)  # 5 seconds max for 10 stores
        self.assertLess(retrieve_time, 2.0)  # 2 seconds max for 3 queries
        
        print(f"Performance test: Store={store_time:.2f}s, Retrieve={retrieve_time:.2f}s")


def run_tests():
    """Run all shadow mode tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestShadowModeProcessor))
    suite.addTest(unittest.makeSuite(TestAdvancedComparator))
    suite.addTest(unittest.makeSuite(TestComparisonLogger))
    suite.addTest(unittest.makeSuite(TestKnowledgeAdapter))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Shadow Mode Tests...")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)