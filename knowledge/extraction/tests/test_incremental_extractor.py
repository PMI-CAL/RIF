"""
Comprehensive tests for the Incremental Entity Extraction system (Issue #65).

This module provides thorough testing coverage for all components of the incremental
extraction system including:
- IncrementalEntityExtractor functionality
- EntityDiffer algorithms
- EntityVersionManager operations
- Performance validation
- Integration with file change detection
"""

import unittest
import tempfile
import os
import time
import json
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, MagicMock

# Import the system under test
from ..incremental_extractor import (
    IncrementalEntityExtractor,
    EntityDiffer,
    EntityVersionManager,
    EntityDiff,
    EntityVersion,
    IncrementalResult,
    create_incremental_extractor
)
from ..entity_types import EntityType, CodeEntity, SourceLocation
from ..storage_integration import EntityStorage


class TestEntityDiff(unittest.TestCase):
    """Test the EntityDiff data structure and functionality."""

    def test_entity_diff_creation(self):
        """Test creating EntityDiff with different change types."""
        # Create test entities
        entity1 = CodeEntity(type=EntityType.FUNCTION, name="func1", file_path="/test.py")
        entity2 = CodeEntity(type=EntityType.CLASS, name="Class1", file_path="/test.py")
        entity3 = CodeEntity(type=EntityType.FUNCTION, name="func2", file_path="/test.py")
        entity4 = CodeEntity(type=EntityType.FUNCTION, name="func1_modified", file_path="/test.py")

        diff = EntityDiff(
            added=[entity1, entity2],
            modified=[(entity3, entity4)],
            removed=[CodeEntity(type=EntityType.VARIABLE, name="old_var", file_path="/test.py")],
            unchanged=[CodeEntity(type=EntityType.MODULE, name="import1", file_path="/test.py")]
        )

        # Test properties
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.total_changes, 4)  # 2 added + 1 modified + 1 removed
        self.assertEqual(len(diff.added), 2)
        self.assertEqual(len(diff.modified), 1)
        self.assertEqual(len(diff.removed), 1)
        self.assertEqual(len(diff.unchanged), 1)

    def test_empty_entity_diff(self):
        """Test EntityDiff with no changes."""
        diff = EntityDiff()
        
        self.assertFalse(diff.has_changes)
        self.assertEqual(diff.total_changes, 0)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.modified), 0)
        self.assertEqual(len(diff.removed), 0)
        self.assertEqual(len(diff.unchanged), 0)


class TestEntityVersion(unittest.TestCase):
    """Test entity version tracking functionality."""

    def test_entity_version_creation(self):
        """Test creating EntityVersion instances."""
        from datetime import datetime
        
        timestamp = datetime.now()
        version = EntityVersion(
            entity_id="test_func_123",
            version_number=5,
            timestamp=timestamp,
            change_type="MODIFIED",
            ast_hash="abc123def456",
            metadata={"complexity": "high"}
        )

        self.assertEqual(version.entity_id, "test_func_123")
        self.assertEqual(version.version_number, 5)
        self.assertEqual(version.timestamp, timestamp)
        self.assertEqual(version.change_type, "MODIFIED")
        self.assertEqual(version.ast_hash, "abc123def456")
        self.assertEqual(version.metadata["complexity"], "high")


class TestEntityDiffer(unittest.TestCase):
    """Test the EntityDiffer component for calculating entity differences."""

    def setUp(self):
        """Set up test fixtures."""
        self.differ = EntityDiffer()

    def create_test_entity(self, name: str, entity_type: EntityType = EntityType.FUNCTION,
                          location_line: int = 1, content_suffix: str = "") -> CodeEntity:
        """Create a test entity with specified parameters."""
        entity = CodeEntity(
            type=entity_type,
            name=name,
            file_path="/test/file.py",
            location=SourceLocation(line_start=location_line, line_end=location_line + 5),
            metadata={"content": f"def {name}(): pass{content_suffix}"}
        )
        # Manually set hash to make testing predictable
        entity.ast_hash = f"hash_{name}_{content_suffix}"
        return entity

    def test_identical_entities(self):
        """Test diff calculation for identical entity lists."""
        entities = [
            self.create_test_entity("func1"),
            self.create_test_entity("func2"),
            self.create_test_entity("func3")
        ]

        diff = self.differ.calculate_diff(entities, entities)

        self.assertEqual(len(diff.unchanged), 3)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.modified), 0)
        self.assertEqual(len(diff.removed), 0)
        self.assertFalse(diff.has_changes)

    def test_added_entities(self):
        """Test diff calculation for added entities."""
        old_entities = [
            self.create_test_entity("func1"),
            self.create_test_entity("func2")
        ]

        new_entities = old_entities + [
            self.create_test_entity("func3"),
            self.create_test_entity("func4")
        ]

        diff = self.differ.calculate_diff(old_entities, new_entities)

        self.assertEqual(len(diff.unchanged), 2)
        self.assertEqual(len(diff.added), 2)
        self.assertEqual(len(diff.modified), 0)
        self.assertEqual(len(diff.removed), 0)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.total_changes, 2)

    def test_removed_entities(self):
        """Test diff calculation for removed entities."""
        old_entities = [
            self.create_test_entity("func1"),
            self.create_test_entity("func2"),
            self.create_test_entity("func3")
        ]

        new_entities = [old_entities[0]]  # Only keep first entity

        diff = self.differ.calculate_diff(old_entities, new_entities)

        self.assertEqual(len(diff.unchanged), 1)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.modified), 0)
        self.assertEqual(len(diff.removed), 2)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.total_changes, 2)

    def test_modified_entities(self):
        """Test diff calculation for modified entities."""
        # Create old entity
        old_entity = self.create_test_entity("func1", content_suffix="")
        
        # Create modified version with same signature but different hash
        new_entity = self.create_test_entity("func1", content_suffix="_modified")

        old_entities = [old_entity]
        new_entities = [new_entity]

        diff = self.differ.calculate_diff(old_entities, new_entities)

        self.assertEqual(len(diff.unchanged), 0)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.modified), 1)
        self.assertEqual(len(diff.removed), 0)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.total_changes, 1)

        # Check that modified tuple contains both old and new
        old_modified, new_modified = diff.modified[0]
        self.assertEqual(old_modified.name, "func1")
        self.assertEqual(new_modified.name, "func1")
        self.assertNotEqual(old_modified.ast_hash, new_modified.ast_hash)

    def test_complex_diff_scenario(self):
        """Test complex scenario with all types of changes."""
        # Old entities
        old_entities = [
            self.create_test_entity("unchanged_func"),
            self.create_test_entity("modified_func"),
            self.create_test_entity("removed_func"),
        ]

        # New entities
        new_entities = [
            old_entities[0],  # unchanged
            self.create_test_entity("modified_func", content_suffix="_modified"),  # modified
            self.create_test_entity("added_func1"),  # added
            self.create_test_entity("added_func2"),  # added
        ]

        diff = self.differ.calculate_diff(old_entities, new_entities)

        self.assertEqual(len(diff.unchanged), 1)
        self.assertEqual(len(diff.added), 2)
        self.assertEqual(len(diff.modified), 1)
        self.assertEqual(len(diff.removed), 1)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.total_changes, 4)

    def test_signature_map_creation(self):
        """Test the internal signature mapping functionality."""
        entities = [
            self.create_test_entity("func1", EntityType.FUNCTION, location_line=10),
            self.create_test_entity("func2", EntityType.FUNCTION, location_line=20),
            CodeEntity(type=EntityType.CLASS, name="TestClass", file_path="/test/file.py",
                      location=SourceLocation(line_start=30, line_end=40))
        ]

        signature_map = self.differ._create_signature_map(entities)

        self.assertEqual(len(signature_map), 3)
        
        # Check signature format: type:name:file_path:line_start
        expected_signatures = [
            "function:func1:/test/file.py:10",
            "function:func2:/test/file.py:20", 
            "class:TestClass:/test/file.py:30"
        ]
        
        for sig in expected_signatures:
            self.assertIn(sig, signature_map)


class TestEntityVersionManager(unittest.TestCase):
    """Test entity version management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_versions.duckdb")
        
        # Mock storage for version manager
        self.mock_storage = Mock(spec=EntityStorage)
        self.mock_storage.get_entities_by_file.return_value = []
        
        self.version_manager = EntityVersionManager(self.mock_storage)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initial_version_tracking(self):
        """Test initial version assignment for new entities."""
        entity_id = "test_func_123"
        file_path = "/test/file.py"

        # First version should be 0 (no existing versions)
        initial_version = self.version_manager.get_entity_version(entity_id, file_path)
        self.assertEqual(initial_version, 0)

        # Increment version
        version = self.version_manager.increment_version(entity_id, file_path, "CREATED")
        
        self.assertEqual(version.entity_id, entity_id)
        self.assertEqual(version.version_number, 1)
        self.assertEqual(version.change_type, "CREATED")
        
        # Check that version is cached
        cached_version = self.version_manager.get_entity_version(entity_id, file_path)
        self.assertEqual(cached_version, 1)

    def test_version_increment_sequence(self):
        """Test sequential version increments."""
        entity_id = "evolving_entity"
        file_path = "/test/evolving.py"

        # Create sequence of version increments
        version1 = self.version_manager.increment_version(entity_id, file_path, "CREATED")
        version2 = self.version_manager.increment_version(entity_id, file_path, "MODIFIED")
        version3 = self.version_manager.increment_version(entity_id, file_path, "MODIFIED")

        self.assertEqual(version1.version_number, 1)
        self.assertEqual(version2.version_number, 2)
        self.assertEqual(version3.version_number, 3)

        self.assertEqual(version1.change_type, "CREATED")
        self.assertEqual(version2.change_type, "MODIFIED")
        self.assertEqual(version3.change_type, "MODIFIED")

    def test_multiple_entities_same_file(self):
        """Test version tracking for multiple entities in same file."""
        file_path = "/test/multi.py"
        
        # Track versions for different entities
        entity1_id = "func1"
        entity2_id = "func2"
        
        version1_1 = self.version_manager.increment_version(entity1_id, file_path, "CREATED")
        version2_1 = self.version_manager.increment_version(entity2_id, file_path, "CREATED")
        version1_2 = self.version_manager.increment_version(entity1_id, file_path, "MODIFIED")

        self.assertEqual(version1_1.version_number, 1)
        self.assertEqual(version2_1.version_number, 1)
        self.assertEqual(version1_2.version_number, 2)

        # Check final versions
        self.assertEqual(self.version_manager.get_entity_version(entity1_id, file_path), 2)
        self.assertEqual(self.version_manager.get_entity_version(entity2_id, file_path), 1)


class TestIncrementalResult(unittest.TestCase):
    """Test the IncrementalResult data structure."""

    def test_incremental_result_creation(self):
        """Test creating IncrementalResult with performance metrics."""
        entity1 = CodeEntity(type=EntityType.FUNCTION, name="func1", file_path="/test.py")
        entity2 = CodeEntity(type=EntityType.CLASS, name="Class1", file_path="/test.py")
        
        diff = EntityDiff(
            added=[entity1],
            modified=[(entity2, entity2)],
            removed=[],
            unchanged=[entity2]
        )

        result = IncrementalResult(
            file_path="/test/file.py",
            processing_time=0.045,  # 45ms
            diff=diff,
            version_info={"added": 1, "modified": 1, "removed": 0},
            success=True
        )

        # Test basic properties
        self.assertEqual(result.file_path, "/test/file.py")
        self.assertEqual(result.processing_time, 0.045)
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)

        # Test performance metrics
        metrics = result.performance_metrics
        self.assertEqual(metrics['processing_time_ms'], 45.0)
        self.assertEqual(metrics['entities_added'], 1)
        self.assertEqual(metrics['entities_modified'], 1)
        self.assertEqual(metrics['entities_removed'], 0)
        self.assertEqual(metrics['entities_unchanged'], 1)
        self.assertEqual(metrics['total_changes'], 2)
        self.assertTrue(metrics['meets_performance_target'])  # <100ms

    def test_failed_result(self):
        """Test IncrementalResult for failed extraction."""
        result = IncrementalResult(
            file_path="/test/failed.py",
            processing_time=0.150,  # 150ms
            diff=EntityDiff(),
            success=False,
            error_message="Syntax error in file"
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Syntax error in file")
        
        metrics = result.performance_metrics
        self.assertEqual(metrics['processing_time_ms'], 150.0)
        self.assertFalse(metrics['meets_performance_target'])  # >100ms


class TestIncrementalEntityExtractor(unittest.TestCase):
    """Test the main IncrementalEntityExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_incremental.duckdb")
        
        # Create test files
        self.test_file_path = os.path.join(self.temp_dir, "test_file.py")
        self.create_test_file("def hello(): return 'world'")
        
        # Mock the storage to avoid database dependencies in unit tests
        with patch('knowledge.extraction.incremental_extractor.EntityStorage'):
            self.extractor = IncrementalEntityExtractor(self.db_path)

    def tearDown(self):
        """Clean up test fixtures.""" 
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_file(self, content: str, file_path: str = None):
        """Create a test file with given content."""
        if file_path is None:
            file_path = self.test_file_path
            
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    def test_extract_incremental_modified_file(self):
        """Test incremental extraction for modified file."""
        # Mock the entity extractor's extract_from_file method directly
        test_entity = CodeEntity(type=EntityType.FUNCTION, name="hello", file_path=self.test_file_path)
        mock_extraction_result = Mock()
        mock_extraction_result.success = True
        mock_extraction_result.entities = [test_entity]
        
        # Mock cached entities (empty for new file) - this simulates no previous entities
        self.extractor._entity_cache[self.test_file_path] = []
        
        with patch.object(self.extractor.entity_extractor, 'extract_from_file', return_value=mock_extraction_result):
            # Test incremental extraction
            result = self.extractor.extract_incremental(self.test_file_path, 'modified')
            
            # Verify results
            self.assertTrue(result.success)
            self.assertEqual(result.file_path, self.test_file_path)
            self.assertIsNotNone(result.diff)
            # Since we had no cached entities and found one new entity, it should be added
            self.assertEqual(len(result.diff.added), 1)
            self.assertLess(result.processing_time, 1.0)  # Should be very fast for test

    def test_extract_incremental_created_file(self):
        """Test incremental extraction for newly created file."""
        with patch.object(self.extractor.entity_extractor, 'extract_from_file') as mock_extract:
            # Mock successful extraction
            test_entity = CodeEntity(type=EntityType.FUNCTION, name="new_func", file_path=self.test_file_path)
            mock_result = Mock()
            mock_result.success = True
            mock_result.entities = [test_entity]
            mock_extract.return_value = mock_result

            # Test created file processing
            result = self.extractor.extract_incremental(self.test_file_path, 'created')

            self.assertTrue(result.success)
            self.assertEqual(len(result.diff.added), 1)
            self.assertEqual(len(result.diff.modified), 0)
            self.assertEqual(len(result.diff.removed), 0)

    def test_extract_incremental_deleted_file(self):
        """Test incremental extraction for deleted file."""
        # Setup cached entities for the file
        cached_entities = [
            CodeEntity(type=EntityType.FUNCTION, name="deleted_func", file_path=self.test_file_path),
            CodeEntity(type=EntityType.CLASS, name="DeletedClass", file_path=self.test_file_path)
        ]
        self.extractor._entity_cache[self.test_file_path] = cached_entities

        # Test deleted file processing
        result = self.extractor.extract_incremental(self.test_file_path, 'deleted')

        self.assertTrue(result.success)
        self.assertEqual(len(result.diff.added), 0)
        self.assertEqual(len(result.diff.modified), 0)
        self.assertEqual(len(result.diff.removed), 2)

        # Check that cache is cleared
        self.assertNotIn(self.test_file_path, self.extractor._entity_cache)

    def test_file_change_detection(self):
        """Test hash-based file change detection."""
        # Initial file content
        initial_content = "def initial(): pass"
        self.create_test_file(initial_content)
        
        # Check initial change detection (should be True since no hash cached)
        self.assertTrue(self.extractor._has_file_changed(self.test_file_path))
        
        # After checking once, should be cached
        self.assertFalse(self.extractor._has_file_changed(self.test_file_path))
        
        # Modify file content
        modified_content = "def modified(): pass"
        self.create_test_file(modified_content)
        
        # Should detect change
        self.assertTrue(self.extractor._has_file_changed(self.test_file_path))

    def test_entity_caching(self):
        """Test entity caching functionality."""
        # Mock storage to return test entities
        test_entities = [
            CodeEntity(type=EntityType.FUNCTION, name="cached_func", file_path=self.test_file_path)
        ]
        self.extractor.storage.get_entities_by_file.return_value = test_entities

        # First call should hit storage
        entities = self.extractor.get_cached_entities(self.test_file_path)
        self.assertEqual(len(entities), 1)
        self.assertEqual(self.extractor.performance_metrics['cache_misses'], 1)
        
        # Second call should hit cache
        entities = self.extractor.get_cached_entities(self.test_file_path)
        self.assertEqual(len(entities), 1)
        self.assertEqual(self.extractor.performance_metrics['cache_hits'], 1)

    def test_performance_metrics_tracking(self):
        """Test performance metrics collection."""
        # Reset metrics
        self.extractor.reset_metrics()
        
        initial_metrics = self.extractor.get_performance_metrics()
        self.assertEqual(initial_metrics['files_processed'], 0)
        self.assertEqual(initial_metrics['total_processing_time'], 0.0)
        
        # Simulate processing
        self.extractor._update_metrics(0.05, 10)  # 50ms, 10 entities
        self.extractor._update_metrics(0.03, 5)   # 30ms, 5 entities
        
        metrics = self.extractor.get_performance_metrics()
        self.assertEqual(metrics['files_processed'], 2)
        self.assertEqual(metrics['total_processing_time'], 0.08)
        self.assertEqual(metrics['avg_processing_time'], 0.04)
        self.assertEqual(metrics['avg_entities_per_file'], 7.5)
        self.assertTrue(metrics['meets_performance_target'])  # <100ms average

    def test_performance_validation(self):
        """Test performance validation functionality."""
        with patch.object(self.extractor, 'extract_incremental') as mock_extract:
            # Mock a fast result
            mock_result = IncrementalResult(
                file_path=self.test_file_path,
                processing_time=0.025,  # 25ms
                diff=EntityDiff(added=[CodeEntity(type=EntityType.FUNCTION, name="fast_func", file_path=self.test_file_path)])
            )
            mock_extract.return_value = mock_result
            
            validation = self.extractor.validate_performance(self.test_file_path)
            
            self.assertTrue(validation['meets_target'])
            self.assertEqual(validation['performance_rating'], 'excellent')
            self.assertEqual(validation['processing_time_ms'], 25.0)
            self.assertEqual(len(validation['recommendations']), 0)

    def test_performance_validation_slow(self):
        """Test performance validation for slow processing."""
        with patch.object(self.extractor, 'extract_incremental') as mock_extract:
            # Mock a slow result
            mock_result = IncrementalResult(
                file_path=self.test_file_path,
                processing_time=0.120,  # 120ms
                diff=EntityDiff(added=[CodeEntity(type=EntityType.FUNCTION, name="slow_func", file_path=self.test_file_path)])
            )
            mock_extract.return_value = mock_result
            
            validation = self.extractor.validate_performance(self.test_file_path)
            
            self.assertFalse(validation['meets_target'])
            self.assertEqual(validation['performance_rating'], 'needs_improvement')
            self.assertEqual(validation['processing_time_ms'], 120.0)
            self.assertGreater(len(validation['recommendations']), 0)

    def test_batch_file_changes_processing(self):
        """Test processing multiple file changes in batch."""
        # Mock FileChange objects
        mock_changes = []
        for i in range(3):
            mock_change = Mock()
            mock_change.path = os.path.join(self.temp_dir, f"file{i}.py")
            mock_change.type = 'modified'
            mock_changes.append(mock_change)
            
            # Create the test file
            self.create_test_file(f"def func{i}(): pass", mock_change.path)

        with patch.object(self.extractor, 'extract_incremental') as mock_extract:
            # Mock results for each file
            mock_results = []
            for i, change in enumerate(mock_changes):
                result = IncrementalResult(
                    file_path=change.path,
                    processing_time=0.030 + i * 0.01,  # Vary processing times
                    diff=EntityDiff()
                )
                mock_results.append(result)
            
            mock_extract.side_effect = mock_results
            
            # Process batch
            results = self.extractor.process_file_changes(mock_changes)
            
            self.assertEqual(len(results), 3)
            self.assertEqual(mock_extract.call_count, 3)
            
            # Check that all files were processed
            for i, result in enumerate(results):
                self.assertEqual(result.file_path, os.path.join(self.temp_dir, f"file{i}.py"))


class TestCreateIncrementalExtractor(unittest.TestCase):
    """Test the convenience factory function."""

    def test_create_incremental_extractor_default(self):
        """Test creating extractor with default parameters."""
        with patch('knowledge.extraction.incremental_extractor.IncrementalEntityExtractor') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            extractor = create_incremental_extractor()
            
            # Verify constructor called with default path
            mock_class.assert_called_once_with("knowledge/chromadb/entities.duckdb")
            self.assertEqual(extractor, mock_instance)

    def test_create_incremental_extractor_custom_path(self):
        """Test creating extractor with custom storage path."""
        custom_path = "/custom/path/entities.duckdb"
        
        with patch('knowledge.extraction.incremental_extractor.IncrementalEntityExtractor') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            extractor = create_incremental_extractor(custom_path)
            
            # Verify constructor called with custom path
            mock_class.assert_called_once_with(custom_path)
            self.assertEqual(extractor, mock_instance)


class TestIncrementalExtractionIntegration(unittest.TestCase):
    """Integration tests for the complete incremental extraction system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.duckdb")

    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @unittest.skipIf(not os.getenv('RUN_INTEGRATION_TESTS'), 
                     "Integration tests require RUN_INTEGRATION_TESTS environment variable")
    def test_end_to_end_incremental_extraction(self):
        """Full end-to-end test of incremental extraction."""
        # Create real extractor (not mocked)
        extractor = create_incremental_extractor(self.db_path)
        
        # Create initial Python file
        test_file = os.path.join(self.temp_dir, "evolving_module.py")
        initial_content = """
def hello_world():
    return "Hello, World!"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
"""
        
        with open(test_file, 'w') as f:
            f.write(initial_content)
        
        # First extraction (created file)
        result1 = extractor.extract_incremental(test_file, 'created')
        
        # Verify initial extraction
        self.assertTrue(result1.success)
        self.assertGreater(len(result1.diff.added), 0)
        self.assertEqual(len(result1.diff.modified), 0)
        self.assertEqual(len(result1.diff.removed), 0)
        
        # Modify the file
        modified_content = """
def hello_world():
    return "Hello, World!"

def hello_universe():
    return "Hello, Universe!"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
    
    def farewell(self, name):
        return f"Goodbye, {name}!"
"""
        
        with open(test_file, 'w') as f:
            f.write(modified_content)
        
        # Second extraction (modified file)
        result2 = extractor.extract_incremental(test_file, 'modified')
        
        # Verify incremental extraction detected changes
        self.assertTrue(result2.success)
        
        # Should have detected additions (new function and method)
        # Note: Exact counts depend on extraction implementation
        self.assertGreater(result2.diff.total_changes, 0)
        
        # Verify performance
        self.assertLess(result1.processing_time, 1.0)  # Should be reasonably fast
        self.assertLess(result2.processing_time, 1.0)
        
        # Check metrics
        metrics = extractor.get_performance_metrics()
        self.assertEqual(metrics['files_processed'], 2)
        self.assertGreater(metrics['total_entities_processed'], 0)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for incremental extraction."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up performance test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_large_python_file(self, function_count: int = 100) -> str:
        """Create a large Python file with many functions for performance testing."""
        file_path = os.path.join(self.temp_dir, f"large_file_{function_count}.py")
        
        content = "# Large Python file for performance testing\n\n"
        content += "import os\nimport sys\nfrom typing import List, Dict\n\n"
        
        for i in range(function_count):
            content += f"""
def function_{i}(param1, param2={i}):
    '''Function {i} documentation.'''
    result = param1 + param2
    if result > {i * 10}:
        return result * 2
    return result

"""

        # Add a few classes too
        for i in range(function_count // 10):
            content += f"""
class TestClass{i}:
    '''Test class {i}.'''
    
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self):
        return self.value * {i}

"""

        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path

    @unittest.skipIf(not os.getenv('RUN_PERFORMANCE_TESTS'), 
                     "Performance tests require RUN_PERFORMANCE_TESTS environment variable")
    def test_performance_target_compliance(self):
        """Test that incremental extraction meets <100ms performance target."""
        extractor = create_incremental_extractor()
        
        # Test with different file sizes
        test_sizes = [10, 50, 100]  # Number of functions
        
        for size in test_sizes:
            with self.subTest(function_count=size):
                test_file = self.create_large_python_file(size)
                
                # Measure processing time
                start_time = time.time()
                result = extractor.extract_incremental(test_file, 'created')
                processing_time = time.time() - start_time
                
                # Verify performance target
                self.assertTrue(result.success, f"Extraction failed for {size} functions")
                self.assertLess(
                    processing_time, 
                    0.1,  # 100ms target
                    f"Processing time {processing_time*1000:.1f}ms exceeds 100ms target for {size} functions"
                )
                
                # Also check the result's internal timing
                self.assertLess(
                    result.processing_time, 
                    0.1,
                    f"Result processing time {result.processing_time*1000:.1f}ms exceeds 100ms target"
                )

    @unittest.skipIf(not os.getenv('RUN_PERFORMANCE_TESTS'), 
                     "Performance tests require RUN_PERFORMANCE_TESTS environment variable")  
    def test_cache_performance(self):
        """Test that caching provides significant performance improvements."""
        extractor = create_incremental_extractor()
        test_file = self.create_large_python_file(50)
        
        # First extraction (cache miss)
        result1 = extractor.extract_incremental(test_file, 'created')
        first_time = result1.processing_time
        
        # Second extraction without changes (should be much faster due to caching)
        result2 = extractor.extract_incremental(test_file, 'modified')
        second_time = result2.processing_time
        
        # Cache should make second extraction significantly faster
        self.assertLess(second_time, first_time / 2, 
                       f"Cache did not improve performance: {first_time*1000:.1f}ms -> {second_time*1000:.1f}ms")
        
        # Check cache hit metrics
        metrics = extractor.get_performance_metrics()
        self.assertGreater(metrics.get('cache_hits', 0), 0)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests
    
    # Set test environment variables for optional tests
    # os.environ['RUN_INTEGRATION_TESTS'] = '1'
    # os.environ['RUN_PERFORMANCE_TESTS'] = '1'
    
    # Run the tests
    unittest.main(verbosity=2)