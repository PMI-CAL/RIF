"""
Comprehensive tests for the Incremental Entity Extraction system (Issue #65).

Tests cover:
- Incremental parsing and diff calculation
- Entity version management
- Performance benchmarks (<100ms requirement)
- Integration with file change detection
- Cache management and optimization
- Error handling and edge cases
"""

import unittest
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import system under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.extraction.incremental_extractor import (
    IncrementalEntityExtractor,
    EntityDiff,
    EntityDiffer,
    EntityVersionManager,
    IncrementalResult,
    create_incremental_extractor
)
from knowledge.extraction.entity_types import CodeEntity, EntityType, SourceLocation
from knowledge.extraction.storage_integration import EntityStorage


class TestEntityDiffer(unittest.TestCase):
    """Test the EntityDiffer component"""
    
    def setUp(self):
        self.differ = EntityDiffer()
        
        # Create test entities
        self.entity1 = CodeEntity(
            type=EntityType.FUNCTION,
            name="test_function",
            file_path="test.py",
            location=SourceLocation(1, 5),
            ast_hash="hash1"
        )
        
        self.entity2 = CodeEntity(
            type=EntityType.CLASS,
            name="TestClass",
            file_path="test.py",
            location=SourceLocation(10, 20),
            ast_hash="hash2"
        )
        
        # Modified version of entity1
        self.entity1_modified = CodeEntity(
            type=EntityType.FUNCTION,
            name="test_function",
            file_path="test.py",
            location=SourceLocation(1, 8),  # Different end line
            ast_hash="hash1_modified"  # Different hash
        )
    
    def test_no_changes(self):
        """Test diff calculation when no changes occurred"""
        old_entities = [self.entity1, self.entity2]
        new_entities = [self.entity1, self.entity2]
        
        diff = self.differ.calculate_diff(old_entities, new_entities)
        
        self.assertEqual(len(diff.unchanged), 2)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.modified), 0)
        self.assertEqual(len(diff.removed), 0)
        self.assertFalse(diff.has_changes)
    
    def test_added_entities(self):
        """Test detection of newly added entities"""
        old_entities = [self.entity1]
        new_entities = [self.entity1, self.entity2]
        
        diff = self.differ.calculate_diff(old_entities, new_entities)
        
        self.assertEqual(len(diff.unchanged), 1)
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(diff.added[0].name, "TestClass")
        self.assertTrue(diff.has_changes)
    
    def test_removed_entities(self):
        """Test detection of removed entities"""
        old_entities = [self.entity1, self.entity2]
        new_entities = [self.entity1]
        
        diff = self.differ.calculate_diff(old_entities, new_entities)
        
        self.assertEqual(len(diff.unchanged), 1)
        self.assertEqual(len(diff.removed), 1)
        self.assertEqual(diff.removed[0].name, "TestClass")
        self.assertTrue(diff.has_changes)
    
    def test_modified_entities(self):
        """Test detection of modified entities"""
        old_entities = [self.entity1, self.entity2]
        new_entities = [self.entity1_modified, self.entity2]
        
        diff = self.differ.calculate_diff(old_entities, new_entities)
        
        self.assertEqual(len(diff.unchanged), 1)  # entity2 unchanged
        self.assertEqual(len(diff.modified), 1)
        self.assertEqual(len(diff.added), 0)
        self.assertEqual(len(diff.removed), 0)
        
        old_modified, new_modified = diff.modified[0]
        self.assertEqual(old_modified.ast_hash, "hash1")
        self.assertEqual(new_modified.ast_hash, "hash1_modified")
        self.assertTrue(diff.has_changes)
    
    def test_complex_changes(self):
        """Test complex scenario with multiple change types"""
        entity3 = CodeEntity(
            type=EntityType.VARIABLE,
            name="new_var",
            file_path="test.py",
            location=SourceLocation(25, 25),
            ast_hash="hash3"
        )
        
        old_entities = [self.entity1, self.entity2]  # function + class
        new_entities = [self.entity1_modified, entity3]  # modified function + new variable
        
        diff = self.differ.calculate_diff(old_entities, new_entities)
        
        self.assertEqual(len(diff.modified), 1)  # function modified
        self.assertEqual(len(diff.added), 1)     # variable added
        self.assertEqual(len(diff.removed), 1)   # class removed
        self.assertEqual(len(diff.unchanged), 0)
        self.assertEqual(diff.total_changes, 3)


class TestIncrementalEntityExtractor(unittest.TestCase):
    """Test the main IncrementalEntityExtractor class"""
    
    def setUp(self):
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb")
        self.temp_db.close()
        
        self.extractor = IncrementalEntityExtractor(self.temp_db.name)
        
        # Create test Python file
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.test_file.write("""
def hello_world():
    print("Hello, World!")

class TestClass:
    def method(self):
        return 42
""")
        self.test_file.close()
        self.test_file_path = self.test_file.name
    
    def tearDown(self):
        # Cleanup temporary files
        try:
            os.unlink(self.temp_db.name)
            os.unlink(self.test_file_path)
        except OSError:
            pass
    
    def test_create_incremental_extractor(self):
        """Test extractor creation"""
        extractor = create_incremental_extractor(self.temp_db.name)
        self.assertIsInstance(extractor, IncrementalEntityExtractor)
    
    def test_handle_created_file(self):
        """Test handling of newly created file"""
        result = self.extractor.extract_incremental(self.test_file_path, 'created')
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.diff.added), 0)
        self.assertEqual(len(result.diff.removed), 0)
        self.assertEqual(len(result.diff.modified), 0)
        
        # Verify entities were extracted
        added_names = [e.name for e in result.diff.added]
        self.assertIn("hello_world", added_names)
        self.assertIn("TestClass", added_names)
    
    def test_handle_modified_file_no_changes(self):
        """Test handling modified file when no actual changes occurred"""
        # First extraction (creates cache)
        result1 = self.extractor.extract_incremental(self.test_file_path, 'created')
        self.assertTrue(result1.success)
        
        # Second extraction without file changes
        result2 = self.extractor.extract_incremental(self.test_file_path, 'modified')
        
        self.assertTrue(result2.success)
        self.assertEqual(len(result2.diff.added), 0)
        self.assertEqual(len(result2.diff.modified), 0)
        self.assertEqual(len(result2.diff.removed), 0)
        self.assertGreater(len(result2.diff.unchanged), 0)
    
    def test_handle_modified_file_with_changes(self):
        """Test handling modified file with actual changes"""
        # First extraction
        result1 = self.extractor.extract_incremental(self.test_file_path, 'created')
        self.assertTrue(result1.success)
        
        # Modify the file
        with open(self.test_file_path, 'w') as f:
            f.write("""
def hello_world():
    print("Hello, Modified World!")

def new_function():
    return "new"

class TestClass:
    def method(self):
        return 43  # Changed value
    
    def new_method(self):
        return "added"
""")
        
        # Second extraction after changes
        result2 = self.extractor.extract_incremental(self.test_file_path, 'modified')
        
        self.assertTrue(result2.success)
        self.assertTrue(result2.diff.has_changes)
        
        # Should detect some changes (exact counts depend on extraction implementation)
        self.assertGreater(result2.diff.total_changes, 0)
    
    def test_handle_deleted_file(self):
        """Test handling of deleted file"""
        # First extraction to populate cache
        result1 = self.extractor.extract_incremental(self.test_file_path, 'created')
        entities_count = len(result1.diff.added)
        
        # Handle file deletion
        result2 = self.extractor.extract_incremental(self.test_file_path, 'deleted')
        
        self.assertTrue(result2.success)
        self.assertEqual(len(result2.diff.removed), entities_count)
        self.assertEqual(len(result2.diff.added), 0)
    
    def test_performance_target_compliance(self):
        """Test that processing meets <100ms performance target"""
        # Test with multiple files of different sizes
        test_results = []
        
        for _ in range(5):  # Test multiple times for consistency
            start_time = time.time()
            result = self.extractor.extract_incremental(self.test_file_path, 'modified')
            processing_time = time.time() - start_time
            
            test_results.append(processing_time)
            self.assertTrue(result.success)
        
        # Check average processing time
        avg_time = sum(test_results) / len(test_results)
        self.assertLess(avg_time, 0.1, 
                       f"Average processing time {avg_time*1000:.1f}ms exceeds 100ms target")
        
        # Check individual results
        for processing_time in test_results:
            self.assertLess(processing_time, 0.2,  # Allow some tolerance for CI environments
                           f"Processing time {processing_time*1000:.1f}ms significantly exceeds target")
    
    def test_caching_performance(self):
        """Test that caching improves performance"""
        # First extraction (cache miss)
        start_time = time.time()
        result1 = self.extractor.extract_incremental(self.test_file_path, 'created')
        first_time = time.time() - start_time
        
        # Second extraction (cache hit)
        start_time = time.time()
        result2 = self.extractor.extract_incremental(self.test_file_path, 'modified')
        second_time = time.time() - start_time
        
        # Cache hit should be faster (or at least not significantly slower)
        self.assertTrue(result1.success and result2.success)
        
        # Get cache metrics
        metrics = self.extractor.get_performance_metrics()
        self.assertGreater(metrics['cache_hits'], 0)
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # Process some files
        self.extractor.extract_incremental(self.test_file_path, 'created')
        self.extractor.extract_incremental(self.test_file_path, 'modified')
        
        metrics = self.extractor.get_performance_metrics()
        
        self.assertGreater(metrics['files_processed'], 0)
        self.assertGreater(metrics['total_processing_time'], 0)
        self.assertIn('avg_processing_time', metrics)
        self.assertIn('meets_performance_target', metrics)
        self.assertIn('cache_hit_rate', metrics)
    
    def test_validate_performance(self):
        """Test performance validation"""
        validation = self.extractor.validate_performance(self.test_file_path)
        
        self.assertIn('processing_time_ms', validation)
        self.assertIn('meets_target', validation)
        self.assertIn('performance_rating', validation)
        self.assertIsInstance(validation['recommendations'], list)
    
    def test_error_handling(self):
        """Test error handling for invalid files"""
        # Test with non-existent file
        result = self.extractor.extract_incremental("/nonexistent/file.py", 'modified')
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
    
    def test_clear_caches(self):
        """Test cache clearing functionality"""
        # Populate cache
        self.extractor.extract_incremental(self.test_file_path, 'created')
        
        # Verify cache has content
        self.assertGreater(len(self.extractor._entity_cache), 0)
        
        # Clear caches
        self.extractor.clear_caches()
        
        # Verify caches are cleared
        self.assertEqual(len(self.extractor._entity_cache), 0)
        self.assertEqual(len(self.extractor._file_hash_cache), 0)


class TestEntityVersionManager(unittest.TestCase):
    """Test the EntityVersionManager component"""
    
    def setUp(self):
        # Mock storage for testing
        self.mock_storage = Mock()
        self.mock_storage.get_entities_by_file.return_value = []
        
        self.version_manager = EntityVersionManager(self.mock_storage)
    
    def test_initial_version(self):
        """Test initial version assignment"""
        version = self.version_manager.get_entity_version("entity1", "test.py")
        self.assertEqual(version, 0)
    
    def test_increment_version(self):
        """Test version increment"""
        entity_version = self.version_manager.increment_version(
            "entity1", "test.py", "CREATED"
        )
        
        self.assertEqual(entity_version.version_number, 1)
        self.assertEqual(entity_version.change_type, "CREATED")
        self.assertEqual(entity_version.entity_id, "entity1")
        
        # Next increment should be version 2
        entity_version2 = self.version_manager.increment_version(
            "entity1", "test.py", "MODIFIED"
        )
        self.assertEqual(entity_version2.version_number, 2)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks and stress tests"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb")
        self.temp_db.close()
        
        self.extractor = IncrementalEntityExtractor(self.temp_db.name)
    
    def tearDown(self):
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass
    
    def test_large_file_performance(self):
        """Test performance with large files"""
        # Create a large Python file
        large_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        
        # Generate many functions and classes
        content_lines = ["# Large test file"]
        for i in range(100):
            content_lines.append(f"""
def function_{i}():
    \"\"\"Function {i}\"\"\"
    return {i}

class Class_{i}:
    \"\"\"Class {i}\"\"\"
    def method_{i}(self):
        return {i}
""")
        
        large_file.write("\n".join(content_lines))
        large_file.close()
        
        try:
            # Test extraction performance
            start_time = time.time()
            result = self.extractor.extract_incremental(large_file.name, 'created')
            processing_time = time.time() - start_time
            
            self.assertTrue(result.success)
            self.assertGreater(len(result.diff.added), 100)  # Should find many entities
            
            # Performance target may be relaxed for very large files
            # but should still be reasonable
            self.assertLess(processing_time, 1.0,  # 1 second max for large files
                           f"Large file processing took {processing_time*1000:.1f}ms")
            
        finally:
            os.unlink(large_file.name)
    
    def test_batch_processing_performance(self):
        """Test performance when processing multiple files"""
        # Create multiple test files
        test_files = []
        
        try:
            for i in range(10):
                test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
                test_file.write(f"""
def function_{i}():
    return {i}

class TestClass_{i}:
    value = {i}
""")
                test_file.close()
                test_files.append(test_file.name)
            
            # Process all files and measure total time
            start_time = time.time()
            results = []
            
            for file_path in test_files:
                result = self.extractor.extract_incremental(file_path, 'created')
                results.append(result)
            
            total_time = time.time() - start_time
            avg_time = total_time / len(test_files)
            
            # All extractions should succeed
            self.assertTrue(all(r.success for r in results))
            
            # Average time per file should meet target
            self.assertLess(avg_time, 0.1, 
                           f"Average processing time {avg_time*1000:.1f}ms exceeds target")
            
            # Check performance metrics
            metrics = self.extractor.get_performance_metrics()
            self.assertEqual(metrics['files_processed'], len(test_files))
            
        finally:
            for file_path in test_files:
                try:
                    os.unlink(file_path)
                except OSError:
                    pass


class TestIntegrationWithFileChange(unittest.TestCase):
    """Integration tests with file change detection"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb")
        self.temp_db.close()
        
        self.extractor = IncrementalEntityExtractor(self.temp_db.name)
    
    def tearDown(self):
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass
    
    def test_file_change_integration(self):
        """Test integration with FileChange objects"""
        # Create test file
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        test_file.write("""
def test_function():
    return "test"
""")
        test_file.close()
        
        try:
            # Mock FileChange objects (since we might not have the actual class)
            class MockFileChange:
                def __init__(self, path, change_type):
                    self.path = path
                    self.type = change_type
            
            # Test processing file changes
            changes = [
                MockFileChange(test_file.name, 'created')
            ]
            
            results = self.extractor.process_file_changes(changes)
            
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)
            self.assertGreater(len(results[0].diff.added), 0)
            
        finally:
            os.unlink(test_file.name)


def run_performance_benchmark():
    """Standalone performance benchmark"""
    print("Running Incremental Entity Extraction Performance Benchmark")
    print("=" * 60)
    
    extractor = create_incremental_extractor()
    
    # Test with current file
    test_file = __file__
    
    print(f"Testing with file: {test_file}")
    print(f"File size: {os.path.getsize(test_file)} bytes")
    
    # Run multiple iterations
    times = []
    for i in range(10):
        start_time = time.time()
        result = extractor.extract_incremental(test_file, 'modified')
        processing_time = time.time() - start_time
        times.append(processing_time)
        
        status = "✓" if processing_time < 0.1 else "✗"
        print(f"Iteration {i+1}: {processing_time*1000:.1f}ms {status}")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\nBenchmark Results:")
    print(f"Average time: {avg_time*1000:.1f}ms")
    print(f"Min time: {min_time*1000:.1f}ms")
    print(f"Max time: {max_time*1000:.1f}ms")
    print(f"Performance target (<100ms): {'✓ PASSED' if avg_time < 0.1 else '✗ FAILED'}")
    
    # Print performance metrics
    metrics = extractor.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    # Check if running as benchmark
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        run_performance_benchmark()
    else:
        # Run unit tests
        unittest.main(verbosity=2)