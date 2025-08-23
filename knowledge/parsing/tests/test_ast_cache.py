"""
Tests for AST cache system with LRU eviction and memory management.

These tests verify cache functionality, file change detection,
memory limits, and performance characteristics.
"""

import unittest
import tempfile
import os
import time
from pathlib import Path

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge.parsing import (
    ASTCache,
    ASTCacheEntry,
    ParserManager,
    CacheError
)


class TestASTCacheEntry(unittest.TestCase):
    """Test individual cache entry functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test file
        self.test_file = os.path.join(self.temp_dir, "test.js")
        with open(self.test_file, 'w') as f:
            f.write("function hello() { console.log('Hello'); }")
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cache_entry_creation(self):
        """Test creation of cache entry."""
        # Mock parse result
        parse_result = {
            'file_path': self.test_file,
            'language': 'javascript',
            'tree': None,  # Mock tree
            'source_size': 100,
            'parse_time': 0.1
        }
        
        entry = ASTCacheEntry(self.test_file, 'javascript', None, parse_result)
        
        # Verify basic properties
        self.assertEqual(entry.file_path, self.test_file)
        self.assertEqual(entry.language, 'javascript')
        self.assertEqual(entry.parse_result, parse_result)
        self.assertGreater(entry.created_at, 0)
        self.assertGreater(entry.accessed_at, 0)
        self.assertEqual(entry.access_count, 1)
        
        # Verify file metadata
        self.assertGreater(entry.file_size, 0)
        self.assertGreater(entry.file_mtime, 0)
        self.assertNotEqual(entry.file_hash, "")
    
    def test_cache_entry_validity_check(self):
        """Test file change detection."""
        parse_result = {'test': 'data'}
        entry = ASTCacheEntry(self.test_file, 'javascript', None, parse_result)
        
        # Should be valid initially
        self.assertTrue(entry.is_valid())
        
        # Modify file
        time.sleep(0.1)  # Ensure different mtime
        with open(self.test_file, 'w') as f:
            f.write("function hello() { console.log('Modified'); }")
        
        # Should now be invalid
        self.assertFalse(entry.is_valid())
    
    def test_cache_entry_touch(self):
        """Test entry access tracking."""
        parse_result = {'test': 'data'}
        entry = ASTCacheEntry(self.test_file, 'javascript', None, parse_result)
        
        original_access_time = entry.accessed_at
        original_count = entry.access_count
        
        time.sleep(0.01)
        entry.touch()
        
        self.assertGreater(entry.accessed_at, original_access_time)
        self.assertEqual(entry.access_count, original_count + 1)
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        parse_result = {
            'file_path': self.test_file,
            'language': 'javascript',
            'data': 'test' * 100  # Some data to estimate
        }
        
        entry = ASTCacheEntry(self.test_file, 'javascript', None, parse_result)
        memory_usage = entry.estimate_memory_usage()
        
        # Should be reasonable estimate
        self.assertGreater(memory_usage, 100)  # At least some overhead
        self.assertLess(memory_usage, 1_000_000)  # Not unreasonably large


class TestASTCache(unittest.TestCase):
    """Test AST cache functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache = ASTCache(max_entries=5, max_memory_mb=1)  # Small for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test files
        self.test_files = {}
        for i, lang in enumerate(['javascript', 'python', 'go']):
            filename = f"test{i}.{lang[:2] if lang != 'javascript' else 'js'}"
            file_path = os.path.join(self.temp_dir, filename)
            content = f"// Test {lang} file {i}\nfunction test() {{}}"
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            self.test_files[lang] = file_path
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_cache_basic_operations(self):
        """Test basic cache put/get operations."""
        file_path = self.test_files['javascript']
        language = 'javascript'
        
        # Mock parse result
        parse_result = {
            'file_path': file_path,
            'language': language,
            'tree': None,
            'source_size': 100
        }
        
        # Initially empty
        result = self.cache.get(file_path, language)
        self.assertIsNone(result)
        
        # Put and get
        success = self.cache.put(file_path, language, None, parse_result)
        self.assertTrue(success)
        
        retrieved = self.cache.get(file_path, language)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['language'], language)
        self.assertEqual(retrieved['file_path'], file_path)
    
    def test_cache_metrics(self):
        """Test cache metrics tracking."""
        file_path = self.test_files['python']
        language = 'python'
        parse_result = {'file_path': file_path, 'language': language}
        
        # Initial metrics
        metrics = self.cache.get_metrics()
        self.assertEqual(metrics['hits'], 0)
        self.assertEqual(metrics['misses'], 0)
        self.assertEqual(metrics['current_entries'], 0)
        
        # Miss
        self.cache.get(file_path, language)
        metrics = self.cache.get_metrics()
        self.assertEqual(metrics['misses'], 1)
        
        # Put and hit
        self.cache.put(file_path, language, None, parse_result)
        self.cache.get(file_path, language)
        
        metrics = self.cache.get_metrics()
        self.assertEqual(metrics['hits'], 1)
        self.assertEqual(metrics['current_entries'], 1)
        self.assertGreater(metrics['hit_rate'], 0)
    
    def test_lru_eviction(self):
        """Test LRU eviction when max entries exceeded."""
        # Fill cache to capacity
        for i, (lang, file_path) in enumerate(self.test_files.items()):
            parse_result = {
                'file_path': file_path,
                'language': lang,
                'data': f'entry_{i}'
            }
            self.cache.put(file_path, lang, None, parse_result)
        
        # Add more files to trigger eviction
        for i in range(3, 7):  # Will exceed max_entries of 5
            extra_file = os.path.join(self.temp_dir, f"extra{i}.js")
            with open(extra_file, 'w') as f:
                f.write(f"// Extra file {i}")
            
            parse_result = {
                'file_path': extra_file,
                'language': 'javascript',
                'data': f'extra_{i}'
            }
            self.cache.put(extra_file, 'javascript', None, parse_result)
        
        # Should not exceed max entries
        metrics = self.cache.get_metrics()
        self.assertLessEqual(metrics['current_entries'], 5)
        self.assertGreater(metrics['evictions'], 0)
    
    def test_file_change_invalidation(self):
        """Test that modified files are invalidated."""
        file_path = self.test_files['go']
        language = 'go'
        parse_result = {'file_path': file_path, 'language': language}
        
        # Cache the file
        self.cache.put(file_path, language, None, parse_result)
        
        # Should be cached
        result = self.cache.get(file_path, language)
        self.assertIsNotNone(result)
        
        # Modify file
        time.sleep(0.1)  # Ensure different mtime
        with open(file_path, 'w') as f:
            f.write("// Modified content")
        
        # Should be invalidated (cache miss)
        result = self.cache.get(file_path, language)
        self.assertIsNone(result)
        
        metrics = self.cache.get_metrics()
        self.assertGreater(metrics['invalidations'], 0)
    
    def test_cache_invalidation_methods(self):
        """Test explicit cache invalidation."""
        file_path = self.test_files['javascript']
        
        # Cache multiple languages for same file
        for lang in ['javascript']:  # Only one supported in test
            parse_result = {'file_path': file_path, 'language': lang}
            self.cache.put(file_path, lang, None, parse_result)
        
        # Verify cached
        self.assertIsNotNone(self.cache.get(file_path, 'javascript'))
        
        # Invalidate specific language
        self.cache.invalidate(file_path, 'javascript')
        self.assertIsNone(self.cache.get(file_path, 'javascript'))
        
        # Cache again and invalidate all languages for file
        self.cache.put(file_path, 'javascript', None, {'test': 'data'})
        self.cache.invalidate(file_path)  # All languages
        self.assertIsNone(self.cache.get(file_path, 'javascript'))
    
    def test_cache_clear(self):
        """Test clearing entire cache."""
        # Add some entries
        for lang, file_path in self.test_files.items():
            parse_result = {'file_path': file_path, 'language': lang}
            self.cache.put(file_path, lang, None, parse_result)
        
        # Verify entries exist
        metrics = self.cache.get_metrics()
        self.assertGreater(metrics['current_entries'], 0)
        
        # Clear cache
        self.cache.clear()
        
        # Verify empty
        metrics = self.cache.get_metrics()
        self.assertEqual(metrics['current_entries'], 0)
    
    def test_cleanup_invalid_entries(self):
        """Test cleanup of invalid entries."""
        # Cache some files
        cached_files = []
        for i, (lang, file_path) in enumerate(self.test_files.items()):
            parse_result = {'file_path': file_path, 'language': lang}
            self.cache.put(file_path, lang, None, parse_result)
            cached_files.append(file_path)
        
        # Modify some files to make them invalid
        time.sleep(0.1)
        for file_path in cached_files[:2]:  # Modify first 2 files
            with open(file_path, 'w') as f:
                f.write("// Modified to invalidate cache")
        
        # Cleanup invalid entries
        removed_count = self.cache.cleanup_invalid()
        
        self.assertGreater(removed_count, 0)
        self.assertLessEqual(removed_count, 2)  # Should remove the modified files
    
    def test_cache_info(self):
        """Test detailed cache information."""
        file_path = self.test_files['python']
        language = 'python'
        parse_result = {'file_path': file_path, 'language': language}
        
        self.cache.put(file_path, language, None, parse_result)
        
        cache_info = self.cache.get_cache_info()
        
        self.assertEqual(len(cache_info), 1)
        
        entry_info = cache_info[0]
        self.assertEqual(entry_info['file_path'], file_path)
        self.assertEqual(entry_info['language'], language)
        self.assertIn('created_at', entry_info)
        self.assertIn('access_count', entry_info)
        self.assertIn('estimated_memory_mb', entry_info)
        self.assertTrue(entry_info['valid'])


class TestParserManagerCacheIntegration(unittest.TestCase):
    """Test ParserManager integration with cache system."""
    
    def setUp(self):
        """Set up test environment."""
        self.parser_manager = ParserManager.get_instance()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test file
        self.js_file = os.path.join(self.temp_dir, "cache_test.js")
        with open(self.js_file, 'w') as f:
            f.write("function hello() { console.log('Cache test'); }")
        
        # Clear cache to start fresh
        self.parser_manager.clear_cache()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
        self.parser_manager.reset_metrics()
        self.parser_manager.clear_cache()
    
    def test_cache_integration_basic(self):
        """Test basic cache integration in ParserManager."""
        # First parse - should miss cache
        result1 = self.parser_manager.parse_file(self.js_file)
        self.assertFalse(result1.get('cache_hit', False))
        
        # Second parse - should hit cache
        result2 = self.parser_manager.parse_file(self.js_file)
        self.assertTrue(result2.get('cache_hit', False))
        
        # Results should be equivalent (except cache_hit flag and parse_time)
        for key in ['file_path', 'language', 'source_size']:
            self.assertEqual(result1[key], result2[key])
    
    def test_cache_performance_improvement(self):
        """Test that cache improves performance."""
        # First parse (uncached)
        start_time = time.time()
        result1 = self.parser_manager.parse_file(self.js_file)
        uncached_time = time.time() - start_time
        
        # Second parse (cached)
        start_time = time.time()
        result2 = self.parser_manager.parse_file(self.js_file)
        cached_time = time.time() - start_time
        
        # Cached should be faster (though both are very fast)
        self.assertTrue(result2.get('cache_hit', False))
        self.assertLessEqual(cached_time, uncached_time + 0.01)  # Allow small margin
    
    def test_cache_metrics_integration(self):
        """Test cache metrics in ParserManager."""
        # Parse a file to populate cache
        self.parser_manager.parse_file(self.js_file)
        self.parser_manager.parse_file(self.js_file)  # Second time for cache hit
        
        metrics = self.parser_manager.get_metrics()
        
        # Should have cache metrics
        self.assertIn('cache', metrics)
        cache_metrics = metrics['cache']
        
        self.assertGreater(cache_metrics['hits'], 0)
        self.assertGreater(cache_metrics['total_requests'], 0)
        self.assertGreater(cache_metrics['hit_rate'], 0)
    
    def test_cache_invalidation_integration(self):
        """Test cache invalidation through ParserManager."""
        # Cache a file
        self.parser_manager.parse_file(self.js_file)
        
        # Verify it's cached
        result = self.parser_manager.parse_file(self.js_file)
        self.assertTrue(result.get('cache_hit', False))
        
        # Invalidate cache
        self.parser_manager.invalidate_cache(self.js_file)
        
        # Should miss cache now
        result = self.parser_manager.parse_file(self.js_file)
        self.assertFalse(result.get('cache_hit', False))
    
    def test_cache_disabled_option(self):
        """Test parsing with cache disabled."""
        # Parse with cache disabled
        result1 = self.parser_manager.parse_file(self.js_file, use_cache=False)
        result2 = self.parser_manager.parse_file(self.js_file, use_cache=False)
        
        # Both should be cache misses
        self.assertFalse(result1.get('cache_hit', False))
        self.assertFalse(result2.get('cache_hit', False))
        
        # Cache should still be empty
        cache_info = self.parser_manager.get_cache_info()
        self.assertEqual(len(cache_info), 0)
    
    def test_cache_cleanup_integration(self):
        """Test cache cleanup through ParserManager."""
        # Cache a file
        self.parser_manager.parse_file(self.js_file)
        
        # Verify cached
        cache_info = self.parser_manager.get_cache_info()
        self.assertEqual(len(cache_info), 1)
        
        # Modify file to make cache invalid
        time.sleep(0.1)
        with open(self.js_file, 'w') as f:
            f.write("function hello() { console.log('Modified for cleanup test'); }")
        
        # Cleanup invalid entries
        removed_count = self.parser_manager.cleanup_invalid_cache()
        self.assertGreater(removed_count, 0)
        
        # Cache should be empty now
        cache_info = self.parser_manager.get_cache_info()
        self.assertEqual(len(cache_info), 0)


if __name__ == '__main__':
    unittest.main()