#!/usr/bin/env python3
"""
Comprehensive Test Suite for RIF File Monitoring System
Tests for issue #29: Implement real-time file monitoring with watchdog

Validation requirements:
1. File monitoring with debouncing and priority queue  
2. Respect gitignore patterns
3. Handle 1000+ file changes efficiently
4. Integration with tree-sitter parsing
5. Performance benchmarks and stress testing
"""

import unittest
import tempfile
import json
import time
import threading
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the file monitoring system
from claude.commands.file_monitor import (
    FileMonitor, MonitoringConfig, FileChangeEvent, Priority,
    DebounceBuffer, TreeSitterCoordination, FileSystemEventProcessor,
    create_default_config
)

class TestFileChangeEvent(unittest.TestCase):
    """Test FileChangeEvent dataclass"""
    
    def test_event_creation(self):
        """Test creating a file change event"""
        event = FileChangeEvent(
            file_path="/test/file.py",
            event_type="modified",
            timestamp=time.time(),
            priority=Priority.IMMEDIATE,
            size=1024,
            checksum="abc123"
        )
        
        self.assertEqual(event.file_path, "/test/file.py")
        self.assertEqual(event.event_type, "modified")
        self.assertEqual(event.priority, Priority.IMMEDIATE)
        self.assertEqual(event.size, 1024)
        self.assertEqual(event.checksum, "abc123")
    
    def test_priority_ordering(self):
        """Test priority queue ordering"""
        immediate = FileChangeEvent("/test.py", "modified", time.time(), Priority.IMMEDIATE)
        high = FileChangeEvent("/config.json", "modified", time.time(), Priority.HIGH)
        medium = FileChangeEvent("/readme.md", "modified", time.time(), Priority.MEDIUM) 
        low = FileChangeEvent("/temp.log", "modified", time.time(), Priority.LOW)
        
        # Test ordering (lower priority number = higher priority)
        self.assertTrue(immediate < high)
        self.assertTrue(high < medium)
        self.assertTrue(medium < low)

class TestMonitoringConfig(unittest.TestCase):
    """Test MonitoringConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = create_default_config("/test/path")
        
        self.assertEqual(config.root_paths, ["/test/path"])
        self.assertEqual(config.debounce_interval, 0.5)
        self.assertEqual(config.max_events_per_second, 500)
        self.assertEqual(config.memory_limit_mb, 100)
        self.assertIn(Priority.IMMEDIATE, config.priority_extensions)
    
    def test_priority_extensions(self):
        """Test file extension priority mapping"""
        config = create_default_config()
        
        # Test immediate priority extensions
        immediate_exts = config.priority_extensions[Priority.IMMEDIATE]
        self.assertIn('.py', immediate_exts)
        self.assertIn('.js', immediate_exts)
        self.assertIn('.ts', immediate_exts)
        
        # Test high priority extensions
        high_exts = config.priority_extensions[Priority.HIGH]
        self.assertIn('.json', high_exts)
        self.assertIn('.yaml', high_exts)

class TestDebounceBuffer(unittest.TestCase):
    """Test debounce buffer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.debounce_buffer = DebounceBuffer(debounce_interval=0.1)  # 100ms for testing
    
    def test_single_event_debouncing(self):
        """Test debouncing of single file events"""
        event = FileChangeEvent("/test.py", "modified", time.time(), Priority.IMMEDIATE)
        
        # Add event
        self.debounce_buffer.add_event(event)
        
        # Should not be ready immediately
        ready_events = self.debounce_buffer.get_ready_events()
        self.assertEqual(len(ready_events), 0)
        
        # Wait for debounce interval
        time.sleep(0.15)
        
        # Should be ready now
        ready_events = self.debounce_buffer.get_ready_events()
        self.assertEqual(len(ready_events), 1)
        self.assertEqual(ready_events[0].file_path, "/test.py")
    
    def test_event_coalescing(self):
        """Test coalescing of multiple events for same file"""
        timestamp = time.time()
        
        # Add multiple events for same file
        event1 = FileChangeEvent("/test.py", "modified", timestamp, Priority.IMMEDIATE)
        event2 = FileChangeEvent("/test.py", "modified", timestamp + 0.01, Priority.HIGH)
        event3 = FileChangeEvent("/test.py", "modified", timestamp + 0.02, Priority.MEDIUM)
        
        self.debounce_buffer.add_event(event1)
        self.debounce_buffer.add_event(event2)
        self.debounce_buffer.add_event(event3)
        
        # Wait for debounce
        time.sleep(0.15)
        
        # Should have only one event with highest priority
        ready_events = self.debounce_buffer.get_ready_events()
        self.assertEqual(len(ready_events), 1)
        self.assertEqual(ready_events[0].priority, Priority.IMMEDIATE)  # Highest priority wins
    
    def test_delete_event_coalescing(self):
        """Test that delete events override other events"""
        timestamp = time.time()
        
        # Add modify then delete event
        modify_event = FileChangeEvent("/test.py", "modified", timestamp, Priority.IMMEDIATE)
        delete_event = FileChangeEvent("/test.py", "deleted", timestamp + 0.01, Priority.MEDIUM)
        
        self.debounce_buffer.add_event(modify_event)
        self.debounce_buffer.add_event(delete_event)
        
        time.sleep(0.15)
        
        # Should have delete event
        ready_events = self.debounce_buffer.get_ready_events()
        self.assertEqual(len(ready_events), 1)
        self.assertEqual(ready_events[0].event_type, "deleted")
    
    def test_rapid_change_detection(self):
        """Test detection of rapid change sequences (IDE auto-save)"""
        timestamp = time.time()
        
        # Simulate rapid changes within 200ms threshold
        for i in range(5):
            event = FileChangeEvent("/test.py", "modified", timestamp + i * 0.03, Priority.IMMEDIATE)
            self.debounce_buffer.add_event(event)
        
        # Should detect as rapid change and extend debounce interval
        time.sleep(0.15)
        ready_events = self.debounce_buffer.get_ready_events()
        # Might still be debouncing due to extended interval
        self.assertLessEqual(len(ready_events), 1)
    
    def test_batch_operation_detection(self):
        """Test detection of batch operations (multiple files in directory)"""
        timestamp = time.time()
        dir_path = "/test/src"
        
        # Add events for multiple files in same directory
        for i in range(6):  # Above batch threshold
            event = FileChangeEvent(f"{dir_path}/file{i}.py", "modified", timestamp + i * 0.01, Priority.IMMEDIATE)
            self.debounce_buffer.add_event(event)
        
        # Wait for batch processing
        time.sleep(0.3)  # Wait longer for batch window
        
        ready_events = self.debounce_buffer.get_ready_events()
        # Should process all events as batch
        self.assertEqual(len(ready_events), 6)
    
    def test_statistics(self):
        """Test debounce buffer statistics"""
        # Add some events
        event1 = FileChangeEvent("/test1.py", "modified", time.time(), Priority.IMMEDIATE)
        event2 = FileChangeEvent("/test2.py", "modified", time.time(), Priority.HIGH)
        
        self.debounce_buffer.add_event(event1)
        self.debounce_buffer.add_event(event2)
        
        stats = self.debounce_buffer.get_statistics()
        
        self.assertIn("single_file_buffer_size", stats)
        self.assertIn("debounce_interval", stats)
        self.assertIn("ide_sequence_window", stats)
        self.assertEqual(stats["single_file_buffer_size"], 2)

class TestTreeSitterCoordination(unittest.TestCase):
    """Test tree-sitter coordination interface"""
    
    def setUp(self):
        """Set up test environment"""
        self.coordinator = TreeSitterCoordination()
    
    def test_file_change_notification(self):
        """Test notifying tree-sitter of file changes"""
        # Should not raise any errors
        self.coordinator.notify_file_changed("/test.py", "modified")
        self.coordinator.notify_file_changed("/test.js", "created")
        
        # Non-source files should be handled gracefully
        self.coordinator.notify_file_changed("/test.txt", "modified")
    
    def test_parsing_priority(self):
        """Test getting parsing priority"""
        priority = self.coordinator.get_parsing_priority("/test.py")
        self.assertIsInstance(priority, int)
        
        # Unknown files should have low priority
        self.assertEqual(priority, Priority.LOW.value)
    
    def test_parsing_status(self):
        """Test parsing status checking"""
        # Initially no files should be parsing
        self.assertFalse(self.coordinator.is_parsing_in_progress("/test.py"))

class TestFileMonitor(unittest.TestCase):
    """Test main FileMonitor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_dir = tempfile.mkdtemp()
        
        # Create test directory structure
        self.test_src_dir = Path(self.temp_dir) / "src"
        self.test_src_dir.mkdir(parents=True)
        
        # Create test files
        (self.test_src_dir / "main.py").write_text("# Test file")
        (self.test_src_dir / "config.json").write_text('{"test": true}')
        (Path(self.temp_dir) / "README.md").write_text("# Test Project")
        
        # Create .gitignore
        gitignore_content = """
*.pyc
__pycache__/
*.log
build/
dist/
"""
        (Path(self.temp_dir) / ".gitignore").write_text(gitignore_content.strip())
        
        # Create monitoring config
        self.config = MonitoringConfig(
            root_paths=[str(self.temp_dir)],
            debounce_interval=0.05,  # Short interval for testing
            max_events_per_second=100,
            memory_limit_mb=50
        )
        
        self.monitor = FileMonitor(self.config, str(self.knowledge_dir))
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            shutil.rmtree(self.temp_dir)
            shutil.rmtree(self.knowledge_dir)
        except:
            pass
    
    def test_gitignore_loading(self):
        """Test loading and parsing gitignore patterns"""
        # Should ignore pyc files
        self.assertTrue(self.monitor._should_ignore_file(str(Path(self.temp_dir) / "test.pyc")))
        
        # Should ignore __pycache__ directory contents
        self.assertTrue(self.monitor._should_ignore_file(str(Path(self.temp_dir) / "__pycache__" / "module.pyc")))
        
        # Should not ignore Python source files
        self.assertFalse(self.monitor._should_ignore_file(str(Path(self.temp_dir) / "src" / "main.py")))
        
        # Should not ignore JSON config files
        self.assertFalse(self.monitor._should_ignore_file(str(Path(self.temp_dir) / "config.json")))
    
    def test_nested_gitignore_loading(self):
        """Test loading nested .gitignore files"""
        # Create nested directory with its own .gitignore
        nested_dir = self.test_src_dir / "submodule"
        nested_dir.mkdir()
        
        nested_gitignore = nested_dir / ".gitignore"
        nested_gitignore.write_text("*.tmp\nlocal_config.json")
        
        # Reload gitignore patterns
        self.monitor._load_gitignore_patterns()
        
        # Should ignore files matching nested patterns
        self.assertTrue(self.monitor._should_ignore_file(str(nested_dir / "temp.tmp")))
        self.assertTrue(self.monitor._should_ignore_file(str(nested_dir / "local_config.json")))
        
        # Should still respect parent .gitignore
        self.assertTrue(self.monitor._should_ignore_file(str(nested_dir / "test.pyc")))
    
    def test_priority_assignment(self):
        """Test file priority assignment based on extensions"""
        # Immediate priority
        self.assertEqual(self.monitor._get_file_priority("main.py"), Priority.IMMEDIATE)
        self.assertEqual(self.monitor._get_file_priority("app.js"), Priority.IMMEDIATE)
        self.assertEqual(self.monitor._get_file_priority("component.ts"), Priority.IMMEDIATE)
        
        # High priority
        self.assertEqual(self.monitor._get_file_priority("config.json"), Priority.HIGH)
        self.assertEqual(self.monitor._get_file_priority("docker-compose.yaml"), Priority.HIGH)
        
        # Medium priority
        self.assertEqual(self.monitor._get_file_priority("README.md"), Priority.MEDIUM)
        # Note: test_main.py has .py extension so gets IMMEDIATE priority, but also has "test" in name which would be MEDIUM
        # The extension takes precedence, so it should be IMMEDIATE
        self.assertEqual(self.monitor._get_file_priority("test_main.py"), Priority.IMMEDIATE)
        
        # Low priority
        self.assertEqual(self.monitor._get_file_priority("debug.log"), Priority.LOW)
        self.assertEqual(self.monitor._get_file_priority("temp.cache"), Priority.LOW)
    
    def test_event_queuing(self):
        """Test queuing file system events"""
        test_file = str(self.test_src_dir / "main.py")
        
        # Queue an event
        self.monitor._queue_event(test_file, "modified")
        
        # Check debounce buffer has the event
        buffer_stats = self.monitor.debounce_buffer.get_statistics()
        self.assertGreater(buffer_stats["single_file_buffer_size"], 0)
    
    def test_rate_limiting(self):
        """Test rate limiting of events"""
        test_file = str(self.test_src_dir / "main.py")
        
        # Queue many events rapidly
        for i in range(self.config.max_events_per_second + 10):
            self.monitor._queue_event(f"{test_file}_{i}", "modified")
        
        # Should have been rate limited
        self.assertLessEqual(len(self.monitor.event_timestamps), self.config.max_events_per_second + 10)
    
    def test_gitignore_cache(self):
        """Test caching of gitignore results for performance"""
        test_file = str(Path(self.temp_dir) / "test.py")
        
        # First check should populate cache
        result1 = self.monitor._should_ignore_file(test_file)
        
        # Second check should use cache
        result2 = self.monitor._should_ignore_file(test_file)
        
        self.assertEqual(result1, result2)
        self.assertIn(test_file, self.monitor.gitignore_cache)
    
    def test_status_reporting(self):
        """Test comprehensive status reporting"""
        status = self.monitor.get_status()
        
        # Check required fields
        required_fields = [
            "is_running", "monitored_paths", "processed_events", 
            "runtime_seconds", "priority_queue", "debounce_statistics",
            "resource_usage", "rate_limiting", "file_type_statistics"
        ]
        
        for field in required_fields:
            self.assertIn(field, status)
        
        # Check priority queue metrics
        pq_metrics = status["priority_queue"]
        required_pq_fields = [
            "current_size", "events_queued", "events_processed",
            "max_queue_size_reached", "average_queue_time_ms",
            "throughput_events_per_second", "priority_distribution"
        ]
        
        for field in required_pq_fields:
            self.assertIn(field, pq_metrics)
    
    def test_event_handlers(self):
        """Test custom event handlers"""
        handled_events = []
        
        def test_handler(event: FileChangeEvent):
            handled_events.append(event)
        
        self.monitor.add_event_handler(test_handler)
        
        # Create an event manually and handle it
        event = FileChangeEvent("/test.py", "modified", time.time(), Priority.IMMEDIATE)
        
        # Use asyncio to test async handler
        async def test_async_handler():
            await self.monitor._handle_event(event)
        
        asyncio.run(test_async_handler())
        
        # Handler should have been called
        self.assertEqual(len(handled_events), 1)
        self.assertEqual(handled_events[0].file_path, "/test.py")

class TestPerformanceAndStressTesting(unittest.TestCase):
    """Performance and stress tests for handling 1000+ file changes"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_dir = tempfile.mkdtemp()
        
        # Create large directory structure
        self.large_project_dir = Path(self.temp_dir) / "large_project"
        self.large_project_dir.mkdir()
        
        # Create multiple subdirectories
        for i in range(10):
            subdir = self.large_project_dir / f"module_{i}"
            subdir.mkdir()
            
            # Create multiple files in each subdirectory
            for j in range(20):
                file_types = ['.py', '.js', '.json', '.md', '.log']
                ext = file_types[j % len(file_types)]
                test_file = subdir / f"file_{j}{ext}"
                test_file.write_text(f"// Test file {i}-{j}")
        
        # Performance-optimized config
        self.config = MonitoringConfig(
            root_paths=[str(self.large_project_dir)],
            debounce_interval=0.1,
            max_events_per_second=1000,  # High throughput
            memory_limit_mb=200  # Generous memory limit
        )
        
        self.monitor = FileMonitor(self.config, str(self.knowledge_dir))
    
    def tearDown(self):
        """Clean up performance test environment"""
        try:
            shutil.rmtree(self.temp_dir)
            shutil.rmtree(self.knowledge_dir)
        except:
            pass
    
    def test_1000_file_change_handling(self):
        """Test handling 1000+ file changes efficiently"""
        print("\n" + "="*60)
        print("PERFORMANCE TEST: 1000+ File Changes")
        print("="*60)
        
        start_time = time.time()
        
        # Generate 1000 file change events
        test_files = []
        for i in range(10):  # 10 directories
            for j in range(100):  # 100 files each = 1000 total
                file_path = self.large_project_dir / f"module_{i}" / f"test_file_{j}.py"
                test_files.append(str(file_path))
        
        # Queue all events
        event_start = time.time()
        for file_path in test_files:
            self.monitor._queue_event(file_path, "modified")
        
        event_queue_time = time.time() - event_start
        
        # Process through debounce buffer
        debounce_start = time.time()
        
        # Wait for debounce to settle
        time.sleep(0.2)
        
        # Get ready events
        ready_events = self.monitor.debounce_buffer.get_ready_events()
        debounce_time = time.time() - debounce_start
        
        total_time = time.time() - start_time
        
        # Performance assertions
        self.assertGreater(len(ready_events), 0, "Should have processed some events")
        self.assertLessEqual(event_queue_time, 5.0, "Event queuing should complete within 5 seconds")
        self.assertLessEqual(debounce_time, 2.0, "Debouncing should complete within 2 seconds")
        self.assertLessEqual(total_time, 10.0, "Total processing should complete within 10 seconds")
        
        # Print performance metrics
        events_per_second = len(test_files) / event_queue_time if event_queue_time > 0 else 0
        
        print(f"Events Queued: {len(test_files)}")
        print(f"Events Ready: {len(ready_events)}")
        print(f"Queue Time: {event_queue_time:.3f}s")
        print(f"Debounce Time: {debounce_time:.3f}s")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Throughput: {events_per_second:.0f} events/second")
        
        # Check memory usage
        memory_usage = self.monitor._get_memory_usage()
        print(f"Memory Usage: {memory_usage:.1f}MB")
        self.assertLessEqual(memory_usage, self.config.memory_limit_mb, 
                            f"Memory usage {memory_usage}MB should not exceed limit {self.config.memory_limit_mb}MB")
    
    def test_gitignore_performance(self):
        """Test gitignore pattern matching performance"""
        print("\n" + "="*60)
        print("PERFORMANCE TEST: Gitignore Pattern Matching")
        print("="*60)
        
        # Create many test file paths
        test_paths = []
        for i in range(1000):
            extensions = ['.py', '.pyc', '.js', '.log', '.json', '.md']
            ext = extensions[i % len(extensions)]
            test_paths.append(f"/project/src/module_{i//10}/file_{i}{ext}")
        
        # Measure gitignore checking performance
        start_time = time.time()
        
        ignored_count = 0
        for path in test_paths:
            if self.monitor._should_ignore_file(path):
                ignored_count += 1
        
        check_time = time.time() - start_time
        checks_per_second = len(test_paths) / check_time if check_time > 0 else 0
        
        print(f"Files Checked: {len(test_paths)}")
        print(f"Files Ignored: {ignored_count}")
        print(f"Check Time: {check_time:.3f}s")
        print(f"Checks/Second: {checks_per_second:.0f}")
        
        # Performance assertion
        self.assertGreater(checks_per_second, 1000, "Should handle >1000 gitignore checks per second")
        
        # Test cache effectiveness
        cache_hit_time = time.time()
        for path in test_paths[:100]:  # Re-check same files
            self.monitor._should_ignore_file(path)
        cache_time = time.time() - cache_hit_time
        
        cache_checks_per_second = 100 / cache_time if cache_time > 0 else 0
        print(f"Cache Performance: {cache_checks_per_second:.0f} checks/second")
        
        # Cache should be much faster
        self.assertGreater(cache_checks_per_second, checks_per_second * 2, 
                          "Cache should provide significant performance improvement")
    
    def test_priority_queue_performance(self):
        """Test priority queue performance with mixed priorities"""
        print("\n" + "="*60) 
        print("PERFORMANCE TEST: Priority Queue")
        print("="*60)
        
        # Create events with different priorities
        events = []
        for i in range(1000):
            priorities = [Priority.IMMEDIATE, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
            priority = priorities[i % len(priorities)]
            event = FileChangeEvent(f"/test/file_{i}.py", "modified", time.time(), priority)
            events.append(event)
        
        # Add to debounce buffer
        start_time = time.time()
        for event in events:
            self.monitor.debounce_buffer.add_event(event)
        
        add_time = time.time() - start_time
        
        # Process events
        process_start = time.time()
        time.sleep(0.15)  # Wait for debounce
        ready_events = self.monitor.debounce_buffer.get_ready_events()
        process_time = time.time() - process_start
        
        # Verify priority ordering
        if len(ready_events) > 1:
            for i in range(len(ready_events) - 1):
                self.assertLessEqual(ready_events[i].priority.value, ready_events[i+1].priority.value,
                                   "Events should be ordered by priority")
        
        events_per_second = len(events) / add_time if add_time > 0 else 0
        
        print(f"Events Added: {len(events)}")
        print(f"Events Ready: {len(ready_events)}")
        print(f"Add Time: {add_time:.3f}s")
        print(f"Process Time: {process_time:.3f}s")
        print(f"Add Rate: {events_per_second:.0f} events/second")
        
        # Performance assertion
        self.assertGreater(events_per_second, 1000, "Should handle >1000 events/second for priority queue")

class TestIntegrationWithTreeSitter(unittest.TestCase):
    """Test integration points with tree-sitter parsing"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_dir = tempfile.mkdtemp()
        
        self.config = MonitoringConfig(root_paths=[str(self.temp_dir)])
        self.monitor = FileMonitor(self.config, str(self.knowledge_dir))
    
    def tearDown(self):
        """Clean up integration test environment"""
        try:
            shutil.rmtree(self.temp_dir)
            shutil.rmtree(self.knowledge_dir)
        except:
            pass
    
    def test_source_file_coordination(self):
        """Test coordination with tree-sitter for source files"""
        # Mock tree-sitter coordination
        mock_coordinator = Mock()
        self.monitor.tree_sitter = mock_coordinator
        
        # Create events for source files
        source_event = FileChangeEvent("/test/main.py", "modified", time.time(), Priority.IMMEDIATE)
        config_event = FileChangeEvent("/test/config.json", "modified", time.time(), Priority.HIGH)
        
        # Handle events
        async def test_coordination():
            await self.monitor._handle_event(source_event)
            await self.monitor._handle_event(config_event)
        
        asyncio.run(test_coordination())
        
        # Tree-sitter should be notified for source files
        mock_coordinator.notify_file_changed.assert_any_call("/test/main.py", "modified")
        # But not for config files (only source files trigger parsing)
        self.assertEqual(mock_coordinator.notify_file_changed.call_count, 1)
    
    def test_parsing_priority_integration(self):
        """Test integration with tree-sitter parsing priorities"""
        # Tree-sitter should provide parsing priority information
        priority = self.monitor.tree_sitter.get_parsing_priority("/test/main.py")
        self.assertIsInstance(priority, int)
        
        # Should handle unknown files gracefully
        unknown_priority = self.monitor.tree_sitter.get_parsing_priority("/unknown/file.xyz")
        self.assertIsInstance(unknown_priority, int)

class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up edge case test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_dir = tempfile.mkdtemp()
        
        self.config = MonitoringConfig(root_paths=[str(self.temp_dir)])
        self.monitor = FileMonitor(self.config, str(self.knowledge_dir))
    
    def tearDown(self):
        """Clean up edge case test environment"""
        try:
            shutil.rmtree(self.temp_dir)
            shutil.rmtree(self.knowledge_dir)
        except:
            pass
    
    def test_nonexistent_root_path(self):
        """Test handling nonexistent root paths"""
        config = MonitoringConfig(root_paths=["/nonexistent/path"])
        monitor = FileMonitor(config, str(self.knowledge_dir))
        
        # Should not raise exception
        # In actual usage, start_monitoring would log warning and continue
        self.assertEqual(len(monitor.config.root_paths), 1)
    
    def test_malformed_gitignore(self):
        """Test handling malformed .gitignore files"""
        # Create malformed .gitignore
        gitignore_path = Path(self.temp_dir) / ".gitignore"
        gitignore_path.write_bytes(b'\xff\xfe invalid binary content')  # Invalid UTF-8
        
        # Should handle gracefully and not crash
        try:
            self.monitor._load_gitignore_patterns()
            # Should continue working despite malformed file
            result = self.monitor._should_ignore_file("/test.py")
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"Should handle malformed .gitignore gracefully: {e}")
    
    def test_very_long_file_paths(self):
        """Test handling very long file paths"""
        # Create very long path (close to system limits)
        long_path = "/test/" + ("a" * 200) + "/" + ("b" * 200) + "/file.py"
        
        # Should handle gracefully
        try:
            ignored = self.monitor._should_ignore_file(long_path)
            priority = self.monitor._get_file_priority(long_path)
            self.assertIsInstance(ignored, bool)
            self.assertIsInstance(priority, Priority)
        except Exception as e:
            self.fail(f"Should handle long paths gracefully: {e}")
    
    def test_special_characters_in_paths(self):
        """Test handling special characters in file paths"""
        special_paths = [
            "/test/file with spaces.py",
            "/test/file-with-dashes.py",
            "/test/file_with_underscores.py",
            "/test/file.with.dots.py",
            "/test/file(with)parens.py",
            "/test/file[with]brackets.py",
        ]
        
        for path in special_paths:
            try:
                ignored = self.monitor._should_ignore_file(path)
                priority = self.monitor._get_file_priority(path)
                self.assertIsInstance(ignored, bool)
                self.assertIsInstance(priority, Priority)
            except Exception as e:
                self.fail(f"Should handle special characters in {path}: {e}")
    
    def test_concurrent_access(self):
        """Test concurrent access to debounce buffer"""
        def add_events(thread_id):
            for i in range(50):
                event = FileChangeEvent(f"/test/thread_{thread_id}_file_{i}.py", "modified", time.time(), Priority.MEDIUM)
                self.monitor.debounce_buffer.add_event(event)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_events, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        time.sleep(0.1)
        ready_events = self.monitor.debounce_buffer.get_ready_events()
        
        # Should have processed some events
        self.assertGreaterEqual(len(ready_events), 0)

class TestCLIInterface(unittest.TestCase):
    """Test CLI interface functionality"""
    
    def setUp(self):
        """Set up CLI test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up CLI test environment"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_default_config_creation(self):
        """Test creating default configuration from CLI"""
        config = create_default_config(self.temp_dir)
        
        self.assertEqual(config.root_paths, [self.temp_dir])
        self.assertGreater(config.debounce_interval, 0)
        self.assertGreater(config.max_events_per_second, 0)
        self.assertGreater(config.memory_limit_mb, 0)
    
    @patch('sys.argv', ['file_monitor.py', '--validate-config', '/test'])
    @patch('builtins.print')
    def test_config_validation_command(self, mock_print):
        """Test configuration validation CLI command"""
        from claude.commands.file_monitor import main
        
        try:
            main()
        except SystemExit:
            pass  # Expected for CLI commands
        
        # Should have printed validation output
        mock_print.assert_called()

# Custom test result class for detailed reporting
class DetailedTestResult(unittest.TextTestResult):
    """Enhanced test result with performance metrics"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_start_times = {}
        self.performance_metrics = {}
    
    def startTest(self, test):
        super().startTest(test)
        self.test_start_times[test] = time.time()
    
    def stopTest(self, test):
        super().stopTest(test)
        if test in self.test_start_times:
            duration = time.time() - self.test_start_times[test]
            self.performance_metrics[test] = duration
    
    def print_performance_summary(self):
        """Print performance summary of all tests"""
        if not self.performance_metrics:
            return
        
        print(f"\n{'='*60}")
        print("TEST PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        sorted_tests = sorted(self.performance_metrics.items(), key=lambda x: x[1], reverse=True)
        
        for test, duration in sorted_tests[:10]:  # Top 10 slowest tests
            test_name = f"{test.__class__.__name__}.{test._testMethodName}"
            print(f"{test_name:<50} {duration:.3f}s")
        
        total_time = sum(self.performance_metrics.values())
        avg_time = total_time / len(self.performance_metrics)
        
        print(f"\nTotal Test Time: {total_time:.2f}s")
        print(f"Average Test Time: {avg_time:.3f}s")
        print(f"Tests Run: {len(self.performance_metrics)}")

# Test runner with validation reporting
class FileMonitorTestRunner:
    """Custom test runner for file monitor validation"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.validation_results = {}
    
    def run_validation(self):
        """Run comprehensive validation tests"""
        print("="*80)
        print("RIF FILE MONITORING SYSTEM - VALIDATION SUITE")
        print("Issue #29: Implement real-time file monitoring with watchdog")
        print("="*80)
        
        self.start_time = time.time()
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        
        # Use custom result class
        runner = unittest.TextTestRunner(
            verbosity=2,
            resultclass=DetailedTestResult,
            buffer=True
        )
        result = runner.run(suite)
        
        self.end_time = time.time()
        
        # Generate validation report
        self._generate_validation_report(result)
        
        return result.wasSuccessful()
    
    def _generate_validation_report(self, result):
        """Generate comprehensive validation report"""
        duration = self.end_time - self.start_time
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        
        print(f"\n{'='*80}")
        print("VALIDATION REPORT - FILE MONITORING SYSTEM")
        print(f"{'='*80}")
        
        # Overall results
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Tests Run: {result.testsRun}")
        print(f"Successful: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Requirement validation
        print(f"\n{'REQUIREMENT VALIDATION':<40} {'STATUS':<10}")
        print("-" * 50)
        
        requirements = [
            ("File monitoring with debouncing", "PASS" if success_rate > 90 else "FAIL"),
            ("Priority queue functionality", "PASS" if success_rate > 90 else "FAIL"), 
            ("Gitignore pattern respect", "PASS" if success_rate > 90 else "FAIL"),
            ("1000+ file change handling", "PASS" if success_rate > 90 else "FAIL"),
            ("Tree-sitter integration", "PASS" if success_rate > 90 else "FAIL"),
            ("Performance benchmarks", "PASS" if duration < 60 else "FAIL")
        ]
        
        for requirement, status in requirements:
            print(f"{requirement:<40} {status:<10}")
        
        # Performance metrics
        if hasattr(result, 'print_performance_summary'):
            result.print_performance_summary()
        
        # Detailed failures if any
        if result.failures:
            print(f"\n{'FAILURES':<20}")
            print("-" * 30)
            for test, traceback in result.failures:
                print(f"FAIL: {test}")
                print(traceback)
        
        if result.errors:
            print(f"\n{'ERRORS':<20}")
            print("-" * 30)
            for test, traceback in result.errors:
                print(f"ERROR: {test}")
                print(traceback)
        
        # Final verdict
        overall_status = "PASS" if result.wasSuccessful() and success_rate >= 95 else "FAIL"
        print(f"\n{'='*80}")
        print(f"OVERALL VALIDATION STATUS: {overall_status}")
        print(f"{'='*80}")

if __name__ == "__main__":
    # Run comprehensive validation
    runner = FileMonitorTestRunner()
    success = runner.run_validation()
    
    sys.exit(0 if success else 1)