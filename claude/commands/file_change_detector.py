#!/usr/bin/env python3
"""
RIF File Change Detector for Knowledge Graph Auto-Update

This module implements the FileChangeDetector class as specified in Issue #64,
providing an interface for detecting file system changes, filtering relevant updates,
and batching related changes for knowledge graph auto-update.

This implementation leverages the existing high-performance file monitoring
infrastructure (file_monitor.py) while providing the specific API required
for knowledge graph integration.
"""

import sys
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, DefaultDict
from collections import defaultdict
from queue import PriorityQueue
from dataclasses import dataclass, field
from enum import Enum

# Import existing file monitoring infrastructure
try:
    from .file_monitor import (
        FileMonitor, 
        MonitoringConfig, 
        FileChangeEvent,
        Priority,
        create_default_config
    )
except ImportError:
    # Handle standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from file_monitor import (
        FileMonitor, 
        MonitoringConfig, 
        FileChangeEvent,
        Priority,
        create_default_config
    )

# Import knowledge system interface
try:
    import sys
    import os
    # Add the parent directory to Python path for knowledge import
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from knowledge.interface import get_knowledge_system
    KNOWLEDGE_SYSTEM_AVAILABLE = True
except ImportError:
    KNOWLEDGE_SYSTEM_AVAILABLE = False


@dataclass
class FileChange:
    """File change data structure matching the issue specification"""
    path: str
    type: str  # 'created', 'modified', 'deleted', 'moved'
    priority: int  # 0 = highest priority, higher numbers = lower priority
    timestamp: float = field(default_factory=time.time)
    module: Optional[str] = None
    
    def __lt__(self, other):
        """Priority queue ordering - lower priority number = higher priority"""
        return self.priority < other.priority


class FileChangeDetector:
    """
    File change detector for knowledge graph auto-update.
    
    This class provides the exact API specified in Issue #64 while leveraging
    the existing high-performance file monitoring infrastructure. It implements:
    - File system event monitoring using watchdog
    - Change relevance filtering based on gitignore and file types
    - Related change batching by module/component
    - Priority-based processing with configurable thresholds
    """
    
    def __init__(self, 
                 root_paths: Optional[List[str]] = None,
                 knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """
        Initialize the file change detector.
        
        Args:
            root_paths: List of root paths to monitor (defaults to current directory)
            knowledge_path: Path to knowledge base for logging and integration
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize the underlying file monitor with optimized configuration
        self.config = create_default_config(root_paths[0] if root_paths else ".")
        if root_paths:
            self.config.root_paths = root_paths
            
        # Optimize configuration for knowledge graph updates
        self.config.debounce_interval = 1.0  # Longer debounce for batching
        self.config.max_events_per_second = 1000  # Higher throughput
        
        self.file_monitor = FileMonitor(self.config, knowledge_path)
        
        # Set up priority queue for FileChange objects (as specified in issue)
        self.change_queue = PriorityQueue()
        
        # Module detection and batching
        self.module_detector = ModuleDetector()
        self.batch_processor = BatchProcessor(self.module_detector)
        
        # Event tracking
        self.processed_events = 0
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Knowledge system integration
        self.knowledge_system = None
        if KNOWLEDGE_SYSTEM_AVAILABLE:
            try:
                self.knowledge_system = get_knowledge_system()
                
                # Validate that auto_updates collection is supported
                try:
                    # Test storing a minimal update to verify collection support
                    test_content = {"test": "validation", "timestamp": time.time()}
                    test_metadata = {"type": "validation_test"}
                    test_id = self.knowledge_system.store_knowledge(
                        'auto_updates', test_content, test_metadata, 'validation_test'
                    )
                    
                    if test_id:
                        # Clean up test entry
                        self.knowledge_system.delete_knowledge('auto_updates', 'validation_test')
                        self.logger.info("Knowledge system integration enabled with auto_updates collection")
                    else:
                        raise ValueError("auto_updates collection not supported")
                        
                except Exception as collection_error:
                    self.logger.error(f"auto_updates collection error: {collection_error}")
                    # Disable knowledge system integration if collection is not supported
                    self.knowledge_system = None
                    
            except Exception as e:
                self.logger.warning(f"Knowledge system not available: {e}")
                self.knowledge_system = None
        
        # Set up file monitor event handler
        self.file_monitor.add_event_handler(self._on_file_change)
        
        self.logger.info("FileChangeDetector initialized successfully")
    
    def _on_file_change(self, event: FileChangeEvent):
        """Handle file change events from the underlying monitor"""
        # Convert FileChangeEvent to FileChange (API specification)
        file_change = FileChange(
            path=event.file_path,
            type=event.event_type,
            priority=event.priority.value,  # Convert enum to int
            timestamp=event.timestamp,
            module=self.module_detector.get_module(event.file_path)
        )
        
        # Add to priority queue as specified in the issue
        self.change_queue.put(file_change)
        self.processed_events += 1
        
        self.logger.debug(f"Queued change: {file_change.type} - {file_change.path} "
                         f"(priority: {file_change.priority}, module: {file_change.module})")
    
    def on_file_modified(self, event_path: str):
        """
        Handle file modification events (as specified in issue requirements).
        
        This method provides the exact API from the issue specification.
        In practice, events are handled automatically via the file monitor.
        """
        if self.is_relevant(event_path):
            file_change = FileChange(
                path=event_path,
                type='modified',
                priority=self.calculate_priority(event_path),
                module=self.module_detector.get_module(event_path)
            )
            self.change_queue.put(file_change)
            self.logger.info(f"Manual file modification queued: {event_path}")
    
    def is_relevant(self, file_path: str) -> bool:
        """
        Check if file change is relevant for knowledge graph updates.
        
        This leverages the existing file monitor's sophisticated gitignore
        and pattern matching capabilities with additional checks.
        """
        if not file_path:
            return False
            
        # Always use comprehensive filtering that includes node_modules
        # First check file monitor's filtering if available
        should_ignore = False
        if hasattr(self.file_monitor, '_should_ignore_file'):
            try:
                should_ignore = self.file_monitor._should_ignore_file(file_path)
            except:
                # If file monitor method fails, use fallback
                should_ignore = False
        
        # Always apply our fallback filtering as additional check
        # This ensures consistent node_modules filtering
        if not should_ignore:
            should_ignore = self._fallback_should_ignore(file_path)
        
        return not should_ignore
    
    def _fallback_should_ignore(self, file_path: str) -> bool:
        """Fallback filtering logic when file monitor method not available"""
        import os.path
        
        # Normalize path for consistent checking
        normalized_path = os.path.normpath(file_path)
        
        # Common patterns that should be ignored (exact matches and substrings)
        ignore_patterns = [
            'node_modules',
            '__pycache__',
            '.git',
            '.pytest_cache',
            'build',
            'dist',
            '.vscode',
            '.idea',
            '.DS_Store'
        ]
        
        # Check if path contains any ignore patterns
        path_parts = normalized_path.split(os.sep)
        for pattern in ignore_patterns:
            # Check for exact directory matches
            if pattern in path_parts:
                return True
            # Check for substring matches in any path component
            if any(pattern in part for part in path_parts):
                return True
        
        # Additional check for common ignore patterns in path
        path_lower = normalized_path.lower()
        additional_ignores = [
            '/node_modules/',
            '\\node_modules\\',  # Windows path
            'node_modules/',
            'node_modules\\'
        ]
        
        for ignore_pattern in additional_ignores:
            if ignore_pattern in path_lower:
                return True
        
        # Check file extensions that should be ignored
        ignore_extensions = ['.pyc', '.pyo', '.log', '.tmp', '.cache']
        for ext in ignore_extensions:
            if normalized_path.endswith(ext):
                return True
        
        return False
    
    def calculate_priority(self, file_path: str) -> int:
        """
        Calculate priority for file change (0 = highest priority).
        
        Maps the file monitor's Priority enum to integer values as specified
        in the issue requirements.
        """
        priority_enum = self.file_monitor._get_file_priority(file_path)
        return priority_enum.value
    
    def get_module(self, file_path: str) -> str:
        """Get module/component for a file path."""
        return self.module_detector.get_module(file_path)
    
    def batch_related_changes(self) -> Dict[str, List[FileChange]]:
        """
        Group changes by module/component (exact API from issue specification).
        
        Returns:
            Dictionary mapping module names to lists of related changes
        """
        return self.batch_processor.batch_changes_from_queue(self.change_queue)
    
    def start_monitoring(self):
        """Start file change monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        # Clean up any existing thread first
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.info("Cleaning up previous monitoring thread...")
            self.stop_monitoring()
            
        try:
            self.logger.info("Starting file change detection...")
            self.is_monitoring = True
            
            # Start the underlying file monitor in a separate thread
            if hasattr(self.file_monitor, 'start_monitoring'):
                self.monitoring_thread = threading.Thread(
                    target=self.file_monitor.start_monitoring,
                    daemon=True
                )
                self.monitoring_thread.start()
                
                # Give thread a moment to start
                time.sleep(0.1)
                
                if not self.monitoring_thread.is_alive():
                    raise RuntimeError("Monitoring thread failed to start")
            else:
                self.logger.warning("File monitor does not support start_monitoring method")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.is_monitoring = False
            self.monitoring_thread = None
            raise
        
        self.logger.info("File change detection started successfully")
    
    def stop_monitoring(self):
        """Stop file change monitoring"""
        if not self.is_monitoring:
            self.logger.debug("Monitoring already stopped")
            return
            
        self.logger.info("Stopping file change detection...")
        
        try:
            # Set flag first to prevent race conditions
            self.is_monitoring = False
            
            # Stop the underlying file monitor
            if hasattr(self.file_monitor, 'stop_monitoring'):
                self.file_monitor.stop_monitoring()
            
            # Wait for monitoring thread to complete with timeout
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
                if self.monitoring_thread.is_alive():
                    self.logger.warning("Monitoring thread did not stop cleanly")
            
            # Clean up thread reference
            self.monitoring_thread = None
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
            # Ensure we're in stopped state regardless of errors
            self.is_monitoring = False
            self.monitoring_thread = None
        
        self.logger.info("File change detection stopped")
    
    def get_pending_changes(self) -> List[FileChange]:
        """Get all pending changes from the queue"""
        changes = []
        while not self.change_queue.empty():
            try:
                change = self.change_queue.get_nowait()
                changes.append(change)
            except:
                break
        return changes
    
    def get_status(self) -> Dict:
        """Get current detector status"""
        underlying_status = self.file_monitor.get_status()
        
        return {
            "is_monitoring": self.is_monitoring,
            "processed_events": self.processed_events,
            "pending_changes": self.change_queue.qsize(),
            "monitored_paths": self.config.root_paths,
            "knowledge_integration": KNOWLEDGE_SYSTEM_AVAILABLE,
            "underlying_monitor": underlying_status
        }


class ModuleDetector:
    """Detect module/component for file paths"""
    
    def __init__(self):
        self.module_cache = {}
        
    def get_module(self, file_path: str) -> str:
        """
        Determine the module/component for a file path.
        
        Uses directory structure and file patterns to group related files.
        """
        if file_path in self.module_cache:
            return self.module_cache[file_path]
        
        path = Path(file_path)
        
        # Module detection logic based on common patterns
        module = self._detect_module_from_path(path)
        self.module_cache[file_path] = module
        return module
    
    def _detect_module_from_path(self, path: Path) -> str:
        """Internal module detection logic"""
        parts = path.parts
        
        # Handle common project structures
        if 'src' in parts:
            src_index = parts.index('src')
            if src_index + 1 < len(parts):
                return parts[src_index + 1]
        
        if 'lib' in parts:
            lib_index = parts.index('lib')
            if lib_index + 1 < len(parts):
                return parts[lib_index + 1]
        
        # Language-specific patterns
        if any(part in parts for part in ['claude', 'agents']):
            return 'rif-agents'
        
        if any(part in parts for part in ['knowledge', 'db']):
            return 'knowledge-system'
        
        if any(part in parts for part in ['config', 'configs']):
            return 'configuration'
        
        if any(part in parts for part in ['test', 'tests', 'spec']):
            return 'testing'
        
        if any(part in parts for part in ['docs', 'doc', 'documentation']):
            return 'documentation'
        
        # Default to parent directory
        if len(parts) > 1:
            return parts[-2]  # Parent directory
        
        return 'root'


class BatchProcessor:
    """Process and batch file changes by module"""
    
    def __init__(self, module_detector: ModuleDetector):
        self.module_detector = module_detector
        
    def batch_changes_from_queue(self, queue: PriorityQueue) -> Dict[str, List[FileChange]]:
        """
        Extract and batch changes from the priority queue by module.
        
        This implements the exact batching logic specified in the issue.
        """
        batches: DefaultDict[str, List[FileChange]] = defaultdict(list)
        
        # Extract all changes from queue
        changes = []
        while not queue.empty():
            try:
                change = queue.get_nowait()
                changes.append(change)
            except:
                break
        
        # Group by module
        for change in changes:
            module = change.module or self.module_detector.get_module(change.path)
            batches[module].append(change)
        
        # Sort changes within each batch by priority and timestamp
        for module, module_changes in batches.items():
            module_changes.sort(key=lambda c: (c.priority, c.timestamp))
        
        return dict(batches)


def create_file_change_detector(root_paths: Optional[List[str]] = None) -> FileChangeDetector:
    """
    Convenience function to create a FileChangeDetector instance.
    
    Args:
        root_paths: List of paths to monitor (defaults to current directory)
        
    Returns:
        Configured FileChangeDetector instance
    """
    return FileChangeDetector(root_paths)


# Example usage and testing functions
def main():
    """CLI interface for testing the FileChangeDetector"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: file_change_detector.py <command> [options]")
        print("Commands:")
        print("  --demo         Run demonstration")
        print("  --test-api     Test the API as specified in issue")
        print("  --monitor      Start interactive monitoring")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--demo":
        demo_file_change_detector()
    elif command == "--test-api":
        test_issue_api()
    elif command == "--monitor":
        interactive_monitoring()
    else:
        print(f"Unknown command: {command}")


def demo_file_change_detector():
    """Demonstrate FileChangeDetector functionality"""
    print("RIF FileChangeDetector Demo")
    print("==========================")
    
    # Create detector
    detector = create_file_change_detector(["."])
    
    print(f"Status: {detector.get_status()}")
    
    # Simulate some events
    test_files = [
        "src/main.py",
        "src/utils.py", 
        "config/settings.json",
        "tests/test_main.py",
        "docs/README.md"
    ]
    
    print("\nSimulating file changes...")
    for test_file in test_files:
        detector.on_file_modified(test_file)
    
    print(f"Queue size: {detector.change_queue.qsize()}")
    
    # Batch related changes
    batches = detector.batch_related_changes()
    print(f"\nBatched changes by module:")
    for module, changes in batches.items():
        print(f"  {module}: {len(changes)} changes")
        for change in changes:
            print(f"    - {change.type}: {change.path} (priority: {change.priority})")


def test_issue_api():
    """Test the exact API specified in the issue"""
    print("Testing Issue #64 API Specification")
    print("===================================")
    
    # Test the exact API from the issue
    detector = FileChangeDetector()
    
    # Test is_relevant method
    test_paths = ["src/main.py", "node_modules/package.json", "__pycache__/module.pyc"]
    print("\nTesting relevance filtering:")
    for path in test_paths:
        relevant = detector.is_relevant(path)
        priority = detector.calculate_priority(path)
        module = detector.get_module(path)
        print(f"  {path}: relevant={relevant}, priority={priority}, module={module}")
    
    # Test priority calculation
    print("\nTesting priority calculation:")
    priority_test_files = ["main.py", "config.json", "README.md", "temp.log"]
    for test_file in priority_test_files:
        priority = detector.calculate_priority(test_file)
        print(f"  {test_file}: priority={priority}")
    
    # Test batch processing
    print("\nTesting batch processing:")
    for test_file in ["src/core.py", "src/utils.py", "tests/test_core.py"]:
        detector.on_file_modified(test_file)
    
    batches = detector.batch_related_changes()
    for module, changes in batches.items():
        print(f"  Module '{module}': {len(changes)} changes")


def interactive_monitoring():
    """Interactive monitoring session"""
    print("Starting interactive file monitoring...")
    print("Press Ctrl+C to stop")
    
    detector = create_file_change_detector(["."])
    
    try:
        detector.start_monitoring()
        
        while True:
            time.sleep(2)
            status = detector.get_status()
            print(f"Events processed: {status['processed_events']}, "
                  f"Pending: {status['pending_changes']}")
            
            if status['pending_changes'] > 0:
                batches = detector.batch_related_changes()
                for module, changes in batches.items():
                    print(f"  {module}: {len(changes)} changes")
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
        detector.stop_monitoring()


if __name__ == "__main__":
    main()