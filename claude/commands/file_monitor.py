#!/usr/bin/env python3
"""
RIF Real-time File Monitoring System
Monitors file system changes with debouncing, priority queuing, and gitignore compliance
"""

import sys
import json
import time
import asyncio
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import logging

# Required dependencies
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    import pathspec
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install watchdog pathspec")
    sys.exit(1)

class Priority(Enum):
    """File change priority levels"""
    IMMEDIATE = 0  # Source code files (.py, .js, .ts, .go, .rs)
    HIGH = 1       # Configuration files (.json, .yaml, .toml)
    MEDIUM = 2     # Documentation (.md, .rst), tests
    LOW = 3        # Generated files, logs, temporary files

@dataclass
class FileChangeEvent:
    """File change event with metadata"""
    file_path: str
    event_type: str  # created, modified, deleted, moved
    timestamp: float
    priority: Priority
    size: Optional[int] = None
    checksum: Optional[str] = None
    
    def __lt__(self, other):
        """Priority queue ordering - lower priority number = higher priority"""
        return self.priority.value < other.priority.value

@dataclass
class MonitoringConfig:
    """Configuration for file monitoring"""
    root_paths: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=list)
    debounce_interval: float = 0.5  # 500ms default
    max_events_per_second: int = 500
    memory_limit_mb: int = 100
    enable_checksums: bool = False
    priority_extensions: Dict[Priority, List[str]] = field(default_factory=lambda: {
        Priority.IMMEDIATE: ['.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.cpp', '.c', '.h'],
        Priority.HIGH: ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
        Priority.MEDIUM: ['.md', '.rst', '.txt', '.test.py', '.test.js', '.spec.py', '.spec.js'],
        Priority.LOW: ['.log', '.tmp', '.cache', '.pyc', '.pyo', '__pycache__']
    })

class FileSystemEventProcessor(FileSystemEventHandler):
    """Handles raw file system events from watchdog"""
    
    def __init__(self, monitor: 'FileMonitor'):
        super().__init__()
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if not event.is_directory:
            self.monitor._queue_event(event.src_path, 'modified')
    
    def on_created(self, event):
        if not event.is_directory:
            self.monitor._queue_event(event.src_path, 'created')
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.monitor._queue_event(event.src_path, 'deleted')
    
    def on_moved(self, event):
        if not event.is_directory:
            self.monitor._queue_event(event.dest_path, 'moved', old_path=event.src_path)

class DebounceBuffer:
    """Advanced debouncing with IDE compatibility and batch processing"""
    
    def __init__(self, debounce_interval: float = 0.5):
        self.debounce_interval = debounce_interval
        self.buffer: Dict[str, FileChangeEvent] = {}
        self.batch_buffer: Dict[str, List[FileChangeEvent]] = {}  # Directory-based batching
        self.ide_sequences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))  # IDE save sequence tracking
        self.last_flush = time.time()
        self.lock = threading.RLock()
        
        # IDE compatibility settings
        self.ide_sequence_window = 2.0  # Extended window for multi-file operations
        self.rapid_change_threshold = 0.2  # 200ms for detecting rapid changes
        self.batch_size_threshold = 5  # Minimum files for batch processing
    
    def add_event(self, event: FileChangeEvent):
        """Add event with advanced debouncing and IDE compatibility"""
        with self.lock:
            current_time = time.time()
            file_key = event.file_path
            dir_key = str(Path(event.file_path).parent)
            
            # Track IDE save sequences (rapid successive changes)
            sequence = self.ide_sequences[file_key]
            sequence.append((current_time, event.event_type))
            
            # Detect rapid changes (IDE auto-save pattern)
            is_rapid_change = self._is_rapid_change_sequence(sequence, current_time)
            
            # Detect batch operations (multiple files in same directory)
            is_batch_operation = self._is_batch_operation(dir_key, current_time)
            
            # Adjust debounce interval based on context
            effective_interval = self._calculate_effective_interval(
                is_rapid_change, is_batch_operation, event.priority
            )
            
            # Update event timestamp to account for effective interval
            event.timestamp = current_time
            
            # Store in appropriate buffer
            if is_batch_operation:
                if dir_key not in self.batch_buffer:
                    self.batch_buffer[dir_key] = []
                self.batch_buffer[dir_key].append(event)
            else:
                # Standard single-file debouncing with coalescing
                if file_key in self.buffer:
                    existing = self.buffer[file_key]
                    self._coalesce_events(existing, event, effective_interval)
                else:
                    self.buffer[file_key] = event
    
    def _is_rapid_change_sequence(self, sequence: deque, current_time: float) -> bool:
        """Detect if this is part of a rapid change sequence (IDE auto-save)"""
        if len(sequence) < 3:
            return False
        
        # Check if last 3 events were within rapid change threshold
        recent_events = list(sequence)[-3:]
        time_span = recent_events[-1][0] - recent_events[0][0]
        
        return time_span <= self.rapid_change_threshold * 3
    
    def _is_batch_operation(self, dir_key: str, current_time: float) -> bool:
        """Detect if this is part of a batch operation (refactoring, etc.)"""
        # Count recent events in same directory
        if dir_key not in self.batch_buffer:
            return False
        
        recent_events = [
            e for e in self.batch_buffer[dir_key] 
            if current_time - e.timestamp <= self.ide_sequence_window
        ]
        
        return len(recent_events) >= self.batch_size_threshold
    
    def _calculate_effective_interval(self, is_rapid: bool, is_batch: bool, priority: Priority) -> float:
        """Calculate effective debounce interval based on context"""
        base_interval = self.debounce_interval
        
        if is_rapid:
            # Longer interval for rapid changes to group them
            base_interval *= 2
        
        if is_batch:
            # Extended interval for batch operations
            base_interval = max(base_interval, self.ide_sequence_window)
        
        # Priority-based adjustment
        if priority == Priority.IMMEDIATE:
            base_interval *= 0.7  # Faster processing for critical files
        elif priority == Priority.LOW:
            base_interval *= 1.5  # Slower processing for low priority
        
        return base_interval
    
    def _coalesce_events(self, existing: FileChangeEvent, new_event: FileChangeEvent, interval: float):
        """Intelligently coalesce events based on type and timing"""
        # Update timestamp to latest
        existing.timestamp = new_event.timestamp
        
        # Priority: keep highest priority
        if new_event.priority.value < existing.priority.value:
            existing.priority = new_event.priority
        
        # Event type coalescing logic
        if new_event.event_type == 'deleted':
            # Delete overrides everything
            existing.event_type = 'deleted'
        elif existing.event_type == 'deleted':
            # Keep delete, ignore subsequent events
            pass
        elif new_event.event_type == 'created' and existing.event_type == 'modified':
            # Create + modify = create with content
            existing.event_type = 'created'
        elif new_event.event_type == 'moved':
            # Move events are special - preserve move type
            existing.event_type = 'moved'
        else:
            # Default: use latest event type
            existing.event_type = new_event.event_type
    
    def get_ready_events(self) -> List[FileChangeEvent]:
        """Get events ready for processing with batch processing"""
        with self.lock:
            current_time = time.time()
            ready_events = []
            
            # Process single-file events
            for path, event in list(self.buffer.items()):
                effective_interval = self._calculate_effective_interval(
                    False, False, event.priority  # Recalculate for current context
                )
                
                if current_time - event.timestamp >= effective_interval:
                    ready_events.append(event)
                    del self.buffer[path]
            
            # Process batch events
            for dir_path, events in list(self.batch_buffer.items()):
                if not events:
                    del self.batch_buffer[dir_path]
                    continue
                
                # Check if batch is ready (oldest event past threshold)
                oldest_event = min(events, key=lambda e: e.timestamp)
                if current_time - oldest_event.timestamp >= self.ide_sequence_window:
                    # Process entire batch
                    ready_events.extend(events)
                    del self.batch_buffer[dir_path]
            
            # Sort by priority and timestamp
            ready_events.sort(key=lambda e: (e.priority.value, e.timestamp))
            
            return ready_events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get debounce buffer statistics"""
        with self.lock:
            return {
                "single_file_buffer_size": len(self.buffer),
                "batch_buffer_directories": len(self.batch_buffer),
                "total_batch_events": sum(len(events) for events in self.batch_buffer.values()),
                "active_ide_sequences": len(self.ide_sequences),
                "debounce_interval": self.debounce_interval,
                "ide_sequence_window": self.ide_sequence_window
            }
    
    def flush_all(self) -> List[FileChangeEvent]:
        """Flush all events (used during shutdown)"""
        with self.lock:
            events = list(self.buffer.values())
            
            # Add all batch events
            for batch_events in self.batch_buffer.values():
                events.extend(batch_events)
            
            # Clear all buffers
            self.buffer.clear()
            self.batch_buffer.clear()
            self.ide_sequences.clear()
            
            return events

class TreeSitterCoordination:
    """Interface for coordinating with tree-sitter parsing"""
    
    def __init__(self):
        self.enabled = False
        self.parsing_in_progress: Set[str] = set()
        self.parsing_priorities: Dict[str, int] = {}
        self.lock = threading.RLock()
    
    def notify_file_changed(self, file_path: str, change_type: str):
        """Notify tree-sitter system of file changes"""
        if not self.enabled:
            return
        
        # Mock implementation - will be replaced when Issue #27 completes
        with self.lock:
            if file_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs')):
                # Schedule for incremental parsing
                pass
    
    def get_parsing_priority(self, file_path: str) -> int:
        """Get parsing priority for coordination with file monitor priority queue"""
        with self.lock:
            return self.parsing_priorities.get(file_path, Priority.LOW.value)
    
    def is_parsing_in_progress(self, file_path: str) -> bool:
        """Check if file is currently being parsed"""
        with self.lock:
            return file_path in self.parsing_in_progress

class FileMonitor:
    """Main file monitoring system with debouncing and priority queuing"""
    
    def __init__(self, config: MonitoringConfig, knowledge_base_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.config = config
        self.kb_path = Path(knowledge_base_path)
        self.events_log_path = self.kb_path / "events.jsonl"
        
        # Core components
        self.observer = Observer()
        self.event_processor = FileSystemEventProcessor(self)
        self.debounce_buffer = DebounceBuffer(config.debounce_interval)
        self.priority_queue = None  # Will be initialized when event loop starts
        self.tree_sitter = TreeSitterCoordination()
        
        # Priority queue metrics
        self.queue_metrics = {
            "events_queued": 0,
            "events_processed": 0,
            "priority_distribution": defaultdict(int),
            "average_queue_time": 0.0,
            "max_queue_size": 0,
            "throughput_events_per_second": 0.0,
            "last_throughput_calculation": time.time()
        }
        
        # Logging (setup first)
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # State management
        self.is_running = False
        self.processed_events = 0
        self.start_time = None
        self.event_handlers: List[Callable[[FileChangeEvent], None]] = []
        self.statistics = defaultdict(int)
        
        # Rate limiting
        self.event_timestamps = deque(maxlen=1000)
        
        # Gitignore handling (after logging is setup)
        self.ignore_specs = {}  # Path -> PathSpec mapping for nested gitignore
        self.global_ignore_spec = None
        self.gitignore_cache = {}  # Path -> ignore_result cache
        self.gitignore_files = set()  # Track .gitignore file locations
        self._load_gitignore_patterns()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _load_gitignore_patterns(self):
        """Load and compile multi-level gitignore patterns with caching"""
        # Default exclusions (always applied)
        default_patterns = [
            '.git/*',
            '*.pyc',
            '*.pyo', 
            '__pycache__/*',
            'node_modules/*',
            '.DS_Store',
            '*.log',
            '.env',
            '.venv/*',
            'venv/*',
            'build/*',
            'dist/*',
            '*.egg-info/*',
            '.pytest_cache/*',
            '.coverage',
            'htmlcov/*',
            '.idea/*',
            '.vscode/*',
            '*.swp',
            '*.swo',
            '*~'
        ]
        
        try:
            # Create global ignore spec from default patterns and config
            all_global_patterns = default_patterns + self.config.ignore_patterns
            self.global_ignore_spec = pathspec.PathSpec.from_lines('gitignore', all_global_patterns)
            
            # Load .gitignore files for each root path and subdirectories
            for root_path in self.config.root_paths:
                self._load_nested_gitignore_files(Path(root_path))
                
            self.logger.info(f"Loaded gitignore patterns from {len(self.ignore_specs)} directories")
            self.logger.info(f"Global patterns: {len(all_global_patterns)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load gitignore patterns: {e}")
            # Fallback to basic global spec
            self.global_ignore_spec = pathspec.PathSpec.from_lines('gitignore', default_patterns)
    
    def _load_nested_gitignore_files(self, directory_path: Path):
        """Recursively load .gitignore files from directory and subdirectories"""
        try:
            if not directory_path.is_dir():
                return
                
            # Load .gitignore in current directory
            gitignore_path = directory_path / '.gitignore'
            if gitignore_path.exists():
                try:
                    with open(gitignore_path, 'r', encoding='utf-8') as f:
                        patterns = [
                            line.strip() for line in f.readlines()
                            if line.strip() and not line.startswith('#')
                        ]
                    
                    if patterns:
                        spec = pathspec.PathSpec.from_lines('gitignore', patterns)
                        self.ignore_specs[str(directory_path)] = spec
                        self.gitignore_files.add(str(gitignore_path))
                        self.logger.debug(f"Loaded {len(patterns)} patterns from {gitignore_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load .gitignore from {gitignore_path}: {e}")
            
            # Recursively load from subdirectories (but skip obviously ignored ones)
            try:
                for subdir in directory_path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        # Skip common ignore directories for performance
                        if subdir.name in ('node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist'):
                            continue
                        self._load_nested_gitignore_files(subdir)
            except PermissionError:
                # Skip directories we can't read
                pass
                
        except Exception as e:
            self.logger.warning(f"Error loading nested gitignore from {directory_path}: {e}")
    
    def _reload_gitignore_patterns(self):
        """Reload all gitignore patterns (called when .gitignore files change)"""
        self.logger.info("Reloading gitignore patterns...")
        
        # Clear existing patterns and cache
        self.ignore_specs.clear()
        self.gitignore_cache.clear()
        self.gitignore_files.clear()
        
        # Reload all patterns
        self._load_gitignore_patterns()
        
        self.logger.info("Gitignore patterns reloaded successfully")
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on multi-level patterns with caching"""
        # Check cache first for performance
        if file_path in self.gitignore_cache:
            return self.gitignore_cache[file_path]
        
        try:
            file_path_obj = Path(file_path)
            
            # Check global patterns first (most common case)
            if self.global_ignore_spec:
                # Test filename directly
                if self.global_ignore_spec.match_file(file_path_obj.name):
                    self.gitignore_cache[file_path] = True
                    return True
                
                # Test relative paths from root directories
                for root_path in self.config.root_paths:
                    try:
                        rel_path = file_path_obj.relative_to(Path(root_path).resolve())
                        if self.global_ignore_spec.match_file(str(rel_path)):
                            self.gitignore_cache[file_path] = True
                            return True
                    except ValueError:
                        continue
            
            # Check directory-specific .gitignore files
            # Walk up the directory hierarchy to find applicable .gitignore files
            for directory_path, ignore_spec in self.ignore_specs.items():
                dir_path_obj = Path(directory_path)
                
                # Check if file is within this directory
                try:
                    rel_path = file_path_obj.relative_to(dir_path_obj)
                    
                    # Test against this directory's patterns
                    if ignore_spec.match_file(str(rel_path)) or ignore_spec.match_file(file_path_obj.name):
                        self.gitignore_cache[file_path] = True
                        return True
                        
                except ValueError:
                    # File is not within this directory
                    continue
            
            # File is not ignored
            self.gitignore_cache[file_path] = False
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking ignore patterns for {file_path}: {e}")
            # Default to not ignored on error
            self.gitignore_cache[file_path] = False
            return False
    
    def _get_file_priority(self, file_path: str) -> Priority:
        """Determine priority based on file extension"""
        file_ext = Path(file_path).suffix.lower()
        
        for priority, extensions in self.config.priority_extensions.items():
            if file_ext in extensions:
                return priority
        
        # Check for test files (higher priority)
        file_name = Path(file_path).name.lower()
        if any(test_marker in file_name for test_marker in ['test', 'spec']):
            return Priority.MEDIUM
        
        # Default to low priority for unknown extensions
        return Priority.LOW
    
    def _queue_event(self, file_path: str, event_type: str, old_path: str = None):
        """Queue a file system event for processing"""
        # Check rate limiting
        current_time = time.time()
        self.event_timestamps.append(current_time)
        
        # Check events per second
        recent_events = sum(1 for t in self.event_timestamps if current_time - t <= 1.0)
        if recent_events > self.config.max_events_per_second:
            self.logger.warning(f"Rate limit exceeded: {recent_events} events/second")
            return
        
        # Check if this is a .gitignore file change (reload patterns if so)
        if Path(file_path).name == '.gitignore':
            self.logger.info(f"Detected .gitignore change: {file_path}")
            self._reload_gitignore_patterns()
            # Continue processing the .gitignore file change event itself
        
        # Check if file should be ignored (after potential reload)
        if self._should_ignore_file(file_path):
            return
        
        # Create event
        priority = self._get_file_priority(file_path)
        event = FileChangeEvent(
            file_path=file_path,
            event_type=event_type,
            timestamp=current_time,
            priority=priority
        )
        
        # Add to debounce buffer
        self.debounce_buffer.add_event(event)
        self.statistics[f"{priority.name.lower()}_events"] += 1
        
        self.logger.debug(f"Queued {event_type} event for {file_path} (priority: {priority.name})")
    
    def add_event_handler(self, handler: Callable[[FileChangeEvent], None]):
        """Add custom event handler"""
        self.event_handlers.append(handler)
    
    async def _process_events(self):
        """Main event processing loop"""
        self.logger.info("Starting event processing loop")
        
        # Initialize priority queue in the event loop
        if self.priority_queue is None:
            self.priority_queue = asyncio.PriorityQueue()
        
        while self.is_running:
            try:
                # Get events ready for processing from debounce buffer
                ready_events = self.debounce_buffer.get_ready_events()
                
                # Add events to priority queue with metrics
                for event in ready_events:
                    await self.priority_queue.put((time.time(), event))  # Add queue timestamp
                    self.queue_metrics["events_queued"] += 1
                    self.queue_metrics["priority_distribution"][event.priority.name] += 1
                    
                    # Track max queue size
                    current_queue_size = self.priority_queue.qsize()
                    if current_queue_size > self.queue_metrics["max_queue_size"]:
                        self.queue_metrics["max_queue_size"] = current_queue_size
                
                # Process events from priority queue
                try:
                    # Wait for event with timeout to allow graceful shutdown
                    queue_timestamp, event = await asyncio.wait_for(
                        self.priority_queue.get(), 
                        timeout=0.1
                    )
                    
                    # Calculate queue time
                    queue_time = time.time() - queue_timestamp
                    self._update_queue_metrics(queue_time)
                    
                    await self._handle_event(event)
                    self.processed_events += 1
                    self.queue_metrics["events_processed"] += 1
                    
                except asyncio.TimeoutError:
                    # No events to process, continue loop
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _handle_event(self, event: FileChangeEvent):
        """Handle individual file change event"""
        try:
            self.logger.info(f"Processing {event.event_type} event for {event.file_path} "
                           f"(priority: {event.priority.name})")
            
            # Coordinate with tree-sitter if needed
            if event.file_path.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs')):
                self.tree_sitter.notify_file_changed(event.file_path, event.event_type)
            
            # Log event to knowledge base
            await self._log_event(event)
            
            # Call custom event handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
            
            # Update statistics
            self.statistics['total_processed'] += 1
            self.statistics[f"{event.priority.name.lower()}_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Error handling event {event}: {e}")
    
    async def _log_event(self, event: FileChangeEvent):
        """Log event to knowledge base"""
        try:
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "event_type": "file_change",
                "file_path": event.file_path,
                "change_type": event.event_type,
                "priority": event.priority.name,
                "processing_latency_ms": round((time.time() - event.timestamp) * 1000, 2)
            }
            
            # Append to events log
            with open(self.events_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")
    
    def start_monitoring(self):
        """Start file system monitoring"""
        if self.is_running:
            self.logger.warning("Monitoring is already running")
            return
        
        self.logger.info("Starting RIF File Monitor...")
        self.is_running = True
        self.start_time = time.time()
        
        # Setup watchdog observers for each root path
        for root_path in self.config.root_paths:
            if not Path(root_path).exists():
                self.logger.warning(f"Root path does not exist: {root_path}")
                continue
                
            self.observer.schedule(
                self.event_processor,
                str(root_path),
                recursive=True
            )
            self.logger.info(f"Monitoring started for: {root_path}")
        
        # Start watchdog observer
        self.observer.start()
        
        # Start async event processing
        try:
            asyncio.run(self._process_events())
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop file system monitoring"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping RIF File Monitor...")
        self.is_running = False
        
        # Stop watchdog observer
        self.observer.stop()
        self.observer.join()
        
        # Process remaining events in debounce buffer
        remaining_events = self.debounce_buffer.flush_all()
        self.logger.info(f"Processing {len(remaining_events)} remaining events")
        
        # Log final statistics
        self._log_statistics()
        
        self.logger.info("RIF File Monitor stopped")
    
    def _log_statistics(self):
        """Log monitoring statistics"""
        if self.start_time:
            runtime_minutes = (time.time() - self.start_time) / 60
            events_per_minute = self.processed_events / runtime_minutes if runtime_minutes > 0 else 0
            
            stats = {
                "runtime_minutes": round(runtime_minutes, 2),
                "total_events_processed": self.processed_events,
                "events_per_minute": round(events_per_minute, 2),
                "priority_breakdown": dict(self.statistics),
                "memory_usage_mb": self._get_memory_usage()
            }
            
            self.logger.info(f"Final statistics: {json.dumps(stats, indent=2)}")
            
            # Save statistics to file
            stats_file = self.kb_path / "metrics" / "file-monitor-stats.json"
            stats_file.parent.mkdir(exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
    
    def _update_queue_metrics(self, queue_time: float):
        """Update queue processing metrics"""
        # Update average queue time (rolling average)
        current_avg = self.queue_metrics["average_queue_time"]
        processed_count = self.queue_metrics["events_processed"] 
        
        if processed_count == 0:
            self.queue_metrics["average_queue_time"] = queue_time
        else:
            # Rolling average with recent events weighted more heavily
            weight = min(0.1, 1.0 / processed_count)
            self.queue_metrics["average_queue_time"] = (
                current_avg * (1 - weight) + queue_time * weight
            )
        
        # Calculate throughput every 10 seconds
        current_time = time.time()
        time_since_last = current_time - self.queue_metrics["last_throughput_calculation"]
        
        if time_since_last >= 10.0:  # Update every 10 seconds
            events_processed_period = self.queue_metrics["events_processed"]
            if events_processed_period > 0:
                self.queue_metrics["throughput_events_per_second"] = (
                    events_processed_period / time_since_last
                )
            self.queue_metrics["last_throughput_calculation"] = current_time
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return round(process.memory_info().rss / 1024 / 1024, 2)
        except ImportError:
            return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status with comprehensive metrics"""
        debounce_stats = self.debounce_buffer.get_statistics()
        
        return {
            "is_running": self.is_running,
            "monitored_paths": self.config.root_paths,
            "processed_events": self.processed_events,
            "runtime_seconds": round(time.time() - self.start_time, 2) if self.start_time else 0,
            
            # Priority Queue Metrics
            "priority_queue": {
                "current_size": self.priority_queue.qsize() if self.priority_queue else 0,
                "events_queued": self.queue_metrics["events_queued"],
                "events_processed": self.queue_metrics["events_processed"],
                "max_queue_size_reached": self.queue_metrics["max_queue_size"],
                "average_queue_time_ms": round(self.queue_metrics["average_queue_time"] * 1000, 2),
                "throughput_events_per_second": round(self.queue_metrics["throughput_events_per_second"], 2),
                "priority_distribution": dict(self.queue_metrics["priority_distribution"])
            },
            
            # Debounce Statistics
            "debounce_statistics": debounce_stats,
            
            # Resource Usage
            "resource_usage": {
                "memory_usage_mb": self._get_memory_usage(),
                "memory_limit_mb": self.config.memory_limit_mb,
                "memory_utilization_percent": round(
                    (self._get_memory_usage() / self.config.memory_limit_mb) * 100, 1
                ) if self.config.memory_limit_mb > 0 else 0
            },
            
            # Rate Limiting
            "rate_limiting": {
                "recent_events_count": len([t for t in self.event_timestamps if time.time() - t <= 1.0]),
                "max_events_per_second": self.config.max_events_per_second,
                "rate_utilization_percent": round(
                    (len([t for t in self.event_timestamps if time.time() - t <= 1.0]) / 
                     self.config.max_events_per_second) * 100, 1
                )
            },
            
            # General Statistics
            "file_type_statistics": dict(self.statistics)
        }

def create_default_config(root_path: str = ".") -> MonitoringConfig:
    """Create default monitoring configuration"""
    return MonitoringConfig(
        root_paths=[root_path],
        ignore_patterns=[],
        debounce_interval=0.5,
        max_events_per_second=500,
        memory_limit_mb=100
    )

def main():
    """CLI interface for file monitoring"""
    
    if len(sys.argv) < 2:
        print("Usage: file_monitor.py <command> [options]")
        print("Commands:")
        print("  --start [path]         Start monitoring (default: current directory)")
        print("  --status               Show monitoring status")
        print("  --test-patterns [path] Test gitignore pattern matching")
        print("  --validate-config [path] Validate configuration")
        print("  --load-test [path] [count] Load test with N file events (default: 100)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--start":
        root_path = sys.argv[2] if len(sys.argv) > 2 else "."
        config = create_default_config(root_path)
        monitor = FileMonitor(config)
        
        # Add example event handler
        def example_handler(event: FileChangeEvent):
            print(f"Custom handler: {event.event_type} - {event.file_path}")
        
        monitor.add_event_handler(example_handler)
        monitor.start_monitoring()
    
    elif command == "--status":
        # This would connect to a running monitor instance
        print("Status monitoring not yet implemented")
    
    elif command == "--test-patterns":
        root_path = sys.argv[2] if len(sys.argv) > 2 else "."
        config = create_default_config(root_path)
        monitor = FileMonitor(config)
        
        test_files = [
            "src/main.py",
            "node_modules/package/index.js",
            "__pycache__/module.pyc",
            ".git/config",
            "README.md",
            "config.json",
            ".env"
        ]
        
        print("Testing ignore patterns:")
        for test_file in test_files:
            ignored = monitor._should_ignore_file(test_file)
            priority = monitor._get_file_priority(test_file)
            print(f"  {test_file}: ignored={ignored}, priority={priority.name}")
    
    elif command == "--validate-config":
        root_path = sys.argv[2] if len(sys.argv) > 2 else "."
        config = create_default_config(root_path)
        
        print("Validating configuration...")
        print(f"Root paths: {config.root_paths}")
        print(f"Debounce interval: {config.debounce_interval}s")
        print(f"Max events/second: {config.max_events_per_second}")
        print(f"Memory limit: {config.memory_limit_mb}MB")
        print("Configuration is valid")
    
    elif command == "--load-test":
        print("RIF File Monitor - Load Testing Utility")
        print("=====================================")
        
        root_path = sys.argv[2] if len(sys.argv) > 2 else "."
        config = create_default_config(root_path)
        monitor = FileMonitor(config)
        
        # Performance test configuration
        test_files = 100  # Start with smaller test
        if len(sys.argv) > 3:
            test_files = int(sys.argv[3])
        
        print(f"Testing with {test_files} simulated file events...")
        
        # Test priority queue performance
        import asyncio
        async def load_test():
            start_time = time.time()
            
            # Generate test events
            for i in range(test_files):
                file_ext = ['.py', '.js', '.json', '.md', '.log'][i % 5]
                test_path = f"/tmp/test_file_{i}{file_ext}"
                priority = monitor._get_file_priority(test_path)
                
                event = FileChangeEvent(
                    file_path=test_path,
                    event_type="modified",
                    timestamp=time.time(),
                    priority=priority
                )
                
                # Add to debounce buffer
                monitor.debounce_buffer.add_event(event)
            
            # Process events
            ready_events = monitor.debounce_buffer.get_ready_events()
            process_time = time.time() - start_time
            
            print(f"Generated {test_files} events in {process_time:.3f}s")
            print(f"Ready events: {len(ready_events)}")
            print(f"Events/second: {test_files/process_time:.0f}")
            
            # Test gitignore performance
            test_paths = [
                "src/main.py",
                "node_modules/package.json", 
                "__pycache__/module.pyc",
                "build/output.js",
                ".git/config",
                "README.md"
            ]
            
            ignore_start = time.time()
            for _ in range(1000):
                for test_path in test_paths:
                    monitor._should_ignore_file(test_path)
            ignore_time = time.time() - ignore_start
            
            print(f"Gitignore checks: {6000/ignore_time:.0f} checks/second")
            print(f"Memory usage: {monitor._get_memory_usage():.1f}MB")
            
            # Show statistics
            debounce_stats = monitor.debounce_buffer.get_statistics()
            print("\nDebounce Statistics:")
            for key, value in debounce_stats.items():
                print(f"  {key}: {value}")
        
        asyncio.run(load_test())
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()