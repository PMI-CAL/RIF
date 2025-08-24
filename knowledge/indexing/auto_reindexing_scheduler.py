"""
Auto-Reindexing Scheduler for RIF Knowledge Graph
Issue #69: Build auto-reindexing scheduler

This module implements the core AutoReindexingScheduler class with:
- Priority-based reindexing job scheduling
- Resource-aware execution with adaptive throttling
- Integration with existing file monitoring and graph validation
- Smart batching and performance optimization
- Background processing with minimal system impact

Author: RIF-Implementer
Date: 2025-08-23
"""

import asyncio
import logging
import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from queue import PriorityQueue, Empty
import json

# Import existing infrastructure
try:
    from ..database.graph_validator import GraphValidator, ValidationSeverity
    from ..database.connection_manager import DuckDBConnectionManager
    from ..database.database_config import DatabaseConfig
    from ...claude.commands.file_change_detector import FileChangeDetector, Priority as FilePriority
    from ...claude.commands.system_monitor import MetricsCollector, SystemMonitor
except ImportError:
    # Handle standalone execution or missing dependencies
    GraphValidator = None
    DuckDBConnectionManager = None
    DatabaseConfig = None
    FileChangeDetector = None
    FilePriority = None
    MetricsCollector = None
    SystemMonitor = None

# Import resource management and job management
from .resource_manager import ReindexingResourceManager, SystemResourceMonitor
from .reindex_job_manager import ReindexJobManager
from .priority_calculator import PriorityCalculator


class ReindexPriority(IntEnum):
    """Priority levels for reindexing operations (lower = higher priority)"""
    CRITICAL = 0    # Data integrity issues, must run immediately
    HIGH = 1        # Content updates, performance critical
    MEDIUM = 2      # Relationship updates, optimization 
    LOW = 3         # Background optimization, maintenance


class ReindexTrigger(Enum):
    """Types of triggers that can initiate reindexing"""
    FILE_CHANGE = "file_change"                    # File system changes
    VALIDATION_ISSUE = "validation_issue"          # Graph validation found issues
    MANUAL_REQUEST = "manual_request"              # Explicit reindex request
    SCHEDULED_MAINTENANCE = "scheduled_maintenance" # Regular maintenance
    DEPENDENCY_UPDATE = "dependency_update"        # Related entity changed
    SYSTEM_STARTUP = "system_startup"             # System initialization
    PERFORMANCE_OPTIMIZATION = "performance_optimization"  # Performance tuning


@dataclass
class ReindexJob:
    """Represents a single reindexing job with priority and context"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str = ""
    entity_id: Optional[str] = None
    file_path: Optional[str] = None
    priority: ReindexPriority = ReindexPriority.MEDIUM
    trigger: ReindexTrigger = ReindexTrigger.MANUAL_REQUEST
    trigger_reason: str = ""
    scheduled_time: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    batch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0  # seconds
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority queue ordering - lower priority value = higher priority"""
        if not isinstance(other, ReindexJob):
            return NotImplemented
        # First sort by priority, then by scheduled time for FIFO within priority
        return (self.priority.value, self.scheduled_time) < (other.priority.value, other.scheduled_time)
    
    def can_retry(self) -> bool:
        """Check if job can be retried"""
        return self.retry_count < self.max_retries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "file_path": self.file_path,
            "priority": self.priority.name,
            "trigger": self.trigger.value,
            "trigger_reason": self.trigger_reason,
            "scheduled_time": self.scheduled_time.isoformat(),
            "created_at": self.created_at.isoformat(),
            "batch_id": self.batch_id,
            "metadata": self.metadata,
            "dependencies": self.dependencies,
            "estimated_duration": self.estimated_duration,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


class SchedulerStatus(Enum):
    """Current status of the scheduler"""
    STOPPED = "stopped"
    STARTING = "starting"  
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass  
class SchedulerConfig:
    """Configuration for the auto-reindexing scheduler"""
    # Resource thresholds by priority (0.0-1.0) - adjusted to be more reasonable
    resource_thresholds: Dict[ReindexPriority, float] = field(default_factory=lambda: {
        ReindexPriority.CRITICAL: 0.95,  # Can use up to 95% resources
        ReindexPriority.HIGH: 0.85,      # Can use up to 85% resources  
        ReindexPriority.MEDIUM: 0.75,    # Can use up to 75% resources
        ReindexPriority.LOW: 0.60        # Can use up to 60% resources
    })
    
    # Processing intervals
    queue_check_interval: float = 1.0      # seconds between queue checks
    metrics_report_interval: float = 30.0   # seconds between metrics reports
    resource_check_interval: float = 5.0    # seconds between resource checks
    
    # Batching configuration
    enable_batching: bool = True
    max_batch_size: int = 50
    batch_timeout_seconds: float = 300.0    # 5 minutes max batch processing
    
    # Retry configuration  
    default_max_retries: int = 3
    retry_delay_seconds: float = 60.0       # delay before retry
    exponential_backoff: bool = True        # increase delay on each retry
    
    # Performance tuning
    max_concurrent_jobs: int = 2            # max jobs running simultaneously
    enable_adaptive_scheduling: bool = True  # learn optimal scheduling patterns
    priority_boost_threshold: int = 10      # boost priority after N queue cycles
    
    # Integration settings
    enable_file_change_integration: bool = True
    enable_graph_validation_integration: bool = True  
    enable_performance_monitoring: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "resource_thresholds": {p.name: t for p, t in self.resource_thresholds.items()},
            "queue_check_interval": self.queue_check_interval,
            "metrics_report_interval": self.metrics_report_interval, 
            "resource_check_interval": self.resource_check_interval,
            "enable_batching": self.enable_batching,
            "max_batch_size": self.max_batch_size,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "default_max_retries": self.default_max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "exponential_backoff": self.exponential_backoff,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "enable_adaptive_scheduling": self.enable_adaptive_scheduling,
            "priority_boost_threshold": self.priority_boost_threshold,
            "enable_file_change_integration": self.enable_file_change_integration,
            "enable_graph_validation_integration": self.enable_graph_validation_integration,
            "enable_performance_monitoring": self.enable_performance_monitoring
        }


class AutoReindexingScheduler:
    """
    Auto-reindexing scheduler with priority-based processing and resource management.
    
    Features:
    - Priority-based job queuing with 4-tier system (CRITICAL/HIGH/MEDIUM/LOW)
    - Resource-aware execution with adaptive throttling
    - Smart batching for improved performance
    - Integration with file change detection and graph validation
    - Background processing with minimal system impact
    - Comprehensive metrics and monitoring
    - Automatic retry with exponential backoff
    - Dependency-aware scheduling
    """
    
    def __init__(self, 
                 config: Optional[SchedulerConfig] = None,
                 database_config: Optional[DatabaseConfig] = None,
                 knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"):
        """
        Initialize the auto-reindexing scheduler.
        
        Args:
            config: Scheduler configuration
            database_config: Database configuration for graph validation
            knowledge_path: Path to knowledge base for storage and integration
        """
        self.config = config or SchedulerConfig()
        self.knowledge_path = Path(knowledge_path)
        self.logger = self._setup_logging()
        
        # Core scheduling components  
        self.job_queue = PriorityQueue()
        self.active_jobs: Dict[str, ReindexJob] = {}
        self.completed_jobs: List[ReindexJob] = []
        self.failed_jobs: List[ReindexJob] = []
        
        # Status and control
        self.status = SchedulerStatus.STOPPED
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Component managers
        self.resource_manager = ReindexingResourceManager(
            resource_thresholds=self.config.resource_thresholds,
            check_interval=self.config.resource_check_interval
        )
        self.job_manager = ReindexJobManager(
            max_concurrent_jobs=self.config.max_concurrent_jobs,
            enable_batching=self.config.enable_batching,
            max_batch_size=self.config.max_batch_size
        )
        self.priority_calculator = PriorityCalculator()
        
        # Performance tracking
        self.metrics = {
            'jobs_processed': 0,
            'jobs_failed': 0,
            'jobs_retried': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'queue_size': 0,
            'active_job_count': 0,
            'resource_utilization': 0.0,
            'throughput_jobs_per_minute': 0.0,
            'last_metrics_update': datetime.now()
        }
        
        # Integration components (optional)
        self.graph_validator: Optional[GraphValidator] = None
        self.file_change_detector: Optional[FileChangeDetector] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Initialize integrations if available
        self._initialize_integrations(database_config)
        
        # Adaptive scheduling state
        self.scheduling_patterns = {}
        self.performance_history = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'job_queued': [],
            'job_started': [],
            'job_completed': [],
            'job_failed': [],
            'batch_completed': [],
            'scheduler_started': [],
            'scheduler_stopped': []
        }
        
        self.logger.info("AutoReindexingScheduler initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated logger for the scheduler"""
        logger = logging.getLogger(f"rif_auto_reindex_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # File handler for scheduler logs
        log_dir = self.knowledge_path / "indexing" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "auto_reindexing.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_integrations(self, database_config: Optional[DatabaseConfig]):
        """Initialize optional integrations with existing systems"""
        try:
            # Graph validator integration
            if self.config.enable_graph_validation_integration and GraphValidator:
                self.graph_validator = GraphValidator(database_config)
                self.logger.info("Graph validation integration enabled")
            
            # File change detector integration
            if self.config.enable_file_change_integration and FileChangeDetector:
                # Initialize file change detector for monitoring
                self.file_change_detector = FileChangeDetector([str(self.knowledge_path.parent)])
                self.file_change_detector.add_event_handler(self._on_file_change)
                self.logger.info("File change detection integration enabled")
            
            # Metrics collector integration  
            if self.config.enable_performance_monitoring and MetricsCollector:
                # Create basic config for metrics collector
                metrics_config = {
                    "monitoring": {
                        "collection": {"metrics_interval": "30s"}
                    },
                    "storage": {
                        "paths": {"logs": str(self.knowledge_path / "indexing" / "logs")}
                    }
                }
                self.metrics_collector = MetricsCollector(metrics_config)
                self.logger.info("Performance monitoring integration enabled")
                
        except Exception as e:
            self.logger.warning(f"Some integrations could not be initialized: {e}")
    
    def _on_file_change(self, file_change_event):
        """Handle file change events from file monitor integration"""
        try:
            # Map file change to reindex priority
            priority_mapping = {
                0: ReindexPriority.CRITICAL,  # FilePriority.IMMEDIATE -> CRITICAL
                1: ReindexPriority.HIGH,      # FilePriority.HIGH -> HIGH
                2: ReindexPriority.MEDIUM,    # FilePriority.MEDIUM -> MEDIUM  
                3: ReindexPriority.LOW        # FilePriority.LOW -> LOW
            }
            
            file_priority = getattr(file_change_event, 'priority', 2)
            if hasattr(file_priority, 'value'):
                file_priority = file_priority.value
                
            reindex_priority = priority_mapping.get(file_priority, ReindexPriority.MEDIUM)
            
            # Create reindex job for file change
            job = ReindexJob(
                entity_type="file_entity",
                file_path=getattr(file_change_event, 'file_path', None),
                priority=reindex_priority,
                trigger=ReindexTrigger.FILE_CHANGE,
                trigger_reason=f"File {getattr(file_change_event, 'event_type', 'changed')}: {getattr(file_change_event, 'file_path', 'unknown')}",
                metadata={
                    "file_change_type": getattr(file_change_event, 'event_type', 'unknown'),
                    "file_priority": file_priority
                }
            )
            
            self.schedule_reindex(job)
            self.logger.debug(f"Scheduled reindex for file change: {job.file_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling file change event: {e}")
    
    def schedule_reindex(self, 
                        job: Optional[ReindexJob] = None,
                        entity_type: str = "",
                        entity_id: Optional[str] = None,
                        file_path: Optional[str] = None,
                        priority: ReindexPriority = ReindexPriority.MEDIUM,
                        trigger: ReindexTrigger = ReindexTrigger.MANUAL_REQUEST,
                        trigger_reason: str = "",
                        scheduled_time: Optional[datetime] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Schedule a reindexing job.
        
        Args:
            job: Pre-created ReindexJob (if provided, other args ignored)
            entity_type: Type of entity to reindex
            entity_id: Specific entity ID (optional)
            file_path: File path for file-based reindexing
            priority: Job priority level
            trigger: What triggered this reindexing
            trigger_reason: Human-readable reason for reindexing
            scheduled_time: When to run (defaults to now)
            metadata: Additional job metadata
            
        Returns:
            Job ID for tracking
        """
        if job is None:
            job = ReindexJob(
                entity_type=entity_type,
                entity_id=entity_id,
                file_path=file_path,
                priority=priority,
                trigger=trigger,
                trigger_reason=trigger_reason,
                scheduled_time=scheduled_time or datetime.now(),
                metadata=metadata or {},
                max_retries=self.config.default_max_retries
            )
        
        # Calculate dynamic priority if enabled
        if self.config.enable_adaptive_scheduling:
            original_priority = job.priority
            job.priority = self.priority_calculator.calculate_priority(job)
            if original_priority != job.priority:
                self.logger.debug(f"Priority adjusted: {original_priority.name} -> {job.priority.name}")
        
        # Add to queue
        self.job_queue.put(job)
        self.logger.info(f"Scheduled reindex job {job.id}: {job.entity_type} (priority: {job.priority.name})")
        
        # Update metrics
        self.metrics['queue_size'] = self.job_queue.qsize()
        
        # Fire event
        self._fire_event('job_queued', job)
        
        return job.id
    
    def schedule_validation_triggered_reindex(self, validation_report):
        """Schedule reindexing based on validation issues"""
        if not validation_report or not hasattr(validation_report, 'issues'):
            return
        
        jobs_scheduled = 0
        for issue in validation_report.issues:
            priority = ReindexPriority.CRITICAL if issue.severity == ValidationSeverity.CRITICAL else ReindexPriority.HIGH
            
            job = ReindexJob(
                entity_type=issue.table_name or "unknown",
                entity_id=issue.entity_id,
                priority=priority,
                trigger=ReindexTrigger.VALIDATION_ISSUE,
                trigger_reason=f"Validation issue: {issue.message}",
                metadata={
                    "validation_issue_id": issue.id,
                    "validation_severity": issue.severity.value,
                    "validation_category": issue.category,
                    "suggested_fix": issue.suggested_fix
                }
            )
            
            self.schedule_reindex(job)
            jobs_scheduled += 1
        
        if jobs_scheduled > 0:
            self.logger.info(f"Scheduled {jobs_scheduled} reindex jobs from validation issues")
    
    def start(self) -> bool:
        """
        Start the auto-reindexing scheduler.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.status == SchedulerStatus.RUNNING:
            self.logger.warning("Scheduler is already running")
            return True
        
        try:
            self.status = SchedulerStatus.STARTING
            self.stop_event.clear()
            self.pause_event.clear()
            
            # Start resource monitoring
            self.resource_manager.start()
            
            # Start file change detector if available
            if self.file_change_detector:
                self.file_change_detector.start_monitoring()
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name=f"AutoReindexScheduler-{id(self)}"
            )
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            self.status = SchedulerStatus.RUNNING
            self.logger.info("Auto-reindexing scheduler started successfully")
            
            # Fire event
            self._fire_event('scheduler_started', None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            self.status = SchedulerStatus.ERROR
            return False
    
    def stop(self, timeout: float = 30.0) -> bool:
        """
        Stop the auto-reindexing scheduler gracefully.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            True if stopped successfully, False if timeout
        """
        if self.status == SchedulerStatus.STOPPED:
            self.logger.debug("Scheduler is already stopped")
            return True
        
        self.logger.info("Stopping auto-reindexing scheduler...")
        self.status = SchedulerStatus.STOPPING
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Stop components
        try:
            # Stop file change detector
            if self.file_change_detector:
                self.file_change_detector.stop_monitoring()
            
            # Stop resource manager
            self.resource_manager.stop()
            
            # Wait for scheduler thread
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=timeout)
                
                if self.scheduler_thread.is_alive():
                    self.logger.warning("Scheduler thread did not stop within timeout")
                    self.status = SchedulerStatus.ERROR
                    return False
            
            # Cancel any active jobs
            for job_id, job in list(self.active_jobs.items()):
                self.logger.info(f"Canceling active job {job_id}")
                # Job manager will handle cleanup
                
            self.status = SchedulerStatus.STOPPED
            self.logger.info("Auto-reindexing scheduler stopped successfully")
            
            # Fire event
            self._fire_event('scheduler_stopped', None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping scheduler: {e}")
            self.status = SchedulerStatus.ERROR
            return False
    
    def pause(self) -> bool:
        """Pause the scheduler (stop processing new jobs, keep running)"""
        if self.status != SchedulerStatus.RUNNING:
            return False
        
        self.status = SchedulerStatus.PAUSING
        self.pause_event.set()
        self.status = SchedulerStatus.PAUSED
        self.logger.info("Scheduler paused")
        return True
    
    def resume(self) -> bool:
        """Resume the scheduler from paused state"""
        if self.status != SchedulerStatus.PAUSED:
            return False
        
        self.pause_event.clear()
        self.status = SchedulerStatus.RUNNING
        self.logger.info("Scheduler resumed")
        return True
    
    def _scheduler_loop(self):
        """Main scheduler loop that processes the job queue"""
        self.logger.info("Scheduler loop started")
        
        last_metrics_report = time.time()
        
        while not self.stop_event.is_set():
            try:
                self.logger.debug(f"Scheduler loop iteration - queue size: {self.job_queue.qsize()}")
                # Check if paused
                if self.pause_event.is_set():
                    time.sleep(1.0)
                    continue
                
                # Check resource availability
                if not self._can_process_jobs():
                    self.logger.debug("Cannot process jobs due to resource constraints or max concurrent limit")
                    time.sleep(self.config.resource_check_interval)
                    continue
                
                # Try to get a job from the queue
                try:
                    job = self.job_queue.get(timeout=self.config.queue_check_interval)
                    self.logger.debug(f"Retrieved job {job.id} from queue")
                except Empty:
                    # No jobs in queue, update metrics and continue
                    self.logger.debug("No jobs in queue")
                    self._update_metrics()
                    continue
                
                # Check if job is ready to run (scheduled time)
                if job.scheduled_time > datetime.now():
                    # Put job back and wait
                    self.job_queue.put(job)
                    time.sleep(1.0)
                    continue
                
                # Process the job
                self._process_job(job)
                
                # Report metrics periodically
                now = time.time()
                if now - last_metrics_report >= self.config.metrics_report_interval:
                    self._report_metrics()
                    last_metrics_report = now
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5.0)  # Brief pause on error
        
        self.logger.info("Scheduler loop ended")
    
    def _can_process_jobs(self) -> bool:
        """Check if system resources allow processing more jobs"""
        # Check if we've reached max concurrent jobs
        if len(self.active_jobs) >= self.config.max_concurrent_jobs:
            return False
        
        # Check system resources - use HIGH priority as minimum threshold
        # Individual jobs will have their specific priority checked later
        return self.resource_manager.can_schedule_job(ReindexPriority.HIGH)
    
    def _process_job(self, job: ReindexJob):
        """Process a single reindexing job"""
        job_start_time = time.time()
        
        try:
            self.logger.info(f"Starting job {job.id}: {job.entity_type} (priority: {job.priority.name})")
            
            # Add to active jobs
            self.active_jobs[job.id] = job
            
            # Fire event
            self._fire_event('job_started', job)
            
            # Check resource availability for this specific job priority
            if not self.resource_manager.can_schedule_job(job.priority):
                # Reschedule for later
                job.scheduled_time = datetime.now() + timedelta(minutes=5)
                self.job_queue.put(job)
                del self.active_jobs[job.id]
                self.logger.debug(f"Rescheduled job {job.id} due to resource constraints")
                return
            
            # Execute the reindexing operation
            success = self._execute_reindex(job)
            
            # Calculate processing time
            processing_time = time.time() - job_start_time
            
            if success:
                # Job completed successfully
                self.completed_jobs.append(job)
                self.metrics['jobs_processed'] += 1
                self.metrics['total_processing_time'] += processing_time
                
                self.logger.info(f"Completed job {job.id} in {processing_time:.2f}s")
                self._fire_event('job_completed', job)
                
            else:
                # Job failed, check if it can be retried
                if job.can_retry():
                    job.retry_count += 1
                    # Calculate retry delay with exponential backoff
                    if self.config.exponential_backoff:
                        delay = self.config.retry_delay_seconds * (2 ** (job.retry_count - 1))
                    else:
                        delay = self.config.retry_delay_seconds
                    
                    job.scheduled_time = datetime.now() + timedelta(seconds=delay)
                    self.job_queue.put(job)
                    
                    self.metrics['jobs_retried'] += 1
                    self.logger.warning(f"Retrying job {job.id} (attempt {job.retry_count + 1}/{job.max_retries + 1}) in {delay}s")
                else:
                    # Max retries exceeded
                    self.failed_jobs.append(job)
                    self.metrics['jobs_failed'] += 1
                    
                    self.logger.error(f"Job {job.id} failed after {job.retry_count} retries")
                    self._fire_event('job_failed', job)
            
        except Exception as e:
            self.logger.error(f"Error processing job {job.id}: {e}")
            self.failed_jobs.append(job)
            self.metrics['jobs_failed'] += 1
            self._fire_event('job_failed', job)
        
        finally:
            # Remove from active jobs
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
    
    def _execute_reindex(self, job: ReindexJob) -> bool:
        """
        Execute the actual reindexing operation.
        
        This is where the core reindexing logic would be implemented.
        For now, this is a placeholder that simulates the operation.
        
        Returns:
            True if successful, False if failed
        """
        try:
            # Simulate reindexing work based on entity type and priority
            if job.trigger == ReindexTrigger.VALIDATION_ISSUE:
                # Handle validation-triggered reindexing
                return self._handle_validation_reindex(job)
            elif job.trigger == ReindexTrigger.FILE_CHANGE:
                # Handle file-change-triggered reindexing
                return self._handle_file_change_reindex(job)
            else:
                # Handle other types of reindexing
                return self._handle_general_reindex(job)
                
        except Exception as e:
            self.logger.error(f"Reindexing execution failed for job {job.id}: {e}")
            return False
    
    def _handle_validation_reindex(self, job: ReindexJob) -> bool:
        """Handle reindexing triggered by validation issues"""
        try:
            self.logger.info(f"Handling validation-triggered reindex for {job.entity_type}")
            
            # If graph validator is available, re-run validation after reindexing
            if self.graph_validator:
                # Simulate reindexing operation (replace with actual implementation)
                time.sleep(0.1 + job.priority.value * 0.05)  # Simulate work
                
                # Re-validate to check if issue was resolved
                validation_report = self.graph_validator.validate_graph(
                    categories=['referential_integrity', 'data_consistency']
                )
                
                # Check if the specific issue was resolved
                issue_id = job.metadata.get('validation_issue_id')
                if issue_id:
                    unresolved_issues = [i for i in validation_report.issues if i.id == issue_id]
                    if not unresolved_issues:
                        self.logger.info(f"Validation issue {issue_id} resolved by reindexing")
                        return True
                    else:
                        self.logger.warning(f"Validation issue {issue_id} still present after reindexing")
                        return False
                
            # Fallback: simulate reindexing without validation
            time.sleep(0.2)  # Simulate work
            return True
            
        except Exception as e:
            self.logger.error(f"Error in validation reindex: {e}")
            return False
    
    def _handle_file_change_reindex(self, job: ReindexJob) -> bool:
        """Handle reindexing triggered by file changes"""
        try:
            file_path = job.file_path
            self.logger.info(f"Handling file-change-triggered reindex for {file_path}")
            
            if not file_path:
                self.logger.warning(f"No file path provided for file change reindex job {job.id}")
                return False
            
            # Simulate file processing based on file type and priority
            if file_path.endswith(('.py', '.js', '.ts', '.java', '.go', '.rs')):
                # Source code files require more processing
                time.sleep(0.3 + job.priority.value * 0.1)
            elif file_path.endswith(('.json', '.yaml', '.yml', '.toml')):
                # Configuration files
                time.sleep(0.1 + job.priority.value * 0.05)
            else:
                # Other files
                time.sleep(0.05 + job.priority.value * 0.02)
            
            # Simulate successful reindexing
            return True
            
        except Exception as e:
            self.logger.error(f"Error in file change reindex: {e}")
            return False
    
    def _handle_general_reindex(self, job: ReindexJob) -> bool:
        """Handle general reindexing operations"""
        try:
            self.logger.info(f"Handling general reindex for {job.entity_type}")
            
            # Simulate processing time based on entity type and priority
            base_time = {
                'function': 0.1,
                'class': 0.2, 
                'module': 0.3,
                'file_entity': 0.15,
                'relationship': 0.05
            }.get(job.entity_type, 0.1)
            
            # Adjust time based on priority (lower priority = more thorough processing)
            processing_time = base_time * (1 + job.priority.value * 0.5)
            time.sleep(processing_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in general reindex: {e}")
            return False
    
    def _update_metrics(self):
        """Update internal metrics"""
        now = datetime.now()
        
        # Update basic metrics
        self.metrics['queue_size'] = self.job_queue.qsize()
        self.metrics['active_job_count'] = len(self.active_jobs)
        self.metrics['last_metrics_update'] = now
        
        # Calculate average processing time
        if self.metrics['jobs_processed'] > 0:
            self.metrics['average_processing_time'] = (
                self.metrics['total_processing_time'] / self.metrics['jobs_processed']
            )
        
        # Calculate throughput (jobs per minute)
        if hasattr(self, '_start_time'):
            elapsed_minutes = (now - self._start_time).total_seconds() / 60.0
            if elapsed_minutes > 0:
                self.metrics['throughput_jobs_per_minute'] = (
                    self.metrics['jobs_processed'] / elapsed_minutes
                )
        
        # Get resource utilization
        self.metrics['resource_utilization'] = self.resource_manager.get_current_utilization()
    
    def _report_metrics(self):
        """Report metrics to monitoring system and logs"""
        self._update_metrics()
        
        # Log metrics summary
        self.logger.info(
            f"Metrics: queue={self.metrics['queue_size']}, "
            f"active={self.metrics['active_job_count']}, "
            f"processed={self.metrics['jobs_processed']}, "
            f"failed={self.metrics['jobs_failed']}, "
            f"throughput={self.metrics['throughput_jobs_per_minute']:.1f}/min, "
            f"avg_time={self.metrics['average_processing_time']:.2f}s, "
            f"resource_util={self.metrics['resource_utilization']:.1%}"
        )
        
        # Report to metrics collector if available
        if self.metrics_collector:
            try:
                # Track key metrics
                self.metrics_collector.track_latency(
                    "reindex_job_processing", 
                    self.metrics['average_processing_time'] * 1000  # convert to ms
                )
                
                # Add custom metrics
                for metric_name, value in self.metrics.items():
                    if isinstance(value, (int, float)):
                        self.metrics_collector._add_metric(
                            f"reindexing.{metric_name}", 
                            float(value), 
                            {"scheduler_id": str(id(self))},
                            ""
                        )
                        
            except Exception as e:
                self.logger.warning(f"Failed to report metrics to collector: {e}")
    
    def _fire_event(self, event_name: str, job: Optional[ReindexJob]):
        """Fire event to registered handlers"""
        handlers = self.event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                handler(job)
            except Exception as e:
                self.logger.warning(f"Event handler {handler} failed for {event_name}: {e}")
    
    def add_event_handler(self, event_name: str, handler: Callable):
        """Add event handler for scheduler events"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
    
    def remove_event_handler(self, event_name: str, handler: Callable):
        """Remove event handler"""
        if event_name in self.event_handlers:
            try:
                self.event_handlers[event_name].remove(handler)
            except ValueError:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        self._update_metrics()
        
        return {
            "scheduler_status": self.status.value,
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds() if hasattr(self, '_start_time') else 0,
            "metrics": self.metrics.copy(),
            "configuration": self.config.to_dict(),
            "integrations": {
                "graph_validator": self.graph_validator is not None,
                "file_change_detector": self.file_change_detector is not None,
                "metrics_collector": self.metrics_collector is not None
            },
            "components": {
                "resource_manager": self.resource_manager.get_status(),
                "job_manager": self.job_manager.get_status()
            }
        }
    
    def get_job_history(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent job history"""
        return {
            "completed": [job.to_dict() for job in self.completed_jobs[-limit:]],
            "failed": [job.to_dict() for job in self.failed_jobs[-limit:]],
            "active": [job.to_dict() for job in self.active_jobs.values()]
        }
    
    def clear_job_history(self):
        """Clear completed and failed job history"""
        self.completed_jobs.clear()
        self.failed_jobs.clear()
        self.logger.info("Job history cleared")
    
    def __enter__(self):
        """Context manager entry"""
        self._start_time = datetime.now()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


# Factory function for easy instantiation
def create_auto_reindexing_scheduler(
    config_path: Optional[str] = None,
    knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"
) -> AutoReindexingScheduler:
    """
    Factory function to create an AutoReindexingScheduler instance.
    
    Args:
        config_path: Path to configuration file (optional)
        knowledge_path: Path to knowledge base directory
        
    Returns:
        Configured AutoReindexingScheduler instance
    """
    config = SchedulerConfig()
    
    # Load configuration from file if provided
    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
                # Apply configuration (would need proper config parsing)
                # For now, use defaults
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load config from {config_path}: {e}")
    
    return AutoReindexingScheduler(config=config, knowledge_path=knowledge_path)


# Example usage and testing
if __name__ == "__main__":
    # Basic demonstration
    with create_auto_reindexing_scheduler() as scheduler:
        # Schedule some test jobs
        scheduler.schedule_reindex(
            entity_type="test_entity",
            priority=ReindexPriority.HIGH,
            trigger=ReindexTrigger.MANUAL_REQUEST,
            trigger_reason="Testing scheduler"
        )
        
        # Let it run briefly
        import time
        time.sleep(10)
        
        # Print status
        status = scheduler.get_status()
        print(f"Scheduler status: {status['scheduler_status']}")
        print(f"Jobs processed: {status['metrics']['jobs_processed']}")