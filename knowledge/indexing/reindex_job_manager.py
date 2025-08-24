"""
Reindex Job Manager for Auto-Reindexing Scheduler
Issue #69: Build auto-reindexing scheduler

This module provides job lifecycle management and batch processing optimization:
- Job lifecycle tracking and state management
- Batch processing optimization for related jobs
- Dependency resolution and scheduling
- Performance monitoring and job metrics

Author: RIF-Implementer
Date: 2025-08-23
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid


class JobStatus(Enum):
    """Current status of a reindexing job"""
    QUEUED = "queued"           # Job is in queue waiting to be processed
    SCHEDULED = "scheduled"     # Job is scheduled for future execution  
    RUNNING = "running"         # Job is currently being processed
    COMPLETED = "completed"     # Job completed successfully
    FAILED = "failed"           # Job failed after retries
    CANCELLED = "cancelled"     # Job was cancelled before execution
    RETRYING = "retrying"       # Job is being retried after failure


@dataclass
class JobMetrics:
    """Metrics tracking for individual jobs"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    retry_count: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    io_operations: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "io_operations": self.io_operations,
            "error_message": self.error_message
        }


@dataclass
class JobBatch:
    """Represents a batch of related jobs for optimized processing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    jobs: List[str] = field(default_factory=list)  # Job IDs
    batch_type: str = "general"  # file_based, entity_based, priority_based
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.QUEUED
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "jobs": self.jobs.copy(),
            "batch_type": self.batch_type,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "job_count": len(self.jobs)
        }


class JobLifecycleManager:
    """
    Manages the complete lifecycle of reindexing jobs from creation to completion.
    Provides state tracking, metrics collection, and event handling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"job_lifecycle_{id(self)}")
        
        # Job state tracking
        self.job_states: Dict[str, JobStatus] = {}
        self.job_metrics: Dict[str, JobMetrics] = {}
        
        # State transition history
        self.state_history: Dict[str, List[Tuple[JobStatus, datetime]]] = defaultdict(list)
        
        # Event handlers for job state changes
        self.state_change_handlers: Dict[JobStatus, List[callable]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("JobLifecycleManager initialized")
    
    def register_job(self, job_id: str, initial_status: JobStatus = JobStatus.QUEUED) -> bool:
        """
        Register a new job for lifecycle management.
        
        Args:
            job_id: Unique job identifier
            initial_status: Initial job status
            
        Returns:
            True if registered successfully, False if job already exists
        """
        with self.lock:
            if job_id in self.job_states:
                self.logger.warning(f"Job {job_id} already registered")
                return False
            
            # Initialize job state
            self.job_states[job_id] = initial_status
            self.job_metrics[job_id] = JobMetrics()
            
            # Record initial state transition
            self.state_history[job_id].append((initial_status, datetime.now()))
            
            self.logger.debug(f"Registered job {job_id} with status {initial_status.value}")
            
            # Fire state change event
            self._fire_state_change_event(job_id, initial_status)
            
            return True
    
    def transition_job_state(self, job_id: str, new_status: JobStatus, error_message: Optional[str] = None) -> bool:
        """
        Transition a job to a new status.
        
        Args:
            job_id: Job identifier
            new_status: New status to transition to
            error_message: Error message if transitioning to failed status
            
        Returns:
            True if transition was successful, False otherwise
        """
        with self.lock:
            if job_id not in self.job_states:
                self.logger.error(f"Cannot transition unknown job {job_id}")
                return False
            
            old_status = self.job_states[job_id]
            
            # Validate state transition
            if not self._is_valid_transition(old_status, new_status):
                self.logger.error(f"Invalid state transition for job {job_id}: {old_status.value} -> {new_status.value}")
                return False
            
            # Update job state
            self.job_states[job_id] = new_status
            
            # Record state transition
            transition_time = datetime.now()
            self.state_history[job_id].append((new_status, transition_time))
            
            # Update metrics based on new status
            self._update_job_metrics(job_id, new_status, error_message)
            
            self.logger.debug(f"Job {job_id} transitioned: {old_status.value} -> {new_status.value}")
            
            # Fire state change event
            self._fire_state_change_event(job_id, new_status)
            
            return True
    
    def _is_valid_transition(self, from_status: JobStatus, to_status: JobStatus) -> bool:
        """Check if a state transition is valid"""
        # Define valid state transitions
        valid_transitions = {
            JobStatus.QUEUED: {JobStatus.SCHEDULED, JobStatus.RUNNING, JobStatus.CANCELLED},
            JobStatus.SCHEDULED: {JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.QUEUED},
            JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
            JobStatus.FAILED: {JobStatus.RETRYING, JobStatus.CANCELLED},
            JobStatus.RETRYING: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED},
            JobStatus.COMPLETED: set(),  # Terminal state
            JobStatus.CANCELLED: set()   # Terminal state
        }
        
        return to_status in valid_transitions.get(from_status, set())
    
    def _update_job_metrics(self, job_id: str, new_status: JobStatus, error_message: Optional[str]):
        """Update job metrics based on status transition"""
        metrics = self.job_metrics[job_id]
        now = datetime.now()
        
        if new_status == JobStatus.RUNNING:
            metrics.start_time = now
        elif new_status == JobStatus.COMPLETED:
            metrics.end_time = now
            if metrics.start_time:
                metrics.duration_seconds = (now - metrics.start_time).total_seconds()
        elif new_status == JobStatus.FAILED:
            metrics.end_time = now
            metrics.error_message = error_message
            if metrics.start_time:
                metrics.duration_seconds = (now - metrics.start_time).total_seconds()
        elif new_status == JobStatus.RETRYING:
            metrics.retry_count += 1
    
    def _fire_state_change_event(self, job_id: str, new_status: JobStatus):
        """Fire state change event to registered handlers"""
        handlers = self.state_change_handlers[new_status]
        for handler in handlers:
            try:
                handler(job_id, new_status)
            except Exception as e:
                self.logger.error(f"State change handler failed: {e}")
    
    def add_state_change_handler(self, status: JobStatus, handler: callable):
        """Add a handler for job state changes"""
        self.state_change_handlers[status].append(handler)
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current status of a job"""
        return self.job_states.get(job_id)
    
    def get_job_metrics(self, job_id: str) -> Optional[JobMetrics]:
        """Get metrics for a job"""
        return self.job_metrics.get(job_id)
    
    def get_job_history(self, job_id: str) -> List[Tuple[JobStatus, datetime]]:
        """Get state transition history for a job"""
        return self.state_history.get(job_id, [])
    
    def get_jobs_by_status(self, status: JobStatus) -> List[str]:
        """Get list of job IDs with the specified status"""
        return [job_id for job_id, job_status in self.job_states.items() if job_status == status]
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed jobs from memory.
        
        Args:
            max_age_hours: Maximum age in hours for completed jobs
            
        Returns:
            Number of jobs cleaned up
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            jobs_to_remove = []
            
            for job_id in self.job_states:
                if self.job_states[job_id] in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
                    # Check if job is old enough to clean up
                    history = self.state_history[job_id]
                    if history and history[-1][1] < cutoff_time:
                        jobs_to_remove.append(job_id)
            
            # Remove old jobs
            for job_id in jobs_to_remove:
                del self.job_states[job_id]
                del self.job_metrics[job_id]
                del self.state_history[job_id]
            
            if jobs_to_remove:
                self.logger.info(f"Cleaned up {len(jobs_to_remove)} old completed jobs")
            
            return len(jobs_to_remove)


class ReindexJobManager:
    """
    Manages reindexing job execution with batch processing optimization,
    dependency resolution, and performance monitoring.
    """
    
    def __init__(self,
                 max_concurrent_jobs: int = 2,
                 enable_batching: bool = True,
                 max_batch_size: int = 50,
                 batch_timeout_seconds: float = 300.0):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        
        self.logger = logging.getLogger(f"job_manager_{id(self)}")
        
        # Job management
        self.lifecycle_manager = JobLifecycleManager()
        self.active_jobs: Dict[str, Any] = {}  # job_id -> job_info
        self.pending_batches: Dict[str, JobBatch] = {}
        
        # Performance tracking
        self.job_performance_history = deque(maxlen=1000)
        self.batch_performance_history = deque(maxlen=100)
        
        # Dependency tracking
        self.job_dependencies: Dict[str, Set[str]] = {}  # job_id -> set of dependency job_ids
        self.dependency_waiters: Dict[str, Set[str]] = {}  # dependency_id -> jobs waiting for it
        
        # Batch optimization
        self.batch_formation_rules = {
            "file_based": self._create_file_based_batches,
            "entity_based": self._create_entity_based_batches,
            "priority_based": self._create_priority_based_batches
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("ReindexJobManager initialized")
    
    def can_accept_job(self) -> bool:
        """Check if job manager can accept more jobs"""
        with self.lock:
            return len(self.active_jobs) < self.max_concurrent_jobs
    
    def submit_job(self, job: Any) -> bool:
        """
        Submit a job for processing.
        
        Args:
            job: ReindexJob instance to process
            
        Returns:
            True if job was accepted, False otherwise
        """
        if not self.can_accept_job():
            return False
        
        with self.lock:
            job_id = job.id
            
            # Register job with lifecycle manager
            if not self.lifecycle_manager.register_job(job_id):
                return False
            
            # Check for dependencies
            if job.dependencies:
                # Add to dependency tracking
                self.job_dependencies[job_id] = set(job.dependencies)
                
                # Register as waiter for each dependency
                for dep_id in job.dependencies:
                    if dep_id not in self.dependency_waiters:
                        self.dependency_waiters[dep_id] = set()
                    self.dependency_waiters[dep_id].add(job_id)
                
                # Check if dependencies are satisfied
                if not self._are_dependencies_satisfied(job_id):
                    self.lifecycle_manager.transition_job_state(job_id, JobStatus.SCHEDULED)
                    self.logger.debug(f"Job {job_id} waiting for dependencies: {job.dependencies}")
                    return True
            
            # Job is ready to run
            return self._execute_job(job)
    
    def _are_dependencies_satisfied(self, job_id: str) -> bool:
        """Check if all dependencies for a job are satisfied"""
        dependencies = self.job_dependencies.get(job_id, set())
        
        for dep_id in dependencies:
            dep_status = self.lifecycle_manager.get_job_status(dep_id)
            if dep_status != JobStatus.COMPLETED:
                return False
        
        return True
    
    def _execute_job(self, job: Any) -> bool:
        """Execute a single job"""
        job_id = job.id
        
        try:
            # Transition to running
            if not self.lifecycle_manager.transition_job_state(job_id, JobStatus.RUNNING):
                return False
            
            # Add to active jobs
            self.active_jobs[job_id] = {
                "job": job,
                "start_time": datetime.now(),
                "thread": None  # Would be set if using threading
            }
            
            self.logger.info(f"Executing job {job_id}: {job.entity_type}")
            
            # This is where actual job execution would happen
            # For now, simulate with a placeholder
            success = self._simulate_job_execution(job)
            
            # Update job state based on result
            if success:
                self.lifecycle_manager.transition_job_state(job_id, JobStatus.COMPLETED)
                self._handle_job_completion(job_id)
            else:
                self.lifecycle_manager.transition_job_state(job_id, JobStatus.FAILED, "Execution failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing job {job_id}: {e}")
            self.lifecycle_manager.transition_job_state(job_id, JobStatus.FAILED, str(e))
            return False
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    def _simulate_job_execution(self, job: Any) -> bool:
        """
        Simulate job execution (placeholder for actual implementation).
        
        In a real implementation, this would call the actual reindexing logic.
        """
        # Simulate processing time based on job characteristics
        import time
        processing_time = 0.1 + (job.priority.value * 0.05)
        time.sleep(processing_time)
        
        # Simulate 90% success rate
        import random
        return random.random() > 0.1
    
    def _handle_job_completion(self, job_id: str):
        """Handle job completion - check for dependent jobs"""
        # Check if any jobs were waiting for this one
        waiting_jobs = self.dependency_waiters.get(job_id, set())
        
        for waiting_job_id in waiting_jobs:
            if self._are_dependencies_satisfied(waiting_job_id):
                # Dependencies satisfied, job can now run
                current_status = self.lifecycle_manager.get_job_status(waiting_job_id)
                if current_status == JobStatus.SCHEDULED:
                    # Find the job object and execute it
                    # This would require maintaining a job registry
                    self.logger.info(f"Dependencies satisfied for job {waiting_job_id}")
        
        # Clean up dependency tracking
        if job_id in self.dependency_waiters:
            del self.dependency_waiters[job_id]
    
    def create_batch(self, jobs: List[Any], batch_type: str = "general") -> Optional[JobBatch]:
        """
        Create a batch of related jobs for optimized processing.
        
        Args:
            jobs: List of ReindexJob instances to batch
            batch_type: Type of batching strategy used
            
        Returns:
            JobBatch instance if successful, None otherwise
        """
        if not self.enable_batching or not jobs:
            return None
        
        if len(jobs) > self.max_batch_size:
            self.logger.warning(f"Batch size {len(jobs)} exceeds maximum {self.max_batch_size}")
            jobs = jobs[:self.max_batch_size]
        
        batch = JobBatch(
            jobs=[job.id for job in jobs],
            batch_type=batch_type,
            estimated_duration=sum(getattr(job, 'estimated_duration', 1.0) for job in jobs)
        )
        
        # Register batch
        self.pending_batches[batch.id] = batch
        
        # Register individual jobs with lifecycle manager
        for job in jobs:
            if not self.lifecycle_manager.register_job(job.id, JobStatus.SCHEDULED):
                self.logger.warning(f"Failed to register job {job.id} in batch {batch.id}")
        
        self.logger.info(f"Created batch {batch.id} with {len(jobs)} jobs (type: {batch_type})")
        return batch
    
    def _create_file_based_batches(self, jobs: List[Any]) -> List[List[Any]]:
        """Group jobs by file path for batch processing"""
        file_groups = defaultdict(list)
        
        for job in jobs:
            file_path = getattr(job, 'file_path', None)
            if file_path:
                # Group by directory
                import os
                directory = os.path.dirname(file_path)
                file_groups[directory].append(job)
            else:
                file_groups['no_file'].append(job)
        
        return list(file_groups.values())
    
    def _create_entity_based_batches(self, jobs: List[Any]) -> List[List[Any]]:
        """Group jobs by entity type for batch processing"""
        entity_groups = defaultdict(list)
        
        for job in jobs:
            entity_type = getattr(job, 'entity_type', 'unknown')
            entity_groups[entity_type].append(job)
        
        return list(entity_groups.values())
    
    def _create_priority_based_batches(self, jobs: List[Any]) -> List[List[Any]]:
        """Group jobs by priority for batch processing"""
        priority_groups = defaultdict(list)
        
        for job in jobs:
            priority = getattr(job, 'priority', 'medium')
            priority_groups[str(priority)].append(job)
        
        return list(priority_groups.values())
    
    def optimize_job_scheduling(self, pending_jobs: List[Any]) -> List[Any]:
        """
        Optimize job scheduling order for better performance.
        
        Args:
            pending_jobs: List of jobs to optimize
            
        Returns:
            Reordered list of jobs for optimal processing
        """
        if not pending_jobs:
            return []
        
        # Sort by priority first (lower value = higher priority)
        jobs_by_priority = sorted(pending_jobs, key=lambda j: (j.priority.value, j.scheduled_time))
        
        # Apply dependency constraints
        optimized_jobs = []
        remaining_jobs = jobs_by_priority.copy()
        
        while remaining_jobs:
            # Find jobs with no unfulfilled dependencies
            ready_jobs = []
            for job in remaining_jobs:
                if not job.dependencies or all(
                    dep_id in [completed.id for completed in optimized_jobs]
                    for dep_id in job.dependencies
                ):
                    ready_jobs.append(job)
            
            if not ready_jobs:
                # Break circular dependencies by taking highest priority job
                ready_jobs = [min(remaining_jobs, key=lambda j: j.priority.value)]
            
            # Add ready jobs to optimized list
            optimized_jobs.extend(ready_jobs)
            
            # Remove from remaining
            for job in ready_jobs:
                remaining_jobs.remove(job)
        
        return optimized_jobs
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive job manager status"""
        with self.lock:
            # Get job counts by status
            status_counts = defaultdict(int)
            for status in self.lifecycle_manager.job_states.values():
                status_counts[status.value] += 1
            
            return {
                "active_jobs": len(self.active_jobs),
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "can_accept_jobs": self.can_accept_job(),
                "job_status_counts": dict(status_counts),
                "pending_batches": len(self.pending_batches),
                "dependency_chains": len(self.job_dependencies),
                "configuration": {
                    "batching_enabled": self.enable_batching,
                    "max_batch_size": self.max_batch_size,
                    "batch_timeout": self.batch_timeout_seconds
                }
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get job performance metrics"""
        if not self.job_performance_history:
            return {"no_data": True}
        
        # Calculate performance statistics
        durations = [perf.get("duration", 0) for perf in self.job_performance_history]
        
        return {
            "total_jobs_processed": len(self.job_performance_history),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "job_history_size": len(self.job_performance_history),
            "batch_history_size": len(self.batch_performance_history)
        }
    
    def cleanup(self):
        """Clean up old job data and resources"""
        # Clean up completed jobs from lifecycle manager
        cleaned_jobs = self.lifecycle_manager.cleanup_completed_jobs()
        
        # Clean up old performance data
        if len(self.job_performance_history) > 500:
            # Keep only recent half
            recent_history = list(self.job_performance_history)[-250:]
            self.job_performance_history.clear()
            self.job_performance_history.extend(recent_history)
        
        self.logger.info(f"Cleanup completed: {cleaned_jobs} jobs cleaned")


# Factory function for easy instantiation
def create_job_manager(max_concurrent_jobs: int = 2, enable_batching: bool = True) -> ReindexJobManager:
    """Create a ReindexJobManager with specified configuration"""
    return ReindexJobManager(
        max_concurrent_jobs=max_concurrent_jobs,
        enable_batching=enable_batching
    )


# Example usage and testing
if __name__ == "__main__":
    # Test job management system
    print("Testing Job Management System")
    print("=" * 40)
    
    # Create job manager
    manager = create_job_manager()
    
    # Test job lifecycle
    test_job_id = "test-job-123"
    
    # Register job
    if manager.lifecycle_manager.register_job(test_job_id):
        print(f"Job {test_job_id} registered successfully")
        
        # Test state transitions
        transitions = [
            JobStatus.SCHEDULED,
            JobStatus.RUNNING,
            JobStatus.COMPLETED
        ]
        
        for status in transitions:
            if manager.lifecycle_manager.transition_job_state(test_job_id, status):
                print(f"Job transitioned to {status.value}")
            time.sleep(0.1)
        
        # Get job metrics
        metrics = manager.lifecycle_manager.get_job_metrics(test_job_id)
        if metrics:
            print(f"Job metrics: {metrics.to_dict()}")
        
        # Get job history
        history = manager.lifecycle_manager.get_job_history(test_job_id)
        print(f"Job history: {[(status.value, time.isoformat()) for status, time in history]}")
    
    # Get manager status
    status = manager.get_status()
    print(f"\nJob Manager Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nJob management system test completed")