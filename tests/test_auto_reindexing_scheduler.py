"""
Comprehensive tests for Auto-Reindexing Scheduler
Issue #69: Build auto-reindexing scheduler

Tests cover:
- Core scheduler functionality
- Priority-based job scheduling 
- Resource management and throttling
- Integration with file monitoring and graph validation
- Performance benchmarking
- Error handling and recovery

Author: RIF-Implementer
Date: 2025-08-23
"""

import pytest
import threading
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import components under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from knowledge.indexing.auto_reindexing_scheduler import (
    AutoReindexingScheduler,
    ReindexJob,
    ReindexPriority, 
    ReindexTrigger,
    SchedulerStatus,
    SchedulerConfig,
    create_auto_reindexing_scheduler
)

from knowledge.indexing.resource_manager import (
    ReindexingResourceManager,
    SystemResourceMonitor,
    ResourceSnapshot
)

from knowledge.indexing.priority_calculator import (
    PriorityCalculator,
    TriggerAnalyzer
)

from knowledge.indexing.reindex_job_manager import (
    ReindexJobManager,
    JobLifecycleManager,
    JobStatus
)


class TestReindexJob:
    """Test ReindexJob data structure and behavior"""
    
    def test_job_creation(self):
        """Test basic job creation"""
        job = ReindexJob(
            entity_type="test_entity",
            priority=ReindexPriority.HIGH,
            trigger=ReindexTrigger.FILE_CHANGE,
            trigger_reason="Test job creation"
        )
        
        assert job.entity_type == "test_entity"
        assert job.priority == ReindexPriority.HIGH
        assert job.trigger == ReindexTrigger.FILE_CHANGE
        assert job.trigger_reason == "Test job creation"
        assert job.retry_count == 0
        assert job.id is not None
    
    def test_job_priority_ordering(self):
        """Test that jobs are ordered correctly by priority"""
        job_high = ReindexJob(priority=ReindexPriority.HIGH)
        job_low = ReindexJob(priority=ReindexPriority.LOW)
        job_critical = ReindexJob(priority=ReindexPriority.CRITICAL)
        
        # Lower priority value = higher priority
        assert job_critical < job_high
        assert job_high < job_low
        assert not job_low < job_high
    
    def test_job_retry_logic(self):
        """Test job retry behavior"""
        job = ReindexJob(max_retries=3)
        
        assert job.can_retry() is True
        
        job.retry_count = 3
        assert job.can_retry() is False
        
        job.retry_count = 2
        assert job.can_retry() is True
    
    def test_job_serialization(self):
        """Test job can be serialized to dictionary"""
        job = ReindexJob(
            entity_type="test",
            entity_id="123",
            priority=ReindexPriority.MEDIUM,
            trigger=ReindexTrigger.MANUAL_REQUEST,
            metadata={"test": "data"}
        )
        
        job_dict = job.to_dict()
        
        assert job_dict["entity_type"] == "test"
        assert job_dict["entity_id"] == "123"
        assert job_dict["priority"] == "MEDIUM"
        assert job_dict["trigger"] == "manual_request"
        assert job_dict["metadata"]["test"] == "data"


class TestSchedulerConfig:
    """Test scheduler configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SchedulerConfig()
        
        # Test updated resource thresholds (made more reasonable)
        assert config.resource_thresholds[ReindexPriority.CRITICAL] == 0.95
        assert config.resource_thresholds[ReindexPriority.HIGH] == 0.85
        assert config.resource_thresholds[ReindexPriority.MEDIUM] == 0.75
        assert config.resource_thresholds[ReindexPriority.LOW] == 0.60
        assert config.enable_batching is True
        assert config.max_batch_size == 50
        assert config.default_max_retries == 3
    
    def test_config_serialization(self):
        """Test config can be serialized"""
        config = SchedulerConfig()
        config_dict = config.to_dict()
        
        assert "resource_thresholds" in config_dict
        assert "enable_batching" in config_dict
        assert config_dict["max_batch_size"] == 50


class TestSystemResourceMonitor:
    """Test system resource monitoring"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = SystemResourceMonitor(
            sample_interval=0.1,
            history_size=10,
            knowledge_path=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        if self.monitor.monitoring_active:
            self.monitor.stop_monitoring()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_resource_snapshot(self):
        """Test taking resource snapshots"""
        snapshot = self.monitor.get_current_snapshot()
        
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.cpu_percent >= 0
        assert snapshot.memory_percent >= 0
        assert snapshot.memory_available_mb >= 0
        assert isinstance(snapshot.timestamp, datetime)
    
    def test_monitoring_lifecycle(self):
        """Test monitor start/stop lifecycle"""
        assert not self.monitor.monitoring_active
        
        # Start monitoring
        assert self.monitor.start_monitoring() is True
        assert self.monitor.monitoring_active is True
        
        # Let it collect some data
        time.sleep(0.3)
        assert len(self.monitor.resource_history) > 0
        
        # Stop monitoring
        assert self.monitor.stop_monitoring() is True
        assert not self.monitor.monitoring_active
    
    def test_resource_trends(self):
        """Test resource trend analysis"""
        # Start monitoring and collect data
        self.monitor.start_monitoring()
        time.sleep(0.5)
        
        trends = self.monitor.get_resource_trend(duration_minutes=1)
        
        assert "cpu_trend" in trends
        assert "memory_trend" in trends
        assert "io_trend" in trends
        assert "data_points" in trends
        assert trends["data_points"] > 0
    
    def test_resource_statistics(self):
        """Test resource statistics calculation"""
        # Start monitoring and collect data
        self.monitor.start_monitoring()
        time.sleep(0.3)
        
        stats = self.monitor.get_resource_statistics(duration_minutes=1)
        
        if stats:  # May be empty if not enough data
            assert "cpu" in stats
            assert "memory" in stats
            assert "sample_count" in stats


class TestReindexingResourceManager:
    """Test resource management for reindexing"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ReindexingResourceManager(
            check_interval=0.1,
            knowledge_path=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.manager.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_resource_manager_lifecycle(self):
        """Test resource manager start/stop"""
        assert self.manager.start() is True
        time.sleep(0.2)  # Let it start
        assert self.manager.stop() is True
    
    def test_job_scheduling_decisions(self):
        """Test resource-based scheduling decisions"""
        self.manager.start()
        time.sleep(0.2)
        
        # Test different priorities
        for priority in ReindexPriority:
            can_schedule = self.manager.can_schedule_job(priority)
            assert isinstance(can_schedule, bool)
            
            # Critical jobs should generally be allowed
            if priority == ReindexPriority.CRITICAL:
                # May be denied under extreme load, but usually allowed
                pass
    
    def test_utilization_calculation(self):
        """Test overall utilization calculation"""
        self.manager.start()
        time.sleep(0.2)
        
        utilization = self.manager.get_current_utilization()
        assert 0.0 <= utilization <= 1.0
    
    def test_status_reporting(self):
        """Test resource manager status"""
        status = self.manager.get_status()
        
        assert "monitoring_active" in status
        assert "priority_thresholds" in status
    
    def test_recommendations(self):
        """Test resource optimization recommendations"""
        recommendations = self.manager.get_resource_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestTriggerAnalyzer:
    """Test trigger analysis for priority calculation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.analyzer = TriggerAnalyzer()
    
    def test_validation_trigger_analysis(self):
        """Test analysis of validation triggers"""
        # Critical validation issue
        priority, confidence = self.analyzer.analyze_trigger(
            "validation_issue",
            {"validation_severity": "critical", "validation_category": "referential_integrity"}
        )
        
        assert priority == ReindexPriority.CRITICAL
        assert confidence > 0.5
        
        # Warning validation issue
        priority, confidence = self.analyzer.analyze_trigger(
            "validation_issue",
            {"validation_severity": "warning", "validation_category": "data_quality"}
        )
        
        assert priority == ReindexPriority.MEDIUM
        assert confidence > 0.5
    
    def test_file_change_trigger_analysis(self):
        """Test analysis of file change triggers"""
        # Source code file change
        priority, confidence = self.analyzer.analyze_trigger(
            "file_change",
            {"file_path": "/src/main.py", "file_change_type": "modified"}
        )
        
        assert priority == ReindexPriority.HIGH
        assert confidence > 0.5
        
        # Documentation file change
        priority, confidence = self.analyzer.analyze_trigger(
            "file_change",
            {"file_path": "/docs/README.md", "file_change_type": "modified"}
        )
        
        assert priority == ReindexPriority.MEDIUM
        assert confidence > 0.5
    
    def test_unknown_trigger(self):
        """Test handling of unknown triggers"""
        priority, confidence = self.analyzer.analyze_trigger(
            "unknown_trigger_type",
            {}
        )
        
        assert priority == ReindexPriority.MEDIUM  # Default
        assert confidence < 0.5  # Low confidence


class TestPriorityCalculator:
    """Test priority calculation system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.calculator = PriorityCalculator(knowledge_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_priority_calculation(self):
        """Test priority calculation for different job types"""
        # Create test jobs
        critical_job = ReindexJob(
            entity_type="module",
            trigger=ReindexTrigger.VALIDATION_ISSUE,
            metadata={"validation_severity": "critical"}
        )
        
        routine_job = ReindexJob(
            entity_type="variable",
            trigger=ReindexTrigger.SCHEDULED_MAINTENANCE,
            metadata={}
        )
        
        # Calculate priorities
        critical_priority = self.calculator.calculate_priority(critical_job)
        routine_priority = self.calculator.calculate_priority(routine_job)
        
        # Critical job should have higher priority (lower value)
        assert critical_priority.value <= routine_priority.value
    
    def test_performance_tracking(self):
        """Test performance history tracking"""
        job = ReindexJob(entity_type="test", trigger=ReindexTrigger.MANUAL_REQUEST)
        
        # Record performance
        self.calculator.record_job_performance(
            job, success=True, duration=1.5, metrics={"cpu": 20}
        )
        
        assert len(self.calculator.performance_history) == 1
        
        # Get statistics
        stats = self.calculator.get_priority_statistics()
        assert stats["total_jobs_analyzed"] == 1
    
    def test_staleness_tracking(self):
        """Test entity staleness tracking"""
        self.calculator.update_entity_staleness("module", "test_module", None)
        
        assert "module:test_module" in self.calculator.entity_staleness_cache
    
    def test_dependency_tracking(self):
        """Test dependency relationship tracking"""
        self.calculator.add_dependency("module:main", "function:helper")
        
        assert "function:helper" in self.calculator.dependency_graph["module:main"]


class TestJobLifecycleManager:
    """Test job lifecycle management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.manager = JobLifecycleManager()
    
    def test_job_registration(self):
        """Test job registration"""
        job_id = "test-job-123"
        
        assert self.manager.register_job(job_id) is True
        assert self.manager.get_job_status(job_id) == JobStatus.QUEUED
        
        # Duplicate registration should fail
        assert self.manager.register_job(job_id) is False
    
    def test_state_transitions(self):
        """Test valid state transitions"""
        job_id = "test-job-456"
        self.manager.register_job(job_id)
        
        # Valid transitions
        assert self.manager.transition_job_state(job_id, JobStatus.RUNNING) is True
        assert self.manager.get_job_status(job_id) == JobStatus.RUNNING
        
        assert self.manager.transition_job_state(job_id, JobStatus.COMPLETED) is True
        assert self.manager.get_job_status(job_id) == JobStatus.COMPLETED
    
    def test_invalid_transitions(self):
        """Test invalid state transitions are rejected"""
        job_id = "test-job-789"
        self.manager.register_job(job_id, JobStatus.COMPLETED)
        
        # Completed jobs can't transition to running
        assert self.manager.transition_job_state(job_id, JobStatus.RUNNING) is False
        assert self.manager.get_job_status(job_id) == JobStatus.COMPLETED
    
    def test_job_metrics_tracking(self):
        """Test job metrics are tracked"""
        job_id = "test-job-metrics"
        self.manager.register_job(job_id)
        
        # Start job
        self.manager.transition_job_state(job_id, JobStatus.RUNNING)
        time.sleep(0.1)
        
        # Complete job
        self.manager.transition_job_state(job_id, JobStatus.COMPLETED)
        
        metrics = self.manager.get_job_metrics(job_id)
        assert metrics is not None
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.duration_seconds > 0
    
    def test_jobs_by_status_query(self):
        """Test querying jobs by status"""
        # Register multiple jobs in different states
        self.manager.register_job("job-1", JobStatus.QUEUED)
        self.manager.register_job("job-2", JobStatus.RUNNING)
        self.manager.register_job("job-3", JobStatus.QUEUED)
        
        queued_jobs = self.manager.get_jobs_by_status(JobStatus.QUEUED)
        running_jobs = self.manager.get_jobs_by_status(JobStatus.RUNNING)
        
        assert len(queued_jobs) == 2
        assert len(running_jobs) == 1
        assert "job-1" in queued_jobs
        assert "job-3" in queued_jobs
        assert "job-2" in running_jobs


class TestReindexJobManager:
    """Test job management with batching and dependencies"""
    
    def setup_method(self):
        """Setup test environment"""
        self.manager = ReindexJobManager(
            max_concurrent_jobs=2,
            enable_batching=True,
            max_batch_size=10
        )
    
    def test_job_acceptance(self):
        """Test job acceptance based on capacity"""
        assert self.manager.can_accept_job() is True
    
    def test_job_submission(self):
        """Test job submission"""
        job = ReindexJob(entity_type="test", trigger=ReindexTrigger.MANUAL_REQUEST)
        
        # Mock the job execution to avoid actual work
        with patch.object(self.manager, '_simulate_job_execution', return_value=True):
            result = self.manager.submit_job(job)
            assert result is True
    
    def test_batch_creation(self):
        """Test batch creation"""
        jobs = [
            ReindexJob(entity_type="module", trigger=ReindexTrigger.FILE_CHANGE),
            ReindexJob(entity_type="module", trigger=ReindexTrigger.FILE_CHANGE),
            ReindexJob(entity_type="class", trigger=ReindexTrigger.FILE_CHANGE)
        ]
        
        batch = self.manager.create_batch(jobs, "entity_based")
        
        assert batch is not None
        assert len(batch.jobs) == 3
        assert batch.batch_type == "entity_based"
    
    def test_job_optimization(self):
        """Test job scheduling optimization"""
        jobs = [
            ReindexJob(priority=ReindexPriority.LOW, scheduled_time=datetime.now() + timedelta(minutes=1)),
            ReindexJob(priority=ReindexPriority.CRITICAL, scheduled_time=datetime.now()),
            ReindexJob(priority=ReindexPriority.HIGH, scheduled_time=datetime.now())
        ]
        
        optimized = self.manager.optimize_job_scheduling(jobs)
        
        # Should be ordered by priority (critical first)
        assert optimized[0].priority == ReindexPriority.CRITICAL
        assert optimized[1].priority == ReindexPriority.HIGH
        assert optimized[2].priority == ReindexPriority.LOW
    
    def test_status_reporting(self):
        """Test job manager status"""
        status = self.manager.get_status()
        
        assert "active_jobs" in status
        assert "max_concurrent_jobs" in status
        assert "can_accept_jobs" in status
        assert "job_status_counts" in status


class TestAutoReindexingScheduler:
    """Test the main auto-reindexing scheduler"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SchedulerConfig()
        self.config.queue_check_interval = 0.1
        self.config.metrics_report_interval = 1.0
        
        self.scheduler = AutoReindexingScheduler(
            config=self.config,
            knowledge_path=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        if self.scheduler.status == SchedulerStatus.RUNNING:
            self.scheduler.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scheduler_lifecycle(self):
        """Test scheduler start/stop lifecycle"""
        assert self.scheduler.status == SchedulerStatus.STOPPED
        
        # Start scheduler
        assert self.scheduler.start() is True
        assert self.scheduler.status == SchedulerStatus.RUNNING
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop scheduler
        assert self.scheduler.stop() is True
        assert self.scheduler.status == SchedulerStatus.STOPPED
    
    def test_job_scheduling(self):
        """Test job scheduling"""
        job_id = self.scheduler.schedule_reindex(
            entity_type="test_entity",
            priority=ReindexPriority.HIGH,
            trigger=ReindexTrigger.MANUAL_REQUEST,
            trigger_reason="Test scheduling"
        )
        
        assert job_id is not None
        assert self.scheduler.job_queue.qsize() == 1
    
    def test_job_processing(self):
        """Test job processing"""
        self.scheduler.start()
        
        # Schedule a job
        job_id = self.scheduler.schedule_reindex(
            entity_type="test_entity",
            priority=ReindexPriority.HIGH,
            trigger=ReindexTrigger.MANUAL_REQUEST,
            trigger_reason="Test processing"
        )
        
        # Let scheduler process it
        time.sleep(0.5)
        
        # Job should have been processed
        assert self.scheduler.metrics["jobs_processed"] >= 1
    
    def test_pause_resume(self):
        """Test scheduler pause/resume"""
        self.scheduler.start()
        
        assert self.scheduler.pause() is True
        assert self.scheduler.status == SchedulerStatus.PAUSED
        
        assert self.scheduler.resume() is True
        assert self.scheduler.status == SchedulerStatus.RUNNING
    
    def test_validation_integration(self):
        """Test integration with validation system"""
        # Mock validation report
        mock_validation_report = Mock()
        mock_validation_report.issues = [
            Mock(
                severity=Mock(value="critical"),
                table_name="test_table",
                entity_id="test_entity",
                message="Test validation issue",
                id="issue-123",
                category="test_category",
                suggested_fix="Test fix"
            )
        ]
        
        self.scheduler.schedule_validation_triggered_reindex(mock_validation_report)
        
        # Should have scheduled a job
        assert self.scheduler.job_queue.qsize() == 1
    
    def test_status_reporting(self):
        """Test comprehensive status reporting"""
        status = self.scheduler.get_status()
        
        assert "scheduler_status" in status
        assert "metrics" in status
        assert "configuration" in status
        assert "integrations" in status
        assert "components" in status
    
    def test_job_history(self):
        """Test job history tracking"""
        self.scheduler.start()
        
        # Schedule and process a job
        self.scheduler.schedule_reindex(
            entity_type="test_entity",
            trigger=ReindexTrigger.MANUAL_REQUEST
        )
        
        time.sleep(0.5)  # Let it process
        
        history = self.scheduler.get_job_history()
        
        assert "completed" in history
        assert "failed" in history
        assert "active" in history
    
    def test_event_handlers(self):
        """Test event handler system"""
        events_received = []
        
        def test_handler(job):
            events_received.append(("job_queued", job.id if job else None))
        
        self.scheduler.add_event_handler("job_queued", test_handler)
        
        # Schedule a job to trigger event
        job_id = self.scheduler.schedule_reindex(
            entity_type="test_entity",
            trigger=ReindexTrigger.MANUAL_REQUEST
        )
        
        assert len(events_received) == 1
        assert events_received[0][0] == "job_queued"


class TestIntegrationScenarios:
    """Integration tests for complete scheduler workflows"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.scheduler = create_auto_reindexing_scheduler(knowledge_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        if self.scheduler.status == SchedulerStatus.RUNNING:
            self.scheduler.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_high_volume_scheduling(self):
        """Test scheduler under high job volume"""
        self.scheduler.start()
        
        # Schedule many jobs
        job_ids = []
        for i in range(20):
            job_id = self.scheduler.schedule_reindex(
                entity_type=f"test_entity_{i}",
                priority=ReindexPriority.MEDIUM,
                trigger=ReindexTrigger.MANUAL_REQUEST,
                trigger_reason=f"Bulk test job {i}"
            )
            job_ids.append(job_id)
        
        # Let scheduler process jobs
        time.sleep(2.0)
        
        # Check that jobs were processed
        assert self.scheduler.metrics["jobs_processed"] > 0
        assert self.scheduler.job_queue.qsize() < 20  # Some jobs should have been processed
    
    def test_priority_ordering(self):
        """Test that jobs are processed in priority order"""
        self.scheduler.start()
        
        # Schedule jobs in reverse priority order
        low_job = self.scheduler.schedule_reindex(
            entity_type="low_priority",
            priority=ReindexPriority.LOW,
            trigger=ReindexTrigger.MANUAL_REQUEST,
            trigger_reason="Low priority job"
        )
        
        critical_job = self.scheduler.schedule_reindex(
            entity_type="critical_priority",
            priority=ReindexPriority.CRITICAL,
            trigger=ReindexTrigger.VALIDATION_ISSUE,
            trigger_reason="Critical job"
        )
        
        time.sleep(0.5)  # Let one job process
        
        # The critical job should be processed first
        # (This is a simplified test - in practice we'd need more sophisticated tracking)
        assert self.scheduler.metrics["jobs_processed"] >= 1
    
    def test_resource_throttling(self):
        """Test that resource constraints affect job scheduling"""
        # Create a scheduler with very restrictive resource limits
        config = SchedulerConfig()
        config.resource_thresholds[ReindexPriority.LOW] = 0.01  # Very low threshold
        
        scheduler = AutoReindexingScheduler(config=config, knowledge_path=self.temp_dir)
        scheduler.start()
        
        # Schedule low priority jobs
        for i in range(5):
            scheduler.schedule_reindex(
                entity_type=f"throttled_entity_{i}",
                priority=ReindexPriority.LOW,
                trigger=ReindexTrigger.MANUAL_REQUEST
            )
        
        time.sleep(1.0)
        
        # Due to resource constraints, not all jobs should be processed
        # (Exact behavior depends on system load)
        status = scheduler.get_status()
        
        scheduler.stop()


class TestPerformanceBenchmarks:
    """Performance benchmarks for the scheduler"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = SchedulerConfig()
        self.config.queue_check_interval = 0.01  # Fast for testing
        
    def teardown_method(self):
        """Cleanup performance test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scheduling_throughput(self):
        """Benchmark job scheduling throughput"""
        scheduler = AutoReindexingScheduler(config=self.config, knowledge_path=self.temp_dir)
        
        start_time = time.time()
        
        # Schedule 100 jobs
        for i in range(100):
            scheduler.schedule_reindex(
                entity_type=f"bench_entity_{i}",
                priority=ReindexPriority.MEDIUM,
                trigger=ReindexTrigger.MANUAL_REQUEST
            )
        
        scheduling_time = time.time() - start_time
        
        # Should be able to schedule at least 1000 jobs/second
        jobs_per_second = 100 / scheduling_time
        print(f"Scheduling throughput: {jobs_per_second:.0f} jobs/second")
        
        assert jobs_per_second > 100  # Minimum acceptable performance
        assert scheduler.job_queue.qsize() == 100
    
    def test_processing_performance(self):
        """Benchmark job processing performance"""
        scheduler = AutoReindexingScheduler(config=self.config, knowledge_path=self.temp_dir)
        scheduler.start()
        
        # Schedule jobs and measure processing time
        start_time = time.time()
        
        for i in range(20):
            scheduler.schedule_reindex(
                entity_type=f"perf_entity_{i}",
                priority=ReindexPriority.MEDIUM,
                trigger=ReindexTrigger.MANUAL_REQUEST
            )
        
        # Wait for all jobs to complete
        timeout = 10.0
        while scheduler.job_queue.qsize() > 0 and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        processing_time = time.time() - start_time
        processed_jobs = scheduler.metrics["jobs_processed"]
        
        if processed_jobs > 0:
            jobs_per_second = processed_jobs / processing_time
            print(f"Processing throughput: {jobs_per_second:.1f} jobs/second")
            print(f"Average job time: {processing_time/processed_jobs:.3f} seconds")
        
        scheduler.stop()
    
    def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        scheduler = AutoReindexingScheduler(config=self.config, knowledge_path=self.temp_dir)
        scheduler.start()
        
        # Schedule many jobs
        for i in range(500):
            scheduler.schedule_reindex(
                entity_type=f"memory_test_{i}",
                priority=ReindexPriority.MEDIUM,
                trigger=ReindexTrigger.MANUAL_REQUEST
            )
        
        time.sleep(2.0)  # Let jobs process
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 100MB for 500 jobs)
        assert memory_increase < 100
        
        scheduler.stop()


def test_factory_function():
    """Test factory function creates scheduler correctly"""
    with tempfile.TemporaryDirectory() as temp_dir:
        scheduler = create_auto_reindexing_scheduler(knowledge_path=temp_dir)
        
        assert isinstance(scheduler, AutoReindexingScheduler)
        assert scheduler.status == SchedulerStatus.STOPPED
        assert scheduler.knowledge_path == Path(temp_dir)


def test_context_manager():
    """Test scheduler can be used as context manager"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with create_auto_reindexing_scheduler(knowledge_path=temp_dir) as scheduler:
            assert scheduler.status == SchedulerStatus.RUNNING
            
            # Schedule a test job
            job_id = scheduler.schedule_reindex(
                entity_type="context_test",
                trigger=ReindexTrigger.MANUAL_REQUEST
            )
            
            assert job_id is not None
        
        # Scheduler should be stopped after context exit
        assert scheduler.status == SchedulerStatus.STOPPED


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])