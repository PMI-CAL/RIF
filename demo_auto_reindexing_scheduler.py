"""
Demonstration of Auto-Reindexing Scheduler
Issue #69: Build auto-reindexing scheduler

This script demonstrates the capabilities of the auto-reindexing scheduler:
- Priority-based job scheduling
- Resource-aware execution
- Integration with file monitoring and graph validation
- Batch processing optimization
- Performance monitoring

Author: RIF-Implementer
Date: 2025-08-23
"""

import time
import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

# Import the scheduler system
from knowledge.indexing.auto_reindexing_scheduler import (
    create_auto_reindexing_scheduler,
    ReindexJob,
    ReindexPriority,
    ReindexTrigger,
    SchedulerConfig
)

from knowledge.indexing.resource_manager import create_resource_manager
from knowledge.indexing.priority_calculator import create_priority_calculator


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")


def print_status(scheduler):
    """Print scheduler status summary"""
    status = scheduler.get_status()
    metrics = status["metrics"]
    
    print(f"Status: {status['scheduler_status']}")
    print(f"Queue Size: {metrics['queue_size']}")
    print(f"Active Jobs: {metrics['active_job_count']}")
    print(f"Processed: {metrics['jobs_processed']}")
    print(f"Failed: {metrics['jobs_failed']}")
    print(f"Throughput: {metrics['throughput_jobs_per_minute']:.1f} jobs/min")
    print(f"Resource Util: {metrics['resource_utilization']:.1%}")


@contextmanager
def demo_scheduler():
    """Context manager for demo scheduler"""
    scheduler = None
    try:
        print_section("Initializing Scheduler")
        
        # Create scheduler with demo configuration
        config = SchedulerConfig()
        config.queue_check_interval = 1.0
        config.metrics_report_interval = 10.0
        config.enable_batching = True
        config.max_batch_size = 5
        
        scheduler = create_auto_reindexing_scheduler("/Users/cal/DEV/RIF/knowledge")
        scheduler.config = config
        
        print("✓ Auto-reindexing scheduler created")
        print(f"✓ Configuration: {config.to_dict()}")
        
        # Start scheduler
        if scheduler.start():
            print("✓ Scheduler started successfully")
        else:
            raise RuntimeError("Failed to start scheduler")
        
        yield scheduler
        
    finally:
        if scheduler:
            print_section("Shutting Down Scheduler")
            if scheduler.stop():
                print("✓ Scheduler stopped gracefully")
            else:
                print("⚠ Scheduler shutdown with warnings")


def demo_basic_scheduling():
    """Demonstrate basic job scheduling"""
    print_header("Basic Job Scheduling Demo")
    
    with demo_scheduler() as scheduler:
        print_section("Scheduling Various Job Types")
        
        # Schedule different types of jobs
        jobs = [
            {
                "entity_type": "module",
                "priority": ReindexPriority.HIGH,
                "trigger": ReindexTrigger.FILE_CHANGE,
                "reason": "Source code modified"
            },
            {
                "entity_type": "class",
                "priority": ReindexPriority.MEDIUM,
                "trigger": ReindexTrigger.DEPENDENCY_UPDATE,
                "reason": "Related entity changed"
            },
            {
                "entity_type": "function",
                "priority": ReindexPriority.LOW,
                "trigger": ReindexTrigger.SCHEDULED_MAINTENANCE,
                "reason": "Regular maintenance"
            },
            {
                "entity_type": "configuration",
                "priority": ReindexPriority.CRITICAL,
                "trigger": ReindexTrigger.VALIDATION_ISSUE,
                "reason": "Validation error detected"
            }
        ]
        
        scheduled_jobs = []
        for job_spec in jobs:
            job_id = scheduler.schedule_reindex(
                entity_type=job_spec["entity_type"],
                priority=job_spec["priority"],
                trigger=job_spec["trigger"],
                trigger_reason=job_spec["reason"]
            )
            scheduled_jobs.append(job_id)
            print(f"✓ Scheduled {job_spec['priority'].name} priority {job_spec['entity_type']} job: {job_id}")
        
        print(f"\n✓ Total jobs scheduled: {len(scheduled_jobs)}")
        
        # Let scheduler process jobs
        print_section("Processing Jobs")
        print("Waiting for job processing...")
        
        for i in range(10):
            time.sleep(1)
            print_status(scheduler)
            
            if scheduler.job_queue.qsize() == 0:
                print("\n✓ All jobs processed!")
                break
        
        # Show final results
        print_section("Final Results")
        history = scheduler.get_job_history()
        print(f"Completed jobs: {len(history['completed'])}")
        print(f"Failed jobs: {len(history['failed'])}")
        print(f"Active jobs: {len(history['active'])}")


def demo_priority_system():
    """Demonstrate priority-based scheduling"""
    print_header("Priority-Based Scheduling Demo")
    
    with demo_scheduler() as scheduler:
        print_section("Scheduling Jobs in Mixed Priority Order")
        
        # Schedule jobs in non-priority order to show reordering
        job_specs = [
            ("Low priority background job", ReindexPriority.LOW),
            ("Critical validation fix", ReindexPriority.CRITICAL),
            ("Medium priority update", ReindexPriority.MEDIUM),
            ("High priority file change", ReindexPriority.HIGH),
            ("Another low priority job", ReindexPriority.LOW),
        ]
        
        for description, priority in job_specs:
            job_id = scheduler.schedule_reindex(
                entity_type="test_entity",
                priority=priority,
                trigger=ReindexTrigger.MANUAL_REQUEST,
                trigger_reason=description
            )
            print(f"Scheduled {priority.name}: {description}")
        
        print(f"\n✓ Queue size: {scheduler.job_queue.qsize()}")
        
        # Process jobs and observe priority ordering
        print_section("Observing Priority Processing")
        
        processed_order = []
        original_fire_event = scheduler._fire_event
        
        def track_events(event_name, job):
            if event_name == "job_started" and job:
                processed_order.append((job.priority.name, job.trigger_reason))
            original_fire_event(event_name, job)
        
        scheduler._fire_event = track_events
        
        # Let jobs process
        time.sleep(5)
        
        print("Processing order observed:")
        for i, (priority, reason) in enumerate(processed_order):
            print(f"  {i+1}. {priority}: {reason}")
        
        # Verify priority ordering
        if processed_order:
            priorities_processed = [p[0] for p in processed_order]
            print(f"\n✓ Priorities processed: {' → '.join(priorities_processed)}")


def demo_resource_management():
    """Demonstrate resource-aware scheduling"""
    print_header("Resource-Aware Scheduling Demo")
    
    print_section("Resource Monitor Status")
    
    # Create standalone resource manager for demonstration
    resource_manager = create_resource_manager()
    resource_manager.start()
    
    try:
        time.sleep(2)  # Let it collect some data
        
        status = resource_manager.get_status()
        print("Current resource status:")
        if "current_snapshot" in status:
            snapshot = status["current_snapshot"]
            print(f"  CPU Usage: {snapshot['cpu_percent']:.1f}%")
            print(f"  Memory Usage: {snapshot['memory_percent']:.1f}%")
            print(f"  Available Memory: {snapshot['memory_available_mb']:.0f}MB")
            print(f"  I/O Rate: {snapshot['io_read_rate']:.1f}MB/s read, {snapshot['io_write_rate']:.1f}MB/s write")
        
        print(f"  Overall Utilization: {resource_manager.get_current_utilization():.1%}")
        
        # Test scheduling decisions for different priorities
        print_section("Resource-Based Scheduling Decisions")
        
        for priority in ReindexPriority:
            can_schedule = resource_manager.can_schedule_job(priority)
            status_text = "✓ ALLOW" if can_schedule else "✗ DENY"
            print(f"  {priority.name} priority job: {status_text}")
        
        # Show resource recommendations
        print_section("Resource Optimization Recommendations")
        recommendations = resource_manager.get_resource_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    finally:
        resource_manager.stop()


def demo_validation_integration():
    """Demonstrate integration with validation system"""
    print_header("Graph Validation Integration Demo")
    
    with demo_scheduler() as scheduler:
        print_section("Simulating Validation Issues")
        
        # Create mock validation report
        from dataclasses import dataclass
        from enum import Enum
        
        class MockSeverity(Enum):
            CRITICAL = "critical"
            ERROR = "error"
            WARNING = "warning"
        
        @dataclass
        class MockIssue:
            id: str
            severity: MockSeverity
            table_name: str
            entity_id: str
            message: str
            category: str
            suggested_fix: str
        
        @dataclass
        class MockValidationReport:
            issues: list
        
        # Create mock validation issues
        mock_issues = [
            MockIssue(
                id="issue-1",
                severity=MockSeverity.CRITICAL,
                table_name="entities",
                entity_id="entity-123",
                message="Orphaned entity reference",
                category="referential_integrity",
                suggested_fix="Remove orphaned reference"
            ),
            MockIssue(
                id="issue-2",
                severity=MockSeverity.WARNING,
                table_name="relationships",
                entity_id="rel-456",
                message="Low confidence relationship",
                category="data_quality",
                suggested_fix="Review relationship confidence"
            )
        ]
        
        mock_report = MockValidationReport(issues=mock_issues)
        
        print(f"Mock validation report with {len(mock_issues)} issues:")
        for issue in mock_issues:
            print(f"  - {issue.severity.value.upper()}: {issue.message}")
        
        # Schedule validation-triggered reindexing
        initial_queue_size = scheduler.job_queue.qsize()
        scheduler.schedule_validation_triggered_reindex(mock_report)
        
        jobs_added = scheduler.job_queue.qsize() - initial_queue_size
        print(f"\n✓ Scheduled {jobs_added} reindexing jobs based on validation issues")
        
        # Process the validation-triggered jobs
        print_section("Processing Validation-Triggered Jobs")
        time.sleep(3)
        
        print_status(scheduler)


def demo_batch_processing():
    """Demonstrate batch processing optimization"""
    print_header("Batch Processing Optimization Demo")
    
    with demo_scheduler() as scheduler:
        print_section("Creating Batch-Optimizable Jobs")
        
        # Schedule related jobs that can be batched
        file_jobs = []
        for i in range(8):
            job_id = scheduler.schedule_reindex(
                entity_type="function",
                file_path=f"/src/module_{i//3}.py",  # Group by file
                priority=ReindexPriority.MEDIUM,
                trigger=ReindexTrigger.FILE_CHANGE,
                trigger_reason=f"Function update in module_{i//3}.py"
            )
            file_jobs.append(job_id)
        
        print(f"✓ Scheduled {len(file_jobs)} related jobs")
        print(f"✓ Queue size: {scheduler.job_queue.qsize()}")
        
        # The scheduler will automatically batch related jobs
        print_section("Batch Processing in Action")
        print("Scheduler will automatically batch related jobs for efficiency...")
        
        # Let scheduler process batches
        time.sleep(5)
        
        print_status(scheduler)


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    print_header("Performance Monitoring Demo")
    
    print_section("Priority Calculator Statistics")
    
    # Create priority calculator and simulate some history
    calculator = create_priority_calculator()
    
    # Simulate some job performance history
    from knowledge.indexing.auto_reindexing_scheduler import ReindexJob
    
    for i in range(20):
        mock_job = ReindexJob(
            entity_type=["module", "class", "function"][i % 3],
            trigger=ReindexTrigger.FILE_CHANGE
        )
        
        calculator.record_job_performance(
            mock_job,
            success=i % 10 != 0,  # 90% success rate
            duration=1.0 + i * 0.1,
            metrics={"cpu": 10 + i, "memory": 50 + i * 2}
        )
    
    stats = calculator.get_priority_statistics()
    print("Priority calculation statistics:")
    for key, value in stats.items():
        if key not in ["current_weights"]:  # Skip detailed weights
            print(f"  {key}: {value}")
    
    # Demonstrate with scheduler
    print_section("Live Performance Monitoring")
    
    with demo_scheduler() as scheduler:
        # Schedule jobs with performance tracking
        for i in range(10):
            scheduler.schedule_reindex(
                entity_type=f"perf_test_{i}",
                priority=ReindexPriority.MEDIUM,
                trigger=ReindexTrigger.MANUAL_REQUEST,
                trigger_reason=f"Performance test job {i}"
            )
        
        print("Processing jobs with performance monitoring...")
        time.sleep(4)
        
        # Show performance metrics
        status = scheduler.get_status()
        metrics = status["metrics"]
        
        print("Performance metrics:")
        print(f"  Average processing time: {metrics['average_processing_time']:.3f}s")
        print(f"  Throughput: {metrics['throughput_jobs_per_minute']:.1f} jobs/min")
        print(f"  Jobs processed: {metrics['jobs_processed']}")
        total_jobs = metrics['jobs_processed'] + metrics['jobs_failed']
        if total_jobs > 0:
            success_rate = (metrics['jobs_processed'] / total_jobs) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        else:
            print("  Success rate: N/A (no jobs completed yet)")


def interactive_demo():
    """Interactive demo mode"""
    print_header("Interactive Auto-Reindexing Scheduler Demo")
    
    with demo_scheduler() as scheduler:
        print("\nInteractive mode started. Available commands:")
        print("  schedule <priority> <type> <reason> - Schedule a reindexing job")
        print("  status - Show scheduler status")
        print("  history - Show job history")
        print("  pause - Pause scheduler")
        print("  resume - Resume scheduler")
        print("  quit - Exit interactive mode")
        print("\nPriorities: critical, high, medium, low")
        print("Types: module, class, function, variable, etc.")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                
                elif command[0] == "schedule" and len(command) >= 4:
                    priority_name = command[1].upper()
                    entity_type = command[2]
                    reason = " ".join(command[3:])
                    
                    try:
                        priority = ReindexPriority[priority_name]
                        job_id = scheduler.schedule_reindex(
                            entity_type=entity_type,
                            priority=priority,
                            trigger=ReindexTrigger.MANUAL_REQUEST,
                            trigger_reason=reason
                        )
                        print(f"✓ Scheduled job {job_id}")
                    except KeyError:
                        print(f"Invalid priority: {priority_name}")
                
                elif command[0] == "status":
                    print_status(scheduler)
                
                elif command[0] == "history":
                    history = scheduler.get_job_history()
                    print(f"Completed: {len(history['completed'])}")
                    print(f"Failed: {len(history['failed'])}")
                    print(f"Active: {len(history['active'])}")
                
                elif command[0] == "pause":
                    if scheduler.pause():
                        print("✓ Scheduler paused")
                    else:
                        print("✗ Failed to pause scheduler")
                
                elif command[0] == "resume":
                    if scheduler.resume():
                        print("✓ Scheduler resumed")
                    else:
                        print("✗ Failed to resume scheduler")
                
                else:
                    print("Unknown command or invalid arguments")
            
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main demo function"""
    print_header("RIF Auto-Reindexing Scheduler Demonstration")
    print("This demo showcases the capabilities of the auto-reindexing scheduler")
    print("built for Issue #69.")
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            interactive_demo()
        else:
            # Run all demos
            demo_basic_scheduling()
            time.sleep(2)
            
            demo_priority_system()
            time.sleep(2)
            
            demo_resource_management()
            time.sleep(2)
            
            demo_validation_integration()
            time.sleep(2)
            
            demo_batch_processing()
            time.sleep(2)
            
            demo_performance_monitoring()
            
            print_header("Demo Complete")
            print("✅ All demonstrations completed successfully!")
            print("\nKey features demonstrated:")
            print("  • Priority-based job scheduling")
            print("  • Resource-aware execution")
            print("  • Graph validation integration")
            print("  • Batch processing optimization")
            print("  • Performance monitoring")
            print("  • Comprehensive testing framework")
            print("\nFor interactive mode, run: python demo_auto_reindexing_scheduler.py --interactive")
    
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()