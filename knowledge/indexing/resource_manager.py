"""
Resource Manager for Auto-Reindexing Scheduler
Issue #69: Build auto-reindexing scheduler

This module provides resource monitoring and management for the auto-reindexing scheduler:
- System resource monitoring (CPU, memory, I/O)
- Adaptive resource allocation based on priority
- Throttling mechanisms to prevent system overload
- Integration with existing system monitoring

Author: RIF-Implementer
Date: 2025-08-23
"""

import psutil
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime, timedelta
import statistics
from collections import deque


class ReindexPriority(IntEnum):
    """Priority levels for reindexing operations (lower = higher priority)"""
    CRITICAL = 0    # Data integrity issues, must run immediately
    HIGH = 1        # Content updates, performance critical
    MEDIUM = 2      # Relationship updates, optimization 
    LOW = 3         # Background optimization, maintenance


@dataclass
class ResourceSnapshot:
    """Point-in-time snapshot of system resources"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    io_read_rate: float      # MB/s
    io_write_rate: float     # MB/s
    disk_usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_mb": self.memory_available_mb,
            "io_read_rate": self.io_read_rate,
            "io_write_rate": self.io_write_rate,
            "disk_usage_percent": self.disk_usage_percent
        }


@dataclass
class ResourceThresholds:
    """Resource utilization thresholds for different priorities"""
    cpu_threshold: float        # 0.0-1.0
    memory_threshold: float     # 0.0-1.0
    io_threshold: float         # MB/s
    disk_threshold: float       # 0.0-1.0
    
    def allows_execution(self, snapshot: ResourceSnapshot) -> bool:
        """Check if current resource usage allows execution at this threshold"""
        return (
            snapshot.cpu_percent / 100.0 <= self.cpu_threshold and
            snapshot.memory_percent / 100.0 <= self.memory_threshold and
            (snapshot.io_read_rate + snapshot.io_write_rate) <= self.io_threshold and
            snapshot.disk_usage_percent / 100.0 <= self.disk_threshold
        )


class SystemResourceMonitor:
    """
    Monitors system resource utilization with configurable sampling intervals.
    Provides historical data and trend analysis for adaptive scheduling.
    """
    
    def __init__(self, 
                 sample_interval: float = 5.0,
                 history_size: int = 720,  # 1 hour at 5s intervals
                 knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.knowledge_path = knowledge_path
        self.logger = logging.getLogger(f"resource_monitor_{id(self)}")
        
        # Resource history
        self.resource_history = deque(maxlen=history_size)
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Cached values for performance
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._last_snapshot_time = 0.0
        self._cache_duration = 2.0  # seconds
        
        # I/O tracking for rate calculation
        self._last_io_counters = None
        self._last_io_time = None
        
        self.logger.info("SystemResourceMonitor initialized")
    
    def start_monitoring(self) -> bool:
        """Start background resource monitoring"""
        if self.monitoring_active:
            self.logger.warning("Resource monitoring already active")
            return True
        
        try:
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name=f"ResourceMonitor-{id(self)}"
            )
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.monitoring_active = True
            self.logger.info("Resource monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start resource monitoring: {e}")
            return False
    
    def stop_monitoring(self, timeout: float = 10.0) -> bool:
        """Stop background resource monitoring"""
        if not self.monitoring_active:
            return True
        
        self.logger.info("Stopping resource monitoring...")
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=timeout)
            
            if self.monitoring_thread.is_alive():
                self.logger.warning("Resource monitoring thread did not stop cleanly")
                return False
        
        self.monitoring_active = False
        self.logger.info("Resource monitoring stopped")
        return True
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        self.logger.info("Resource monitoring loop started")
        
        while not self.stop_event.is_set():
            try:
                # Take resource snapshot
                snapshot = self._take_snapshot()
                
                # Add to history
                self.resource_history.append(snapshot)
                
                # Wait for next sample
                self.stop_event.wait(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring loop: {e}")
                self.stop_event.wait(5.0)  # Brief pause on error
        
        self.logger.info("Resource monitoring loop ended")
    
    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot with caching"""
        current_time = time.time()
        
        # Return cached snapshot if recent enough
        if (self._last_snapshot and 
            current_time - self._last_snapshot_time < self._cache_duration):
            return self._last_snapshot
        
        # Take new snapshot
        snapshot = self._take_snapshot()
        self._last_snapshot = snapshot
        self._last_snapshot_time = current_time
        
        return snapshot
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a point-in-time resource snapshot"""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # I/O rates (calculated from deltas)
            io_read_rate, io_write_rate = self._calculate_io_rates()
            
            # Disk usage (for knowledge directory)
            try:
                disk_usage = psutil.disk_usage(self.knowledge_path)
                disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            except:
                disk_usage_percent = 0.0
            
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                io_read_rate=io_read_rate,
                io_write_rate=io_write_rate,
                disk_usage_percent=disk_usage_percent
            )
            
        except Exception as e:
            self.logger.error(f"Error taking resource snapshot: {e}")
            # Return default snapshot on error
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=1024.0,  # 1GB default
                io_read_rate=0.0,
                io_write_rate=0.0,
                disk_usage_percent=0.0
            )
    
    def _calculate_io_rates(self) -> Tuple[float, float]:
        """Calculate I/O read/write rates in MB/s"""
        try:
            current_time = time.time()
            io_counters = psutil.disk_io_counters()
            
            if io_counters is None:
                return 0.0, 0.0
            
            if self._last_io_counters is None or self._last_io_time is None:
                # First measurement, no rate calculation possible
                self._last_io_counters = io_counters
                self._last_io_time = current_time
                return 0.0, 0.0
            
            # Calculate time delta
            time_delta = current_time - self._last_io_time
            if time_delta <= 0:
                return 0.0, 0.0
            
            # Calculate byte deltas
            read_bytes_delta = io_counters.read_bytes - self._last_io_counters.read_bytes
            write_bytes_delta = io_counters.write_bytes - self._last_io_counters.write_bytes
            
            # Calculate rates in MB/s
            read_rate = (read_bytes_delta / time_delta) / (1024 * 1024)
            write_rate = (write_bytes_delta / time_delta) / (1024 * 1024)
            
            # Update cached values
            self._last_io_counters = io_counters
            self._last_io_time = current_time
            
            return max(0.0, read_rate), max(0.0, write_rate)
            
        except Exception as e:
            self.logger.warning(f"Error calculating I/O rates: {e}")
            return 0.0, 0.0
    
    def get_resource_trend(self, duration_minutes: int = 5) -> Dict[str, float]:
        """
        Analyze resource trends over the specified duration.
        
        Returns:
            Dictionary with trend analysis (positive = increasing, negative = decreasing)
        """
        if not self.resource_history:
            return {
                "cpu_trend": 0.0,
                "memory_trend": 0.0,
                "io_trend": 0.0,
                "data_points": 0
            }
        
        # Filter history to requested duration
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_snapshots = [
            s for s in self.resource_history 
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return {
                "cpu_trend": 0.0,
                "memory_trend": 0.0,
                "io_trend": 0.0,
                "data_points": len(recent_snapshots)
            }
        
        # Calculate simple linear trends
        def calculate_trend(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            # Simple slope calculation
            n = len(values)
            x_values = list(range(n))
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            return numerator / denominator if denominator != 0 else 0.0
        
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        io_values = [s.io_read_rate + s.io_write_rate for s in recent_snapshots]
        
        return {
            "cpu_trend": calculate_trend(cpu_values),
            "memory_trend": calculate_trend(memory_values),
            "io_trend": calculate_trend(io_values),
            "data_points": len(recent_snapshots)
        }
    
    def get_resource_statistics(self, duration_minutes: int = 10) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of recent resource usage"""
        if not self.resource_history:
            return {}
        
        # Filter to requested duration
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_snapshots = [
            s for s in self.resource_history 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        def calc_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            
            return {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0
            }
        
        return {
            "cpu": calc_stats([s.cpu_percent for s in recent_snapshots]),
            "memory": calc_stats([s.memory_percent for s in recent_snapshots]),
            "io_read": calc_stats([s.io_read_rate for s in recent_snapshots]),
            "io_write": calc_stats([s.io_write_rate for s in recent_snapshots]),
            "disk": calc_stats([s.disk_usage_percent for s in recent_snapshots]),
            "sample_count": len(recent_snapshots)
        }


class ReindexingResourceManager:
    """
    Manages resource allocation for reindexing jobs based on priority and system load.
    Integrates with SystemResourceMonitor to make scheduling decisions.
    """
    
    def __init__(self,
                 resource_thresholds: Optional[Dict[ReindexPriority, float]] = None,
                 check_interval: float = 5.0,
                 knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.knowledge_path = knowledge_path
        self.check_interval = check_interval
        self.logger = logging.getLogger(f"resource_manager_{id(self)}")
        
        # Initialize resource monitor
        self.monitor = SystemResourceMonitor(
            sample_interval=check_interval,
            knowledge_path=knowledge_path
        )
        
        # Set up resource thresholds for each priority level
        if resource_thresholds:
            self.priority_thresholds = resource_thresholds
        else:
            # Default thresholds (0.0-1.0 scale) - adjusted to be more reasonable
            self.priority_thresholds = {
                ReindexPriority.CRITICAL: 0.95,  # Can use up to 95% resources
                ReindexPriority.HIGH: 0.85,      # Can use up to 85% resources  
                ReindexPriority.MEDIUM: 0.75,    # Can use up to 75% resources
                ReindexPriority.LOW: 0.60        # Can use up to 60% resources
            }
        
        # Convert to ResourceThresholds objects
        self.threshold_objects = {}
        for priority, threshold in self.priority_thresholds.items():
            self.threshold_objects[priority] = ResourceThresholds(
                cpu_threshold=threshold,
                memory_threshold=threshold,
                io_threshold=threshold * 100.0,  # MB/s threshold
                disk_threshold=min(threshold + 0.1, 0.95)  # Allow slightly higher disk usage
            )
        
        # Adaptive scheduling state
        self.load_history = deque(maxlen=100)
        self.job_performance_history = deque(maxlen=50)
        
        # Current resource reservation
        self.reserved_resources = {
            "cpu": 0.0,
            "memory": 0.0,
            "io": 0.0
        }
        
        self.logger.info("ReindexingResourceManager initialized")
    
    def start(self) -> bool:
        """Start resource monitoring"""
        return self.monitor.start_monitoring()
    
    def stop(self, timeout: float = 10.0) -> bool:
        """Stop resource monitoring"""
        return self.monitor.stop_monitoring(timeout)
    
    def can_schedule_job(self, priority: ReindexPriority) -> bool:
        """
        Check if system resources allow scheduling a job with the given priority.
        
        Args:
            priority: Job priority level
            
        Returns:
            True if job can be scheduled, False otherwise
        """
        try:
            # Get current resource snapshot
            snapshot = self.monitor.get_current_snapshot()
            
            # Get threshold for this priority
            threshold = self.threshold_objects.get(priority)
            if not threshold:
                # Default to most restrictive
                threshold = self.threshold_objects[ReindexPriority.LOW]
            
            # Check if current usage allows execution
            can_schedule = threshold.allows_execution(snapshot)
            
            # Consider resource trends for adaptive scheduling
            if can_schedule and self._should_consider_trends():
                trends = self.monitor.get_resource_trend(duration_minutes=3)
                
                # If resources are trending up rapidly, be more conservative
                if (trends["cpu_trend"] > 5.0 or trends["memory_trend"] > 5.0 or 
                    trends["io_trend"] > 10.0):
                    
                    # Only allow higher priority jobs when resources are trending up
                    if priority.value > ReindexPriority.HIGH.value:
                        can_schedule = False
                        self.logger.debug(f"Rejected {priority.name} job due to upward resource trends")
            
            # Log decision for debugging
            self.logger.debug(
                f"Schedule check for {priority.name}: "
                f"CPU {snapshot.cpu_percent:.1f}%/{threshold.cpu_threshold*100:.1f}%, "
                f"Memory {snapshot.memory_percent:.1f}%/{threshold.memory_threshold*100:.1f}%, "
                f"Result: {'ALLOW' if can_schedule else 'DENY'}"
            )
            
            return can_schedule
            
        except Exception as e:
            self.logger.error(f"Error checking resource availability: {e}")
            # Conservative approach - deny on error
            return False
    
    def _should_consider_trends(self) -> bool:
        """Check if we have enough data to consider resource trends"""
        return len(self.monitor.resource_history) > 10
    
    def reserve_resources(self, priority: ReindexPriority, estimated_usage: Dict[str, float]):
        """
        Reserve resources for a job (for future enhancement).
        
        Args:
            priority: Job priority
            estimated_usage: Estimated resource usage dict
        """
        # This is a placeholder for future resource reservation system
        # For now, we just log the reservation
        self.logger.debug(f"Reserved resources for {priority.name}: {estimated_usage}")
    
    def release_resources(self, priority: ReindexPriority, actual_usage: Dict[str, float]):
        """
        Release reserved resources and update performance history.
        
        Args:
            priority: Job priority
            actual_usage: Actual resource usage dict
        """
        # Update performance history for adaptive learning
        self.job_performance_history.append({
            "priority": priority,
            "usage": actual_usage,
            "timestamp": datetime.now()
        })
        
        self.logger.debug(f"Released resources for {priority.name}: {actual_usage}")
    
    def get_current_utilization(self) -> float:
        """
        Get overall current resource utilization as a single metric.
        
        Returns:
            Float between 0.0 and 1.0 representing overall system load
        """
        try:
            snapshot = self.monitor.get_current_snapshot()
            
            # Weighted combination of resource metrics
            cpu_weight = 0.4
            memory_weight = 0.3
            io_weight = 0.2
            disk_weight = 0.1
            
            # Normalize I/O to 0-100 scale (assume 100 MB/s is 100%)
            io_percent = min((snapshot.io_read_rate + snapshot.io_write_rate) / 100.0 * 100, 100)
            
            overall_utilization = (
                snapshot.cpu_percent * cpu_weight +
                snapshot.memory_percent * memory_weight +
                io_percent * io_weight +
                snapshot.disk_usage_percent * disk_weight
            ) / 100.0
            
            return min(overall_utilization, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating utilization: {e}")
            return 0.0
    
    def get_optimal_scheduling_window(self) -> Tuple[datetime, datetime]:
        """
        Analyze historical data to suggest optimal scheduling window.
        
        Returns:
            Tuple of (start_time, end_time) for optimal scheduling
        """
        # This is a placeholder for future implementation
        # For now, return next 5-minute window
        start = datetime.now()
        end = start + timedelta(minutes=5)
        return start, end
    
    def adjust_thresholds(self, performance_metrics: Dict[str, float]):
        """
        Adaptively adjust resource thresholds based on system performance.
        
        Args:
            performance_metrics: Recent system performance data
        """
        # This is a placeholder for adaptive threshold adjustment
        # In a full implementation, this would analyze performance trends
        # and automatically adjust thresholds to optimize system performance
        
        self.logger.debug(f"Performance metrics for threshold adjustment: {performance_metrics}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive resource manager status"""
        try:
            snapshot = self.monitor.get_current_snapshot()
            statistics = self.monitor.get_resource_statistics()
            trends = self.monitor.get_resource_trend()
            
            return {
                "monitoring_active": self.monitor.monitoring_active,
                "current_snapshot": snapshot.to_dict(),
                "overall_utilization": self.get_current_utilization(),
                "resource_statistics": statistics,
                "resource_trends": trends,
                "priority_thresholds": {
                    p.name: t for p, t in self.priority_thresholds.items()
                },
                "history_size": len(self.monitor.resource_history),
                "reserved_resources": self.reserved_resources.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting resource manager status: {e}")
            return {
                "monitoring_active": False,
                "error": str(e)
            }
    
    def get_resource_recommendations(self) -> List[str]:
        """Get recommendations for resource optimization"""
        recommendations = []
        
        try:
            snapshot = self.monitor.get_current_snapshot()
            trends = self.monitor.get_resource_trend()
            
            # CPU recommendations
            if snapshot.cpu_percent > 80:
                recommendations.append("High CPU usage detected - consider reducing concurrent jobs")
            elif trends["cpu_trend"] > 10:
                recommendations.append("CPU usage trending upward - monitor for potential issues")
            
            # Memory recommendations
            if snapshot.memory_percent > 85:
                recommendations.append("High memory usage - consider reducing batch sizes")
            elif snapshot.memory_available_mb < 500:
                recommendations.append("Low available memory - schedule only critical jobs")
            
            # I/O recommendations
            total_io = snapshot.io_read_rate + snapshot.io_write_rate
            if total_io > 50:  # MB/s
                recommendations.append("High I/O activity - defer low priority indexing jobs")
            
            # Disk recommendations
            if snapshot.disk_usage_percent > 90:
                recommendations.append("Disk usage critical - clean up temporary files")
            elif snapshot.disk_usage_percent > 80:
                recommendations.append("High disk usage - monitor storage capacity")
            
            if not recommendations:
                recommendations.append("Resource usage within normal parameters")
            
        except Exception as e:
            recommendations.append(f"Unable to analyze resources: {e}")
        
        return recommendations


# Factory functions for easy instantiation
def create_resource_manager(knowledge_path: str = "/Users/cal/DEV/RIF/knowledge") -> ReindexingResourceManager:
    """Create a ReindexingResourceManager with default configuration"""
    return ReindexingResourceManager(knowledge_path=knowledge_path)


def create_resource_monitor(knowledge_path: str = "/Users/cal/DEV/RIF/knowledge") -> SystemResourceMonitor:
    """Create a SystemResourceMonitor with default configuration"""
    return SystemResourceMonitor(knowledge_path=knowledge_path)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Test resource monitoring
    print("Testing Resource Management System")
    print("=" * 40)
    
    # Create resource manager
    manager = create_resource_manager()
    
    try:
        # Start monitoring
        if manager.start():
            print("Resource monitoring started successfully")
            
            # Let it collect some data
            time.sleep(10)
            
            # Test scheduling decisions
            for priority in ReindexPriority:
                can_schedule = manager.can_schedule_job(priority)
                print(f"Can schedule {priority.name} job: {can_schedule}")
            
            # Get status
            status = manager.get_status()
            print(f"\nCurrent utilization: {status.get('overall_utilization', 0):.1%}")
            
            # Get recommendations
            recommendations = manager.get_resource_recommendations()
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        else:
            print("Failed to start resource monitoring")
    
    finally:
        manager.stop()
        print("\nResource monitoring stopped")