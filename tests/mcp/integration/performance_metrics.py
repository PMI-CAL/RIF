"""
Performance Metrics Collection for MCP Integration Testing

Provides comprehensive metrics collection, analysis, and reporting capabilities
for MCP integration test performance evaluation.

Issue: #86 - Build MCP integration tests
Component: Performance Metrics
"""

import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Deque
from enum import Enum
import json


class MetricType(Enum):
    """Types of metrics collected"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"  
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    CONCURRENCY = "concurrency"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class OperationMetric:
    """Represents a single operation metric"""
    timestamp: float
    operation: str
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    concurrency_level: int = 1
    server_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark run"""
    concurrency_level: int
    total_requests: int
    successful_requests: int
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_requests_per_second: float
    success_rate: float
    timestamp: float
    percentiles: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestMetrics:
    """Aggregated metrics for a test session"""
    test_name: str
    start_time: float
    end_time: Optional[float] = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    operations_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    success_rate: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    performance_percentiles: Dict[str, float] = field(default_factory=dict)


class PerformanceMetrics:
    """
    Comprehensive performance metrics collection and analysis
    
    Collects, analyzes, and reports performance metrics for MCP integration tests
    with support for real-time monitoring and historical analysis.
    """
    
    def __init__(self, test_name: str, max_history: int = 10000):
        """Initialize performance metrics collection"""
        self.test_name = test_name
        self.max_history = max_history
        self.start_time = time.time()
        
        # Metrics storage
        self.operation_history: Deque[OperationMetric] = deque(maxlen=max_history)
        self.current_session_metrics: List[OperationMetric] = []
        
        # Aggregated statistics
        self._response_times: List[float] = []
        self._operations_by_type: Dict[str, List[OperationMetric]] = defaultdict(list)
        self._errors_by_type: Dict[str, int] = defaultdict(int)
        
        # Real-time monitoring
        self._current_concurrency = 0
        self._peak_concurrency = 0
        self._last_reset_time = time.time()
    
    def record_operation(self,
                        operation: str,
                        duration_ms: float,
                        success: bool,
                        error_type: Optional[str] = None,
                        server_name: Optional[str] = None,
                        concurrency_level: int = 1,
                        **metadata) -> None:
        """Record a single operation metric"""
        
        metric = OperationMetric(
            timestamp=time.time(),
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            concurrency_level=concurrency_level,
            server_name=server_name,
            metadata=metadata
        )
        
        # Store metric
        self.operation_history.append(metric)
        self.current_session_metrics.append(metric)
        
        # Update aggregated statistics
        self._response_times.append(duration_ms)
        self._operations_by_type[operation].append(metric)
        
        if not success and error_type:
            self._errors_by_type[error_type] += 1
        
        # Update concurrency tracking
        self._peak_concurrency = max(self._peak_concurrency, concurrency_level)
    
    def start_operation_tracking(self) -> None:
        """Start tracking concurrent operations"""
        self._current_concurrency += 1
        self._peak_concurrency = max(self._peak_concurrency, self._current_concurrency)
    
    def end_operation_tracking(self) -> None:
        """End tracking concurrent operations"""
        self._current_concurrency = max(0, self._current_concurrency - 1)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        if not self.current_session_metrics:
            return {
                "test_name": self.test_name,
                "total_operations": 0,
                "message": "No operations recorded"
            }
        
        successful_ops = [m for m in self.current_session_metrics if m.success]
        failed_ops = [m for m in self.current_session_metrics if not m.success]
        
        response_times = [m.duration_ms for m in self.current_session_metrics]
        successful_response_times = [m.duration_ms for m in successful_ops]
        
        session_duration = time.time() - self._last_reset_time
        
        summary = {
            "test_name": self.test_name,
            "session_duration_seconds": session_duration,
            "total_operations": len(self.current_session_metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.current_session_metrics) if self.current_session_metrics else 0,
            
            # Response time statistics
            "average_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "median_response_time_ms": statistics.median(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "response_time_std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            
            # Percentiles
            "percentiles": self._calculate_percentiles(response_times),
            
            # Throughput
            "throughput_ops_per_second": len(self.current_session_metrics) / session_duration if session_duration > 0 else 0,
            "successful_throughput_ops_per_second": len(successful_ops) / session_duration if session_duration > 0 else 0,
            
            # Concurrency
            "peak_concurrency": self._peak_concurrency,
            "current_concurrency": self._current_concurrency,
            
            # Operations by type
            "operations_by_type": {
                op_type: len(ops) for op_type, ops in self._operations_by_type.items()
            },
            
            # Error analysis
            "errors_by_type": dict(self._errors_by_type),
            "error_rate": len(failed_ops) / len(self.current_session_metrics) if self.current_session_metrics else 0,
            
            # Performance classification
            "performance_class": self._classify_performance(
                statistics.mean(successful_response_times) if successful_response_times else float('inf'),
                len(successful_ops) / len(self.current_session_metrics) if self.current_session_metrics else 0
            )
        }
        
        return summary
    
    def get_operation_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed breakdown by operation type"""
        breakdown = {}
        
        for operation, metrics in self._operations_by_type.items():
            successful = [m for m in metrics if m.success]
            failed = [m for m in metrics if not m.success]
            response_times = [m.duration_ms for m in metrics]
            
            breakdown[operation] = {
                "total_count": len(metrics),
                "successful_count": len(successful),
                "failed_count": len(failed),
                "success_rate": len(successful) / len(metrics) if metrics else 0,
                "average_response_time_ms": statistics.mean(response_times) if response_times else 0,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "percentiles": self._calculate_percentiles(response_times)
            }
        
        return breakdown
    
    def get_server_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed breakdown by server"""
        server_metrics = defaultdict(list)
        
        for metric in self.current_session_metrics:
            if metric.server_name:
                server_metrics[metric.server_name].append(metric)
        
        breakdown = {}
        for server_name, metrics in server_metrics.items():
            successful = [m for m in metrics if m.success]
            response_times = [m.duration_ms for m in metrics]
            
            breakdown[server_name] = {
                "total_requests": len(metrics),
                "successful_requests": len(successful),
                "success_rate": len(successful) / len(metrics) if metrics else 0,
                "average_response_time_ms": statistics.mean(response_times) if response_times else 0,
                "throughput_ops_per_second": len(metrics) / (time.time() - self._last_reset_time)
            }
        
        return breakdown
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        length = len(sorted_values)
        
        def percentile(p):
            index = int((p / 100.0) * (length - 1))
            return sorted_values[min(index, length - 1)]
        
        return {
            "p50": percentile(50),
            "p75": percentile(75),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99)
        }
    
    def _classify_performance(self, avg_response_time_ms: float, success_rate: float) -> str:
        """Classify performance based on response time and success rate"""
        if success_rate < 0.8:
            return "poor"
        elif success_rate < 0.95:
            if avg_response_time_ms > 1000:
                return "poor"
            elif avg_response_time_ms > 500:
                return "fair"
            else:
                return "good"
        else:  # success_rate >= 0.95
            if avg_response_time_ms > 1000:
                return "fair"
            elif avg_response_time_ms > 200:
                return "good"
            else:
                return "excellent"
    
    def reset(self) -> None:
        """Reset session metrics while preserving history"""
        self.current_session_metrics.clear()
        self._response_times.clear()
        self._operations_by_type.clear()
        self._errors_by_type.clear()
        self._current_concurrency = 0
        self._peak_concurrency = 0
        self._last_reset_time = time.time()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all metrics to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time,
            "session_metrics": [
                {
                    "timestamp": m.timestamp,
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "success": m.success,
                    "error_type": m.error_type,
                    "concurrency_level": m.concurrency_level,
                    "server_name": m.server_name,
                    "metadata": m.metadata
                }
                for m in self.current_session_metrics
            ],
            "summary": self.get_summary(),
            "operation_breakdown": self.get_operation_breakdown(),
            "server_breakdown": self.get_server_breakdown()
        }
    
    def export_to_json(self, file_path: str) -> None:
        """Export metrics to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.export_to_dict(), f, indent=2)
    
    def get_benchmark_comparison(self, baseline_metrics: 'PerformanceMetrics') -> Dict[str, Any]:
        """Compare current metrics against baseline"""
        current_summary = self.get_summary()
        baseline_summary = baseline_metrics.get_summary()
        
        def percentage_change(current, baseline):
            if baseline == 0:
                return 0 if current == 0 else float('inf')
            return ((current - baseline) / baseline) * 100
        
        comparison = {
            "response_time_change_percent": percentage_change(
                current_summary['average_response_time_ms'],
                baseline_summary['average_response_time_ms']
            ),
            "success_rate_change_percent": percentage_change(
                current_summary['success_rate'],
                baseline_summary['success_rate']  
            ),
            "throughput_change_percent": percentage_change(
                current_summary['throughput_ops_per_second'],
                baseline_summary['throughput_ops_per_second']
            ),
            "performance_regression": {
                "response_time_regression": current_summary['average_response_time_ms'] > baseline_summary['average_response_time_ms'] * 1.1,
                "success_rate_regression": current_summary['success_rate'] < baseline_summary['success_rate'] * 0.95,
                "throughput_regression": current_summary['throughput_ops_per_second'] < baseline_summary['throughput_ops_per_second'] * 0.9
            },
            "overall_performance": "improved" if (
                current_summary['average_response_time_ms'] <= baseline_summary['average_response_time_ms'] * 1.05 and
                current_summary['success_rate'] >= baseline_summary['success_rate'] * 0.98 and
                current_summary['throughput_ops_per_second'] >= baseline_summary['throughput_ops_per_second'] * 0.95
            ) else "regressed"
        }
        
        return comparison


class MetricsCollector:
    """Centralized metrics collection and management"""
    
    def __init__(self):
        self.test_sessions: Dict[str, PerformanceMetrics] = {}
        self.global_metrics = PerformanceMetrics("global")
    
    def create_test_session(self, test_name: str) -> PerformanceMetrics:
        """Create a new test session"""
        metrics = PerformanceMetrics(test_name)
        self.test_sessions[test_name] = metrics
        return metrics
    
    def get_test_session(self, test_name: str) -> Optional[PerformanceMetrics]:
        """Get existing test session"""
        return self.test_sessions.get(test_name)
    
    def record_global_metric(self, **kwargs) -> None:
        """Record metric in global session"""
        self.global_metrics.record_operation(**kwargs)
    
    def get_session_comparison(self, test_name1: str, test_name2: str) -> Dict[str, Any]:
        """Compare two test sessions"""
        session1 = self.get_test_session(test_name1)
        session2 = self.get_test_session(test_name2)
        
        if not session1 or not session2:
            return {"error": "One or both test sessions not found"}
        
        return session1.get_benchmark_comparison(session2)
    
    def export_all_sessions(self, directory_path: str) -> None:
        """Export all test sessions to directory"""
        import os
        os.makedirs(directory_path, exist_ok=True)
        
        for test_name, metrics in self.test_sessions.items():
            file_path = os.path.join(directory_path, f"{test_name}_metrics.json")
            metrics.export_to_json(file_path)
        
        # Export global metrics
        global_file_path = os.path.join(directory_path, "global_metrics.json")
        self.global_metrics.export_to_json(global_file_path)
    
    def get_all_sessions_summary(self) -> Dict[str, Any]:
        """Get summary of all test sessions"""
        return {
            "total_sessions": len(self.test_sessions),
            "sessions": {
                name: metrics.get_summary()
                for name, metrics in self.test_sessions.items()
            },
            "global_summary": self.global_metrics.get_summary()
        }


# Global metrics collector instance
default_collector = MetricsCollector()