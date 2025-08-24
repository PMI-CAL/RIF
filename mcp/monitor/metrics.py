"""
MCP Performance Metrics Collector
Real-time performance data collection, aggregation, and analysis for MCP servers

Features:
- Real-time performance metrics collection
- Time-series aggregation with trend analysis  
- Performance benchmarking and baseline tracking
- Resource utilization monitoring
- Integration with enterprise monitoring systems

Issue: #84 - Create MCP health monitor
Component: Performance metrics collection and analysis
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement point"""
    timestamp: float
    metric_name: str
    value: float
    server_id: str
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric measurements"""
    metric_name: str
    server_id: str
    points: Deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""
    
    def add_point(self, value: float, timestamp: Optional[float] = None, tags: Optional[Dict[str, str]] = None):
        """Add a new metric point"""
        point = MetricPoint(
            timestamp=timestamp or time.time(),
            metric_name=self.metric_name,
            value=value,
            server_id=self.server_id,
            unit=self.unit,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_recent_values(self, duration_seconds: int = 300) -> List[float]:
        """Get values from the last N seconds"""
        cutoff = time.time() - duration_seconds
        return [p.value for p in self.points if p.timestamp >= cutoff]
    
    def get_statistics(self, duration_seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        values = self.get_recent_values(duration_seconds)
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison and alerting"""
    server_id: str
    metric_name: str
    baseline_value: float
    threshold_warning: float  # Percentage deviation for warning
    threshold_critical: float  # Percentage deviation for critical
    established_at: datetime
    sample_count: int
    confidence: float = 0.8  # Confidence in the baseline (0.0-1.0)


class PerformanceMetricsCollector:
    """
    Enterprise-grade performance metrics collector for MCP servers
    
    Provides real-time collection, aggregation, and analysis of performance
    metrics with trend detection and baseline comparison capabilities.
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 retention_hours: int = 168,  # 1 week
                 collection_interval: int = 30):
        """
        Initialize the performance metrics collector
        
        Args:
            storage_path: Path for storing metrics data
            retention_hours: Hours to retain metrics data
            collection_interval: Seconds between metric collection cycles
        """
        self.storage_path = Path(storage_path) if storage_path else Path("knowledge/monitoring/metrics")
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval
        
        # Metrics storage
        self.metrics: Dict[str, Dict[str, MetricSeries]] = defaultdict(dict)  # server_id -> metric_name -> series
        self.baselines: Dict[str, Dict[str, PerformanceBaseline]] = defaultdict(dict)  # server_id -> metric_name -> baseline
        
        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.collection_stats = {
            "total_collections": 0,
            "collection_errors": 0,
            "last_collection_time": None,
            "collection_duration_ms": deque(maxlen=100)
        }
        
        # Server callbacks for metric collection
        self.server_callbacks: Dict[str, Callable] = {}
        
        # Storage setup
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PerformanceMetricsCollector initialized (interval: {collection_interval}s, retention: {retention_hours}h)")
    
    async def start_collection(self):
        """Start the metrics collection system"""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self._shutdown_event.clear()
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop the metrics collection system"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        self._shutdown_event.set()
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        try:
            while self.is_collecting and not self._shutdown_event.is_set():
                collection_start = time.time()
                
                try:
                    # Collect metrics from all registered servers
                    await self._collect_all_metrics()
                    
                    # Update baselines periodically
                    await self._update_baselines()
                    
                    # Clean up old metrics
                    await self._cleanup_old_metrics()
                    
                    # Store metrics to persistent storage
                    await self._store_metrics_batch()
                    
                    # Update collection statistics
                    collection_time_ms = (time.time() - collection_start) * 1000
                    self.collection_stats["collection_duration_ms"].append(collection_time_ms)
                    self.collection_stats["total_collections"] += 1
                    self.collection_stats["last_collection_time"] = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection: {e}")
                    self.collection_stats["collection_errors"] += 1
                
                # Wait for next collection interval
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.collection_interval)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal interval timeout, continue collecting
                    
        except asyncio.CancelledError:
            logger.info("Metrics collection loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in metrics collection loop: {e}")
    
    async def _collect_all_metrics(self):
        """Collect metrics from all registered servers"""
        if not self.server_callbacks:
            return
        
        # Collect metrics from all servers in parallel
        tasks = []
        for server_id in list(self.server_callbacks.keys()):
            task = asyncio.create_task(self._collect_server_metrics(server_id))
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error collecting server metrics: {e}")
    
    async def _collect_server_metrics(self, server_id: str):
        """Collect metrics for a specific server"""
        if server_id not in self.server_callbacks:
            return
        
        try:
            callback = self.server_callbacks[server_id]
            server_metrics = await callback()
            
            if not isinstance(server_metrics, dict):
                logger.warning(f"Invalid metrics format from server {server_id}")
                return
            
            # Store each metric
            for metric_name, metric_data in server_metrics.items():
                await self._record_metric(server_id, metric_name, metric_data)
                
        except Exception as e:
            logger.error(f"Failed to collect metrics for server {server_id}: {e}")
    
    async def _record_metric(self, server_id: str, metric_name: str, metric_data: Any):
        """Record a single metric measurement"""
        try:
            # Handle different metric data formats
            if isinstance(metric_data, (int, float)):
                value = float(metric_data)
                unit = ""
                tags = {}
            elif isinstance(metric_data, dict):
                value = float(metric_data.get('value', 0))
                unit = metric_data.get('unit', '')
                tags = metric_data.get('tags', {})
            else:
                logger.warning(f"Unsupported metric data type for {metric_name}: {type(metric_data)}")
                return
            
            # Get or create metric series
            if metric_name not in self.metrics[server_id]:
                self.metrics[server_id][metric_name] = MetricSeries(
                    metric_name=metric_name,
                    server_id=server_id,
                    unit=unit
                )
            
            metric_series = self.metrics[server_id][metric_name]
            metric_series.add_point(value, tags=tags)
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name} for server {server_id}: {e}")
    
    async def _update_baselines(self):
        """Update performance baselines for all metrics"""
        for server_id, server_metrics in self.metrics.items():
            for metric_name, metric_series in server_metrics.items():
                await self._update_baseline(server_id, metric_name, metric_series)
    
    async def _update_baseline(self, server_id: str, metric_name: str, metric_series: MetricSeries):
        """Update baseline for a specific metric"""
        try:
            # Only update baselines if we have sufficient data
            recent_values = metric_series.get_recent_values(3600)  # Last hour
            if len(recent_values) < 10:
                return
            
            # Calculate baseline statistics
            mean_value = statistics.mean(recent_values)
            std_value = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            # Determine thresholds based on metric type
            if 'response_time' in metric_name.lower() or 'latency' in metric_name.lower():
                # For response time metrics, higher is worse
                threshold_warning = 50.0  # 50% increase
                threshold_critical = 100.0  # 100% increase
            elif 'error_rate' in metric_name.lower() or 'failure' in metric_name.lower():
                # For error metrics, any increase is concerning
                threshold_warning = 25.0  # 25% increase
                threshold_critical = 50.0  # 50% increase
            else:
                # Default thresholds
                threshold_warning = 30.0  # 30% deviation
                threshold_critical = 50.0  # 50% deviation
            
            # Calculate confidence based on data stability
            coefficient_of_variation = (std_value / mean_value) if mean_value > 0 else 1.0
            confidence = max(0.1, min(1.0, 1.0 - coefficient_of_variation))
            
            # Update or create baseline
            baseline = PerformanceBaseline(
                server_id=server_id,
                metric_name=metric_name,
                baseline_value=mean_value,
                threshold_warning=threshold_warning,
                threshold_critical=threshold_critical,
                established_at=datetime.utcnow(),
                sample_count=len(recent_values),
                confidence=confidence
            )
            
            self.baselines[server_id][metric_name] = baseline
            
        except Exception as e:
            logger.error(f"Failed to update baseline for {metric_name} on server {server_id}: {e}")
    
    async def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        for server_id in list(self.metrics.keys()):
            for metric_name in list(self.metrics[server_id].keys()):
                metric_series = self.metrics[server_id][metric_name]
                
                # Remove old points
                while metric_series.points and metric_series.points[0].timestamp < cutoff_time:
                    metric_series.points.popleft()
                
                # Remove empty series
                if not metric_series.points:
                    del self.metrics[server_id][metric_name]
            
            # Remove empty server entries
            if not self.metrics[server_id]:
                del self.metrics[server_id]
    
    async def _store_metrics_batch(self):
        """Store current metrics batch to persistent storage"""
        try:
            today = datetime.now().strftime("%Y%m%d")
            metrics_file = self.storage_path / f"performance_metrics_{today}.jsonl"
            
            current_time = time.time()
            stored_count = 0
            
            with open(metrics_file, 'a') as f:
                for server_id, server_metrics in self.metrics.items():
                    for metric_name, metric_series in server_metrics.items():
                        # Store recent points (last 5 minutes)
                        recent_points = [p for p in metric_series.points if current_time - p.timestamp <= 300]
                        
                        for point in recent_points:
                            metric_record = {
                                "timestamp": point.timestamp,
                                "server_id": server_id,
                                "metric_name": metric_name,
                                "value": point.value,
                                "unit": point.unit,
                                "tags": point.tags
                            }
                            f.write(json.dumps(metric_record) + '\n')
                            stored_count += 1
            
            if stored_count > 0:
                logger.debug(f"Stored {stored_count} metric points to {metrics_file}")
                
        except Exception as e:
            logger.error(f"Failed to store metrics batch: {e}")
    
    # Public API methods
    
    def register_server_callback(self, server_id: str, callback: Callable):
        """
        Register callback function for collecting server metrics
        
        Args:
            server_id: Server identifier
            callback: Async function that returns metrics dict
        """
        self.server_callbacks[server_id] = callback
        logger.info(f"Registered metrics callback for server: {server_id}")
    
    def unregister_server_callback(self, server_id: str):
        """Remove server metrics callback"""
        if server_id in self.server_callbacks:
            del self.server_callbacks[server_id]
            logger.info(f"Unregistered metrics callback for server: {server_id}")
    
    async def get_server_metrics(self, server_id: str, 
                               duration_seconds: int = 3600) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics summary for a specific server
        
        Args:
            server_id: Server identifier
            duration_seconds: Time window for metrics (default: 1 hour)
            
        Returns:
            Dictionary of metrics with statistics
        """
        if server_id not in self.metrics:
            return {}
        
        result = {}
        for metric_name, metric_series in self.metrics[server_id].items():
            stats = metric_series.get_statistics(duration_seconds)
            recent_values = metric_series.get_recent_values(duration_seconds)
            
            # Add baseline comparison if available
            baseline_info = {}
            if server_id in self.baselines and metric_name in self.baselines[server_id]:
                baseline = self.baselines[server_id][metric_name]
                current_value = stats.get('mean', 0)
                deviation = ((current_value - baseline.baseline_value) / baseline.baseline_value * 100) if baseline.baseline_value > 0 else 0
                
                baseline_info = {
                    "baseline_value": baseline.baseline_value,
                    "deviation_percent": round(deviation, 2),
                    "confidence": baseline.confidence,
                    "threshold_warning": baseline.threshold_warning,
                    "threshold_critical": baseline.threshold_critical
                }
            
            result[metric_name] = {
                "statistics": stats,
                "unit": metric_series.unit,
                "baseline": baseline_info,
                "trend": self._calculate_trend(recent_values)
            }
        
        return result
    
    async def get_all_metrics_summary(self, duration_seconds: int = 3600) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get metrics summary for all servers"""
        result = {}
        for server_id in self.metrics.keys():
            result[server_id] = await self.get_server_metrics(server_id, duration_seconds)
        return result
    
    async def check_metric_alerts(self, server_id: str) -> List[Dict[str, Any]]:
        """
        Check for metric-based alerts for a server
        
        Args:
            server_id: Server identifier
            
        Returns:
            List of alert conditions that are triggered
        """
        alerts = []
        
        if server_id not in self.metrics or server_id not in self.baselines:
            return alerts
        
        for metric_name, metric_series in self.metrics[server_id].items():
            if metric_name not in self.baselines[server_id]:
                continue
            
            baseline = self.baselines[server_id][metric_name]
            recent_values = metric_series.get_recent_values(300)  # Last 5 minutes
            
            if not recent_values:
                continue
            
            current_value = statistics.mean(recent_values)
            deviation = ((current_value - baseline.baseline_value) / baseline.baseline_value * 100) if baseline.baseline_value > 0 else 0
            
            # Check for threshold breaches
            if abs(deviation) >= baseline.threshold_critical:
                alerts.append({
                    "severity": "critical",
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "baseline_value": baseline.baseline_value,
                    "deviation_percent": round(deviation, 2),
                    "message": f"Metric {metric_name} is {abs(deviation):.1f}% from baseline (critical threshold: {baseline.threshold_critical}%)"
                })
            elif abs(deviation) >= baseline.threshold_warning:
                alerts.append({
                    "severity": "warning", 
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "baseline_value": baseline.baseline_value,
                    "deviation_percent": round(deviation, 2),
                    "message": f"Metric {metric_name} is {abs(deviation):.1f}% from baseline (warning threshold: {baseline.threshold_warning}%)"
                })
        
        return alerts
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        # Calculate slope using simple linear regression
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return "stable"
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend based on slope
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    async def get_collection_statistics(self) -> Dict[str, Any]:
        """Get metrics collection system statistics"""
        overhead_times = list(self.collection_stats["collection_duration_ms"])
        avg_overhead = sum(overhead_times) / len(overhead_times) if overhead_times else 0
        
        return {
            "is_collecting": self.is_collecting,
            "total_collections": self.collection_stats["total_collections"],
            "collection_errors": self.collection_stats["collection_errors"],
            "last_collection_time": self.collection_stats["last_collection_time"].isoformat() if self.collection_stats["last_collection_time"] else None,
            "average_collection_time_ms": round(avg_overhead, 2),
            "registered_servers": len(self.server_callbacks),
            "total_metrics": sum(len(server_metrics) for server_metrics in self.metrics.values()),
            "total_baselines": sum(len(server_baselines) for server_baselines in self.baselines.values()),
            "collection_interval_seconds": self.collection_interval,
            "retention_hours": self.retention_hours
        }