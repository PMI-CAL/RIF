#!/usr/bin/env python3
"""
Context Performance Monitor - Real-time Metrics and Optimization
Issue #123: DPIBS Development Phase 1

Implements comprehensive performance monitoring with:
- Real-time metrics collection and analysis
- Automated optimization triggers
- Performance alerting and degradation detection
- Sub-200ms performance target enforcement
- Intelligent threshold adjustment

Based on RIF-Architect specifications for Context Intelligence Platform.
"""

import asyncio
import json
import time
import threading
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import sqlite3
import os
import math

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"
    RESOURCE_UTILIZATION = "resource_utilization"
    CONTEXT_RELEVANCE = "context_relevance"

class OptimizationAction(Enum):
    """Automated optimization actions"""
    INCREASE_CACHE_TTL = "increase_cache_ttl"
    DECREASE_CACHE_TTL = "decrease_cache_ttl"
    ADJUST_QUEUE_SIZE = "adjust_queue_size"
    SCALE_SERVICE_INSTANCES = "scale_service_instances"
    TRIGGER_CACHE_WARMUP = "trigger_cache_warmup"
    ADJUST_RELEVANCE_THRESHOLD = "adjust_relevance_threshold"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    OPTIMIZE_QUERY_BATCH_SIZE = "optimize_query_batch_size"

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    measurement_window_seconds: int
    min_sample_size: int

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    metric_type: MetricType
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    service_component: str
    suggested_actions: List[OptimizationAction]

@dataclass
class MetricSample:
    """Individual metric sample"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    tags: Dict[str, str]
    service_component: str

@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_type: MetricType
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    predicted_value: float
    confidence: float

class MetricCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_samples: int = 10000, aggregation_window_seconds: int = 60):
        self.max_samples = max_samples
        self.aggregation_window_seconds = aggregation_window_seconds
        self.samples: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.lock = threading.RLock()
        self.aggregated_metrics: Dict[str, Dict] = {}
        
        # Start background aggregation
        self.aggregator_running = True
        self.aggregator_thread = threading.Thread(target=self._aggregation_worker, daemon=True)
        self.aggregator_thread.start()
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     tags: Optional[Dict[str, str]] = None, 
                     service_component: str = "unknown"):
        """Record a performance metric sample"""
        sample = MetricSample(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            tags=tags or {},
            service_component=service_component
        )
        
        with self.lock:
            self.samples[metric_type].append(sample)
    
    def get_recent_samples(self, metric_type: MetricType, 
                          window_seconds: int = 300) -> List[MetricSample]:
        """Get recent samples within time window"""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        with self.lock:
            return [
                sample for sample in self.samples[metric_type]
                if sample.timestamp > cutoff_time
            ]
    
    def get_aggregated_metrics(self, metric_type: MetricType) -> Optional[Dict[str, Any]]:
        """Get aggregated metrics for a metric type"""
        return self.aggregated_metrics.get(metric_type.value)
    
    def _aggregation_worker(self):
        """Background worker for metric aggregation"""
        while self.aggregator_running:
            try:
                self._aggregate_metrics()
                time.sleep(self.aggregation_window_seconds)
            except Exception as e:
                logger.error(f"Metric aggregation error: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate metrics over time windows"""
        current_time = datetime.now()
        
        for metric_type in MetricType:
            recent_samples = self.get_recent_samples(metric_type, self.aggregation_window_seconds)
            
            if not recent_samples:
                continue
            
            values = [sample.value for sample in recent_samples]
            
            aggregated = {
                "timestamp": current_time.isoformat(),
                "sample_count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            }
            
            # Service component breakdown
            component_breakdown = defaultdict(list)
            for sample in recent_samples:
                component_breakdown[sample.service_component].append(sample.value)
            
            aggregated["component_breakdown"] = {
                component: {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "p95": self._percentile(values, 95)
                }
                for component, values in component_breakdown.items()
            }
            
            self.aggregated_metrics[metric_type.value] = aggregated
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

class TrendAnalyzer:
    """Analyzes performance trends and predicts future values"""
    
    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
    
    def analyze_trend(self, samples: List[MetricSample]) -> PerformanceTrend:
        """Analyze trend in metric samples"""
        if len(samples) < self.min_samples:
            return PerformanceTrend(
                metric_type=samples[0].metric_type if samples else MetricType.RESPONSE_TIME,
                trend_direction="unknown",
                trend_strength=0.0,
                predicted_value=0.0,
                confidence=0.0
            )
        
        # Sort by timestamp
        sorted_samples = sorted(samples, key=lambda x: x.timestamp)
        values = [sample.value for sample in sorted_samples]
        
        # Linear regression for trend analysis
        n = len(values)
        x = list(range(n))
        
        # Calculate slope (trend direction and strength)
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Minimal change threshold
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "increasing"
            trend_strength = min(1.0, abs(slope) / (y_mean + 1e-6))
        else:
            trend_direction = "decreasing"
            trend_strength = min(1.0, abs(slope) / (y_mean + 1e-6))
        
        # Predict next value
        predicted_value = values[-1] + slope
        
        # Calculate confidence based on R-squared
        y_pred = [y_mean + slope * (xi - x_mean) for xi in x]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0.0, r_squared)
        
        return PerformanceTrend(
            metric_type=sorted_samples[0].metric_type,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            predicted_value=max(0, predicted_value),
            confidence=confidence
        )

class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, alert_storage_path: str = "/Users/cal/DEV/RIF/systems/context"):
        self.alert_storage_path = alert_storage_path
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.alert_history: deque = deque(maxlen=1000)
        
        # Initialize storage
        os.makedirs(alert_storage_path, exist_ok=True)
        self.db_path = os.path.join(alert_storage_path, "performance_alerts.db")
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize alert storage database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    metric_type TEXT,
                    severity TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    message TEXT,
                    timestamp TEXT,
                    service_component TEXT,
                    suggested_actions TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                )
            """)
    
    def check_thresholds(self, metric_type: MetricType, current_value: float,
                        thresholds: PerformanceThreshold, 
                        service_component: str = "unknown") -> Optional[PerformanceAlert]:
        """Check if metric value exceeds thresholds"""
        alert_id = f"{metric_type.value}_{service_component}_{int(time.time())}"
        
        # Determine severity and threshold exceeded
        if current_value >= thresholds.emergency_threshold:
            severity = AlertSeverity.EMERGENCY
            threshold_value = thresholds.emergency_threshold
        elif current_value >= thresholds.critical_threshold:
            severity = AlertSeverity.CRITICAL  
            threshold_value = thresholds.critical_threshold
        elif current_value >= thresholds.warning_threshold:
            severity = AlertSeverity.WARNING
            threshold_value = thresholds.warning_threshold
        else:
            return None  # No threshold exceeded
        
        # Create alert
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_type=metric_type,
            severity=severity,
            current_value=current_value,
            threshold_value=threshold_value,
            message=f"{metric_type.value} exceeded {severity.value} threshold: "
                   f"{current_value:.2f} >= {threshold_value:.2f}",
            timestamp=datetime.now(),
            service_component=service_component,
            suggested_actions=self._suggest_optimization_actions(metric_type, severity)
        )
        
        # Store and track alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self._store_alert(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks[severity]:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        return alert
    
    def _suggest_optimization_actions(self, metric_type: MetricType, 
                                    severity: AlertSeverity) -> List[OptimizationAction]:
        """Suggest optimization actions based on metric type and severity"""
        actions = []
        
        if metric_type == MetricType.RESPONSE_TIME:
            actions.extend([
                OptimizationAction.TRIGGER_CACHE_WARMUP,
                OptimizationAction.INCREASE_CACHE_TTL,
                OptimizationAction.OPTIMIZE_QUERY_BATCH_SIZE
            ])
            if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                actions.append(OptimizationAction.SCALE_SERVICE_INSTANCES)
        
        elif metric_type == MetricType.CACHE_HIT_RATE:
            actions.extend([
                OptimizationAction.INCREASE_CACHE_TTL,
                OptimizationAction.TRIGGER_CACHE_WARMUP
            ])
        
        elif metric_type == MetricType.QUEUE_DEPTH:
            actions.extend([
                OptimizationAction.ADJUST_QUEUE_SIZE,
                OptimizationAction.SCALE_SERVICE_INSTANCES
            ])
        
        elif metric_type == MetricType.ERROR_RATE:
            actions.extend([
                OptimizationAction.ENABLE_CIRCUIT_BREAKER,
                OptimizationAction.SCALE_SERVICE_INSTANCES
            ])
        
        elif metric_type == MetricType.CONTEXT_RELEVANCE:
            actions.append(OptimizationAction.ADJUST_RELEVANCE_THRESHOLD)
        
        return actions
    
    def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, metric_type, severity, current_value, threshold_value,
                     message, timestamp, service_component, suggested_actions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.metric_type.value,
                    alert.severity.value,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.service_component,
                    json.dumps([action.value for action in alert.suggested_actions])
                ))
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def register_alert_callback(self, severity: AlertSeverity, callback: Callable):
        """Register callback for alert notifications"""
        self.alert_callbacks[severity].append(callback)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            
            # Update database
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE alerts 
                        SET resolved = TRUE, resolved_at = ?
                        WHERE alert_id = ?
                    """, (datetime.now().isoformat(), alert_id))
            except Exception as e:
                logger.error(f"Failed to resolve alert: {e}")

class OptimizationEngine:
    """Automated optimization engine"""
    
    def __init__(self):
        self.optimization_history: List[Dict] = []
        self.optimization_callbacks: Dict[OptimizationAction, List[Callable]] = defaultdict(list)
        self.cooldown_periods: Dict[OptimizationAction, datetime] = {}
        self.default_cooldown_seconds = 300  # 5 minutes
    
    def execute_optimization(self, action: OptimizationAction, 
                           alert: PerformanceAlert, 
                           parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Execute optimization action"""
        # Check cooldown
        if self._is_in_cooldown(action):
            logger.info(f"Optimization {action.value} in cooldown, skipping")
            return False
        
        # Record optimization attempt
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action.value,
            "trigger_alert": alert.alert_id,
            "parameters": parameters or {},
            "success": False
        }
        
        try:
            # Execute optimization callbacks
            success = False
            for callback in self.optimization_callbacks[action]:
                try:
                    result = callback(alert, parameters)
                    if result:
                        success = True
                except Exception as e:
                    logger.error(f"Optimization callback failed: {e}")
            
            optimization_record["success"] = success
            
            # Set cooldown if successful
            if success:
                self.cooldown_periods[action] = datetime.now()
                logger.info(f"Executed optimization: {action.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Optimization execution failed: {e}")
            return False
        
        finally:
            self.optimization_history.append(optimization_record)
    
    def _is_in_cooldown(self, action: OptimizationAction) -> bool:
        """Check if optimization action is in cooldown period"""
        if action not in self.cooldown_periods:
            return False
        
        cooldown_until = self.cooldown_periods[action] + timedelta(seconds=self.default_cooldown_seconds)
        return datetime.now() < cooldown_until
    
    def register_optimization_callback(self, action: OptimizationAction, callback: Callable):
        """Register callback for optimization actions"""
        self.optimization_callbacks[action].append(callback)

class ContextPerformanceMonitor:
    """Main performance monitor for Context Intelligence Platform"""
    
    def __init__(self, storage_path: str = "/Users/cal/DEV/RIF/systems/context"):
        self.storage_path = storage_path
        self.metric_collector = MetricCollector()
        self.trend_analyzer = TrendAnalyzer()
        self.alert_manager = AlertManager(storage_path)
        self.optimization_engine = OptimizationEngine()
        
        # Performance thresholds based on requirements
        self.thresholds = {
            MetricType.RESPONSE_TIME: PerformanceThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=150.0,    # 150ms
                critical_threshold=200.0,   # 200ms target
                emergency_threshold=500.0,  # P99 target
                measurement_window_seconds=60,
                min_sample_size=10
            ),
            MetricType.CACHE_HIT_RATE: PerformanceThreshold(
                metric_type=MetricType.CACHE_HIT_RATE,
                warning_threshold=0.70,     # 70%
                critical_threshold=0.60,    # 60%
                emergency_threshold=0.50,   # 50%
                measurement_window_seconds=300,
                min_sample_size=20
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=0.01,     # 1%
                critical_threshold=0.05,    # 5%
                emergency_threshold=0.10,   # 10%
                measurement_window_seconds=60,
                min_sample_size=10
            ),
            MetricType.CONTEXT_RELEVANCE: PerformanceThreshold(
                metric_type=MetricType.CONTEXT_RELEVANCE,
                warning_threshold=0.85,     # 85% (inverted - lower is worse)
                critical_threshold=0.80,    # 80%
                emergency_threshold=0.70,   # 70%
                measurement_window_seconds=300,
                min_sample_size=15
            )
        }
        
        # Start monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Register alert callbacks
        self.alert_manager.register_alert_callback(
            AlertSeverity.WARNING, 
            self._handle_warning_alert
        )
        self.alert_manager.register_alert_callback(
            AlertSeverity.CRITICAL, 
            self._handle_critical_alert
        )
        self.alert_manager.register_alert_callback(
            AlertSeverity.EMERGENCY, 
            self._handle_emergency_alert
        )
    
    def record_context_request(self, response_time_ms: float, cache_hit: bool, 
                             context_relevance: float, service_component: str,
                             success: bool = True):
        """Record context request metrics"""
        # Response time
        self.metric_collector.record_metric(
            MetricType.RESPONSE_TIME, 
            response_time_ms,
            {"cache_hit": str(cache_hit)},
            service_component
        )
        
        # Cache hit rate (as success/failure)
        self.metric_collector.record_metric(
            MetricType.CACHE_HIT_RATE,
            1.0 if cache_hit else 0.0,
            {},
            service_component
        )
        
        # Context relevance
        self.metric_collector.record_metric(
            MetricType.CONTEXT_RELEVANCE,
            context_relevance,
            {},
            service_component
        )
        
        # Error rate (as success/failure)
        self.metric_collector.record_metric(
            MetricType.ERROR_RATE,
            0.0 if success else 1.0,
            {},
            service_component
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_all_thresholds()
                self._analyze_trends()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Back off on error
    
    def _check_all_thresholds(self):
        """Check all performance thresholds"""
        for metric_type, threshold in self.thresholds.items():
            aggregated = self.metric_collector.get_aggregated_metrics(metric_type)
            
            if not aggregated or aggregated["sample_count"] < threshold.min_sample_size:
                continue
            
            # Use different aggregation based on metric type
            if metric_type == MetricType.RESPONSE_TIME:
                current_value = aggregated["p95"]  # P95 response time
            elif metric_type in [MetricType.CACHE_HIT_RATE, MetricType.CONTEXT_RELEVANCE]:
                current_value = aggregated["mean"]  # Average rate
            elif metric_type == MetricType.ERROR_RATE:
                current_value = aggregated["mean"]  # Average error rate
            else:
                current_value = aggregated["mean"]
            
            # For context relevance, invert threshold logic (lower values are worse)
            if metric_type == MetricType.CONTEXT_RELEVANCE:
                if current_value <= threshold.emergency_threshold:
                    severity = AlertSeverity.EMERGENCY
                elif current_value <= threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                elif current_value <= threshold.warning_threshold:
                    severity = AlertSeverity.WARNING
                else:
                    continue  # No alert needed
            else:
                # Normal threshold logic (higher values are worse)
                alert = self.alert_manager.check_thresholds(
                    metric_type, current_value, threshold, "context-platform"
                )
                if alert:
                    logger.warning(f"Performance alert: {alert.message}")
    
    def _analyze_trends(self):
        """Analyze performance trends"""
        for metric_type in MetricType:
            samples = self.metric_collector.get_recent_samples(metric_type, 3600)  # Last hour
            
            if len(samples) >= 20:  # Minimum samples for trend analysis
                trend = self.trend_analyzer.analyze_trend(samples)
                
                # Take action on concerning trends
                if (trend.confidence > 0.7 and 
                    trend.trend_direction == "increasing" and 
                    trend.trend_strength > 0.3):
                    
                    if metric_type == MetricType.RESPONSE_TIME:
                        logger.info(f"Detected increasing response time trend, "
                                  f"predicted value: {trend.predicted_value:.2f}ms")
                        # Could trigger preemptive optimizations here
    
    def _handle_warning_alert(self, alert: PerformanceAlert):
        """Handle warning-level alerts"""
        logger.warning(f"Performance warning: {alert.message}")
        
        # Execute light optimizations
        for action in alert.suggested_actions[:2]:  # Limit to first 2 actions
            self.optimization_engine.execute_optimization(action, alert)
    
    def _handle_critical_alert(self, alert: PerformanceAlert):
        """Handle critical-level alerts"""
        logger.error(f"Performance critical: {alert.message}")
        
        # Execute more aggressive optimizations
        for action in alert.suggested_actions:
            self.optimization_engine.execute_optimization(action, alert)
    
    def _handle_emergency_alert(self, alert: PerformanceAlert):
        """Handle emergency-level alerts"""
        logger.critical(f"Performance emergency: {alert.message}")
        
        # Execute all suggested optimizations immediately
        for action in alert.suggested_actions:
            self.optimization_engine.execute_optimization(action, alert, {"urgent": True})
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "metrics": {},
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "trends": {},
            "optimizations": len(self.optimization_engine.optimization_history)
        }
        
        # Collect current metrics
        for metric_type in MetricType:
            aggregated = self.metric_collector.get_aggregated_metrics(metric_type)
            if aggregated:
                dashboard["metrics"][metric_type.value] = aggregated
        
        # Analyze trends
        for metric_type in MetricType:
            samples = self.metric_collector.get_recent_samples(metric_type, 3600)
            if len(samples) >= 10:
                trend = self.trend_analyzer.analyze_trend(samples)
                dashboard["trends"][metric_type.value] = asdict(trend)
        
        # Overall system health
        alerts = self.alert_manager.get_active_alerts()
        if any(alert.severity == AlertSeverity.EMERGENCY for alert in alerts):
            dashboard["status"] = "emergency"
        elif any(alert.severity == AlertSeverity.CRITICAL for alert in alerts):
            dashboard["status"] = "critical"
        elif any(alert.severity == AlertSeverity.WARNING for alert in alerts):
            dashboard["status"] = "warning"
        
        return dashboard
    
    def shutdown(self):
        """Shutdown monitoring gracefully"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# CLI and Testing Interface
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Context Performance Monitor")
    parser.add_argument("--test", action="store_true", help="Run performance monitoring test")
    parser.add_argument("--simulate", action="store_true", help="Simulate performance metrics")
    parser.add_argument("--dashboard", action="store_true", help="Show performance dashboard")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ContextPerformanceMonitor()
    
    if args.test or args.simulate:
        print("=== Context Performance Monitor Test ===\n")
        
        # Simulate some performance data
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            # Simulate context request metrics
            response_time = random.gauss(120, 50)  # Mean 120ms, std 50ms
            response_time = max(10, response_time)  # Minimum 10ms
            
            cache_hit = random.random() < 0.8  # 80% cache hit rate
            context_relevance = random.gauss(0.9, 0.1)  # 90% relevance with variation
            context_relevance = max(0.0, min(1.0, context_relevance))
            
            success = random.random() < 0.98  # 98% success rate
            
            service_component = random.choice([
                "context-optimization", 
                "knowledge-integration", 
                "agent-context-delivery"
            ])
            
            monitor.record_context_request(
                response_time_ms=response_time,
                cache_hit=cache_hit,
                context_relevance=context_relevance,
                service_component=service_component,
                success=success
            )
            
            time.sleep(0.5)  # 2 requests per second
            
            if args.simulate:
                # Occasionally introduce performance issues
                if random.random() < 0.05:  # 5% chance
                    monitor.record_context_request(
                        response_time_ms=random.gauss(300, 100),  # Slow response
                        cache_hit=False,
                        context_relevance=random.gauss(0.6, 0.2),  # Poor relevance
                        service_component=service_component,
                        success=random.random() < 0.8  # More failures
                    )
        
        print(f"Simulated {args.duration} seconds of performance data")
    
    if args.dashboard or args.test:
        print("\n=== Performance Dashboard ===")
        dashboard = monitor.get_performance_dashboard()
        print(json.dumps(dashboard, indent=2, default=str))
        
        # Show active alerts
        alerts = monitor.alert_manager.get_active_alerts()
        if alerts:
            print(f"\n=== Active Alerts ({len(alerts)}) ===")
            for alert in alerts:
                print(f"- {alert.severity.value.upper()}: {alert.message}")
        else:
            print("\n=== No Active Alerts ===")
    
    if not any([args.test, args.simulate, args.dashboard]):
        print("Context Performance Monitor initialized")
        print("Use --test, --simulate, or --dashboard to interact with the monitor")
        print("Monitor is running in background...")
        
        try:
            while True:
                time.sleep(60)
                dashboard = monitor.get_performance_dashboard()
                print(f"Status: {dashboard['status']} | "
                      f"Active alerts: {dashboard['active_alerts']} | "
                      f"Metrics collected: {len(dashboard['metrics'])}")
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    # Cleanup
    monitor.shutdown()