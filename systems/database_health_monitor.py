"""
Database Health Monitoring System for Issue #150
Provides continuous monitoring, alerting, and automated recovery
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from pathlib import Path

from systems.resilient_database_interface import ResilientDatabaseInterface
from knowledge.database.database_config import DatabaseConfig


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringState(Enum):
    """Monitoring system states."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthAlert:
    """Health alert data structure."""
    id: str
    timestamp: float
    severity: AlertSeverity
    category: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    resolution_message: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Configuration for health monitoring."""
    check_interval: float = 30.0  # seconds
    alert_cooldown: float = 300.0  # 5 minutes
    max_alert_history: int = 1000
    enable_auto_recovery: bool = True
    recovery_attempt_limit: int = 3
    recovery_cooldown: float = 600.0  # 10 minutes
    
    # Thresholds
    error_rate_warning: float = 0.1  # 10%
    error_rate_critical: float = 0.3  # 30%
    response_time_warning: float = 1.0  # 1 second
    response_time_critical: float = 5.0  # 5 seconds
    connection_pool_warning: float = 0.8  # 80% usage
    connection_pool_critical: float = 0.95  # 95% usage


class DatabaseHealthMonitor:
    """
    Comprehensive database health monitoring system.
    
    Features:
    - Continuous health monitoring with configurable intervals
    - Multi-level alerting system (INFO, WARNING, ERROR, CRITICAL)
    - Automated recovery attempts for common issues
    - Historical metrics tracking and trend analysis
    - Integration with external alerting systems
    - Performance baseline establishment and deviation detection
    """
    
    def __init__(self, 
                 database_interface: ResilientDatabaseInterface,
                 config: Optional[MonitoringConfig] = None,
                 alert_handlers: Optional[List[Callable]] = None):
        
        self.db_interface = database_interface
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.state = MonitoringState.STARTING
        self.monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._state_lock = threading.Lock()
        
        # Alert system
        self.alert_handlers = alert_handlers or []
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        self.alert_cooldowns: Dict[str, float] = {}
        self._alert_lock = threading.Lock()
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        self.metrics_lock = threading.Lock()
        
        # Recovery system
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_attempt: Dict[str, float] = {}
        self._recovery_lock = threading.Lock()
        
        # Performance baselines
        self.performance_baseline = {
            'response_time_p50': 0.0,
            'response_time_p95': 0.0,
            'error_rate': 0.0,
            'connection_usage': 0.0
        }
        
        self.logger.info("Database Health Monitor initialized")
    
    def start_monitoring(self):
        """Start the health monitoring system."""
        with self._state_lock:
            if self.state in [MonitoringState.RUNNING, MonitoringState.STARTING]:
                self.logger.warning("Monitoring already running or starting")
                return
            
            self.state = MonitoringState.STARTING
            self._shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="DatabaseHealthMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Database health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring system."""
        with self._state_lock:
            if self.state == MonitoringState.STOPPED:
                return
            
            self.state = MonitoringState.STOPPED
            self._shutdown_event.set()
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)
        
        self.logger.info("Database health monitoring stopped")
    
    def pause_monitoring(self):
        """Pause monitoring (can be resumed)."""
        with self._state_lock:
            if self.state == MonitoringState.RUNNING:
                self.state = MonitoringState.PAUSED
                self.logger.info("Database health monitoring paused")
    
    def resume_monitoring(self):
        """Resume monitoring after pause."""
        with self._state_lock:
            if self.state == MonitoringState.PAUSED:
                self.state = MonitoringState.RUNNING
                self.logger.info("Database health monitoring resumed")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        with self._state_lock:
            self.state = MonitoringState.RUNNING
        
        self.logger.info("Health monitoring loop started")
        
        # Establish baseline metrics
        self._establish_baseline()
        
        while not self._shutdown_event.is_set():
            try:
                with self._state_lock:
                    if self.state == MonitoringState.STOPPED:
                        break
                    elif self.state == MonitoringState.PAUSED:
                        time.sleep(1.0)
                        continue
                
                # Perform health check
                self._perform_health_check()
                
                # Sleep until next check
                self._shutdown_event.wait(timeout=self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                with self._state_lock:
                    self.state = MonitoringState.ERROR
                time.sleep(5.0)  # Short sleep on error
        
        self.logger.info("Health monitoring loop finished")
    
    def _establish_baseline(self):
        """Establish performance baseline metrics."""
        self.logger.info("Establishing performance baseline...")
        
        baseline_samples = []
        sample_count = 5
        
        for i in range(sample_count):
            try:
                health_status = self.db_interface.get_health_status()
                
                baseline_samples.append({
                    'response_time': health_status.get('avg_response_time', 0.0),
                    'error_rate': health_status.get('error_rate', 0.0),
                    'connection_usage': (
                        health_status.get('active_connections', 0) / 
                        max(1, health_status.get('operation_metrics', {}).get('total_operations', 1))
                    )
                })
                
                if i < sample_count - 1:
                    time.sleep(2.0)  # Wait between samples
                    
            except Exception as e:
                self.logger.warning(f"Baseline sample {i+1} failed: {e}")
        
        if baseline_samples:
            # Calculate baseline values
            response_times = [s['response_time'] for s in baseline_samples]
            error_rates = [s['error_rate'] for s in baseline_samples]
            connection_usages = [s['connection_usage'] for s in baseline_samples]
            
            self.performance_baseline = {
                'response_time_p50': sorted(response_times)[len(response_times)//2],
                'response_time_p95': sorted(response_times)[int(len(response_times)*0.95)],
                'error_rate': sum(error_rates) / len(error_rates),
                'connection_usage': sum(connection_usages) / len(connection_usages),
                'established_at': time.time()
            }
            
            self.logger.info(f"Performance baseline established: {self.performance_baseline}")
        else:
            self.logger.warning("Failed to establish performance baseline")
    
    def _perform_health_check(self):
        """Perform comprehensive health check and generate alerts."""
        try:
            # Get current health status
            health_status = self.db_interface.get_health_status()
            resilience_metrics = self.db_interface.resilience_manager.get_health_metrics()
            
            # Record metrics
            self._record_metrics(health_status, resilience_metrics)
            
            # Check various health aspects
            self._check_database_availability(health_status)
            self._check_error_rates(health_status, resilience_metrics)
            self._check_response_times(health_status)
            self._check_connection_pool(resilience_metrics)
            self._check_circuit_breaker(resilience_metrics)
            self._check_performance_trends(health_status)
            
            # Clean up resolved alerts
            self._cleanup_resolved_alerts()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._generate_alert(
                "health_check_failure",
                AlertSeverity.ERROR,
                "monitoring",
                f"Health check failed: {e}",
                {}
            )
    
    def _record_metrics(self, health_status: Dict[str, Any], resilience_metrics: Dict[str, Any]):
        """Record current metrics for trend analysis."""
        with self.metrics_lock:
            metrics_record = {
                'timestamp': time.time(),
                'health_state': health_status.get('overall_health', 'unknown'),
                'database_available': health_status.get('database_available', False),
                'error_rate': health_status.get('error_rate', 0.0),
                'avg_response_time': health_status.get('avg_response_time', 0.0),
                'active_connections': health_status.get('active_connections', 0),
                'circuit_breaker_open': health_status.get('circuit_breaker_open', False),
                'total_operations': health_status.get('operation_metrics', {}).get('total_operations', 0),
                'fallback_operations': health_status.get('operation_metrics', {}).get('fallback_operations', 0)
            }
            
            self.metrics_history.append(metrics_record)
            
            # Limit history size
            if len(self.metrics_history) > 1440:  # Keep ~24 hours at 1-minute intervals
                self.metrics_history.pop(0)
    
    def _check_database_availability(self, health_status: Dict[str, Any]):
        """Check database availability."""
        if not health_status.get('database_available', False):
            self._generate_alert(
                "database_unavailable",
                AlertSeverity.CRITICAL,
                "availability",
                "Database is not available",
                health_status
            )
            
            # Trigger recovery if enabled
            if self.config.enable_auto_recovery:
                self._attempt_recovery("database_unavailable", "force_recovery")
    
    def _check_error_rates(self, health_status: Dict[str, Any], resilience_metrics: Dict[str, Any]):
        """Check error rates and generate alerts."""
        error_rate = health_status.get('error_rate', 0.0)
        
        if error_rate >= self.config.error_rate_critical:
            self._generate_alert(
                "high_error_rate_critical",
                AlertSeverity.CRITICAL,
                "performance",
                f"Critical error rate: {error_rate:.1%}",
                {'error_rate': error_rate, 'threshold': self.config.error_rate_critical}
            )
        elif error_rate >= self.config.error_rate_warning:
            self._generate_alert(
                "high_error_rate_warning",
                AlertSeverity.WARNING,
                "performance",
                f"High error rate: {error_rate:.1%}",
                {'error_rate': error_rate, 'threshold': self.config.error_rate_warning}
            )
        else:
            # Resolve error rate alerts if they exist
            self._resolve_alert("high_error_rate_critical", f"Error rate normalized to {error_rate:.1%}")
            self._resolve_alert("high_error_rate_warning", f"Error rate normalized to {error_rate:.1%}")
    
    def _check_response_times(self, health_status: Dict[str, Any]):
        """Check response times and generate alerts."""
        response_time = health_status.get('avg_response_time', 0.0)
        
        if response_time >= self.config.response_time_critical:
            self._generate_alert(
                "slow_response_critical",
                AlertSeverity.CRITICAL,
                "performance",
                f"Critical response time: {response_time:.2f}s",
                {'response_time': response_time, 'threshold': self.config.response_time_critical}
            )
        elif response_time >= self.config.response_time_warning:
            self._generate_alert(
                "slow_response_warning",
                AlertSeverity.WARNING,
                "performance",
                f"Slow response time: {response_time:.2f}s",
                {'response_time': response_time, 'threshold': self.config.response_time_warning}
            )
        else:
            self._resolve_alert("slow_response_critical", f"Response time normalized to {response_time:.2f}s")
            self._resolve_alert("slow_response_warning", f"Response time normalized to {response_time:.2f}s")
    
    def _check_connection_pool(self, resilience_metrics: Dict[str, Any]):
        """Check connection pool usage."""
        pool_stats = resilience_metrics.get('pool_stats', {})
        active_connections = pool_stats.get('active_connections', 0)
        max_connections = pool_stats.get('max_connections', 1)
        
        usage_ratio = active_connections / max_connections
        
        if usage_ratio >= self.config.connection_pool_critical:
            self._generate_alert(
                "connection_pool_critical",
                AlertSeverity.CRITICAL,
                "resources",
                f"Critical connection pool usage: {usage_ratio:.1%}",
                {'usage_ratio': usage_ratio, 'active': active_connections, 'max': max_connections}
            )
        elif usage_ratio >= self.config.connection_pool_warning:
            self._generate_alert(
                "connection_pool_warning",
                AlertSeverity.WARNING,
                "resources",
                f"High connection pool usage: {usage_ratio:.1%}",
                {'usage_ratio': usage_ratio, 'active': active_connections, 'max': max_connections}
            )
        else:
            self._resolve_alert("connection_pool_critical", f"Connection pool usage normalized to {usage_ratio:.1%}")
            self._resolve_alert("connection_pool_warning", f"Connection pool usage normalized to {usage_ratio:.1%}")
    
    def _check_circuit_breaker(self, resilience_metrics: Dict[str, Any]):
        """Check circuit breaker status."""
        circuit_breaker = resilience_metrics.get('circuit_breaker', {})
        breaker_state = circuit_breaker.get('state', 'closed')
        
        if breaker_state == 'open':
            self._generate_alert(
                "circuit_breaker_open",
                AlertSeverity.CRITICAL,
                "availability",
                "Circuit breaker is open - service unavailable",
                circuit_breaker
            )
            
            if self.config.enable_auto_recovery:
                self._attempt_recovery("circuit_breaker_open", "reset_circuit_breaker")
                
        elif breaker_state == 'half_open':
            self._generate_alert(
                "circuit_breaker_half_open",
                AlertSeverity.WARNING,
                "availability",
                "Circuit breaker is half-open - testing recovery",
                circuit_breaker
            )
        else:
            self._resolve_alert("circuit_breaker_open", "Circuit breaker returned to closed state")
            self._resolve_alert("circuit_breaker_half_open", "Circuit breaker returned to closed state")
    
    def _check_performance_trends(self, health_status: Dict[str, Any]):
        """Check for performance degradation trends."""
        if not self.baseline_metrics or len(self.metrics_history) < 10:
            return
        
        # Get recent metrics (last 10 samples)
        recent_metrics = self.metrics_history[-10:]
        avg_recent_response_time = sum(m['avg_response_time'] for m in recent_metrics) / len(recent_metrics)
        
        baseline_response_time = self.performance_baseline.get('response_time_p50', 0.0)
        
        # Check if performance has degraded significantly
        if baseline_response_time > 0 and avg_recent_response_time > baseline_response_time * 2:
            self._generate_alert(
                "performance_degradation",
                AlertSeverity.WARNING,
                "performance",
                f"Performance degraded: {avg_recent_response_time:.2f}s vs baseline {baseline_response_time:.2f}s",
                {
                    'current_avg': avg_recent_response_time,
                    'baseline': baseline_response_time,
                    'degradation_factor': avg_recent_response_time / baseline_response_time
                }
            )
    
    def _generate_alert(self, alert_id: str, severity: AlertSeverity, 
                       category: str, message: str, metrics: Dict[str, Any]):
        """Generate a new alert or update existing one."""
        current_time = time.time()
        
        with self._alert_lock:
            # Check cooldown
            if alert_id in self.alert_cooldowns:
                if current_time - self.alert_cooldowns[alert_id] < self.config.alert_cooldown:
                    return  # Still in cooldown
            
            # Create or update alert
            if alert_id in self.active_alerts:
                # Update existing alert
                alert = self.active_alerts[alert_id]
                alert.timestamp = current_time
                alert.metrics = metrics
                alert.resolved = False
            else:
                # Create new alert
                alert = HealthAlert(
                    id=alert_id,
                    timestamp=current_time,
                    severity=severity,
                    category=category,
                    message=message,
                    metrics=metrics
                )
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Limit history size
                if len(self.alert_history) > self.config.max_alert_history:
                    self.alert_history.pop(0)
            
            # Update cooldown
            self.alert_cooldowns[alert_id] = current_time
        
        # Send alert to handlers
        self._send_alert_to_handlers(alert)
        
        self.logger.warning(f"Alert {severity.value.upper()}: {message}")
    
    def _resolve_alert(self, alert_id: str, resolution_message: str):
        """Resolve an active alert."""
        with self._alert_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                if not alert.resolved:
                    alert.resolved = True
                    alert.resolution_timestamp = time.time()
                    alert.resolution_message = resolution_message
                    
                    self.logger.info(f"Alert resolved: {alert_id} - {resolution_message}")
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        with self._alert_lock:
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if (alert.resolved and 
                    alert.resolution_timestamp and 
                    current_time - alert.resolution_timestamp > cleanup_age):
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
    
    def _send_alert_to_handlers(self, alert: HealthAlert):
        """Send alert to all registered handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def _attempt_recovery(self, issue_type: str, recovery_action: str):
        """Attempt automated recovery for an issue."""
        current_time = time.time()
        
        with self._recovery_lock:
            # Check attempt limits
            attempts = self.recovery_attempts.get(issue_type, 0)
            last_attempt = self.last_recovery_attempt.get(issue_type, 0)
            
            if attempts >= self.config.recovery_attempt_limit:
                return  # Exceeded attempt limit
            
            if current_time - last_attempt < self.config.recovery_cooldown:
                return  # Still in cooldown
            
            # Perform recovery action
            self.logger.info(f"Attempting recovery for {issue_type}: {recovery_action}")
            
            success = False
            try:
                if recovery_action == "force_recovery":
                    result = self.db_interface.force_recovery()
                    success = result.get('success', False)
                elif recovery_action == "reset_circuit_breaker":
                    self.db_interface.resilience_manager.force_circuit_breaker_reset()
                    success = True
                
                if success:
                    self.logger.info(f"Recovery successful for {issue_type}")
                    # Reset attempt counter on success
                    self.recovery_attempts[issue_type] = 0
                else:
                    self.logger.warning(f"Recovery failed for {issue_type}")
                    self.recovery_attempts[issue_type] = attempts + 1
                
            except Exception as e:
                self.logger.error(f"Recovery attempt failed for {issue_type}: {e}")
                self.recovery_attempts[issue_type] = attempts + 1
            
            self.last_recovery_attempt[issue_type] = current_time
    
    # Public API
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        with self._state_lock:
            state = self.state
        
        with self._alert_lock:
            active_alerts_count = len(self.active_alerts)
            critical_alerts = sum(1 for alert in self.active_alerts.values() 
                                if alert.severity == AlertSeverity.CRITICAL)
        
        return {
            'state': state.value,
            'active_alerts': active_alerts_count,
            'critical_alerts': critical_alerts,
            'metrics_history_size': len(self.metrics_history),
            'baseline_established': self.baseline_metrics is not None,
            'config': {
                'check_interval': self.config.check_interval,
                'auto_recovery_enabled': self.config.enable_auto_recovery,
                'alert_cooldown': self.config.alert_cooldown
            }
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        with self._alert_lock:
            return [
                {
                    'id': alert.id,
                    'timestamp': alert.timestamp,
                    'severity': alert.severity.value,
                    'category': alert.category,
                    'message': alert.message,
                    'metrics': alert.metrics,
                    'resolved': alert.resolved,
                    'resolution_timestamp': alert.resolution_timestamp,
                    'resolution_message': alert.resolution_message
                }
                for alert in self.active_alerts.values()
            ]
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.metrics_lock:
            return [
                metrics for metrics in self.metrics_history 
                if metrics['timestamp'] >= cutoff_time
            ]
    
    def add_alert_handler(self, handler: Callable[[HealthAlert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def save_state(self, filepath: str):
        """Save monitoring state to file."""
        state_data = {
            'timestamp': time.time(),
            'performance_baseline': self.performance_baseline,
            'metrics_history': self.metrics_history[-100:],  # Last 100 samples
            'alert_history': [
                {
                    'id': alert.id,
                    'timestamp': alert.timestamp,
                    'severity': alert.severity.value,
                    'category': alert.category,
                    'message': alert.message,
                    'resolved': alert.resolved
                }
                for alert in self.alert_history[-50:]  # Last 50 alerts
            ]
        }
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            self.logger.info(f"Monitoring state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save monitoring state: {e}")


# Default alert handlers
def console_alert_handler(alert: HealthAlert):
    """Simple console alert handler."""
    timestamp = datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {alert.severity.value.upper()}: {alert.message}")


def file_alert_handler(log_file: str):
    """Create a file-based alert handler."""
    def handler(alert: HealthAlert):
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {alert.severity.value.upper()}: {alert.message}\n"
        
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to write alert to {log_file}: {e}")
    
    return handler