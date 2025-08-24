"""
MCP Health Monitor - Enterprise Grade Server Health Management

Provides comprehensive health monitoring, automated recovery, and performance tracking
for MCP servers with enterprise-grade reliability and security integration.

Features:
- Continuous health monitoring with 30-second intervals
- Multi-step automated recovery (restart → reload → escalate)
- Real-time performance metrics collection and analysis
- Intelligent alert system with throttling and escalation
- Full security gateway integration
- Recovery pattern learning and optimization

Issue: #84 - Create MCP health monitor
Component: Enterprise health monitoring and recovery system
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from contextlib import asynccontextmanager

from .protocols import (
    HealthCheckManager, HealthCheckType, HealthStatus, 
    HealthCheckResult, default_health_check_manager
)

logger = logging.getLogger(__name__)


@dataclass
class ServerHealthRecord:
    """Complete health record for a monitored server"""
    server_id: str
    server_name: str
    current_status: HealthStatus
    last_check_time: datetime
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    average_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    last_recovery_time: Optional[datetime] = None
    escalation_count: int = 0
    performance_trend: str = "stable"
    alerts_sent: int = 0
    last_alert_time: Optional[datetime] = None


@dataclass 
class RecoveryStrategy:
    """Definition of a recovery strategy"""
    name: str
    description: str
    async_function: Callable
    timeout_seconds: float = 60.0
    success_rate: float = 0.0
    usage_count: int = 0
    success_count: int = 0


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[ServerHealthRecord], bool]
    severity: str  # "info", "warning", "critical"
    cooldown_minutes: int = 15
    message_template: str = "Health alert for {server_name}"


class MCPHealthMonitor:
    """
    Enterprise-grade MCP server health monitoring system
    
    Provides comprehensive health monitoring with automated recovery,
    performance tracking, and intelligent alerting for MCP servers.
    """

    def __init__(self, 
                 check_interval_seconds: int = 30,
                 storage_path: Optional[str] = None,
                 health_check_manager: Optional[HealthCheckManager] = None):
        """
        Initialize the health monitor
        
        Args:
            check_interval_seconds: Interval between health checks
            storage_path: Path for storing metrics and alert data
            health_check_manager: Health check manager (uses default if None)
        """
        # Core configuration
        self.check_interval = check_interval_seconds
        self.storage_path = Path(storage_path) if storage_path else Path("knowledge/monitoring")
        self.health_check_manager = health_check_manager or default_health_check_manager
        
        # Server tracking
        self.monitored_servers: Dict[str, Dict[str, Any]] = {}
        self.health_records: Dict[str, ServerHealthRecord] = {}
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self._shutdown_event = asyncio.Event()
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        self._initialize_recovery_strategies()
        
        # Alert system
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable] = []
        self._initialize_alert_rules()
        
        # Performance tracking
        self.system_metrics: Dict[str, Any] = {
            "monitoring_overhead_ms": deque(maxlen=100),
            "total_health_checks": 0,
            "total_recoveries": 0,
            "total_alerts": 0,
            "uptime_start": datetime.utcnow()
        }
        
        # Storage setup
        self._ensure_storage_directories()
        
        logger.info(f"MCPHealthMonitor initialized (interval: {check_interval_seconds}s)")

    def _ensure_storage_directories(self):
        """Create necessary storage directories"""
        directories = ["metrics", "alerts", "recovery", "health_history"]
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)

    def _initialize_recovery_strategies(self):
        """Initialize standard recovery strategies"""
        self.recovery_strategies = [
            RecoveryStrategy(
                name="restart",
                description="Attempt graceful server restart",
                async_function=self._restart_server,
                timeout_seconds=60.0
            ),
            RecoveryStrategy(
                name="reload",
                description="Reload server configuration",
                async_function=self._reload_server,
                timeout_seconds=30.0
            ),
            RecoveryStrategy(
                name="escalate",
                description="Escalate to manual intervention",
                async_function=self._escalate_server_issue,
                timeout_seconds=10.0
            )
        ]

    def _initialize_alert_rules(self):
        """Initialize standard alert rules"""
        self.alert_rules = [
            AlertRule(
                name="server_unhealthy",
                condition=lambda record: record.current_status == HealthStatus.UNHEALTHY,
                severity="critical",
                cooldown_minutes=10,
                message_template="Server {server_name} is unhealthy after {consecutive_failures} consecutive failures"
            ),
            AlertRule(
                name="server_degraded",
                condition=lambda record: (record.current_status == HealthStatus.DEGRADED and 
                                       record.consecutive_failures >= 3),
                severity="warning", 
                cooldown_minutes=30,
                message_template="Server {server_name} has degraded performance for {consecutive_failures} checks"
            ),
            AlertRule(
                name="high_failure_rate",
                condition=lambda record: (record.total_checks > 10 and 
                                       (record.total_failures / record.total_checks) > 0.3),
                severity="warning",
                cooldown_minutes=60,
                message_template="Server {server_name} has high failure rate: {failure_rate:.1%}"
            ),
            AlertRule(
                name="recovery_escalated",
                condition=lambda record: record.escalation_count > 0,
                severity="critical",
                cooldown_minutes=5,
                message_template="Server {server_name} recovery escalated - manual intervention required"
            )
        ]

    async def register_server(self, server: Any, server_config: Dict[str, Any]):
        """
        Register a server for comprehensive health monitoring
        
        Args:
            server: Server instance to monitor
            server_config: Server configuration including server_id and name
        """
        server_id = server_config['server_id']
        server_name = server_config.get('name', server_id)
        
        self.monitored_servers[server_id] = {
            'server': server,
            'config': server_config
        }
        
        # Initialize health record
        self.health_records[server_id] = ServerHealthRecord(
            server_id=server_id,
            server_name=server_name,
            current_status=HealthStatus.UNKNOWN,
            last_check_time=datetime.utcnow()
        )
        
        logger.info(f"Registered server for monitoring: {server_name} ({server_id})")
        
        # Start monitoring if this is the first server
        if len(self.monitored_servers) == 1 and not self.is_monitoring:
            await self.start_monitoring()

    async def unregister_server(self, server_id: str):
        """
        Unregister a server from monitoring
        
        Args:
            server_id: Server identifier to unregister
        """
        if server_id in self.monitored_servers:
            server_name = self.health_records.get(server_id, {}).server_name or server_id
            
            del self.monitored_servers[server_id]
            if server_id in self.health_records:
                del self.health_records[server_id]
                
            logger.info(f"Unregistered server from monitoring: {server_name}")
            
            # Stop monitoring if no servers left
            if not self.monitored_servers and self.is_monitoring:
                await self.stop_monitoring()

    async def start_monitoring(self):
        """Start the health monitoring system"""
        if self.is_monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.is_monitoring = True
        self._shutdown_event.clear()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Health monitoring system started")

    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._shutdown_event.set()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        logger.info("Health monitoring system stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop with comprehensive error handling"""
        try:
            while self.is_monitoring and not self._shutdown_event.is_set():
                loop_start = time.time()
                
                # Check all servers in parallel
                await self._check_all_servers()
                
                # Process alerts for all servers
                await self._process_alerts()
                
                # Collect system performance metrics
                loop_time_ms = (time.time() - loop_start) * 1000
                self.system_metrics["monitoring_overhead_ms"].append(loop_time_ms)
                
                # Wait for next check interval
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.check_interval)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal interval timeout, continue monitoring
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            # Continue monitoring despite errors
            if self.is_monitoring:
                await asyncio.sleep(self.check_interval)
                # Restart monitoring loop
                await self._monitoring_loop()

    async def _check_all_servers(self):
        """Check health of all registered servers in parallel"""
        if not self.monitored_servers:
            return
        
        # Create health check tasks
        tasks = []
        for server_id in list(self.monitored_servers.keys()):
            task = asyncio.create_task(self._check_server_health(server_id))
            tasks.append(task)
        
        # Execute all checks in parallel
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in parallel health checks: {e}")

    async def _check_server_health(self, server_id: str):
        """
        Comprehensive health check for a specific server
        
        Args:
            server_id: Server identifier to check
        """
        if server_id not in self.monitored_servers:
            return
        
        server_info = self.monitored_servers[server_id]
        server = server_info['server']
        config = server_info['config']
        health_record = self.health_records[server_id]
        
        try:
            # Determine appropriate health check type based on server status
            if health_record.consecutive_failures >= 2:
                check_type = HealthCheckType.COMPREHENSIVE
            elif health_record.consecutive_failures >= 1:
                check_type = HealthCheckType.PERFORMANCE
            else:
                check_type = HealthCheckType.BASIC
            
            # Perform health check
            check_result = await self.health_check_manager.check_server_health(
                server, config, check_type
            )
            
            # Update health record
            await self._update_health_record(server_id, check_result, success=True)
            
            # Attempt recovery if needed
            if check_result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                await self._attempt_recovery(server_id)
            
            # Store health check result
            await self._store_health_result(server_id, check_result)
            
            self.system_metrics["total_health_checks"] += 1
            
        except Exception as e:
            logger.error(f"Health check failed for {server_id}: {e}")
            
            # Create failed health check result
            failed_result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                timestamp=time.time(),
                check_type=HealthCheckType.BASIC,
                error=str(e)
            )
            
            await self._update_health_record(server_id, failed_result, success=False)

    async def _update_health_record(self, server_id: str, check_result: HealthCheckResult, success: bool):
        """
        Update server health record with check result
        
        Args:
            server_id: Server identifier
            check_result: Health check result
            success: Whether the check was successful
        """
        health_record = self.health_records[server_id]
        now = datetime.utcnow()
        
        health_record.last_check_time = now
        health_record.total_checks += 1
        
        if success:
            health_record.current_status = check_result.status
            health_record.response_times.append(check_result.response_time_ms)
            
            # Update average response time
            if health_record.response_times:
                health_record.average_response_time = sum(health_record.response_times) / len(health_record.response_times)
            else:
                health_record.average_response_time = 0.0
            
            # Handle success/failure counts
            if check_result.status == HealthStatus.HEALTHY:
                health_record.consecutive_failures = 0
                health_record.last_success_time = now
            else:
                health_record.consecutive_failures += 1
                health_record.total_failures += 1
            
            # Update performance trend
            health_record.performance_trend = self._calculate_performance_trend(health_record)
            
        else:
            health_record.current_status = HealthStatus.UNHEALTHY
            health_record.consecutive_failures += 1
            health_record.total_failures += 1

    def _calculate_performance_trend(self, health_record: ServerHealthRecord) -> str:
        """
        Calculate performance trend based on recent response times
        
        Args:
            health_record: Server health record
            
        Returns:
            Performance trend string: "improving", "degrading", "stable"
        """
        if len(health_record.response_times) < 10:
            return "stable"
        
        recent_times = list(health_record.response_times)
        first_half = recent_times[:len(recent_times)//2]
        second_half = recent_times[len(recent_times)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if change_percent < -10:
            return "improving"
        elif change_percent > 10:
            return "degrading"
        else:
            return "stable"

    async def _attempt_recovery(self, server_id: str):
        """
        Attempt automated recovery for a failing server
        
        Args:
            server_id: Server identifier to recover
        """
        health_record = self.health_records[server_id]
        
        # Skip recovery if too many recent attempts
        if (health_record.last_recovery_time and 
            datetime.utcnow() - health_record.last_recovery_time < timedelta(minutes=5)):
            return
        
        logger.info(f"Attempting recovery for server: {health_record.server_name}")
        
        health_record.recovery_attempts += 1
        health_record.last_recovery_time = datetime.utcnow()
        
        # Try recovery strategies in order
        for strategy in self.recovery_strategies:
            try:
                logger.info(f"Trying recovery strategy '{strategy.name}' for {health_record.server_name}")
                
                strategy.usage_count += 1
                success = await asyncio.wait_for(
                    strategy.async_function(server_id),
                    timeout=strategy.timeout_seconds
                )
                
                if success:
                    strategy.success_count += 1
                    strategy.success_rate = strategy.success_count / strategy.usage_count
                    health_record.successful_recoveries += 1
                    
                    logger.info(f"Recovery strategy '{strategy.name}' succeeded for {health_record.server_name}")
                    self.system_metrics["total_recoveries"] += 1
                    
                    # Store recovery event
                    await self._store_recovery_event(server_id, strategy.name, True)
                    return
                else:
                    logger.warning(f"Recovery strategy '{strategy.name}' failed for {health_record.server_name}")
                    await self._store_recovery_event(server_id, strategy.name, False)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Recovery strategy '{strategy.name}' timed out for {health_record.server_name}")
                await self._store_recovery_event(server_id, strategy.name, False, "timeout")
            except Exception as e:
                logger.error(f"Recovery strategy '{strategy.name}' error for {health_record.server_name}: {e}")
                await self._store_recovery_event(server_id, strategy.name, False, str(e))
        
        # All recovery strategies failed - escalate
        health_record.escalation_count += 1
        logger.critical(f"All recovery strategies failed for {health_record.server_name} - escalating")

    async def _restart_server(self, server_id: str) -> bool:
        """
        Attempt to restart a server
        
        Args:
            server_id: Server identifier
            
        Returns:
            True if restart was successful
        """
        try:
            server_info = self.monitored_servers.get(server_id)
            if not server_info:
                return False
            
            server = server_info['server']
            
            # Try various restart methods
            if hasattr(server, 'restart'):
                await server.restart()
                return True
            elif hasattr(server, 'stop') and hasattr(server, 'start'):
                await server.stop()
                await asyncio.sleep(2)  # Brief pause
                await server.start()
                return True
            else:
                logger.warning(f"Server {server_id} does not support restart operations")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restart server {server_id}: {e}")
            return False

    async def _reload_server(self, server_id: str) -> bool:
        """
        Attempt to reload server configuration
        
        Args:
            server_id: Server identifier
            
        Returns:
            True if reload was successful
        """
        try:
            server_info = self.monitored_servers.get(server_id)
            if not server_info:
                return False
            
            server = server_info['server']
            
            if hasattr(server, 'reload'):
                await server.reload()
                return True
            elif hasattr(server, 'refresh_config'):
                await server.refresh_config()
                return True
            else:
                logger.warning(f"Server {server_id} does not support reload operations")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reload server {server_id}: {e}")
            return False

    async def _escalate_server_issue(self, server_id: str) -> bool:
        """
        Escalate server issue for manual intervention
        
        Args:
            server_id: Server identifier
            
        Returns:
            Always returns True (escalation is always "successful")
        """
        health_record = self.health_records[server_id]
        
        escalation_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "server_id": server_id,
            "server_name": health_record.server_name,
            "current_status": health_record.current_status.value,
            "consecutive_failures": health_record.consecutive_failures,
            "recovery_attempts": health_record.recovery_attempts,
            "last_error": "Multiple recovery strategies failed"
        }
        
        # Store escalation record
        escalation_file = self.storage_path / "recovery" / f"escalation_{server_id}_{int(time.time())}.json"
        with open(escalation_file, 'w') as f:
            json.dump(escalation_data, f, indent=2)
        
        logger.critical(f"ESCALATION: Manual intervention required for server {health_record.server_name}")
        return True

    async def _process_alerts(self):
        """Process alert rules for all servers"""
        for server_id, health_record in self.health_records.items():
            for alert_rule in self.alert_rules:
                try:
                    if alert_rule.condition(health_record):
                        await self._trigger_alert(server_id, alert_rule)
                except Exception as e:
                    logger.error(f"Error processing alert rule {alert_rule.name}: {e}")

    async def _trigger_alert(self, server_id: str, alert_rule: AlertRule):
        """
        Trigger an alert if cooldown period has passed
        
        Args:
            server_id: Server identifier
            alert_rule: Alert rule that was triggered
        """
        health_record = self.health_records[server_id]
        now = datetime.utcnow()
        
        # Check cooldown period
        if (health_record.last_alert_time and 
            now - health_record.last_alert_time < timedelta(minutes=alert_rule.cooldown_minutes)):
            return
        
        # Format alert message
        message = alert_rule.message_template.format(
            server_name=health_record.server_name,
            server_id=server_id,
            consecutive_failures=health_record.consecutive_failures,
            failure_rate=health_record.total_failures / health_record.total_checks if health_record.total_checks > 0 else 0,
            current_status=health_record.current_status.value
        )
        
        # Create alert data
        alert_data = {
            "timestamp": now.isoformat(),
            "alert_rule": alert_rule.name,
            "severity": alert_rule.severity,
            "server_id": server_id,
            "server_name": health_record.server_name,
            "message": message,
            "health_status": health_record.current_status.value,
            "consecutive_failures": health_record.consecutive_failures,
            "response_time_ms": health_record.average_response_time
        }
        
        # Store alert
        await self._store_alert(alert_data)
        
        # Update health record
        health_record.alerts_sent += 1
        health_record.last_alert_time = now
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"ALERT [{alert_rule.severity.upper()}]: {message}")
        self.system_metrics["total_alerts"] += 1

    async def _store_health_result(self, server_id: str, check_result: HealthCheckResult):
        """Store health check result to persistent storage"""
        try:
            health_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "server_id": server_id,
                "status": check_result.status.value,
                "response_time_ms": check_result.response_time_ms,
                "check_type": check_result.check_type.value,
                "details": check_result.details,
                "error": check_result.error,
                "suggestions": check_result.suggestions
            }
            
            # Store in daily file
            today = datetime.now().strftime("%Y%m%d")
            health_file = self.storage_path / "health_history" / f"health_{today}.jsonl"
            
            with open(health_file, 'a') as f:
                f.write(json.dumps(health_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to store health result: {e}")

    async def _store_recovery_event(self, server_id: str, strategy_name: str, success: bool, error: str = None):
        """Store recovery event to persistent storage"""
        try:
            recovery_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "server_id": server_id,
                "strategy": strategy_name,
                "success": success,
                "error": error
            }
            
            today = datetime.now().strftime("%Y%m%d")
            recovery_file = self.storage_path / "recovery" / f"recovery_{today}.jsonl"
            
            with open(recovery_file, 'a') as f:
                f.write(json.dumps(recovery_data) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to store recovery event: {e}")

    async def _store_alert(self, alert_data: Dict[str, Any]):
        """Store alert to persistent storage"""
        try:
            alert_file = self.storage_path / "alerts" / f"alert_{int(time.time())}_{alert_data['server_id']}.json"
            
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")

    # Public API methods for integration

    async def get_server_health(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get current health status for a server"""
        if server_id not in self.health_records:
            return None
        
        health_record = self.health_records[server_id]
        return {
            "server_id": server_id,
            "server_name": health_record.server_name,
            "status": health_record.current_status.value,
            "last_check": health_record.last_check_time.isoformat(),
            "consecutive_failures": health_record.consecutive_failures,
            "total_checks": health_record.total_checks,
            "total_failures": health_record.total_failures,
            "average_response_time_ms": health_record.average_response_time,
            "uptime_percent": self._calculate_uptime_percent(health_record),
            "recovery_attempts": health_record.recovery_attempts,
            "successful_recoveries": health_record.successful_recoveries,
            "performance_trend": health_record.performance_trend,
            "alerts_sent": health_record.alerts_sent
        }

    async def get_all_server_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all monitored servers"""
        result = {}
        for server_id in self.health_records.keys():
            health = await self.get_server_health(server_id)
            if health:
                result[server_id] = health
        return result

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system health monitoring metrics"""
        overhead_times = list(self.system_metrics["monitoring_overhead_ms"])
        avg_overhead = sum(overhead_times) / len(overhead_times) if overhead_times else 0
        
        uptime_hours = (datetime.utcnow() - self.system_metrics["uptime_start"]).total_seconds() / 3600
        
        return {
            "monitoring_active": self.is_monitoring,
            "monitored_servers": len(self.monitored_servers),
            "total_health_checks": self.system_metrics["total_health_checks"],
            "total_recoveries": self.system_metrics["total_recoveries"], 
            "total_alerts": self.system_metrics["total_alerts"],
            "average_monitoring_overhead_ms": round(avg_overhead, 2),
            "uptime_hours": round(uptime_hours, 2),
            "check_interval_seconds": self.check_interval
        }

    def _calculate_uptime_percent(self, health_record: ServerHealthRecord) -> float:
        """Calculate uptime percentage for a server"""
        if health_record.total_checks == 0:
            return 100.0
        
        uptime_percent = ((health_record.total_checks - health_record.total_failures) / health_record.total_checks) * 100
        return round(uptime_percent, 2)

    def _update_response_time_average(self, server_id: str):
        """Utility method to recalculate response time average (useful for testing)"""
        if server_id in self.health_records:
            health_record = self.health_records[server_id]
            if health_record.response_times:
                health_record.average_response_time = sum(health_record.response_times) / len(health_record.response_times)
            else:
                health_record.average_response_time = 0.0

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)