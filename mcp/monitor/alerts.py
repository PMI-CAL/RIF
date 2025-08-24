"""
MCP Alert System
Intelligent alert generation, throttling, and multi-channel delivery for MCP server monitoring

Features:
- Rule-based alert generation with complex conditions
- Intelligent throttling and deduplication
- Multi-channel alert delivery (GitHub, logs, console, dashboard)
- Alert escalation and acknowledgment management
- Integration with monitoring dashboard and knowledge system

Issue: #84 - Create MCP health monitor
Component: Alert system with throttling and escalation
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert instance with full metadata"""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    server_id: str
    server_name: str
    message: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    delivery_attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AlertRule:
    """Enhanced alert rule with complex conditions and actions"""
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    cooldown_minutes: int = 15
    message_template: str = "Alert triggered for {server_name}"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    suppression_conditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)


@dataclass
class AlertChannel:
    """Alert delivery channel configuration"""
    name: str
    channel_type: str  # "github", "log", "console", "webhook", "dashboard"
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    delivery_function: Optional[Callable] = None


class AlertManager:
    """
    Enterprise alert management system for MCP server monitoring
    
    Provides intelligent alert generation, throttling, escalation, and
    multi-channel delivery with acknowledgment and resolution tracking.
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 default_cooldown_minutes: int = 15,
                 max_alerts_per_server: int = 100):
        """
        Initialize the alert manager
        
        Args:
            storage_path: Path for storing alert data
            default_cooldown_minutes: Default cooldown period for alerts
            max_alerts_per_server: Maximum active alerts per server
        """
        self.storage_path = Path(storage_path) if storage_path else Path("knowledge/monitoring/alerts")
        self.default_cooldown_minutes = default_cooldown_minutes
        self.max_alerts_per_server = max_alerts_per_server
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self.alert_history: List[Alert] = []
        self.alert_channels: Dict[str, AlertChannel] = {}
        
        # Throttling and deduplication
        self.alert_throttle: Dict[str, datetime] = {}  # rule_name:server_id -> last_trigger_time
        self.alert_suppressions: Set[str] = set()  # Suppressed alert combinations
        
        # Processing state
        self.is_processing = False
        self.processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.alert_stats = {
            "total_alerts": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_server": defaultdict(int),
            "delivery_attempts": 0,
            "delivery_failures": 0,
            "acknowledged_alerts": 0,
            "auto_resolved_alerts": 0
        }
        
        # Storage setup
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default channels
        self._initialize_default_channels()
        
        logger.info(f"AlertManager initialized (cooldown: {default_cooldown_minutes}m)")
    
    def _initialize_default_channels(self):
        """Initialize default alert delivery channels"""
        # Console channel for development
        self.alert_channels["console"] = AlertChannel(
            name="console",
            channel_type="console",
            enabled=True,
            delivery_function=self._deliver_console_alert
        )
        
        # Log file channel
        self.alert_channels["log"] = AlertChannel(
            name="log",
            channel_type="log",
            enabled=True,
            delivery_function=self._deliver_log_alert
        )
        
        # GitHub issues channel for critical alerts
        self.alert_channels["github"] = AlertChannel(
            name="github",
            channel_type="github",
            enabled=True,
            severity_filter=[AlertSeverity.CRITICAL],
            delivery_function=self._deliver_github_alert
        )
        
        # Dashboard channel for real-time display
        self.alert_channels["dashboard"] = AlertChannel(
            name="dashboard",
            channel_type="dashboard", 
            enabled=True,
            delivery_function=self._deliver_dashboard_alert
        )
    
    async def start_processing(self):
        """Start alert processing system"""
        if self.is_processing:
            logger.warning("Alert processing already running")
            return
        
        self.is_processing = True
        self._shutdown_event.clear()
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        logger.info("Alert processing started")
    
    async def stop_processing(self):
        """Stop alert processing system"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self._shutdown_event.set()
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
        
        logger.info("Alert processing stopped")
    
    async def _processing_loop(self):
        """Main alert processing loop"""
        try:
            while self.is_processing and not self._shutdown_event.is_set():
                try:
                    # Process alert escalations
                    await self._process_escalations()
                    
                    # Auto-resolve alerts
                    await self._auto_resolve_alerts()
                    
                    # Clean up old throttle entries
                    await self._cleanup_throttle_entries()
                    
                    # Store alert state periodically
                    await self._store_alert_state()
                    
                except Exception as e:
                    logger.error(f"Error in alert processing: {e}")
                
                # Process alerts every 30 seconds
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal interval timeout
                    
        except asyncio.CancelledError:
            logger.info("Alert processing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in alert processing loop: {e}")
    
    def add_alert_rule(self, rule: AlertRule):
        """
        Add or update an alert rule
        
        Args:
            rule: AlertRule to add
        """
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} (severity: {rule.severity.value})")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_alert_channel(self, channel: AlertChannel):
        """
        Add or update an alert delivery channel
        
        Args:
            channel: AlertChannel to add
        """
        self.alert_channels[channel.name] = channel
        logger.info(f"Added alert channel: {channel.name} (type: {channel.channel_type})")
    
    async def evaluate_alerts(self, server_data: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all alert rules for given server data
        
        Args:
            server_data: Server health and metrics data
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        server_id = server_data.get('server_id', 'unknown')
        server_name = server_data.get('server_name', server_id)
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check suppression conditions first
                if any(condition(server_data) for condition in rule.suppression_conditions):
                    continue
                
                # Evaluate alert condition
                if rule.condition(server_data):
                    # Check throttling
                    throttle_key = f"{rule_name}:{server_id}"
                    if self._is_throttled(throttle_key, rule.cooldown_minutes):
                        continue
                    
                    # Create alert
                    alert = await self._create_alert(rule, server_data)
                    triggered_alerts.append(alert)
                    
                    # Update throttle
                    self.alert_throttle[throttle_key] = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
        
        # Process triggered alerts
        for alert in triggered_alerts:
            await self._process_alert(alert)
        
        return triggered_alerts
    
    def _is_throttled(self, throttle_key: str, cooldown_minutes: int) -> bool:
        """Check if alert is throttled"""
        if throttle_key not in self.alert_throttle:
            return False
        
        last_trigger = self.alert_throttle[throttle_key]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() - last_trigger < cooldown_period
    
    async def _create_alert(self, rule: AlertRule, server_data: Dict[str, Any]) -> Alert:
        """Create a new alert instance"""
        server_id = server_data.get('server_id', 'unknown')
        server_name = server_data.get('server_name', server_id)
        
        # Generate unique alert ID
        alert_id = f"{rule.name}_{server_id}_{int(time.time())}"
        
        # Format alert message
        message = rule.message_template.format(
            server_name=server_name,
            server_id=server_id,
            **server_data
        )
        
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            server_id=server_id,
            server_name=server_name,
            message=message,
            timestamp=datetime.utcnow(),
            context=server_data.copy(),
            tags=rule.tags.copy()
        )
        
        return alert
    
    async def _process_alert(self, alert: Alert):
        """Process a new alert (store, deliver, track)"""
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.alert_stats["total_alerts"] += 1
            self.alert_stats["alerts_by_severity"][alert.severity.value] += 1
            self.alert_stats["alerts_by_server"][alert.server_id] += 1
            
            # Check server alert limits
            server_alerts = [a for a in self.active_alerts.values() if a.server_id == alert.server_id]
            if len(server_alerts) > self.max_alerts_per_server:
                # Auto-acknowledge oldest alerts
                oldest_alerts = sorted(server_alerts, key=lambda x: x.timestamp)[:len(server_alerts) - self.max_alerts_per_server]
                for old_alert in oldest_alerts:
                    await self.acknowledge_alert(old_alert.alert_id, "system", "Auto-acknowledged due to alert limit")
            
            # Deliver alert through channels
            await self._deliver_alert(alert)
            
            # Store alert persistently
            await self._store_alert(alert)
            
            logger.info(f"Processed alert: {alert.alert_id} [{alert.severity.value}] {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to process alert {alert.alert_id}: {e}")
    
    async def _deliver_alert(self, alert: Alert):
        """Deliver alert through all configured channels"""
        for channel_name, channel in self.alert_channels.items():
            if not channel.enabled:
                continue
            
            # Check severity filter
            if alert.severity not in channel.severity_filter:
                continue
            
            try:
                if channel.delivery_function:
                    await channel.delivery_function(alert, channel)
                    alert.delivery_attempts.append({
                        "channel": channel_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "success"
                    })
                    self.alert_stats["delivery_attempts"] += 1
                
            except Exception as e:
                logger.error(f"Failed to deliver alert {alert.alert_id} via {channel_name}: {e}")
                alert.delivery_attempts.append({
                    "channel": channel_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": str(e)
                })
                self.alert_stats["delivery_failures"] += 1
    
    async def _deliver_console_alert(self, alert: Alert, channel: AlertChannel):
        """Deliver alert to console"""
        severity_symbol = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        symbol = severity_symbol.get(alert.severity.value, "🔔")
        print(f"{symbol} ALERT [{alert.severity.value.upper()}]: {alert.message}")
    
    async def _deliver_log_alert(self, alert: Alert, channel: AlertChannel):
        """Deliver alert to log file"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        logger.log(log_level[alert.severity], f"ALERT: {alert.message}")
    
    async def _deliver_github_alert(self, alert: Alert, channel: AlertChannel):
        """Deliver alert to GitHub issues (placeholder)"""
        # This would integrate with GitHub API to create issues
        # For now, just log the intent
        logger.info(f"Would create GitHub issue for critical alert: {alert.message}")
    
    async def _deliver_dashboard_alert(self, alert: Alert, channel: AlertChannel):
        """Deliver alert to monitoring dashboard"""
        # This would update the dashboard in real-time
        # For now, store in format that dashboard can read
        dashboard_alert = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "message": alert.message,
            "server_name": alert.server_name,
            "timestamp": alert.timestamp.isoformat(),
            "resolved": False
        }
        
        dashboard_file = self.storage_path / "dashboard_alerts.json"
        
        # Load existing alerts
        existing_alerts = []
        if dashboard_file.exists():
            try:
                with open(dashboard_file, 'r') as f:
                    existing_alerts = json.load(f)
            except:
                existing_alerts = []
        
        # Add new alert and keep only recent ones
        existing_alerts.append(dashboard_alert)
        existing_alerts = existing_alerts[-50:]  # Keep last 50 alerts
        
        # Save updated alerts
        with open(dashboard_file, 'w') as f:
            json.dump(existing_alerts, f, indent=2)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, note: str = ""):
        """
        Acknowledge an active alert
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: Who acknowledged the alert
            note: Optional acknowledgment note
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Cannot acknowledge unknown alert: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        
        if note:
            alert.context["acknowledgment_note"] = note
        
        self.alert_stats["acknowledged_alerts"] += 1
        
        # Update dashboard
        await self._update_dashboard_alert_status(alert_id, "acknowledged")
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system", note: str = ""):
        """
        Resolve an alert
        
        Args:
            alert_id: Alert identifier
            resolved_by: Who resolved the alert
            note: Optional resolution note
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Cannot resolve unknown alert: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        if note:
            alert.context["resolution_note"] = note
        
        # Move from active to history
        del self.active_alerts[alert_id]
        
        if resolved_by == "system":
            self.alert_stats["auto_resolved_alerts"] += 1
        
        # Update dashboard
        await self._update_dashboard_alert_status(alert_id, "resolved")
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    async def _update_dashboard_alert_status(self, alert_id: str, status: str):
        """Update alert status in dashboard"""
        dashboard_file = self.storage_path / "dashboard_alerts.json"
        
        if not dashboard_file.exists():
            return
        
        try:
            with open(dashboard_file, 'r') as f:
                alerts = json.load(f)
            
            for alert in alerts:
                if alert.get("alert_id") == alert_id:
                    alert["resolved"] = (status == "resolved")
                    alert["status"] = status
                    break
            
            with open(dashboard_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update dashboard alert status: {e}")
    
    async def _process_escalations(self):
        """Process alert escalations based on rules"""
        for alert in list(self.active_alerts.values()):
            if alert.status == AlertStatus.ACKNOWLEDGED:
                continue
            
            rule = self.alert_rules.get(alert.rule_name)
            if not rule or not rule.escalation_rules:
                continue
            
            # Check escalation conditions
            alert_age_minutes = (datetime.utcnow() - alert.timestamp).total_seconds() / 60
            
            for escalation_rule in rule.escalation_rules:
                trigger_minutes = escalation_rule.get('trigger_minutes', 60)
                max_level = escalation_rule.get('max_level', 3)
                
                if (alert_age_minutes >= trigger_minutes * (alert.escalation_level + 1) and
                    alert.escalation_level < max_level):
                    
                    alert.escalation_level += 1
                    
                    # Create escalated alert
                    escalated_message = f"ESCALATED (Level {alert.escalation_level}): {alert.message}"
                    escalated_alert = Alert(
                        alert_id=f"{alert.alert_id}_escalation_{alert.escalation_level}",
                        rule_name=alert.rule_name,
                        severity=AlertSeverity.CRITICAL,
                        server_id=alert.server_id,
                        server_name=alert.server_name,
                        message=escalated_message,
                        timestamp=datetime.utcnow(),
                        context=alert.context.copy(),
                        tags=alert.tags + ["escalated"],
                        escalation_level=alert.escalation_level
                    )
                    
                    await self._process_alert(escalated_alert)
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts based on conditions"""
        for alert_id in list(self.active_alerts.keys()):
            alert = self.active_alerts[alert_id]
            
            # Auto-resolve after 24 hours if acknowledged
            if (alert.status == AlertStatus.ACKNOWLEDGED and
                alert.acknowledged_at and
                datetime.utcnow() - alert.acknowledged_at > timedelta(hours=24)):
                
                await self.resolve_alert(alert_id, "system", "Auto-resolved after 24 hours")
    
    async def _cleanup_throttle_entries(self):
        """Remove old throttle entries"""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.alert_throttle = {k: v for k, v in self.alert_throttle.items() if v > cutoff}
    
    async def _store_alert(self, alert: Alert):
        """Store alert to persistent storage"""
        try:
            alert_file = self.storage_path / f"alert_{alert.alert_id}.json"
            
            alert_data = {
                "alert_id": alert.alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "server_id": alert.server_id,
                "server_name": alert.server_name,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "status": alert.status.value,
                "context": alert.context,
                "tags": alert.tags,
                "escalation_level": alert.escalation_level,
                "delivery_attempts": alert.delivery_attempts,
                "acknowledged_by": alert.acknowledged_by,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to store alert {alert.alert_id}: {e}")
    
    async def _store_alert_state(self):
        """Store current alert state"""
        try:
            state_file = self.storage_path / "alert_state.json"
            
            state_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_alerts_count": len(self.active_alerts),
                "total_alerts": self.alert_stats["total_alerts"],
                "statistics": dict(self.alert_stats["alerts_by_severity"]),
                "alert_rules_count": len(self.alert_rules),
                "channels_count": len(self.alert_channels)
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to store alert state: {e}")
    
    # Public API methods
    
    async def get_active_alerts(self, server_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())
        
        if server_id:
            alerts = [a for a in alerts if a.server_id == server_id]
        
        return [
            {
                "alert_id": alert.alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "server_id": alert.server_id,
                "server_name": alert.server_name,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "status": alert.status.value,
                "escalation_level": alert.escalation_level,
                "acknowledged_by": alert.acknowledged_by
            }
            for alert in alerts
        ]
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        return {
            "total_alerts": self.alert_stats["total_alerts"],
            "active_alerts": len(self.active_alerts),
            "alerts_by_severity": dict(self.alert_stats["alerts_by_severity"]),
            "alerts_by_server": dict(self.alert_stats["alerts_by_server"]),
            "delivery_attempts": self.alert_stats["delivery_attempts"],
            "delivery_failures": self.alert_stats["delivery_failures"],
            "delivery_success_rate": (
                (self.alert_stats["delivery_attempts"] - self.alert_stats["delivery_failures"]) / 
                max(1, self.alert_stats["delivery_attempts"])
            ) * 100,
            "acknowledged_alerts": self.alert_stats["acknowledged_alerts"],
            "auto_resolved_alerts": self.alert_stats["auto_resolved_alerts"],
            "configured_rules": len(self.alert_rules),
            "configured_channels": len(self.alert_channels)
        }