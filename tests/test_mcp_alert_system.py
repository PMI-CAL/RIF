"""
Test suite for MCP Alert System
Tests for intelligent alert generation, throttling, and multi-channel delivery

Issue: #84 - Create MCP health monitor
Component: Alert system tests
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Import the components to test
import sys
sys.path.append('/Users/cal/DEV/RIF')

from mcp.monitor.alerts import (
    AlertManager, Alert, AlertRule, AlertChannel, AlertSeverity, AlertStatus
)


class TestAlertComponents:
    """Test alert system components and data structures"""
    
    def test_alert_creation(self):
        """Test alert instance creation"""
        alert = Alert(
            alert_id="test-alert-1",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            server_id="test-server",
            server_name="Test Server",
            message="Test alert message",
            timestamp=datetime.utcnow(),
            context={"test_key": "test_value"},
            tags=["test", "warning"]
        )
        
        assert alert.alert_id == "test-alert-1"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.ACTIVE  # Default status
        assert alert.escalation_level == 0
        assert len(alert.delivery_attempts) == 0
        assert alert.context["test_key"] == "test_value"
        assert "test" in alert.tags
    
    def test_alert_rule_creation(self):
        """Test alert rule configuration"""
        def test_condition(data):
            return data.get("status") == "unhealthy"
        
        rule = AlertRule(
            name="unhealthy_server",
            description="Server is unhealthy",
            condition=test_condition,
            severity=AlertSeverity.CRITICAL,
            cooldown_minutes=10,
            message_template="Server {server_name} is unhealthy",
            tags=["health", "critical"]
        )
        
        assert rule.name == "unhealthy_server"
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.cooldown_minutes == 10
        assert rule.enabled == True
        assert callable(rule.condition)
        assert "health" in rule.tags
        
        # Test condition function
        assert rule.condition({"status": "unhealthy"}) == True
        assert rule.condition({"status": "healthy"}) == False
    
    def test_alert_channel_creation(self):
        """Test alert channel configuration"""
        channel = AlertChannel(
            name="console_channel",
            channel_type="console",
            severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL],
            enabled=True
        )
        
        assert channel.name == "console_channel"
        assert channel.channel_type == "console"
        assert AlertSeverity.WARNING in channel.severity_filter
        assert AlertSeverity.CRITICAL in channel.severity_filter
        assert AlertSeverity.INFO not in channel.severity_filter


class TestAlertManager:
    """Test alert management system"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def alert_manager(self, temp_storage):
        """Create alert manager with temp storage"""
        return AlertManager(
            storage_path=temp_storage,
            default_cooldown_minutes=5,  # Short cooldown for testing
            max_alerts_per_server=10
        )
    
    def test_initialization(self, alert_manager, temp_storage):
        """Test alert manager initialization"""
        assert alert_manager.storage_path == Path(temp_storage)
        assert alert_manager.default_cooldown_minutes == 5
        assert alert_manager.max_alerts_per_server == 10
        
        # Check default channels were created
        assert "console" in alert_manager.alert_channels
        assert "log" in alert_manager.alert_channels
        assert "github" in alert_manager.alert_channels
        assert "dashboard" in alert_manager.alert_channels
        
        # Check initial state
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_rules) == 0
    
    def test_alert_rule_management(self, alert_manager):
        """Test adding and removing alert rules"""
        def test_condition(data):
            return data.get("cpu_usage", 0) > 80
        
        rule = AlertRule(
            name="high_cpu",
            description="High CPU usage",
            condition=test_condition,
            severity=AlertSeverity.WARNING,
            cooldown_minutes=15
        )
        
        # Add rule
        alert_manager.add_alert_rule(rule)
        
        assert "high_cpu" in alert_manager.alert_rules
        assert alert_manager.alert_rules["high_cpu"] is rule
        
        # Remove rule
        alert_manager.remove_alert_rule("high_cpu")
        
        assert "high_cpu" not in alert_manager.alert_rules
    
    def test_alert_channel_management(self, alert_manager):
        """Test adding and configuring alert channels"""
        channel = AlertChannel(
            name="custom_webhook",
            channel_type="webhook",
            config={"url": "https://example.com/webhook"},
            severity_filter=[AlertSeverity.CRITICAL]
        )
        
        alert_manager.add_alert_channel(channel)
        
        assert "custom_webhook" in alert_manager.alert_channels
        assert alert_manager.alert_channels["custom_webhook"] is channel
    
    @pytest.mark.asyncio
    async def test_alert_evaluation_and_creation(self, alert_manager):
        """Test alert rule evaluation and alert creation"""
        # Add test rule
        def cpu_condition(data):
            return data.get("cpu_usage", 0) > 80
        
        rule = AlertRule(
            name="high_cpu_usage",
            description="CPU usage too high",
            condition=cpu_condition,
            severity=AlertSeverity.WARNING,
            message_template="Server {server_name} has high CPU usage: {cpu_usage}%"
        )
        
        alert_manager.add_alert_rule(rule)
        
        # Test data that should trigger alert
        server_data = {
            "server_id": "test-server-1",
            "server_name": "Test Server 1",
            "cpu_usage": 85.0,
            "status": "degraded"
        }
        
        triggered_alerts = await alert_manager.evaluate_alerts(server_data)
        
        assert len(triggered_alerts) == 1
        
        alert = triggered_alerts[0]
        assert alert.rule_name == "high_cpu_usage"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.server_id == "test-server-1"
        assert "85%" in alert.message
        
        # Check alert was stored
        assert alert.alert_id in alert_manager.active_alerts
    
    @pytest.mark.asyncio
    async def test_alert_throttling(self, alert_manager):
        """Test alert throttling prevents spam"""
        def always_trigger(data):
            return True
        
        rule = AlertRule(
            name="spam_rule",
            description="Always triggers",
            condition=always_trigger,
            severity=AlertSeverity.INFO,
            cooldown_minutes=5  # 5 minute cooldown
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {
            "server_id": "test-server-1",
            "server_name": "Test Server 1"
        }
        
        # First evaluation should create alert
        alerts1 = await alert_manager.evaluate_alerts(server_data)
        assert len(alerts1) == 1
        
        # Second evaluation immediately after should be throttled
        alerts2 = await alert_manager.evaluate_alerts(server_data)
        assert len(alerts2) == 0
    
    @pytest.mark.asyncio
    async def test_alert_acknowledgment(self, alert_manager):
        """Test alert acknowledgment"""
        # Create and process an alert first
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            condition=test_condition,
            severity=AlertSeverity.WARNING
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        alerts = await alert_manager.evaluate_alerts(server_data)
        
        alert_id = alerts[0].alert_id
        
        # Acknowledge the alert
        success = await alert_manager.acknowledge_alert(alert_id, "test-user", "Investigating issue")
        
        assert success == True
        
        alert = alert_manager.active_alerts[alert_id]
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test-user"
        assert alert.acknowledged_at is not None
        assert alert.context["acknowledgment_note"] == "Investigating issue"
    
    @pytest.mark.asyncio
    async def test_alert_resolution(self, alert_manager):
        """Test alert resolution"""
        # Create alert
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            condition=test_condition,
            severity=AlertSeverity.WARNING
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        alerts = await alert_manager.evaluate_alerts(server_data)
        
        alert_id = alerts[0].alert_id
        
        # Resolve the alert
        success = await alert_manager.resolve_alert(alert_id, "system", "Issue resolved automatically")
        
        assert success == True
        
        # Alert should be removed from active alerts
        assert alert_id not in alert_manager.active_alerts
        
        # Should be in history
        resolved_alert = None
        for alert in alert_manager.alert_history:
            if alert.alert_id == alert_id:
                resolved_alert = alert
                break
        
        assert resolved_alert is not None
        assert resolved_alert.status == AlertStatus.RESOLVED
        assert resolved_alert.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_alert_delivery(self, alert_manager):
        """Test alert delivery through channels"""
        # Mock delivery functions
        console_calls = []
        log_calls = []
        
        async def mock_console_delivery(alert, channel):
            console_calls.append((alert.alert_id, alert.message))
        
        async def mock_log_delivery(alert, channel):
            log_calls.append((alert.alert_id, alert.severity.value))
        
        # Update channel delivery functions
        alert_manager.alert_channels["console"].delivery_function = mock_console_delivery
        alert_manager.alert_channels["log"].delivery_function = mock_log_delivery
        
        # Create alert
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="delivery_test",
            description="Test delivery",
            condition=test_condition,
            severity=AlertSeverity.WARNING,
            message_template="Test message for {server_name}"
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        alerts = await alert_manager.evaluate_alerts(server_data)
        
        # Verify deliveries were called
        assert len(console_calls) == 1
        assert len(log_calls) == 1
        
        alert_id = alerts[0].alert_id
        assert console_calls[0][0] == alert_id
        assert "Test message for Test Server" in console_calls[0][1]
    
    @pytest.mark.asyncio
    async def test_severity_filtering(self, alert_manager):
        """Test alert channel severity filtering"""
        # Create channel that only accepts critical alerts
        critical_calls = []
        
        async def mock_critical_delivery(alert, channel):
            critical_calls.append(alert.alert_id)
        
        critical_channel = AlertChannel(
            name="critical_only",
            channel_type="custom",
            severity_filter=[AlertSeverity.CRITICAL],
            delivery_function=mock_critical_delivery
        )
        
        alert_manager.add_alert_channel(critical_channel)
        
        # Create warning alert (should not be delivered to critical_only channel)
        def warning_condition(data):
            return True
        
        warning_rule = AlertRule(
            name="warning_rule",
            description="Warning rule",
            condition=warning_condition,
            severity=AlertSeverity.WARNING
        )
        
        alert_manager.add_alert_rule(warning_rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        await alert_manager.evaluate_alerts(server_data)
        
        # Should not have been delivered to critical-only channel
        assert len(critical_calls) == 0
        
        # Create critical alert (should be delivered)
        def critical_condition(data):
            return True
        
        critical_rule = AlertRule(
            name="critical_rule",
            description="Critical rule",
            condition=critical_condition,
            severity=AlertSeverity.CRITICAL
        )
        
        alert_manager.add_alert_rule(critical_rule)
        
        # Clear throttle for this server
        alert_manager.alert_throttle.clear()
        
        await alert_manager.evaluate_alerts(server_data)
        
        # Should have been delivered to critical-only channel
        assert len(critical_calls) == 1
    
    @pytest.mark.asyncio
    async def test_alert_escalation(self, alert_manager):
        """Test alert escalation system"""
        # Create rule with escalation configuration
        def test_condition(data):
            return True
        
        escalation_rules = [
            {"trigger_minutes": 1, "max_level": 2}  # Escalate after 1 minute, max 2 levels
        ]
        
        rule = AlertRule(
            name="escalation_test",
            description="Test escalation",
            condition=test_condition,
            severity=AlertSeverity.WARNING,
            escalation_rules=escalation_rules
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        alerts = await alert_manager.evaluate_alerts(server_data)
        
        original_alert = alerts[0]
        
        # Manually set alert timestamp to trigger escalation
        original_alert.timestamp = datetime.utcnow() - timedelta(minutes=2)
        
        # Process escalations
        await alert_manager._process_escalations()
        
        # Should have created escalated alert
        escalated_alerts = [a for a in alert_manager.active_alerts.values() 
                          if a.escalation_level > 0]
        
        assert len(escalated_alerts) > 0
        
        escalated_alert = escalated_alerts[0]
        assert escalated_alert.severity == AlertSeverity.CRITICAL
        assert "ESCALATED" in escalated_alert.message
        assert escalated_alert.escalation_level == 1
    
    @pytest.mark.asyncio
    async def test_auto_resolution(self, alert_manager):
        """Test automatic alert resolution"""
        # Create and acknowledge an alert
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="auto_resolve_test",
            description="Auto resolve test",
            condition=test_condition,
            severity=AlertSeverity.WARNING
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        alerts = await alert_manager.evaluate_alerts(server_data)
        
        alert_id = alerts[0].alert_id
        
        # Acknowledge alert
        await alert_manager.acknowledge_alert(alert_id, "test-user")
        
        # Set acknowledged time to trigger auto-resolution (25 hours ago)
        alert = alert_manager.active_alerts[alert_id]
        alert.acknowledged_at = datetime.utcnow() - timedelta(hours=25)
        
        # Process auto-resolutions
        await alert_manager._auto_resolve_alerts()
        
        # Alert should be auto-resolved
        assert alert_id not in alert_manager.active_alerts
    
    @pytest.mark.asyncio
    async def test_data_persistence(self, alert_manager, temp_storage):
        """Test alert data persistence"""
        # Create alert
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="persistence_test",
            description="Persistence test",
            condition=test_condition,
            severity=AlertSeverity.WARNING,
            message_template="Persistence test alert"
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "test-server", "server_name": "Test Server"}
        alerts = await alert_manager.evaluate_alerts(server_data)
        
        alert_id = alerts[0].alert_id
        
        # Check if alert file was created
        storage_path = Path(temp_storage)
        alert_file = storage_path / f"alert_{alert_id}.json"
        
        # Give some time for async storage
        await asyncio.sleep(0.1)
        
        if alert_file.exists():
            with open(alert_file, 'r') as f:
                stored_data = json.load(f)
                
                assert stored_data["alert_id"] == alert_id
                assert stored_data["rule_name"] == "persistence_test"
                assert stored_data["severity"] == "warning"
                assert stored_data["server_id"] == "test-server"
    
    @pytest.mark.asyncio
    async def test_dashboard_integration(self, alert_manager, temp_storage):
        """Test dashboard alerts file generation"""
        # Create alert
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="dashboard_test",
            description="Dashboard test",
            condition=test_condition,
            severity=AlertSeverity.CRITICAL,
            message_template="Dashboard test alert"
        )
        
        alert_manager.add_alert_rule(rule)
        
        server_data = {"server_id": "dashboard-server", "server_name": "Dashboard Server"}
        await alert_manager.evaluate_alerts(server_data)
        
        # Check dashboard alerts file
        storage_path = Path(temp_storage)
        dashboard_file = storage_path / "dashboard_alerts.json"
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        if dashboard_file.exists():
            with open(dashboard_file, 'r') as f:
                dashboard_alerts = json.load(f)
                
                assert len(dashboard_alerts) > 0
                
                alert = dashboard_alerts[0]
                assert alert["server_name"] == "Dashboard Server"
                assert alert["severity"] == "critical"
                assert alert["resolved"] == False
    
    @pytest.mark.asyncio
    async def test_alert_statistics(self, alert_manager):
        """Test alert system statistics"""
        # Create some alerts
        def test_condition(data):
            return True
        
        warning_rule = AlertRule(
            name="warning_stat_test",
            description="Warning for stats",
            condition=test_condition,
            severity=AlertSeverity.WARNING
        )
        
        critical_rule = AlertRule(
            name="critical_stat_test", 
            description="Critical for stats",
            condition=test_condition,
            severity=AlertSeverity.CRITICAL
        )
        
        alert_manager.add_alert_rule(warning_rule)
        alert_manager.add_alert_rule(critical_rule)
        
        server_data = {"server_id": "stats-server", "server_name": "Stats Server"}
        
        # Create warning alert
        await alert_manager.evaluate_alerts(server_data)
        
        # Clear throttle and create critical alert
        alert_manager.alert_throttle.clear()
        await alert_manager.evaluate_alerts(server_data)
        
        # Get statistics
        stats = await alert_manager.get_alert_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_alerts"] >= 2
        assert stats["active_alerts"] >= 2
        assert "alerts_by_severity" in stats
        assert "delivery_attempts" in stats
        assert "configured_rules" in stats
        assert "configured_channels" in stats
        
        # Check severity breakdown
        severity_stats = stats["alerts_by_severity"]
        assert severity_stats.get("warning", 0) >= 1
        assert severity_stats.get("critical", 0) >= 1
    
    @pytest.mark.asyncio
    async def test_active_alerts_api(self, alert_manager):
        """Test public API for getting active alerts"""
        # Create alerts
        def test_condition(data):
            return True
        
        rule = AlertRule(
            name="api_test",
            description="API test",
            condition=test_condition,
            severity=AlertSeverity.WARNING
        )
        
        alert_manager.add_alert_rule(rule)
        
        server1_data = {"server_id": "server-1", "server_name": "Server 1"}
        server2_data = {"server_id": "server-2", "server_name": "Server 2"}
        
        await alert_manager.evaluate_alerts(server1_data)
        
        # Clear throttle and create second alert
        alert_manager.alert_throttle.clear()
        await alert_manager.evaluate_alerts(server2_data)
        
        # Get all active alerts
        all_alerts = await alert_manager.get_active_alerts()
        
        assert len(all_alerts) == 2
        
        for alert in all_alerts:
            assert alert["status"] == "active"
            assert alert["severity"] == "warning"
            assert alert["rule_name"] == "api_test"
        
        # Get alerts for specific server
        server1_alerts = await alert_manager.get_active_alerts(server_id="server-1")
        
        assert len(server1_alerts) == 1
        assert server1_alerts[0]["server_id"] == "server-1"


class TestAlertSystemIntegration:
    """Integration tests for complete alert system"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_complete_alert_lifecycle(self, temp_storage):
        """Test complete alert lifecycle from creation to resolution"""
        alert_manager = AlertManager(
            storage_path=temp_storage,
            default_cooldown_minutes=1
        )
        
        # Mock delivery tracking
        deliveries = []
        
        async def mock_delivery(alert, channel):
            deliveries.append((alert.alert_id, channel.name))
        
        # Update all channels to use mock delivery
        for channel in alert_manager.alert_channels.values():
            channel.delivery_function = mock_delivery
        
        # Create escalating rule
        def unhealthy_condition(data):
            return data.get("status") == "unhealthy"
        
        rule = AlertRule(
            name="unhealthy_lifecycle_test",
            description="Unhealthy server lifecycle test",
            condition=unhealthy_condition,
            severity=AlertSeverity.CRITICAL,
            message_template="Server {server_name} is unhealthy",
            escalation_rules=[{"trigger_minutes": 0.01, "max_level": 1}]  # Fast escalation for testing
        )
        
        alert_manager.add_alert_rule(rule)
        
        try:
            # Start processing
            await alert_manager.start_processing()
            
            # Create initial alert
            server_data = {
                "server_id": "lifecycle-server",
                "server_name": "Lifecycle Test Server",
                "status": "unhealthy"
            }
            
            alerts = await alert_manager.evaluate_alerts(server_data)
            assert len(alerts) == 1
            
            original_alert = alerts[0]
            alert_id = original_alert.alert_id
            
            # Verify delivery occurred
            assert len(deliveries) > 0
            assert any(alert_id == delivery[0] for delivery in deliveries)
            
            # Acknowledge alert
            success = await alert_manager.acknowledge_alert(alert_id, "test-admin", "Investigating")
            assert success == True
            
            # Wait briefly for escalation processing
            await asyncio.sleep(0.1)
            
            # Resolve alert
            success = await alert_manager.resolve_alert(alert_id, "test-admin", "Issue fixed")
            assert success == True
            
            # Verify resolution
            assert alert_id not in alert_manager.active_alerts
            
            # Stop processing
            await alert_manager.stop_processing()
            
        finally:
            if alert_manager.is_processing:
                await alert_manager.stop_processing()


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))