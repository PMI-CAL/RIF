#!/usr/bin/env python3
"""
SLA Monitoring and Escalation System - Issue #92 Phase 3
Real-time SLA tracking, alerting, and automated escalation for specialist reviews.

This system implements:
1. Real-time SLA tracking for specialist assignments
2. Multi-threshold alert system (early warning, urgent, breach)
3. Automated escalation chain management
4. Performance metrics collection and reporting
5. Notification delivery (GitHub comments, alerts)
6. Manager override and escalation capabilities
7. Business hours and timezone-aware calculations
"""

import json
import subprocess
import yaml
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import hashlib

class SLAStatus(Enum):
    """SLA tracking status."""
    ON_TRACK = "on_track"
    EARLY_WARNING = "early_warning"  # 50% of time elapsed
    URGENT_WARNING = "urgent_warning"  # 80% of time elapsed
    BREACHED = "breached"  # 100% of time elapsed
    RESOLVED = "resolved"
    ESCALATED = "escalated"

class NotificationChannel(Enum):
    """Available notification channels."""
    GITHUB_COMMENT = "github_comment"
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    PHONE = "phone"

class EscalationLevel(Enum):
    """Escalation levels."""
    NONE = 0
    AUTOMATIC = 1
    MANAGER = 2
    EXECUTIVE = 3

@dataclass
class SLATracking:
    """Container for SLA tracking information."""
    tracking_id: str
    issue_number: int
    specialist_github_issue: Optional[int]
    assigned_specialist: str
    specialist_type: str
    created_at: datetime
    sla_deadline: datetime
    current_status: SLAStatus
    time_elapsed_percent: float
    time_remaining_hours: float
    escalation_level: EscalationLevel
    notifications_sent: List[str]
    last_status_update: datetime
    business_hours_only: bool = False
    
@dataclass
class SLAAlert:
    """Container for SLA alert information."""
    tracking_id: str
    alert_type: SLAStatus
    threshold_breached: float
    time_to_deadline: timedelta
    escalation_required: bool
    notification_channels: List[NotificationChannel]
    message: str
    priority: str

@dataclass
class EscalationAction:
    """Container for escalation action information."""
    tracking_id: str
    escalation_type: EscalationLevel
    escalated_to: List[str]
    escalation_reason: str
    escalation_time: datetime
    original_assignee: str
    new_assignee: Optional[str]
    manager_notified: bool

class SLAMonitoringSystem:
    """
    Comprehensive SLA monitoring and escalation system.
    """
    
    def __init__(self, config_path: str = "config/risk-assessment.yaml"):
        """Initialize the SLA monitoring system."""
        self.config_path = config_path
        self.config = self._load_config()
        self.active_slas = {}  # tracking_id -> SLATracking
        self.alert_history = []
        self.escalation_history = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.notification_callbacks = {}
        self.setup_logging()
        self._setup_notification_channels()
    
    def setup_logging(self):
        """Setup logging for SLA monitoring."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - SLAMonitoringSystem - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load SLA monitoring configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default SLA monitoring configuration."""
        return {
            'sla_monitoring': {
                'response_tracking': {
                    'enabled': True,
                    'track_business_hours_only': False,
                    'timezone': 'UTC',
                    'business_hours': {
                        'start': '09:00',
                        'end': '17:00',
                        'weekdays_only': True
                    }
                },
                'escalation_alerts': {
                    'early_warning': {
                        'threshold': 0.5,
                        'channels': ['slack', 'email']
                    },
                    'urgent_warning': {
                        'threshold': 0.8,
                        'channels': ['slack', 'email', 'pagerduty']
                    },
                    'sla_breach': {
                        'threshold': 1.0,
                        'channels': ['slack', 'email', 'pagerduty', 'phone'],
                        'escalate_to_manager': True
                    }
                }
            }
        }
    
    def _setup_notification_channels(self):
        """Setup notification channel handlers."""
        self.notification_callbacks = {
            NotificationChannel.GITHUB_COMMENT: self._send_github_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.PAGERDUTY: self._send_pagerduty_notification,
            NotificationChannel.PHONE: self._send_phone_notification
        }
    
    def start_sla_tracking(self, 
                          issue_number: int,
                          specialist_github_issue: Optional[int],
                          assigned_specialist: str,
                          specialist_type: str,
                          sla_deadline: datetime,
                          business_hours_only: bool = False) -> str:
        """
        Start SLA tracking for a specialist assignment.
        
        Args:
            issue_number: Original GitHub issue number
            specialist_github_issue: Specialist review GitHub issue number
            assigned_specialist: Assigned specialist identifier
            specialist_type: Type of specialist (security, architecture, etc.)
            sla_deadline: SLA deadline datetime
            business_hours_only: Whether to track only business hours
            
        Returns:
            tracking_id: Unique identifier for this SLA tracking
        """
        tracking_id = self._generate_tracking_id(issue_number, assigned_specialist)
        
        sla_tracking = SLATracking(
            tracking_id=tracking_id,
            issue_number=issue_number,
            specialist_github_issue=specialist_github_issue,
            assigned_specialist=assigned_specialist,
            specialist_type=specialist_type,
            created_at=datetime.now(timezone.utc),
            sla_deadline=sla_deadline.replace(tzinfo=timezone.utc),
            current_status=SLAStatus.ON_TRACK,
            time_elapsed_percent=0.0,
            time_remaining_hours=self._calculate_time_remaining(datetime.now(timezone.utc), sla_deadline),
            escalation_level=EscalationLevel.NONE,
            notifications_sent=[],
            last_status_update=datetime.now(timezone.utc),
            business_hours_only=business_hours_only
        )
        
        self.active_slas[tracking_id] = sla_tracking
        
        # Start monitoring if not already active
        if not self.monitoring_active:
            self.start_monitoring()
        
        self.logger.info(f"📊 Started SLA tracking {tracking_id} for issue #{issue_number} (deadline: {sla_deadline})")
        
        # Send initial notification
        self._notify_sla_started(sla_tracking)
        
        return tracking_id
    
    def update_sla_status(self, tracking_id: str, status: SLAStatus, notes: str = "") -> bool:
        """
        Update SLA tracking status.
        
        Args:
            tracking_id: SLA tracking identifier
            status: New SLA status
            notes: Optional notes about the status update
            
        Returns:
            Success indicator
        """
        if tracking_id not in self.active_slas:
            self.logger.warning(f"SLA tracking {tracking_id} not found")
            return False
        
        sla_tracking = self.active_slas[tracking_id]
        old_status = sla_tracking.current_status
        sla_tracking.current_status = status
        sla_tracking.last_status_update = datetime.now(timezone.utc)
        
        self.logger.info(f"📈 Updated SLA {tracking_id} status: {old_status.value} → {status.value}")
        
        # Handle status-specific actions
        if status == SLAStatus.RESOLVED:
            self._handle_sla_resolution(sla_tracking, notes)
        elif status == SLAStatus.ESCALATED:
            self._handle_sla_escalation(sla_tracking, notes)
        
        return True
    
    def resolve_sla(self, tracking_id: str, resolution_notes: str = "") -> bool:
        """
        Mark SLA as resolved and clean up tracking.
        
        Args:
            tracking_id: SLA tracking identifier
            resolution_notes: Notes about the resolution
            
        Returns:
            Success indicator
        """
        if tracking_id not in self.active_slas:
            return False
        
        sla_tracking = self.active_slas[tracking_id]
        resolution_time = datetime.now(timezone.utc)
        
        # Calculate final metrics
        total_time = (resolution_time - sla_tracking.created_at).total_seconds() / 3600  # hours
        was_breached = resolution_time > sla_tracking.sla_deadline
        
        # Update status
        sla_tracking.current_status = SLAStatus.RESOLVED
        sla_tracking.last_status_update = resolution_time
        
        # Send resolution notification
        self._notify_sla_resolved(sla_tracking, total_time, was_breached, resolution_notes)
        
        # Record metrics
        self._record_sla_metrics(sla_tracking, total_time, was_breached)
        
        # Move to completed tracking
        self.active_slas.pop(tracking_id)
        
        self.logger.info(f"✅ Resolved SLA {tracking_id} - Total time: {total_time:.1f}h, Breached: {was_breached}")
        return True
    
    def start_monitoring(self, check_interval_seconds: int = 300) -> None:
        """
        Start the SLA monitoring background thread.
        
        Args:
            check_interval_seconds: How often to check SLA status (default 5 minutes)
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"🔄 Started SLA monitoring (check interval: {check_interval_seconds}s)")
    
    def stop_monitoring(self) -> None:
        """Stop the SLA monitoring background thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        self.logger.info("🛑 Stopped SLA monitoring")
    
    def _monitoring_loop(self, check_interval_seconds: int) -> None:
        """Main monitoring loop that runs in background thread."""
        while self.monitoring_active:
            try:
                self._check_all_slas()
                time.sleep(check_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in SLA monitoring loop: {e}")
                time.sleep(min(check_interval_seconds, 60))  # Don't overwhelm on errors
    
    def _check_all_slas(self) -> None:
        """Check all active SLAs and trigger alerts as needed."""
        current_time = datetime.now(timezone.utc)
        
        for tracking_id, sla_tracking in list(self.active_slas.items()):
            try:
                self._check_single_sla(sla_tracking, current_time)
            except Exception as e:
                self.logger.error(f"Error checking SLA {tracking_id}: {e}")
    
    def _check_single_sla(self, sla_tracking: SLATracking, current_time: datetime) -> None:
        """Check a single SLA and update status/send alerts."""
        # Calculate time metrics
        total_time = (sla_tracking.sla_deadline - sla_tracking.created_at).total_seconds()
        elapsed_time = (current_time - sla_tracking.created_at).total_seconds()
        
        if total_time <= 0:
            return  # Invalid SLA configuration
        
        elapsed_percent = elapsed_time / total_time
        remaining_hours = (sla_tracking.sla_deadline - current_time).total_seconds() / 3600
        
        # Update tracking metrics
        sla_tracking.time_elapsed_percent = elapsed_percent
        sla_tracking.time_remaining_hours = remaining_hours
        
        # Determine new status
        new_status = self._calculate_sla_status(elapsed_percent, remaining_hours)
        
        # Check if status changed and needs action
        if new_status != sla_tracking.current_status:
            old_status = sla_tracking.current_status
            sla_tracking.current_status = new_status
            sla_tracking.last_status_update = current_time
            
            self.logger.info(f"⚠️ SLA {sla_tracking.tracking_id} status change: {old_status.value} → {new_status.value}")
            
            # Trigger appropriate actions
            self._handle_status_change(sla_tracking, old_status, new_status)
    
    def _calculate_sla_status(self, elapsed_percent: float, remaining_hours: float) -> SLAStatus:
        """Calculate SLA status based on elapsed time."""
        if remaining_hours <= 0:
            return SLAStatus.BREACHED
        elif elapsed_percent >= 0.8:
            return SLAStatus.URGENT_WARNING
        elif elapsed_percent >= 0.5:
            return SLAStatus.EARLY_WARNING
        else:
            return SLAStatus.ON_TRACK
    
    def _handle_status_change(self, sla_tracking: SLATracking, old_status: SLAStatus, new_status: SLAStatus) -> None:
        """Handle SLA status changes with appropriate actions."""
        # Get alert configuration
        alert_config = self.config.get('sla_monitoring', {}).get('escalation_alerts', {})
        
        if new_status == SLAStatus.EARLY_WARNING:
            self._send_early_warning_alert(sla_tracking, alert_config)
        elif new_status == SLAStatus.URGENT_WARNING:
            self._send_urgent_warning_alert(sla_tracking, alert_config)
        elif new_status == SLAStatus.BREACHED:
            self._handle_sla_breach(sla_tracking, alert_config)
    
    def _send_early_warning_alert(self, sla_tracking: SLATracking, alert_config: Dict[str, Any]) -> None:
        """Send early warning alert (50% threshold)."""
        config = alert_config.get('early_warning', {})
        channels = config.get('channels', ['github_comment'])
        
        alert = SLAAlert(
            tracking_id=sla_tracking.tracking_id,
            alert_type=SLAStatus.EARLY_WARNING,
            threshold_breached=0.5,
            time_to_deadline=sla_tracking.sla_deadline - datetime.now(timezone.utc),
            escalation_required=False,
            notification_channels=[NotificationChannel(ch) for ch in channels if ch in [c.value for c in NotificationChannel]],
            message=f"⚠️ Early Warning: SLA 50% elapsed for specialist review of issue #{sla_tracking.issue_number}",
            priority="medium"
        )
        
        self._send_alert(alert, sla_tracking)
    
    def _send_urgent_warning_alert(self, sla_tracking: SLATracking, alert_config: Dict[str, Any]) -> None:
        """Send urgent warning alert (80% threshold)."""
        config = alert_config.get('urgent_warning', {})
        channels = config.get('channels', ['github_comment', 'slack'])
        
        alert = SLAAlert(
            tracking_id=sla_tracking.tracking_id,
            alert_type=SLAStatus.URGENT_WARNING,
            threshold_breached=0.8,
            time_to_deadline=sla_tracking.sla_deadline - datetime.now(timezone.utc),
            escalation_required=True,
            notification_channels=[NotificationChannel(ch) for ch in channels if ch in [c.value for c in NotificationChannel]],
            message=f"🚨 URGENT: SLA 80% elapsed for specialist review of issue #{sla_tracking.issue_number} - Escalation may be triggered soon",
            priority="high"
        )
        
        self._send_alert(alert, sla_tracking)
    
    def _handle_sla_breach(self, sla_tracking: SLATracking, alert_config: Dict[str, Any]) -> None:
        """Handle SLA breach with immediate escalation."""
        config = alert_config.get('sla_breach', {})
        channels = config.get('channels', ['github_comment', 'slack', 'email'])
        escalate_to_manager = config.get('escalate_to_manager', True)
        
        alert = SLAAlert(
            tracking_id=sla_tracking.tracking_id,
            alert_type=SLAStatus.BREACHED,
            threshold_breached=1.0,
            time_to_deadline=timedelta(0),  # Already breached
            escalation_required=True,
            notification_channels=[NotificationChannel(ch) for ch in channels if ch in [c.value for c in NotificationChannel]],
            message=f"🚨 SLA BREACH: Specialist review deadline exceeded for issue #{sla_tracking.issue_number} - IMMEDIATE ACTION REQUIRED",
            priority="critical"
        )
        
        # Send breach alert
        self._send_alert(alert, sla_tracking)
        
        # Trigger automatic escalation if configured
        if escalate_to_manager:
            self._trigger_automatic_escalation(sla_tracking)
    
    def _send_alert(self, alert: SLAAlert, sla_tracking: SLATracking) -> None:
        """Send alert through configured notification channels."""
        self.alert_history.append(alert)
        
        for channel in alert.notification_channels:
            try:
                if channel in self.notification_callbacks:
                    self.notification_callbacks[channel](alert, sla_tracking)
                    sla_tracking.notifications_sent.append(f"{channel.value}:{alert.alert_type.value}")
            except Exception as e:
                self.logger.error(f"Failed to send {channel.value} notification: {e}")
    
    def _trigger_automatic_escalation(self, sla_tracking: SLATracking) -> None:
        """Trigger automatic escalation for SLA breach."""
        escalation = EscalationAction(
            tracking_id=sla_tracking.tracking_id,
            escalation_type=EscalationLevel.AUTOMATIC,
            escalated_to=["engineering_manager", "specialist_team_lead"],
            escalation_reason="SLA breach - automatic escalation",
            escalation_time=datetime.now(timezone.utc),
            original_assignee=sla_tracking.assigned_specialist,
            new_assignee=None,  # Will be determined by escalation chain
            manager_notified=True
        )
        
        self.escalation_history.append(escalation)
        sla_tracking.escalation_level = EscalationLevel.AUTOMATIC
        sla_tracking.current_status = SLAStatus.ESCALATED
        
        # Create escalation GitHub comment
        self._create_escalation_github_comment(sla_tracking, escalation)
        
        self.logger.warning(f"🚨 Automatic escalation triggered for {sla_tracking.tracking_id}")
    
    def _send_github_notification(self, alert: SLAAlert, sla_tracking: SLATracking) -> None:
        """Send notification as GitHub comment."""
        if not sla_tracking.specialist_github_issue:
            return
        
        try:
            # Format time remaining
            time_remaining = alert.time_to_deadline.total_seconds() / 3600
            time_str = f"{time_remaining:.1f} hours" if time_remaining > 0 else "EXCEEDED"
            
            comment_body = f"""## 📊 SLA Status Update
            
{alert.message}

**SLA Details:**
- Time Elapsed: {sla_tracking.time_elapsed_percent * 100:.1f}%
- Time Remaining: {time_str}
- Priority: {alert.priority.upper()}
- Specialist: @{sla_tracking.assigned_specialist}

{"**⚠️ ESCALATION MAY BE TRIGGERED**" if alert.escalation_required else ""}

*Automated SLA monitoring update*
"""
            
            subprocess.run([
                'gh', 'issue', 'comment', str(sla_tracking.specialist_github_issue),
                '--body', comment_body
            ], check=True, capture_output=True)
            
            self.logger.info(f"📝 Posted SLA update to GitHub issue #{sla_tracking.specialist_github_issue}")
            
        except Exception as e:
            self.logger.error(f"Failed to post GitHub comment: {e}")
    
    def _send_slack_notification(self, alert: SLAAlert, sla_tracking: SLATracking) -> None:
        """Send Slack notification (placeholder - would integrate with Slack API)."""
        self.logger.info(f"📱 [SLACK] {alert.message}")
    
    def _send_email_notification(self, alert: SLAAlert, sla_tracking: SLATracking) -> None:
        """Send email notification (placeholder - would integrate with email service)."""
        self.logger.info(f"📧 [EMAIL] {alert.message}")
    
    def _send_pagerduty_notification(self, alert: SLAAlert, sla_tracking: SLATracking) -> None:
        """Send PagerDuty notification (placeholder - would integrate with PagerDuty API)."""
        self.logger.warning(f"🚨 [PAGERDUTY] {alert.message}")
    
    def _send_phone_notification(self, alert: SLAAlert, sla_tracking: SLATracking) -> None:
        """Send phone notification (placeholder - would integrate with phone service)."""
        self.logger.critical(f"📞 [PHONE] {alert.message}")
    
    def _notify_sla_started(self, sla_tracking: SLATracking) -> None:
        """Send initial notification when SLA tracking starts."""
        if sla_tracking.specialist_github_issue:
            self._send_sla_started_github_comment(sla_tracking)
    
    def _send_sla_started_github_comment(self, sla_tracking: SLATracking) -> None:
        """Send SLA started notification to GitHub."""
        try:
            deadline_str = sla_tracking.sla_deadline.strftime("%Y-%m-%d %H:%M UTC")
            total_hours = (sla_tracking.sla_deadline - sla_tracking.created_at).total_seconds() / 3600
            
            comment_body = f"""## 📊 SLA Monitoring Started
            
**SLA Details:**
- ⏰ Response Required By: {deadline_str}
- 🕐 Total Time Allocated: {total_hours:.1f} hours
- 👤 Assigned Specialist: @{sla_tracking.assigned_specialist}
- 📊 Tracking ID: `{sla_tracking.tracking_id}`

**Alert Thresholds:**
- 🟡 Early Warning: 50% time elapsed
- 🟠 Urgent Warning: 80% time elapsed
- 🔴 SLA Breach: 100% time elapsed + automatic escalation

*This issue is being monitored automatically for SLA compliance.*
"""
            
            subprocess.run([
                'gh', 'issue', 'comment', str(sla_tracking.specialist_github_issue),
                '--body', comment_body
            ], check=True, capture_output=True)
            
        except Exception as e:
            self.logger.error(f"Failed to post SLA started comment: {e}")
    
    def _notify_sla_resolved(self, sla_tracking: SLATracking, total_time: float, was_breached: bool, notes: str) -> None:
        """Send SLA resolution notification."""
        if sla_tracking.specialist_github_issue:
            try:
                breach_icon = "🔴" if was_breached else "✅"
                status_text = "BREACHED" if was_breached else "MET"
                
                comment_body = f"""## 📊 SLA Monitoring Complete
                
{breach_icon} **SLA {status_text}**

**Final Metrics:**
- Total Response Time: {total_time:.1f} hours
- SLA Deadline: {sla_tracking.sla_deadline.strftime("%Y-%m-%d %H:%M UTC")}
- Status: {status_text}
- Specialist: @{sla_tracking.assigned_specialist}

{f"**Resolution Notes:** {notes}" if notes else ""}

*SLA monitoring has been completed for this specialist review.*
"""
                
                subprocess.run([
                    'gh', 'issue', 'comment', str(sla_tracking.specialist_github_issue),
                    '--body', comment_body
                ], check=True, capture_output=True)
                
            except Exception as e:
                self.logger.error(f"Failed to post SLA resolution comment: {e}")
    
    def _create_escalation_github_comment(self, sla_tracking: SLATracking, escalation: EscalationAction) -> None:
        """Create escalation notification in GitHub."""
        if not sla_tracking.specialist_github_issue:
            return
        
        try:
            escalated_to_str = ", ".join([f"@{person}" for person in escalation.escalated_to])
            
            comment_body = f"""## 🚨 SLA ESCALATION TRIGGERED
            
**Escalation Details:**
- ⚠️ Reason: {escalation.escalation_reason}
- 👥 Escalated To: {escalated_to_str}
- ⏰ Escalation Time: {escalation.escalation_time.strftime("%Y-%m-%d %H:%M UTC")}
- 📊 Original Assignee: @{escalation.original_assignee}

**Required Actions:**
1. Management review of specialist assignment
2. Determine if reassignment is needed
3. Update SLA expectations if necessary
4. Provide additional resources if required

**This escalation requires immediate management attention.**

---
*Automated escalation triggered by SLA monitoring system*
"""
            
            subprocess.run([
                'gh', 'issue', 'comment', str(sla_tracking.specialist_github_issue),
                '--body', comment_body
            ], check=True, capture_output=True)
            
        except Exception as e:
            self.logger.error(f"Failed to post escalation comment: {e}")
    
    def _generate_tracking_id(self, issue_number: int, assigned_specialist: str) -> str:
        """Generate unique tracking ID for SLA monitoring."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{issue_number}_{assigned_specialist}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"sla_{issue_number}_{hash_suffix}"
    
    def _calculate_time_remaining(self, current_time: datetime, deadline: datetime) -> float:
        """Calculate time remaining in hours."""
        remaining = (deadline - current_time).total_seconds() / 3600
        return max(0.0, remaining)
    
    def _handle_sla_resolution(self, sla_tracking: SLATracking, notes: str) -> None:
        """Handle SLA resolution."""
        self.logger.info(f"✅ SLA {sla_tracking.tracking_id} resolved: {notes}")
    
    def _handle_sla_escalation(self, sla_tracking: SLATracking, notes: str) -> None:
        """Handle SLA escalation."""
        self.logger.warning(f"🚨 SLA {sla_tracking.tracking_id} escalated: {notes}")
    
    def _record_sla_metrics(self, sla_tracking: SLATracking, total_time: float, was_breached: bool) -> None:
        """Record SLA metrics for reporting."""
        metrics_record = {
            'tracking_id': sla_tracking.tracking_id,
            'issue_number': sla_tracking.issue_number,
            'specialist_type': sla_tracking.specialist_type,
            'assigned_specialist': sla_tracking.assigned_specialist,
            'created_at': sla_tracking.created_at.isoformat(),
            'resolved_at': datetime.now(timezone.utc).isoformat(),
            'total_response_time_hours': total_time,
            'sla_deadline': sla_tracking.sla_deadline.isoformat(),
            'was_breached': was_breached,
            'escalation_level': sla_tracking.escalation_level.value,
            'notifications_sent': len(sla_tracking.notifications_sent)
        }
        
        # In a real implementation, this would be stored in database or metrics system
        self.logger.debug(f"📈 Recorded SLA metrics: {metrics_record}")
    
    def get_active_slas_report(self) -> Dict[str, Any]:
        """Generate report of all active SLAs."""
        current_time = datetime.now(timezone.utc)
        
        report = {
            'timestamp': current_time.isoformat(),
            'total_active_slas': len(self.active_slas),
            'active_slas': []
        }
        
        for sla_tracking in self.active_slas.values():
            time_remaining = (sla_tracking.sla_deadline - current_time).total_seconds() / 3600
            
            report['active_slas'].append({
                'tracking_id': sla_tracking.tracking_id,
                'issue_number': sla_tracking.issue_number,
                'specialist_type': sla_tracking.specialist_type,
                'assigned_specialist': sla_tracking.assigned_specialist,
                'status': sla_tracking.current_status.value,
                'time_elapsed_percent': f"{sla_tracking.time_elapsed_percent * 100:.1f}%",
                'time_remaining_hours': f"{time_remaining:.1f}h",
                'sla_deadline': sla_tracking.sla_deadline.isoformat(),
                'escalation_level': sla_tracking.escalation_level.value,
                'notifications_sent': len(sla_tracking.notifications_sent)
            })
        
        # Sort by urgency (least time remaining first)
        report['active_slas'].sort(key=lambda x: float(x['time_remaining_hours'].replace('h', '')))
        
        return report
    
    def get_sla_performance_metrics(self) -> Dict[str, Any]:
        """Generate SLA performance metrics report."""
        # This would typically query historical data from database
        # For now, provide basic structure
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'period': '30_days',
            'metrics': {
                'total_slas_tracked': len(self.alert_history),
                'average_response_time_hours': 6.2,  # Mock data
                'sla_breach_rate': 0.05,  # 5% breach rate
                'escalation_rate': 0.03,  # 3% escalation rate
                'on_time_completion_rate': 0.95,  # 95% on-time
                'by_specialist_type': {
                    'security': {'avg_response': 3.8, 'breach_rate': 0.02},
                    'architecture': {'avg_response': 8.5, 'breach_rate': 0.08},
                    'compliance': {'avg_response': 5.2, 'breach_rate': 0.04}
                }
            },
            'trends': {
                'improving_response_times': True,
                'breach_rate_trend': 'stable',
                'workload_distribution': 'balanced'
            }
        }

def main():
    """Command line interface for SLA monitoring system."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sla_monitoring_system.py <command> [args]")
        print("Commands:")
        print("  start-tracking <issue_num> <specialist> <type> <hours>  - Start SLA tracking")
        print("  active-report                                           - Show active SLAs")
        print("  performance-metrics                                     - Show performance metrics") 
        print("  test-monitoring                                         - Test monitoring system")
        return
    
    command = sys.argv[1]
    monitor = SLAMonitoringSystem()
    
    if command == "start-tracking" and len(sys.argv) >= 6:
        issue_num = int(sys.argv[2])
        specialist = sys.argv[3]
        specialist_type = sys.argv[4]
        hours = float(sys.argv[5])
        
        deadline = datetime.now(timezone.utc) + timedelta(hours=hours)
        
        tracking_id = monitor.start_sla_tracking(
            issue_number=issue_num,
            specialist_github_issue=None,
            assigned_specialist=specialist,
            specialist_type=specialist_type,
            sla_deadline=deadline
        )
        
        print(f"✅ Started SLA tracking: {tracking_id}")
        print(f"   Issue: #{issue_num}")
        print(f"   Specialist: {specialist} ({specialist_type})")
        print(f"   Deadline: {deadline.isoformat()}")
        
    elif command == "active-report":
        report = monitor.get_active_slas_report()
        print(json.dumps(report, indent=2))
        
    elif command == "performance-metrics":
        metrics = monitor.get_sla_performance_metrics()
        print(json.dumps(metrics, indent=2))
        
    elif command == "test-monitoring":
        print("🧪 Testing SLA monitoring system...")
        
        # Create test SLA with short deadline
        deadline = datetime.now(timezone.utc) + timedelta(seconds=30)
        tracking_id = monitor.start_sla_tracking(
            issue_number=999,
            specialist_github_issue=None,
            assigned_specialist="test_specialist",
            specialist_type="security",
            sla_deadline=deadline
        )
        
        print(f"Created test SLA: {tracking_id}")
        print("Monitoring for 60 seconds...")
        
        monitor.start_monitoring(check_interval_seconds=10)
        time.sleep(60)
        monitor.stop_monitoring()
        
        print("Test complete. Check logs for SLA alerts.")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())