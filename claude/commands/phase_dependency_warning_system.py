"""
Phase Dependency Warning & Prevention System

Provides real-time violation detection, actionable warning messages, 
and automatic redirection to prerequisite phases.

Issue #223: RIF Orchestration Error: Not Following Phase Dependencies
"""

import json
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

from phase_dependency_validator import (
    PhaseDependencyValidator, 
    PhaseValidationResult,
    PhaseDependencyViolation,
    PhaseType
)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PhaseWarningAlert:
    """Real-time phase dependency warning alert"""
    alert_id: str
    alert_level: AlertLevel
    violation_type: str
    affected_issues: List[int]
    attempted_phase: str
    missing_phases: List[str]
    warning_message: str
    actionable_steps: List[str]
    auto_redirect_available: bool
    timestamp: str
    expires_at: Optional[str] = None


@dataclass
class AutoRedirectionSuggestion:
    """Automatic redirection to prerequisite phases"""
    original_request: Dict[str, Any]
    redirected_agents: List[Dict[str, Any]]
    rationale: str
    confidence_score: float
    estimated_completion_time: str
    follow_up_actions: List[str]


class PhaseDependencyWarningSystem:
    """
    Real-time warning and prevention system for phase dependency violations.
    
    Features:
    - Real-time violation detection during orchestration
    - Actionable warning messages with specific remediation steps
    - Automatic redirection to prerequisite phases
    - Prevention of resource waste on blocked work
    - Integration with GitHub issue notifications
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.validator = PhaseDependencyValidator(knowledge_base_path)
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        self.active_alerts = {}
        self.alert_history = []
        self.auto_redirect_enabled = True
        
    def detect_violations_real_time(
        self,
        github_issues: List[Dict[str, Any]],
        proposed_agent_launches: List[Dict[str, Any]]
    ) -> List[PhaseWarningAlert]:
        """
        Real-time detection of phase dependency violations
        """
        alerts = []
        
        # Run phase dependency validation
        validation_result = self.validator.validate_phase_dependencies(
            github_issues, proposed_agent_launches
        )
        
        # Convert violations to warning alerts
        for violation in validation_result.violations:
            alert = self._create_warning_alert(violation)
            alerts.append(alert)
            
            # Store active alert
            self.active_alerts[alert.alert_id] = alert
            
        # Add warning alerts for non-critical issues
        warning_alerts = self._generate_warning_alerts(validation_result.warnings, github_issues)
        alerts.extend(warning_alerts)
        
        # Store alerts in history
        self.alert_history.extend(alerts)
        
        return alerts
        
    def generate_actionable_messages(
        self, 
        alerts: List[PhaseWarningAlert]
    ) -> Dict[str, str]:
        """
        Generate user-friendly, actionable warning messages
        """
        messages = {}
        
        for alert in alerts:
            if alert.alert_level == AlertLevel.CRITICAL:
                message = self._generate_critical_message(alert)
            elif alert.alert_level == AlertLevel.ERROR:
                message = self._generate_error_message(alert)
            elif alert.alert_level == AlertLevel.WARNING:
                message = self._generate_warning_message(alert)
            else:
                message = self._generate_info_message(alert)
                
            messages[alert.alert_id] = message
            
        return messages
        
    def auto_redirect_to_prerequisites(
        self,
        github_issues: List[Dict[str, Any]],
        blocked_agent_launches: List[Dict[str, Any]]
    ) -> List[AutoRedirectionSuggestion]:
        """
        Automatically generate alternative agent launches for prerequisite phases
        """
        redirections = []
        
        if not self.auto_redirect_enabled:
            return redirections
            
        for blocked_launch in blocked_agent_launches:
            redirection = self._generate_prerequisite_redirection(blocked_launch, github_issues)
            if redirection:
                redirections.append(redirection)
                
        return redirections
        
    def prevent_resource_waste(
        self,
        validation_result: PhaseValidationResult,
        resource_budget: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate resource waste prevention metrics
        """
        prevented_waste = {
            "blocked_agents": len([v for v in validation_result.violations if v.severity in ["critical", "high"]]),
            "saved_agent_hours": 0,
            "prevented_rework_cycles": 0,
            "efficiency_gain_percentage": 0
        }
        
        # Estimate saved resources
        for violation in validation_result.violations:
            if violation.severity == "critical":
                prevented_waste["saved_agent_hours"] += 4  # Average implementation time
                prevented_waste["prevented_rework_cycles"] += 2  # Typical rework cycles
            elif violation.severity == "high":
                prevented_waste["saved_agent_hours"] += 2
                prevented_waste["prevented_rework_cycles"] += 1
                
        # Calculate efficiency gain
        total_potential_waste = prevented_waste["saved_agent_hours"]
        if total_potential_waste > 0:
            prevented_waste["efficiency_gain_percentage"] = min(100, (total_potential_waste / 8) * 100)
            
        return prevented_waste
        
    def integrate_github_notifications(
        self, 
        alerts: List[PhaseWarningAlert],
        github_issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate GitHub issue comments/notifications for phase dependency violations
        """
        notifications = []
        
        # Group alerts by issue
        issue_alerts = {}
        for alert in alerts:
            for issue_num in alert.affected_issues:
                if issue_num not in issue_alerts:
                    issue_alerts[issue_num] = []
                issue_alerts[issue_num].append(alert)
                
        # Generate notifications for each issue
        for issue_num, issue_alerts_list in issue_alerts.items():
            notification = self._generate_github_notification(issue_num, issue_alerts_list)
            notifications.append(notification)
            
        return notifications
        
    def _create_warning_alert(self, violation: PhaseDependencyViolation) -> PhaseWarningAlert:
        """Create warning alert from validation violation"""
        alert_level = self._map_severity_to_alert_level(violation.severity)
        alert_id = f"phase_violation_{int(time.time())}_{hash(str(violation.issue_numbers))}"
        
        # Generate actionable steps
        actionable_steps = self._generate_actionable_steps(violation)
        
        # Determine if auto-redirect is available
        auto_redirect = violation.severity in ["critical", "high"] and self.auto_redirect_enabled
        
        return PhaseWarningAlert(
            alert_id=alert_id,
            alert_level=alert_level,
            violation_type=violation.violation_type,
            affected_issues=violation.issue_numbers,
            attempted_phase=violation.attempted_phase.value,
            missing_phases=[p.value for p in violation.missing_prerequisite_phases],
            warning_message=violation.description,
            actionable_steps=actionable_steps,
            auto_redirect_available=auto_redirect,
            timestamp=datetime.utcnow().isoformat(),
            expires_at=(datetime.utcnow() + timedelta(hours=24)).isoformat()
        )
        
    def _generate_warning_alerts(
        self, 
        warnings: List[str],
        github_issues: List[Dict[str, Any]]
    ) -> List[PhaseWarningAlert]:
        """Generate warning-level alerts from validation warnings"""
        alerts = []
        
        for warning in warnings:
            # Extract issue numbers from warning
            import re
            issue_matches = re.findall(r"Issue #(\d+)", warning)
            issue_numbers = [int(match) for match in issue_matches]
            
            if issue_numbers:
                alert_id = f"phase_warning_{int(time.time())}_{hash(warning)}"
                
                alert = PhaseWarningAlert(
                    alert_id=alert_id,
                    alert_level=AlertLevel.WARNING,
                    violation_type="phase_transition_warning",
                    affected_issues=issue_numbers,
                    attempted_phase="unknown",
                    missing_phases=[],
                    warning_message=warning,
                    actionable_steps=["Review phase completion evidence", "Consider sequential approach"],
                    auto_redirect_available=False,
                    timestamp=datetime.utcnow().isoformat()
                )
                alerts.append(alert)
                
        return alerts
        
    def _map_severity_to_alert_level(self, severity: str) -> AlertLevel:
        """Map violation severity to alert level"""
        mapping = {
            "critical": AlertLevel.CRITICAL,
            "high": AlertLevel.ERROR,
            "medium": AlertLevel.WARNING,
            "low": AlertLevel.INFO
        }
        return mapping.get(severity.lower(), AlertLevel.WARNING)
        
    def _generate_actionable_steps(self, violation: PhaseDependencyViolation) -> List[str]:
        """Generate specific actionable steps for violation"""
        steps = []
        
        # Add phase-specific steps
        for missing_phase in violation.missing_prerequisite_phases:
            if missing_phase == PhaseType.ANALYSIS:
                steps.append("🔍 Launch RIF-Analyst to complete requirements analysis")
            elif missing_phase == PhaseType.PLANNING:
                steps.append("📋 Launch RIF-Planner to create detailed execution plan")  
            elif missing_phase == PhaseType.ARCHITECTURE:
                steps.append("🏗️ Launch RIF-Architect to design technical solution")
            elif missing_phase == PhaseType.IMPLEMENTATION:
                steps.append("💻 Launch RIF-Implementer to write and test code")
                
        # Add workflow steps
        steps.append("⏳ Wait for prerequisite phases to complete before retrying")
        steps.append("📊 Monitor phase completion in GitHub issue labels and comments")
        
        # Add prevention steps
        if violation.severity in ["critical", "high"]:
            steps.append("🚫 Do not launch implementation/validation agents for these issues")
            steps.append("🔄 Use sequential agent launching instead of parallel launching")
            
        return steps
        
    def _generate_critical_message(self, alert: PhaseWarningAlert) -> str:
        """Generate critical alert message"""
        issues_str = ", ".join([f"#{i}" for i in alert.affected_issues])
        missing_str = ", ".join(alert.missing_phases)
        
        message = f"""🚨 CRITICAL: Phase Dependency Violation Detected

**Issues Affected**: {issues_str}
**Attempted Phase**: {alert.attempted_phase.title()}
**Missing Prerequisites**: {missing_str}

**Impact**: This violation would waste agent resources and likely result in implementation failure.

**Immediate Actions Required**:"""
        
        for step in alert.actionable_steps:
            message += f"\n• {step}"
            
        if alert.auto_redirect_available:
            message += f"\n\n✨ **Auto-Redirect Available**: Alternative prerequisite agents can be launched automatically."
            
        return message
        
    def _generate_error_message(self, alert: PhaseWarningAlert) -> str:
        """Generate error alert message"""
        issues_str = ", ".join([f"#{i}" for i in alert.affected_issues])
        
        message = f"""❌ ERROR: Phase Dependency Issue

**Issues**: {issues_str}
**Problem**: {alert.warning_message}

**Recommended Actions**:"""
        
        for step in alert.actionable_steps:
            message += f"\n• {step}"
            
        return message
        
    def _generate_warning_message(self, alert: PhaseWarningAlert) -> str:
        """Generate warning alert message"""
        issues_str = ", ".join([f"#{i}" for i in alert.affected_issues])
        
        message = f"""⚠️ WARNING: Phase Dependency Concern

**Issues**: {issues_str}
**Concern**: {alert.warning_message}

**Suggested Actions**:"""
        
        for step in alert.actionable_steps:
            message += f"\n• {step}"
            
        return message
        
    def _generate_info_message(self, alert: PhaseWarningAlert) -> str:
        """Generate info alert message"""
        return f"ℹ️ INFO: {alert.warning_message}"
        
    def _generate_prerequisite_redirection(
        self, 
        blocked_launch: Dict[str, Any],
        github_issues: List[Dict[str, Any]]
    ) -> Optional[AutoRedirectionSuggestion]:
        """Generate redirection to prerequisite phases"""
        
        # Extract issue numbers from blocked launch
        import re
        description = blocked_launch.get("description", "")
        prompt = blocked_launch.get("prompt", "")
        text = f"{description} {prompt}"
        
        issue_matches = re.findall(r"#(\d+)", text)
        if not issue_matches:
            return None
            
        issue_numbers = [int(match) for match in issue_matches]
        
        # Find the issues and determine missing prerequisites
        redirected_agents = []
        missing_phases = set()
        
        for issue_num in issue_numbers:
            issue = next((i for i in github_issues if i.get("number") == issue_num), None)
            if not issue:
                continue
                
            # Determine what phases are missing
            for phase_type in [PhaseType.ANALYSIS, PhaseType.PLANNING, PhaseType.ARCHITECTURE]:
                if not self.validator.validate_phase_completion(issue, phase_type):
                    missing_phases.add(phase_type)
                    
        # Generate redirected agents for missing phases
        for phase_type in missing_phases:
            agent_name = self._get_agent_for_phase(phase_type)
            if agent_name:
                for issue_num in issue_numbers:
                    redirected_agents.append({
                        "description": f"{agent_name}: Complete {phase_type.value} for issue #{issue_num}",
                        "prompt": f"You are {agent_name}. Complete {phase_type.value} phase for issue #{issue_num}. Follow all instructions in claude/agents/{agent_name.lower().replace('-', '_')}.md.",
                        "subagent_type": "general-purpose"
                    })
                    
        if not redirected_agents:
            return None
            
        return AutoRedirectionSuggestion(
            original_request=blocked_launch,
            redirected_agents=redirected_agents,
            rationale=f"Redirected to complete missing prerequisite phases: {[p.value for p in missing_phases]}",
            confidence_score=0.9,
            estimated_completion_time=self._estimate_completion_time(missing_phases),
            follow_up_actions=[
                f"After {len(missing_phases)} prerequisite phases complete, retry original request",
                "Monitor GitHub issue states for phase completion",
                "Validate phase dependencies before launching implementation agents"
            ]
        )
        
    def _get_agent_for_phase(self, phase_type: PhaseType) -> Optional[str]:
        """Get appropriate agent for phase"""
        mapping = {
            PhaseType.ANALYSIS: "RIF-Analyst",
            PhaseType.PLANNING: "RIF-Planner", 
            PhaseType.ARCHITECTURE: "RIF-Architect",
            PhaseType.IMPLEMENTATION: "RIF-Implementer",
            PhaseType.VALIDATION: "RIF-Validator",
            PhaseType.DOCUMENTATION: "RIF-Documenter",
            PhaseType.LEARNING: "RIF-Learner"
        }
        return mapping.get(phase_type)
        
    def _estimate_completion_time(self, missing_phases: Set[PhaseType]) -> str:
        """Estimate time to complete missing phases"""
        phase_times = {
            PhaseType.ANALYSIS: 1,      # 1 hour
            PhaseType.PLANNING: 1,      # 1 hour  
            PhaseType.ARCHITECTURE: 2,  # 2 hours
            PhaseType.IMPLEMENTATION: 4, # 4 hours
            PhaseType.VALIDATION: 2     # 2 hours
        }
        
        total_hours = sum(phase_times.get(phase, 1) for phase in missing_phases)
        
        if total_hours <= 2:
            return f"{total_hours} hour{'s' if total_hours > 1 else ''}"
        elif total_hours <= 8:
            return f"{total_hours} hours"
        else:
            return f"{total_hours // 8} day{'s' if total_hours > 8 else ''}"
            
    def _generate_github_notification(
        self, 
        issue_num: int, 
        alerts: List[PhaseWarningAlert]
    ) -> Dict[str, Any]:
        """Generate GitHub notification for issue"""
        
        # Determine highest alert level
        max_level = max(alerts, key=lambda a: a.alert_level.value).alert_level
        
        if max_level == AlertLevel.CRITICAL:
            title = "🚨 CRITICAL: Phase Dependency Violation"
        elif max_level == AlertLevel.ERROR:
            title = "❌ Phase Dependency Error"
        else:
            title = "⚠️ Phase Dependency Warning"
            
        comment_body = f"{title}\n\n"
        
        for alert in alerts:
            comment_body += f"**{alert.violation_type.replace('_', ' ').title()}**\n"
            comment_body += f"- {alert.warning_message}\n"
            
            if alert.actionable_steps:
                comment_body += "\n**Required Actions:**\n"
                for step in alert.actionable_steps:
                    comment_body += f"- {step}\n"
                    
            comment_body += "\n"
            
        comment_body += f"*Generated by Phase Dependency Warning System at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*"
        
        return {
            "issue_number": issue_num,
            "comment_type": "phase_dependency_warning",
            "comment_body": comment_body,
            "labels_to_add": [f"phase-dependency-{max_level.value}"],
            "priority": max_level.value
        }
        
    def get_active_alerts(self) -> List[PhaseWarningAlert]:
        """Get currently active alerts"""
        now = datetime.utcnow()
        active = []
        
        for alert in self.active_alerts.values():
            if alert.expires_at:
                expires = datetime.fromisoformat(alert.expires_at.replace('Z', '+00:00'))
                if now < expires:
                    active.append(alert)
                    
        return active
        
    def clear_resolved_alerts(self, resolved_issues: List[int]):
        """Clear alerts for resolved issues"""
        to_remove = []
        
        for alert_id, alert in self.active_alerts.items():
            if any(issue in resolved_issues for issue in alert.affected_issues):
                to_remove.append(alert_id)
                
        for alert_id in to_remove:
            del self.active_alerts[alert_id]


# Convenience functions
def detect_phase_violations(
    github_issues: List[Dict[str, Any]],
    proposed_launches: List[Dict[str, Any]]
) -> List[PhaseWarningAlert]:
    """Quick function to detect phase violations"""
    warning_system = PhaseDependencyWarningSystem()
    return warning_system.detect_violations_real_time(github_issues, proposed_launches)


def generate_prevention_report(
    validation_result: PhaseValidationResult
) -> Dict[str, Any]:
    """Generate resource waste prevention report"""
    warning_system = PhaseDependencyWarningSystem()
    return warning_system.prevent_resource_waste(validation_result)


if __name__ == "__main__":
    # Test the warning system
    test_issues = [
        {
            "number": 1,
            "title": "Research authentication patterns",
            "labels": [{"name": "state:new"}],
            "body": "Research auth approaches"
        },
        {
            "number": 2,  
            "title": "Implement authentication",
            "labels": [{"name": "state:implementing"}],
            "body": "Build auth system"
        }
    ]
    
    test_launches = [
        {
            "description": "RIF-Implementer: Authentication implementation",
            "prompt": "Implement authentication for issue #2",
            "subagent_type": "general-purpose"
        }
    ]
    
    warning_system = PhaseDependencyWarningSystem()
    alerts = warning_system.detect_violations_real_time(test_issues, test_launches)
    
    print(f"Detected {len(alerts)} alerts")
    for alert in alerts:
        print(f"- {alert.alert_level.value.upper()}: {alert.warning_message}")
        
    messages = warning_system.generate_actionable_messages(alerts)
    for alert_id, message in messages.items():
        print(f"\nMessage for {alert_id}:")
        print(message)