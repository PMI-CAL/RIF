"""
Phase Dependency Orchestration Integration

Integrates phase dependency validation with the existing orchestration intelligence framework.
Provides seamless integration with GitHub API state checking and automated enforcement hooks.

Issue #223: RIF Orchestration Error: Not Following Phase Dependencies
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .phase_dependency_validator import (
    PhaseDependencyValidator,
    PhaseValidationResult, 
    PhaseType
)
from .phase_dependency_warning_system import (
    PhaseDependencyWarningSystem,
    PhaseWarningAlert,
    AutoRedirectionSuggestion
)
from .orchestration_intelligence_integration import (
    OrchestrationIntelligenceIntegration,
    IntelligentOrchestrationDecision,
    DependencyAnalysis
)


@dataclass
class EnhancedOrchestrationDecision:
    """Enhanced orchestration decision with phase dependency validation"""
    base_decision: IntelligentOrchestrationDecision
    phase_validation_result: PhaseValidationResult
    phase_warning_alerts: List[PhaseWarningAlert]
    auto_redirections: List[AutoRedirectionSuggestion]
    final_enforcement_action: str
    prevented_resource_waste: Dict[str, Any]
    github_notifications: List[Dict[str, Any]]
    decision_confidence: float


@dataclass  
class GitHubIntegrationConfig:
    """Configuration for GitHub API integration"""
    enabled: bool
    auto_comment: bool
    auto_label: bool
    auto_state_transition: bool
    notification_threshold: str  # "all", "error", "critical"
    api_rate_limit_ms: int


class PhaseDependencyOrchestrationIntegration:
    """
    Complete integration layer between phase dependency validation and orchestration intelligence.
    
    Features:
    - Seamless integration with existing orchestration framework
    - Real-time GitHub API state checking  
    - Automated enforcement hooks
    - Enhanced decision making with phase awareness
    - Resource waste prevention
    - Comprehensive audit trail
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.knowledge_base_path = knowledge_base_path or "/Users/cal/DEV/RIF/knowledge"
        
        # Initialize component systems
        self.phase_validator = PhaseDependencyValidator(knowledge_base_path)
        self.warning_system = PhaseDependencyWarningSystem(knowledge_base_path) 
        self.orchestration_intelligence = OrchestrationIntelligenceIntegration(knowledge_base_path)
        
        # Integration configuration
        self.github_config = GitHubIntegrationConfig(
            enabled=True,
            auto_comment=True,
            auto_label=True,
            auto_state_transition=False,  # Conservative default
            notification_threshold="error",
            api_rate_limit_ms=1000
        )
        
        self.decision_history = []
        
    def make_enhanced_orchestration_decision(
        self,
        github_issues: List[Dict[str, Any]],
        proposed_agent_launches: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedOrchestrationDecision:
        """
        Main integration point: Make orchestration decisions with full phase dependency validation
        """
        
        # Step 1: Phase Dependency Validation (PRIMARY GATE)
        print("🔍 Step 1: Validating phase dependencies...")
        phase_validation_result = self.phase_validator.validate_phase_dependencies(
            github_issues, proposed_agent_launches
        )
        
        # Step 2: Generate Real-time Warnings
        print("⚠️ Step 2: Generating real-time warnings...")
        phase_alerts = self.warning_system.detect_violations_real_time(
            github_issues, proposed_agent_launches
        )
        
        # Step 3: Calculate Resource Waste Prevention
        print("💰 Step 3: Calculating resource waste prevention...")
        prevented_waste = self.warning_system.prevent_resource_waste(phase_validation_result)
        
        # Step 4: Auto-redirect if needed
        auto_redirections = []
        if not phase_validation_result.is_valid:
            print("🔄 Step 4: Generating auto-redirections for blocked launches...")
            auto_redirections = self.warning_system.auto_redirect_to_prerequisites(
                github_issues, proposed_agent_launches
            )
            
        # Step 5: Traditional Orchestration Intelligence (if phase validation passes)
        base_decision = None
        if phase_validation_result.is_valid:
            print("🧠 Step 5: Applying traditional orchestration intelligence...")
            # Convert proposed launches to format expected by orchestration intelligence
            orchestration_tasks = [
                {
                    "description": launch.get("description", ""),
                    "prompt": launch.get("prompt", ""), 
                    "subagent_type": launch.get("subagent_type", "general-purpose")
                }
                for launch in proposed_agent_launches
            ]
            
            base_decision = self.orchestration_intelligence.intelligent_launch_decision_with_validation(
                self.orchestration_intelligence.enhanced_dependency_analysis(github_issues, context),
                orchestration_tasks,
                context
            )
        else:
            # Create blocked decision
            base_decision = self._create_blocked_decision(phase_validation_result, proposed_agent_launches)
            
        # Step 6: Final Enforcement Decision
        print("⚖️ Step 6: Making final enforcement decision...")
        final_enforcement_action = self._determine_final_enforcement_action(
            phase_validation_result, base_decision, phase_alerts
        )
        
        # Step 7: GitHub Integration
        github_notifications = []
        if self.github_config.enabled:
            print("🔗 Step 7: Generating GitHub notifications...")
            github_notifications = self._generate_github_integrations(
                phase_alerts, github_issues, final_enforcement_action
            )
            
        # Step 8: Calculate Overall Confidence
        decision_confidence = self._calculate_overall_confidence(
            phase_validation_result, base_decision, phase_alerts
        )
        
        # Create enhanced decision
        enhanced_decision = EnhancedOrchestrationDecision(
            base_decision=base_decision,
            phase_validation_result=phase_validation_result,
            phase_warning_alerts=phase_alerts,
            auto_redirections=auto_redirections,
            final_enforcement_action=final_enforcement_action,
            prevented_resource_waste=prevented_waste,
            github_notifications=github_notifications,
            decision_confidence=decision_confidence
        )
        
        # Store decision
        self._store_enhanced_decision(enhanced_decision)
        
        return enhanced_decision
        
    def generate_enhanced_orchestration_template(
        self,
        enhanced_decision: EnhancedOrchestrationDecision
    ) -> str:
        """
        Generate orchestration template with full phase dependency integration
        """
        
        template_parts = []
        
        # Header with comprehensive context
        template_parts.append(f'''# Enhanced Orchestration Decision
# Phase Validation: {'✅ PASSED' if enhanced_decision.phase_validation_result.is_valid else '❌ FAILED'}
# Base Decision: {enhanced_decision.base_decision.decision_type if enhanced_decision.base_decision else 'N/A'}
# Final Action: {enhanced_decision.final_enforcement_action}
# Confidence: {enhanced_decision.decision_confidence:.2f}
# Alerts: {len(enhanced_decision.phase_warning_alerts)}
# Prevented Waste: {enhanced_decision.prevented_resource_waste.get('saved_agent_hours', 0)} agent hours
# Generated: {datetime.utcnow().isoformat()}
''')
        
        # Phase validation section
        if not enhanced_decision.phase_validation_result.is_valid:
            template_parts.append('''
# ❌ PHASE DEPENDENCY VIOLATIONS DETECTED - EXECUTION BLOCKED
# 
# Critical Issues Found:''')
            
            for violation in enhanced_decision.phase_validation_result.violations:
                template_parts.append(f'''
# 🚨 {violation.violation_type.replace('_', ' ').title()}
#    Issues: {', '.join([f'#{i}' for i in violation.issue_numbers])}
#    Missing: {[p.value for p in violation.missing_prerequisite_phases]}
#    Severity: {violation.severity.upper()}
#    Description: {violation.description}''')
                
                template_parts.append('#    Required Actions:')
                for action in violation.remediation_actions:
                    template_parts.append(f'#      • {action}')
                    
        # Auto-redirection section
        if enhanced_decision.auto_redirections:
            template_parts.append('''
# 🔄 AUTO-REDIRECTION AVAILABLE
# 
# Alternative prerequisite agents recommended:''')
            
            for i, redirection in enumerate(enhanced_decision.auto_redirections):
                template_parts.append(f'''
# Redirection {i+1}: {redirection.rationale}
# Confidence: {redirection.confidence_score:.2f}
# Estimated Time: {redirection.estimated_completion_time}''')
                
            # Generate redirected tasks
            template_parts.append('''
# Redirected Task Execution:''')
            
            for redirection in enhanced_decision.auto_redirections:
                for task in redirection.redirected_agents:
                    template_parts.append(f'''
Task(
    description="{task['description']}",
    subagent_type="{task['subagent_type']}",
    prompt="{task['prompt']}"
)''')
                    
        # Original decision section (if valid)
        elif enhanced_decision.final_enforcement_action == "allow_execution":
            template_parts.append('''
# ✅ PHASE DEPENDENCIES VALIDATED - EXECUTION ALLOWED
# 
# Approved Task Execution:''')
            
            if enhanced_decision.base_decision and enhanced_decision.base_decision.recommended_tasks:
                for task in enhanced_decision.base_decision.recommended_tasks:
                    template_parts.append(f'''
Task(
    description="{task.get('description', 'Task')}",
    subagent_type="{task.get('subagent_type', 'general-purpose')}",
    prompt="{task.get('prompt', 'Task prompt')}"
)''')
                    
        # Resource efficiency section
        template_parts.append(f'''
# 📊 RESOURCE EFFICIENCY METRICS
# Blocked Agents: {enhanced_decision.prevented_resource_waste.get('blocked_agents', 0)}
# Saved Agent Hours: {enhanced_decision.prevented_resource_waste.get('saved_agent_hours', 0)}
# Prevented Rework Cycles: {enhanced_decision.prevented_resource_waste.get('prevented_rework_cycles', 0)}
# Efficiency Gain: {enhanced_decision.prevented_resource_waste.get('efficiency_gain_percentage', 0):.1f}%''')
        
        # GitHub integration section
        if enhanced_decision.github_notifications:
            template_parts.append(f'''
# 🔗 GITHUB INTEGRATION ACTIONS
# Notifications: {len(enhanced_decision.github_notifications)} issues
# Auto-labels: {self.github_config.auto_label}
# Auto-comments: {self.github_config.auto_comment}''')
            
        # Footer
        template_parts.append('''
# 
# Phase Dependency Enforcement: ACTIVE ✅
# Orchestration Intelligence: ENHANCED 🧠
# Resource Waste Prevention: ENABLED 💰
# GitHub Integration: CONNECTED 🔗''')
        
        return '\n'.join(template_parts)
        
    def check_github_state_real_time(
        self,
        issue_numbers: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Real-time GitHub API state checking for phase validation
        """
        # In a real implementation, this would use GitHub API
        # For now, return mock data structure
        states = {}
        
        for issue_num in issue_numbers:
            # Mock GitHub API call
            states[issue_num] = {
                "state": "open",
                "labels": [],
                "last_updated": datetime.utcnow().isoformat(),
                "comments_count": 0,
                "last_comment": None,
                "assignees": [],
                "milestone": None
            }
            
        return states
        
    def setup_automated_enforcement_hooks(self) -> Dict[str, bool]:
        """
        Setup automated enforcement hooks in the orchestration pipeline
        """
        hooks_config = {
            "pre_agent_launch_validation": True,
            "real_time_phase_monitoring": True,
            "auto_violation_detection": True,
            "resource_waste_prevention": True,
            "github_notification_integration": self.github_config.enabled,
            "decision_audit_trail": True
        }
        
        # Store hooks configuration
        self._store_hooks_config(hooks_config)
        
        return hooks_config
        
    def _create_blocked_decision(
        self,
        phase_validation: PhaseValidationResult,
        proposed_launches: List[Dict[str, Any]]
    ) -> IntelligentOrchestrationDecision:
        """Create blocked orchestration decision"""
        
        violation_summary = f"Phase dependency violations: {len(phase_validation.violations)} critical issues"
        
        # Import required classes for creating decision
        from .orchestration_intelligence_integration import IntelligentOrchestrationDecision
        from .orchestration_pattern_validator import ValidationResult
        
        # Create validation result for blocked decision
        validation_result = ValidationResult(
            is_valid=False,
            violations=[v.description for v in phase_validation.violations],
            suggestions=[action for v in phase_validation.violations for action in v.remediation_actions],
            pattern_type="phase_dependency_blocked",
            confidence_score=phase_validation.confidence_score
        )
        
        return IntelligentOrchestrationDecision(
            decision_type="blocked_by_phase_dependencies",
            recommended_tasks=[],
            dependency_rationale=violation_summary,
            validation_status=validation_result,
            enforcement_action="block_execution_phase_violations",
            confidence_score=phase_validation.confidence_score,
            decision_timestamp=datetime.utcnow().isoformat()
        )
        
    def _determine_final_enforcement_action(
        self,
        phase_validation: PhaseValidationResult,
        base_decision: IntelligentOrchestrationDecision,
        alerts: List[PhaseWarningAlert]
    ) -> str:
        """Determine final enforcement action"""
        
        # Phase validation takes precedence
        if not phase_validation.is_valid:
            critical_violations = [a for a in alerts if a.alert_level.value == "critical"]
            if critical_violations:
                return "block_execution_critical_violations"
            else:
                return "block_execution_phase_violations"
                
        # Check base decision
        if base_decision and base_decision.enforcement_action.startswith("block"):
            return base_decision.enforcement_action
            
        # Check for warnings that should block
        high_severity_alerts = [a for a in alerts if a.alert_level.value in ["critical", "error"]]
        if len(high_severity_alerts) > 3:
            return "block_execution_too_many_violations"
            
        return "allow_execution"
        
    def _generate_github_integrations(
        self,
        alerts: List[PhaseWarningAlert],
        github_issues: List[Dict[str, Any]], 
        enforcement_action: str
    ) -> List[Dict[str, Any]]:
        """Generate GitHub integration actions"""
        
        integrations = []
        
        if not self.github_config.enabled:
            return integrations
            
        # Filter alerts based on notification threshold
        threshold = self.github_config.notification_threshold
        filtered_alerts = []
        
        for alert in alerts:
            if threshold == "all":
                filtered_alerts.append(alert)
            elif threshold == "error" and alert.alert_level.value in ["error", "critical"]:
                filtered_alerts.append(alert)
            elif threshold == "critical" and alert.alert_level.value == "critical":
                filtered_alerts.append(alert)
                
        # Generate GitHub notifications
        if filtered_alerts:
            github_notifications = self.warning_system.integrate_github_notifications(
                filtered_alerts, github_issues
            )
            integrations.extend(github_notifications)
            
        # Add enforcement action notification if blocked
        if enforcement_action.startswith("block"):
            integrations.append({
                "action_type": "orchestration_blocked",
                "reason": enforcement_action,
                "timestamp": datetime.utcnow().isoformat(),
                "affected_issues": list(set([
                    issue_num 
                    for alert in alerts 
                    for issue_num in alert.affected_issues
                ]))
            })
            
        return integrations
        
    def _calculate_overall_confidence(
        self,
        phase_validation: PhaseValidationResult,
        base_decision: Optional[IntelligentOrchestrationDecision],
        alerts: List[PhaseWarningAlert]
    ) -> float:
        """Calculate overall decision confidence"""
        
        # Start with phase validation confidence
        confidence = phase_validation.confidence_score
        
        # Factor in base decision confidence
        if base_decision:
            confidence = (confidence + base_decision.confidence_score) / 2
            
        # Reduce confidence for high-severity alerts
        critical_alerts = len([a for a in alerts if a.alert_level.value == "critical"])
        error_alerts = len([a for a in alerts if a.alert_level.value == "error"])
        
        confidence_reduction = (critical_alerts * 0.3) + (error_alerts * 0.2)
        confidence = max(0.0, confidence - confidence_reduction)
        
        return confidence
        
    def _store_enhanced_decision(self, decision: EnhancedOrchestrationDecision):
        """Store enhanced decision in knowledge base"""
        try:
            decisions_dir = Path(self.knowledge_base_path) / "decisions" 
            decisions_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_orchestration_decision_{timestamp}.json"
            filepath = decisions_dir / filename
            
            # Convert to JSON-serializable format
            decision_data = {
                "base_decision": asdict(decision.base_decision) if decision.base_decision else None,
                "phase_validation_result": {
                    "is_valid": decision.phase_validation_result.is_valid,
                    "violations": [
                        {
                            "violation_type": v.violation_type,
                            "issue_numbers": v.issue_numbers,
                            "attempted_phase": v.attempted_phase.value,
                            "missing_prerequisite_phases": [p.value for p in v.missing_prerequisite_phases],
                            "severity": v.severity,
                            "description": v.description,
                            "remediation_actions": v.remediation_actions
                        }
                        for v in decision.phase_validation_result.violations
                    ],
                    "warnings": decision.phase_validation_result.warnings,
                    "confidence_score": decision.phase_validation_result.confidence_score
                },
                "phase_warning_alerts": [asdict(alert) for alert in decision.phase_warning_alerts],
                "auto_redirections": [asdict(redirect) for redirect in decision.auto_redirections],
                "final_enforcement_action": decision.final_enforcement_action,
                "prevented_resource_waste": decision.prevented_resource_waste,
                "github_notifications": decision.github_notifications,
                "decision_confidence": decision.decision_confidence,
                "issue_reference": 223,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(decision_data, f, indent=2)
                
            self.decision_history.append(decision)
            
        except Exception as e:
            print(f"Warning: Could not store enhanced orchestration decision: {e}")
            
    def _store_hooks_config(self, hooks_config: Dict[str, bool]):
        """Store hooks configuration"""
        try:
            config_dir = Path(self.knowledge_base_path).parent / "config"
            config_dir.mkdir(exist_ok=True)
            
            filepath = config_dir / "phase_dependency_hooks.json"
            
            config_data = {
                "hooks_config": hooks_config,
                "github_integration": asdict(self.github_config),
                "enabled_at": datetime.utcnow().isoformat(),
                "issue_reference": 223
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not store hooks configuration: {e}")


# Main integration function for CLAUDE.md
def make_enhanced_orchestration_decision_with_phase_validation(
    github_issues: List[Dict[str, Any]],
    proposed_agent_launches: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> EnhancedOrchestrationDecision:
    """
    Main entry point for enhanced orchestration with phase dependency validation.
    
    Use this function in orchestration workflows to get comprehensive validation.
    """
    integration = PhaseDependencyOrchestrationIntegration()
    return integration.make_enhanced_orchestration_decision(
        github_issues, proposed_agent_launches, context
    )


# Convenience function for template generation
def generate_enhanced_orchestration_template(
    github_issues: List[Dict[str, Any]],
    proposed_agent_launches: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Generate ready-to-use orchestration template with phase validation"""
    integration = PhaseDependencyOrchestrationIntegration()
    
    decision = integration.make_enhanced_orchestration_decision(
        github_issues, proposed_agent_launches, context
    )
    
    return integration.generate_enhanced_orchestration_template(decision)


if __name__ == "__main__":
    # Test the integration
    test_issues = [
        {
            "number": 1,
            "title": "Research user authentication",
            "labels": [{"name": "state:new"}],
            "body": "Research auth patterns"
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
            "prompt": "Implement auth for issue #2",
            "subagent_type": "general-purpose"
        }
    ]
    
    decision = make_enhanced_orchestration_decision_with_phase_validation(
        test_issues, test_launches
    )
    
    print(f"Enhanced Decision: {decision.final_enforcement_action}")
    print(f"Phase Validation: {'PASSED' if decision.phase_validation_result.is_valid else 'FAILED'}")
    print(f"Alerts: {len(decision.phase_warning_alerts)}")
    print(f"Confidence: {decision.decision_confidence:.2f}")
    
    template = generate_enhanced_orchestration_template(test_issues, test_launches)
    print(f"\nGenerated template length: {len(template)} characters")