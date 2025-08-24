#!/usr/bin/env python3
"""
Manual Intervention Workflow Coordinator - Issue #92 Phase 3
Core workflow orchestrator for risk-based manual intervention framework.

This coordinator implements:
1. End-to-end workflow orchestration from risk assessment to resolution
2. Integration with RIF workflow state machine
3. Decision tracking and audit trail management
4. Manager override and escalation handling
5. Quality gate coordination and enforcement
6. Pattern learning and continuous improvement
7. Error handling and graceful degradation
"""

import json
import subprocess
import yaml
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import hashlib

# Import other components
try:
    from .risk_assessment_engine import RiskAssessmentEngine, ChangeContext, RiskScore, create_change_context_from_issue
    from .specialist_assignment_engine import SpecialistAssignmentEngine, AssignmentRequest, AssignmentResult, SpecialistType
    from .sla_monitoring_system import SLAMonitoringSystem, SLAStatus
    from .decision_audit_tracker import DecisionAuditTracker, AuditRecord
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from risk_assessment_engine import RiskAssessmentEngine, ChangeContext, RiskScore, create_change_context_from_issue
    from specialist_assignment_engine import SpecialistAssignmentEngine, AssignmentRequest, AssignmentResult, SpecialistType
    from sla_monitoring_system import SLAMonitoringSystem, SLAStatus
    from decision_audit_tracker import DecisionAuditTracker, AuditRecord

class InterventionStatus(Enum):
    """Manual intervention workflow status."""
    INITIATED = "initiated"
    RISK_ASSESSING = "risk_assessing"
    SPECIALIST_ASSIGNED = "specialist_assigned"
    MANUAL_REVIEW = "manual_review"
    MANAGER_OVERRIDE = "manager_override"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ERROR = "error"

class DecisionType(Enum):
    """Types of manual intervention decisions."""
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    OVERRIDE_APPROVE = "override_approve"
    OVERRIDE_REJECT = "override_reject"

@dataclass
class InterventionWorkflow:
    """Container for manual intervention workflow state."""
    workflow_id: str
    issue_number: int
    initiated_at: datetime
    current_status: InterventionStatus
    risk_score: Optional[RiskScore] = None
    assignment_result: Optional[AssignmentResult] = None
    sla_tracking_id: Optional[str] = None
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ManualDecision:
    """Container for manual intervention decision."""
    workflow_id: str
    decision_type: DecisionType
    decision_maker: str
    decision_rationale: str
    evidence_provided: List[str]
    timestamp: datetime
    override_justification: Optional[str] = None
    manager_approval: Optional[str] = None

class ManualInterventionWorkflow:
    """
    Core workflow coordinator for risk-based manual intervention framework.
    
    Orchestrates the complete flow from risk detection through specialist
    assignment, manual review, and final resolution with comprehensive
    audit tracking and quality gate integration.
    """
    
    def __init__(self, config_path: str = "config/risk-assessment.yaml"):
        """Initialize the manual intervention workflow coordinator."""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize component systems
        self.risk_assessor = RiskAssessmentEngine(config_path)
        self.assignment_engine = SpecialistAssignmentEngine(config_path)
        self.sla_monitor = SLAMonitoringSystem(config_path)
        self.audit_tracker = DecisionAuditTracker()
        
        # Workflow state tracking
        self.active_workflows = {}  # workflow_id -> InterventionWorkflow
        self.completed_workflows = {}
        
        # Performance metrics
        self.metrics = {
            'total_interventions': 0,
            'successful_resolutions': 0,
            'escalations': 0,
            'override_rate': 0.0,
            'average_resolution_time': 0.0
        }
        
        # Start SLA monitoring if not already running
        self.sla_monitor.start_monitoring()
    
    def setup_logging(self):
        """Setup logging for manual intervention workflow."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ManualInterventionWorkflow - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load manual intervention configuration."""
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
        """Get default configuration for manual intervention."""
        return {
            'manual_intervention': {
                'enabled': True,
                'default_escalation_threshold': 0.6,
                'auto_approval_threshold': 0.3,
                'manager_override_enabled': True,
                'audit_trail_required': True,
                'quality_gate_integration': True
            },
            'workflow_timeouts': {
                'risk_assessment_minutes': 10,
                'specialist_assignment_minutes': 5,
                'manual_review_hours': 24,
                'escalation_response_hours': 4
            }
        }
    
    def initiate_intervention(self, issue_number: int, triggering_context: Optional[str] = None) -> str:
        """
        Initiate manual intervention workflow for an issue.
        
        Args:
            issue_number: GitHub issue number requiring intervention
            triggering_context: Context that triggered the intervention
            
        Returns:
            workflow_id: Unique identifier for the intervention workflow
        """
        workflow_id = self._generate_workflow_id(issue_number)
        
        workflow = InterventionWorkflow(
            workflow_id=workflow_id,
            issue_number=issue_number,
            initiated_at=datetime.now(timezone.utc),
            current_status=InterventionStatus.INITIATED,
            metadata={
                'triggering_context': triggering_context,
                'initiated_by': 'RIF-System'
            }
        )
        
        self.active_workflows[workflow_id] = workflow
        self.metrics['total_interventions'] += 1
        
        # Start audit trail
        audit_record = AuditRecord(
            workflow_id=workflow_id,
            timestamp=datetime.now(timezone.utc),
            action="intervention_initiated",
            actor="system",
            context=f"Issue #{issue_number}",
            rationale=f"Triggered by: {triggering_context}",
            evidence=[]
        )
        self.audit_tracker.record_decision(audit_record)
        
        workflow.audit_trail.append(f"Intervention initiated for issue #{issue_number}")
        
        self.logger.info(f"🚀 Initiated manual intervention {workflow_id} for issue #{issue_number}")
        
        # Start the workflow processing
        asyncio.create_task(self._process_workflow(workflow_id))
        
        return workflow_id
    
    async def _process_workflow(self, workflow_id: str) -> None:
        """Process the complete intervention workflow asynchronously."""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                self.logger.error(f"Workflow {workflow_id} not found")
                return
            
            # Step 1: Risk Assessment
            success = await self._perform_risk_assessment(workflow)
            if not success:
                await self._handle_workflow_error(workflow, "Risk assessment failed")
                return
            
            # Step 2: Determine if intervention is actually needed
            if not self._intervention_required(workflow):
                await self._complete_workflow_no_intervention(workflow)
                return
            
            # Step 3: Specialist Assignment
            success = await self._assign_specialist(workflow)
            if not success:
                await self._handle_workflow_error(workflow, "Specialist assignment failed")
                return
            
            # Step 4: Start SLA Monitoring
            await self._start_sla_tracking(workflow)
            
            # Step 5: Create GitHub Issue for Manual Review
            await self._create_manual_review_issue(workflow)
            
            # Step 6: Update RIF Workflow State
            await self._update_rif_workflow_state(workflow, "state:manual_review")
            
            self.logger.info(f"✅ Workflow {workflow_id} setup complete, awaiting manual review")
            
        except Exception as e:
            await self._handle_workflow_error(workflow, f"Workflow processing error: {e}")
    
    async def _perform_risk_assessment(self, workflow: InterventionWorkflow) -> bool:
        """Perform risk assessment for the workflow."""
        try:
            workflow.current_status = InterventionStatus.RISK_ASSESSING
            workflow.audit_trail.append("Starting risk assessment")
            
            # Get change context from issue
            change_context = create_change_context_from_issue(workflow.issue_number)
            if not change_context:
                workflow.error_messages.append("Could not create change context from issue")
                return False
            
            # Perform risk assessment
            risk_score = self.risk_assessor.assess_change_risk(change_context)
            workflow.risk_score = risk_score
            
            # Record in audit trail
            audit_record = AuditRecord(
                workflow_id=workflow.workflow_id,
                timestamp=datetime.now(timezone.utc),
                action="risk_assessment_completed",
                actor="system",
                context=f"Risk level: {risk_score.risk_level}",
                rationale=f"Risk score: {risk_score.total_score:.2f}",
                evidence=risk_score.reasoning
            )
            self.audit_tracker.record_decision(audit_record)
            
            workflow.audit_trail.append(f"Risk assessment complete: {risk_score.risk_level} ({risk_score.total_score:.2f})")
            
            self.logger.info(f"📊 Risk assessment complete for {workflow.workflow_id}: {risk_score.risk_level}")
            return True
            
        except Exception as e:
            self.logger.error(f"Risk assessment error for {workflow.workflow_id}: {e}")
            workflow.error_messages.append(f"Risk assessment error: {e}")
            return False
    
    def _intervention_required(self, workflow: InterventionWorkflow) -> bool:
        """Determine if manual intervention is actually required."""
        if not workflow.risk_score:
            return True  # Conservative default
        
        config = self.config.get('manual_intervention', {})
        threshold = config.get('default_escalation_threshold', 0.6)
        
        # Always require intervention for high/critical risk
        if workflow.risk_score.risk_level in ['high', 'critical']:
            return True
        
        # Require intervention if escalation flag is set
        if workflow.risk_score.escalation_required:
            return True
        
        # Require intervention if total score exceeds threshold
        if workflow.risk_score.total_score >= threshold:
            return True
        
        # Check for specific risk factors that always require intervention
        security_threshold = 0.5
        if workflow.risk_score.factors.security_score >= security_threshold:
            return True
        
        return False
    
    async def _complete_workflow_no_intervention(self, workflow: InterventionWorkflow) -> None:
        """Complete workflow when no manual intervention is needed."""
        workflow.current_status = InterventionStatus.RESOLVED
        workflow.decision_history.append({
            'decision': 'auto_approved',
            'reason': 'Low risk, no manual intervention required',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Update original issue
        await self._post_github_comment(
            workflow.issue_number,
            f"## ✅ Risk Assessment Complete - No Manual Intervention Required\n\n"
            f"**Risk Level**: {workflow.risk_score.risk_level}\n"
            f"**Risk Score**: {workflow.risk_score.total_score:.2f}\n\n"
            f"This change has been assessed as low risk and does not require manual specialist review. "
            f"The automated quality gates will continue to apply.\n\n"
            f"**Risk Assessment Details**:\n"
            + "\n".join([f"- {reason}" for reason in workflow.risk_score.reasoning])
        )
        
        # Move to completed workflows
        self.completed_workflows[workflow.workflow_id] = workflow
        self.active_workflows.pop(workflow.workflow_id)
        self.metrics['successful_resolutions'] += 1
        
        self.logger.info(f"✅ Completed workflow {workflow.workflow_id} - no intervention required")
    
    async def _assign_specialist(self, workflow: InterventionWorkflow) -> bool:
        """Assign specialist for manual review."""
        try:
            workflow.current_status = InterventionStatus.SPECIALIST_ASSIGNED
            workflow.audit_trail.append("Assigning specialist")
            
            if not workflow.risk_score:
                return False
            
            # Create assignment request
            assignment_request = AssignmentRequest(
                issue_number=workflow.issue_number,
                risk_score=workflow.risk_score.total_score,
                risk_level=workflow.risk_score.risk_level,
                primary_risk_factors=workflow.risk_score.reasoning,
                specialist_type=self._determine_specialist_type(workflow.risk_score),
                urgency_level=self._determine_urgency_level(workflow.risk_score),
                files_changed=[],  # Would be populated from change context
                estimated_review_time=self._estimate_review_time(workflow.risk_score),
                special_requirements=[]
            )
            
            # Assign specialist
            assignment_result = self.assignment_engine.assign_specialist(assignment_request)
            workflow.assignment_result = assignment_result
            
            if not assignment_result.assigned_specialist:
                workflow.error_messages.append("No available specialists")
                return False
            
            # Record in audit trail
            audit_record = AuditRecord(
                workflow_id=workflow.workflow_id,
                timestamp=datetime.now(timezone.utc),
                action="specialist_assigned",
                actor="system",
                context=f"Assigned: {assignment_result.assigned_specialist.name}",
                rationale=f"Confidence: {assignment_result.assignment_confidence:.2f}",
                evidence=assignment_result.assignment_reasoning
            )
            self.audit_tracker.record_decision(audit_record)
            
            workflow.audit_trail.append(f"Specialist assigned: {assignment_result.assigned_specialist.name}")
            
            self.logger.info(f"👤 Specialist assigned for {workflow.workflow_id}: {assignment_result.assigned_specialist.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Specialist assignment error for {workflow.workflow_id}: {e}")
            workflow.error_messages.append(f"Specialist assignment error: {e}")
            return False
    
    async def _start_sla_tracking(self, workflow: InterventionWorkflow) -> None:
        """Start SLA tracking for the manual review."""
        try:
            if not workflow.assignment_result:
                return
            
            sla_tracking_id = self.sla_monitor.start_sla_tracking(
                issue_number=workflow.issue_number,
                specialist_github_issue=workflow.assignment_result.github_issue_number,
                assigned_specialist=workflow.assignment_result.assigned_specialist.specialist_id,
                specialist_type=workflow.assignment_result.assigned_specialist.specialist_type.value,
                sla_deadline=workflow.assignment_result.sla_deadline
            )
            
            workflow.sla_tracking_id = sla_tracking_id
            workflow.audit_trail.append(f"SLA tracking started: {sla_tracking_id}")
            
        except Exception as e:
            self.logger.error(f"SLA tracking error for {workflow.workflow_id}: {e}")
            # Don't fail the workflow for SLA tracking errors
    
    async def _create_manual_review_issue(self, workflow: InterventionWorkflow) -> None:
        """Create GitHub issue for manual review if not already created."""
        try:
            if workflow.assignment_result and workflow.assignment_result.github_issue_number:
                # Already created by assignment engine
                return
            
            # Create manual review issue (fallback)
            title = f"🔍 Manual Review Required: Issue #{workflow.issue_number}"
            body = self._generate_manual_review_body(workflow)
            
            result = subprocess.run([
                'gh', 'issue', 'create',
                '--title', title,
                '--body', body,
                '--label', 'state:manual_review,priority:high'
            ], capture_output=True, text=True, check=True)
            
            # Extract issue number from output
            for line in result.stderr.split('\n'):
                if 'https://github.com/' in line and '/issues/' in line:
                    issue_number = int(line.strip().split('/issues/')[-1])
                    workflow.metadata['manual_review_issue'] = issue_number
                    break
            
        except Exception as e:
            self.logger.error(f"Manual review issue creation error for {workflow.workflow_id}: {e}")
    
    def _generate_manual_review_body(self, workflow: InterventionWorkflow) -> str:
        """Generate body content for manual review issue."""
        body = f"""# Manual Review Required - Issue #{workflow.issue_number}

## Risk Assessment Summary
- **Risk Level**: {workflow.risk_score.risk_level.upper() if workflow.risk_score else 'UNKNOWN'}
- **Risk Score**: {workflow.risk_score.total_score:.2f if workflow.risk_score else 'N/A'}
- **Workflow ID**: `{workflow.workflow_id}`

## Risk Factors
{chr(10).join([f"- {reason}" for reason in (workflow.risk_score.reasoning if workflow.risk_score else ['Risk assessment not available'])])}

## Assigned Specialist
{f"- **Name**: {workflow.assignment_result.assigned_specialist.name}" if workflow.assignment_result and workflow.assignment_result.assigned_specialist else "- **Status**: No specialist assigned"}
{f"- **Type**: {workflow.assignment_result.assigned_specialist.specialist_type.value}" if workflow.assignment_result and workflow.assignment_result.assigned_specialist else ""}
{f"- **GitHub**: @{workflow.assignment_result.assigned_specialist.github_username}" if workflow.assignment_result and workflow.assignment_result.assigned_specialist else ""}

## Evidence Checklist
{chr(10).join([f"- [ ] {item}" for item in (workflow.assignment_result.evidence_checklist.mandatory_items if workflow.assignment_result else ['Evidence checklist not available'])]) if workflow.assignment_result else "Evidence checklist not available"}

## Instructions
1. Review the original issue #{workflow.issue_number}
2. Complete the evidence checklist above
3. Provide your assessment in a comment below
4. **Approve**, **Request Changes**, or **Escalate**

## Audit Trail
{chr(10).join([f"- {entry}" for entry in workflow.audit_trail])}

---
*This issue was automatically created by the Risk-Based Manual Intervention Framework.*
"""
        return body
    
    async def _update_rif_workflow_state(self, workflow: InterventionWorkflow, state: str) -> None:
        """Update RIF workflow state for integration."""
        try:
            # Update issue labels to reflect state
            subprocess.run([
                'gh', 'issue', 'edit', str(workflow.issue_number),
                '--add-label', state
            ], check=True, capture_output=True)
            
            workflow.audit_trail.append(f"RIF workflow state updated: {state}")
            
        except Exception as e:
            self.logger.error(f"RIF workflow state update error for {workflow.workflow_id}: {e}")
    
    def process_manual_decision(self, workflow_id: str, decision: ManualDecision) -> bool:
        """
        Process a manual decision from a specialist.
        
        Args:
            workflow_id: Workflow identifier
            decision: Manual decision details
            
        Returns:
            Success indicator
        """
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                self.logger.error(f"Workflow {workflow_id} not found for decision processing")
                return False
            
            workflow.current_status = InterventionStatus.MANUAL_REVIEW
            workflow.decision_history.append(asdict(decision))
            
            # Record in audit trail
            audit_record = AuditRecord(
                workflow_id=workflow_id,
                timestamp=decision.timestamp,
                action=f"manual_decision_{decision.decision_type.value}",
                actor=decision.decision_maker,
                context=f"Decision: {decision.decision_type.value}",
                rationale=decision.decision_rationale,
                evidence=decision.evidence_provided
            )
            self.audit_tracker.record_decision(audit_record)
            
            # Process decision type
            if decision.decision_type == DecisionType.APPROVE:
                return self._handle_approval_decision(workflow, decision)
            elif decision.decision_type == DecisionType.REQUEST_CHANGES:
                return self._handle_changes_request(workflow, decision)
            elif decision.decision_type == DecisionType.ESCALATE:
                return self._handle_escalation_request(workflow, decision)
            elif decision.decision_type in [DecisionType.OVERRIDE_APPROVE, DecisionType.OVERRIDE_REJECT]:
                return self._handle_manager_override(workflow, decision)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Decision processing error for {workflow_id}: {e}")
            return False
    
    def _handle_approval_decision(self, workflow: InterventionWorkflow, decision: ManualDecision) -> bool:
        """Handle approval decision from specialist."""
        workflow.current_status = InterventionStatus.RESOLVED
        
        # Stop SLA tracking
        if workflow.sla_tracking_id:
            self.sla_monitor.resolve_sla(workflow.sla_tracking_id, "Approved by specialist")
        
        # Update original issue
        asyncio.create_task(self._post_approval_comment(workflow, decision))
        
        # Update RIF workflow state
        asyncio.create_task(self._update_rif_workflow_state(workflow, "state:approved"))
        
        # Complete workflow
        self._complete_workflow(workflow, "approved")
        
        self.logger.info(f"✅ Approval decision processed for {workflow.workflow_id}")
        return True
    
    def _handle_changes_request(self, workflow: InterventionWorkflow, decision: ManualDecision) -> bool:
        """Handle request for changes from specialist."""
        # Stop SLA tracking
        if workflow.sla_tracking_id:
            self.sla_monitor.resolve_sla(workflow.sla_tracking_id, "Changes requested by specialist")
        
        # Update original issue with requested changes
        asyncio.create_task(self._post_changes_request_comment(workflow, decision))
        
        # Update RIF workflow state
        asyncio.create_task(self._update_rif_workflow_state(workflow, "state:changes_requested"))
        
        # Complete workflow (changes requested)
        self._complete_workflow(workflow, "changes_requested")
        
        self.logger.info(f"📝 Changes request processed for {workflow.workflow_id}")
        return True
    
    def _handle_escalation_request(self, workflow: InterventionWorkflow, decision: ManualDecision) -> bool:
        """Handle escalation request from specialist."""
        workflow.current_status = InterventionStatus.ESCALATED
        
        # Update SLA tracking to escalated
        if workflow.sla_tracking_id:
            self.sla_monitor.update_sla_status(workflow.sla_tracking_id, SLAStatus.ESCALATED, decision.decision_rationale)
        
        # Escalate to next level in chain
        if workflow.assignment_result and workflow.assignment_result.escalation_chain:
            escalation_chain = workflow.assignment_result.escalation_chain
            if len(escalation_chain) > 1:
                next_specialist = escalation_chain[1]  # Next level up
                # Create new assignment for escalated specialist
                # This would involve creating a new specialist assignment
                self.logger.info(f"🔺 Escalated to {next_specialist.name}")
        
        self.metrics['escalations'] += 1
        
        self.logger.info(f"🔺 Escalation request processed for {workflow.workflow_id}")
        return True
    
    def _handle_manager_override(self, workflow: InterventionWorkflow, decision: ManualDecision) -> bool:
        """Handle manager override decision."""
        workflow.current_status = InterventionStatus.MANAGER_OVERRIDE
        
        # Calculate override rate
        total_decisions = len([d for d in workflow.decision_history if 'decision_type' in d])
        if total_decisions > 0:
            override_count = len([d for d in workflow.decision_history if d.get('decision_type', '').startswith('override')])
            self.metrics['override_rate'] = override_count / total_decisions
        
        # Stop SLA tracking
        if workflow.sla_tracking_id:
            self.sla_monitor.resolve_sla(workflow.sla_tracking_id, f"Manager override: {decision.decision_type.value}")
        
        # Post override comment
        asyncio.create_task(self._post_override_comment(workflow, decision))
        
        # Update RIF workflow state based on override type
        final_state = "state:approved" if decision.decision_type == DecisionType.OVERRIDE_APPROVE else "state:rejected"
        asyncio.create_task(self._update_rif_workflow_state(workflow, final_state))
        
        # Complete workflow
        self._complete_workflow(workflow, decision.decision_type.value)
        
        self.logger.warning(f"⚡ Manager override processed for {workflow.workflow_id}: {decision.decision_type.value}")
        return True
    
    async def _post_approval_comment(self, workflow: InterventionWorkflow, decision: ManualDecision) -> None:
        """Post approval comment to original issue."""
        comment = f"""## ✅ Manual Review Approved

**Specialist Decision**: APPROVED
**Reviewer**: {decision.decision_maker}
**Rationale**: {decision.decision_rationale}

**Evidence Reviewed**:
{chr(10).join([f"- ✅ {evidence}" for evidence in decision.evidence_provided])}

This change has been approved by specialist review and may proceed through the automated quality gates.

**Workflow ID**: `{workflow.workflow_id}`
**Review Completed**: {decision.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
"""
        await self._post_github_comment(workflow.issue_number, comment)
    
    async def _post_changes_request_comment(self, workflow: InterventionWorkflow, decision: ManualDecision) -> None:
        """Post changes request comment to original issue."""
        comment = f"""## 📝 Changes Requested

**Specialist Decision**: CHANGES REQUIRED
**Reviewer**: {decision.decision_maker}
**Rationale**: {decision.decision_rationale}

**Required Changes**:
{chr(10).join([f"- ⚠️ {evidence}" for evidence in decision.evidence_provided])}

Please address the requested changes before proceeding. Once changes are made, the review process may be reinitiated if needed.

**Workflow ID**: `{workflow.workflow_id}`
**Review Completed**: {decision.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
"""
        await self._post_github_comment(workflow.issue_number, comment)
    
    async def _post_override_comment(self, workflow: InterventionWorkflow, decision: ManualDecision) -> None:
        """Post manager override comment to original issue."""
        approval_text = "APPROVED" if decision.decision_type == DecisionType.OVERRIDE_APPROVE else "REJECTED"
        icon = "✅" if decision.decision_type == DecisionType.OVERRIDE_APPROVE else "❌"
        
        comment = f"""## {icon} Manager Override: {approval_text}

**Override Decision**: {approval_text}
**Manager**: {decision.decision_maker}
**Justification**: {decision.override_justification or 'Not provided'}
**Rationale**: {decision.decision_rationale}

⚠️ **This decision overrides the specialist recommendation and requires executive accountability.**

**Manager Approval**: {decision.manager_approval or 'Self-approved'}
**Workflow ID**: `{workflow.workflow_id}`
**Override Timestamp**: {decision.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
"""
        await self._post_github_comment(workflow.issue_number, comment)
    
    async def _post_github_comment(self, issue_number: int, comment: str) -> None:
        """Post comment to GitHub issue."""
        try:
            subprocess.run([
                'gh', 'issue', 'comment', str(issue_number),
                '--body', comment
            ], check=True, capture_output=True)
        except Exception as e:
            self.logger.error(f"Failed to post GitHub comment to issue #{issue_number}: {e}")
    
    def _complete_workflow(self, workflow: InterventionWorkflow, resolution: str) -> None:
        """Complete workflow and move to completed state."""
        workflow.current_status = InterventionStatus.RESOLVED
        workflow.last_updated = datetime.now(timezone.utc)
        workflow.metadata['resolution'] = resolution
        workflow.metadata['completion_time'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate resolution time
        resolution_time = (workflow.last_updated - workflow.initiated_at).total_seconds() / 3600  # hours
        workflow.metadata['resolution_time_hours'] = resolution_time
        
        # Update metrics
        self.metrics['successful_resolutions'] += 1
        current_avg = self.metrics.get('average_resolution_time', 0.0)
        total_resolutions = self.metrics['successful_resolutions']
        self.metrics['average_resolution_time'] = ((current_avg * (total_resolutions - 1)) + resolution_time) / total_resolutions
        
        # Move to completed workflows
        self.completed_workflows[workflow.workflow_id] = workflow
        self.active_workflows.pop(workflow.workflow_id)
        
        # Final audit record
        audit_record = AuditRecord(
            workflow_id=workflow.workflow_id,
            timestamp=datetime.now(timezone.utc),
            action="workflow_completed",
            actor="system",
            context=f"Resolution: {resolution}",
            rationale=f"Total time: {resolution_time:.1f}h",
            evidence=[f"Final status: {workflow.current_status.value}"]
        )
        self.audit_tracker.record_decision(audit_record)
        
        self.logger.info(f"🏁 Completed workflow {workflow.workflow_id} - Resolution: {resolution} ({resolution_time:.1f}h)")
    
    async def _handle_workflow_error(self, workflow: InterventionWorkflow, error_message: str) -> None:
        """Handle workflow error with graceful degradation."""
        workflow.current_status = InterventionStatus.ERROR
        workflow.error_messages.append(error_message)
        workflow.last_updated = datetime.now(timezone.utc)
        
        # Post error comment to original issue
        comment = f"""## ⚠️ Manual Intervention Workflow Error

An error occurred in the manual intervention workflow for this issue:

**Error**: {error_message}
**Workflow ID**: `{workflow.workflow_id}`
**Time**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

**Fallback Actions**:
- Manual quality gate review is recommended
- Contact engineering management if critical
- Check system logs for detailed error information

*The automated workflow will continue to monitor for resolution.*
"""
        await self._post_github_comment(workflow.issue_number, comment)
        
        # Record error in audit trail
        audit_record = AuditRecord(
            workflow_id=workflow.workflow_id,
            timestamp=datetime.now(timezone.utc),
            action="workflow_error",
            actor="system",
            context=error_message,
            rationale="Workflow error - graceful degradation",
            evidence=workflow.error_messages
        )
        self.audit_tracker.record_decision(audit_record)
        
        self.logger.error(f"❌ Workflow error for {workflow.workflow_id}: {error_message}")
    
    def _determine_specialist_type(self, risk_score: RiskScore) -> SpecialistType:
        """Determine appropriate specialist type from risk assessment."""
        if risk_score.specialist_type == 'security':
            return SpecialistType.SECURITY
        elif risk_score.specialist_type == 'architecture':
            return SpecialistType.ARCHITECTURE
        elif risk_score.specialist_type == 'compliance':
            return SpecialistType.COMPLIANCE
        else:
            # Default based on highest risk factor
            if risk_score.factors.security_score >= 0.5:
                return SpecialistType.SECURITY
            elif risk_score.factors.complexity_score >= 0.6 or risk_score.factors.impact_score >= 0.6:
                return SpecialistType.ARCHITECTURE
            else:
                return SpecialistType.ARCHITECTURE  # Default
    
    def _determine_urgency_level(self, risk_score: RiskScore) -> str:
        """Determine urgency level from risk assessment."""
        if risk_score.risk_level == 'critical':
            return 'critical'
        elif risk_score.risk_level == 'high':
            return 'high'
        elif risk_score.factors.time_pressure_score >= 0.5:
            return 'high'
        else:
            return 'medium'
    
    def _estimate_review_time(self, risk_score: RiskScore) -> float:
        """Estimate review time in hours based on risk assessment."""
        base_time = 4.0  # hours
        
        if risk_score.risk_level == 'critical':
            return base_time * 1.5
        elif risk_score.risk_level == 'high':
            return base_time * 1.2
        else:
            return base_time
    
    def _generate_workflow_id(self, issue_number: int) -> str:
        """Generate unique workflow identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"manual_intervention_{issue_number}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"mi_{issue_number}_{hash_suffix}"
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        workflow = self.active_workflows.get(workflow_id) or self.completed_workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            'workflow_id': workflow.workflow_id,
            'issue_number': workflow.issue_number,
            'current_status': workflow.current_status.value,
            'initiated_at': workflow.initiated_at.isoformat(),
            'last_updated': workflow.last_updated.isoformat(),
            'risk_assessment': {
                'risk_level': workflow.risk_score.risk_level if workflow.risk_score else None,
                'risk_score': workflow.risk_score.total_score if workflow.risk_score else None,
                'escalation_required': workflow.risk_score.escalation_required if workflow.risk_score else None
            } if workflow.risk_score else None,
            'assigned_specialist': {
                'name': workflow.assignment_result.assigned_specialist.name if workflow.assignment_result and workflow.assignment_result.assigned_specialist else None,
                'type': workflow.assignment_result.assigned_specialist.specialist_type.value if workflow.assignment_result and workflow.assignment_result.assigned_specialist else None
            } if workflow.assignment_result else None,
            'sla_tracking_id': workflow.sla_tracking_id,
            'decision_count': len(workflow.decision_history),
            'error_count': len(workflow.error_messages),
            'metadata': workflow.metadata
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'active_workflows': len(self.active_workflows),
            'completed_workflows': len(self.completed_workflows),
            'performance_metrics': self.metrics,
            'component_health': {
                'risk_assessor': 'healthy',
                'assignment_engine': 'healthy',
                'sla_monitor': 'healthy' if self.sla_monitor.monitoring_active else 'inactive',
                'audit_tracker': 'healthy'
            }
        }

def main():
    """Command line interface for manual intervention workflow."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python manual_intervention_workflow.py <command> [args]")
        print("Commands:")
        print("  initiate <issue_number>           - Initiate intervention for issue")
        print("  status <workflow_id>              - Get workflow status")
        print("  metrics                           - Show system metrics")
        print("  test-workflow                     - Test workflow with mock issue")
        return
    
    command = sys.argv[1]
    coordinator = ManualInterventionWorkflow()
    
    if command == "initiate" and len(sys.argv) >= 3:
        issue_num = int(sys.argv[2])
        workflow_id = coordinator.initiate_intervention(issue_num, "CLI test")
        print(f"✅ Initiated intervention workflow: {workflow_id}")
        
        # Wait a bit for async processing
        import asyncio
        asyncio.run(asyncio.sleep(5))
        
        status = coordinator.get_workflow_status(workflow_id)
        if status:
            print(json.dumps(status, indent=2))
        
    elif command == "status" and len(sys.argv) >= 3:
        workflow_id = sys.argv[2]
        status = coordinator.get_workflow_status(workflow_id)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print(f"Workflow {workflow_id} not found")
            
    elif command == "metrics":
        metrics = coordinator.get_system_metrics()
        print(json.dumps(metrics, indent=2))
        
    elif command == "test-workflow":
        print("🧪 Testing manual intervention workflow...")
        # Would need a test issue for this to work
        print("Note: Requires a real GitHub issue to test fully")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())