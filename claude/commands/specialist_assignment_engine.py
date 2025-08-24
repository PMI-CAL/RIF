#!/usr/bin/env python3
"""
Specialist Assignment Engine - Issue #92 Phase 2
Automated specialist routing and assignment system for risk-based manual intervention.

This engine implements:
1. Pattern-based specialist routing logic
2. Workload balancing across specialist teams
3. Expertise-based assignment optimization
4. SLA tracking and response time monitoring
5. Escalation chain management
6. Evidence checklist generation per specialist type
7. GitHub issue creation for specialist reviews
"""

import json
import subprocess
import yaml
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

class SpecialistType(Enum):
    """Enumeration of specialist types."""
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    DATABASE = "database"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DEVOPS = "devops"

class SpecialistStatus(Enum):
    """Enumeration of specialist availability status."""
    AVAILABLE = "available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ON_CALL = "on_call"

@dataclass
class SpecialistProfile:
    """Container for specialist profile information."""
    specialist_id: str
    name: str
    email: str
    github_username: str
    specialist_type: SpecialistType
    expertise_areas: List[str]
    current_workload: int
    max_concurrent_reviews: int
    average_response_time_hours: float
    status: SpecialistStatus
    timezone: str
    escalation_level: int  # 1 = primary, 2 = senior, 3 = principal
    
@dataclass
class EvidenceChecklist:
    """Container for specialist-specific evidence requirements."""
    specialist_type: SpecialistType
    mandatory_items: List[str]
    recommended_items: List[str]
    custom_items: List[str]
    completion_criteria: str
    
@dataclass
class AssignmentRequest:
    """Container for specialist assignment request."""
    issue_number: int
    risk_score: float
    risk_level: str
    primary_risk_factors: List[str]
    specialist_type: SpecialistType
    urgency_level: str
    files_changed: List[str]
    estimated_review_time: float
    special_requirements: List[str]

@dataclass  
class AssignmentResult:
    """Container for specialist assignment result."""
    assigned_specialist: Optional[SpecialistProfile]
    assignment_confidence: float
    estimated_response_time: float
    sla_deadline: datetime
    evidence_checklist: EvidenceChecklist
    escalation_chain: List[SpecialistProfile]
    github_issue_number: Optional[int]
    assignment_reasoning: List[str]
    backup_specialists: List[SpecialistProfile]

class SpecialistAssignmentEngine:
    """
    Intelligent specialist assignment and routing system.
    """
    
    def __init__(self, config_path: str = "config/risk-assessment.yaml"):
        """Initialize the specialist assignment engine."""
        self.config_path = config_path
        self.setup_logging()
        self.config = self._load_config()
        self.specialists = self._load_specialist_registry()
        self.assignment_history = self._load_assignment_history()
    
    def setup_logging(self):
        """Setup logging for specialist assignment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - SpecialistAssignmentEngine - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load specialist assignment configuration."""
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
        """Get default specialist assignment configuration."""
        return {
            'specialist_routing': {
                'security': {
                    'triggers': ['security_changes', 'auth_modifications'],
                    'sla_hours': 4,
                    'blocking': True
                },
                'architecture': {
                    'triggers': ['large_changes', 'database_changes'],
                    'sla_hours': 12,
                    'blocking': False
                }
            },
            'sla_monitoring': {
                'default_sla_hours': 8,
                'escalation_thresholds': [0.5, 0.8, 1.0]
            }
        }
    
    def _load_specialist_registry(self) -> Dict[SpecialistType, List[SpecialistProfile]]:
        """Load specialist registry from configuration or external source."""
        specialists = {specialist_type: [] for specialist_type in SpecialistType}
        
        try:
            # In a real implementation, this would load from LDAP, database, or config file
            # For now, create some example specialists
            specialists[SpecialistType.SECURITY] = [
                SpecialistProfile(
                    specialist_id="sec_001",
                    name="Alice Security",
                    email="alice@company.com",
                    github_username="alice-sec",
                    specialist_type=SpecialistType.SECURITY,
                    expertise_areas=["authentication", "authorization", "cryptography", "vulnerability_assessment"],
                    current_workload=2,
                    max_concurrent_reviews=5,
                    average_response_time_hours=3.5,
                    status=SpecialistStatus.AVAILABLE,
                    timezone="UTC",
                    escalation_level=1
                ),
                SpecialistProfile(
                    specialist_id="sec_002", 
                    name="Bob SecOps",
                    email="bob@company.com",
                    github_username="bob-secops",
                    specialist_type=SpecialistType.SECURITY,
                    expertise_areas=["penetration_testing", "compliance", "incident_response"],
                    current_workload=1,
                    max_concurrent_reviews=4,
                    average_response_time_hours=2.0,
                    status=SpecialistStatus.AVAILABLE,
                    timezone="UTC",
                    escalation_level=2
                )
            ]
            
            specialists[SpecialistType.ARCHITECTURE] = [
                SpecialistProfile(
                    specialist_id="arch_001",
                    name="Carol Architect",
                    email="carol@company.com",
                    github_username="carol-arch",
                    specialist_type=SpecialistType.ARCHITECTURE,
                    expertise_areas=["system_design", "scalability", "microservices", "api_design"],
                    current_workload=3,
                    max_concurrent_reviews=6,
                    average_response_time_hours=8.0,
                    status=SpecialistStatus.AVAILABLE,
                    timezone="UTC",
                    escalation_level=1
                )
            ]
            
            specialists[SpecialistType.COMPLIANCE] = [
                SpecialistProfile(
                    specialist_id="comp_001",
                    name="David Compliance",
                    email="david@company.com", 
                    github_username="david-compliance",
                    specialist_type=SpecialistType.COMPLIANCE,
                    expertise_areas=["gdpr", "sox", "pci_dss", "audit_trails"],
                    current_workload=1,
                    max_concurrent_reviews=3,
                    average_response_time_hours=6.0,
                    status=SpecialistStatus.AVAILABLE,
                    timezone="UTC",
                    escalation_level=1
                )
            ]
            
            self.logger.info(f"Loaded {sum(len(specs) for specs in specialists.values())} specialists")
            return specialists
            
        except Exception as e:
            self.logger.error(f"Error loading specialist registry: {e}")
            return specialists
    
    def _load_assignment_history(self) -> Dict[str, Any]:
        """Load assignment history for workload tracking."""
        try:
            history_file = Path("knowledge/decisions/assignment_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
            else:
                return {"assignments": [], "metrics": {}}
        except Exception as e:
            self.logger.error(f"Error loading assignment history: {e}")
            return {"assignments": [], "metrics": {}}
    
    def assign_specialist(self, assignment_request: AssignmentRequest) -> AssignmentResult:
        """
        Assign the optimal specialist for a given request.
        
        Args:
            assignment_request: Request details for specialist assignment
            
        Returns:
            AssignmentResult with detailed assignment information
        """
        self.logger.info(f"🎯 Assigning specialist for issue #{assignment_request.issue_number} ({assignment_request.specialist_type.value})")
        
        try:
            # Get candidates for the specialist type
            candidates = self._get_specialist_candidates(assignment_request)
            
            if not candidates:
                return self._create_no_specialist_result(assignment_request)
            
            # Score and rank candidates
            scored_candidates = self._score_candidates(candidates, assignment_request)
            
            # Select best candidate
            best_candidate = scored_candidates[0] if scored_candidates else None
            
            if not best_candidate:
                return self._create_no_specialist_result(assignment_request)
            
            # Calculate SLA deadline
            sla_deadline = self._calculate_sla_deadline(assignment_request, best_candidate[0])
            
            # Generate evidence checklist
            evidence_checklist = self._generate_evidence_checklist(assignment_request)
            
            # Build escalation chain
            escalation_chain = self._build_escalation_chain(assignment_request.specialist_type, best_candidate[0])
            
            # Create GitHub issue for specialist review
            github_issue_number = self._create_specialist_github_issue(assignment_request, best_candidate[0], evidence_checklist)
            
            # Generate assignment reasoning
            reasoning = self._generate_assignment_reasoning(best_candidate[0], assignment_request, best_candidate[1])
            
            # Get backup specialists
            backup_specialists = [candidate[0] for candidate in scored_candidates[1:3]]  # Top 2 backups
            
            result = AssignmentResult(
                assigned_specialist=best_candidate[0],
                assignment_confidence=best_candidate[1],
                estimated_response_time=best_candidate[0].average_response_time_hours,
                sla_deadline=sla_deadline,
                evidence_checklist=evidence_checklist,
                escalation_chain=escalation_chain,
                github_issue_number=github_issue_number,
                assignment_reasoning=reasoning,
                backup_specialists=backup_specialists
            )
            
            # Update specialist workload
            self._update_specialist_workload(best_candidate[0], assignment_request)
            
            # Record assignment in history
            self._record_assignment(assignment_request, result)
            
            self.logger.info(f"✅ Assigned {best_candidate[0].name} to issue #{assignment_request.issue_number}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error assigning specialist: {e}")
            return self._create_error_result(assignment_request, str(e))
    
    def _get_specialist_candidates(self, request: AssignmentRequest) -> List[SpecialistProfile]:
        """Get available specialist candidates for the request."""
        candidates = self.specialists.get(request.specialist_type, [])
        
        # Filter by availability and workload
        available_candidates = []
        for specialist in candidates:
            if (specialist.status in [SpecialistStatus.AVAILABLE, SpecialistStatus.ON_CALL] and
                specialist.current_workload < specialist.max_concurrent_reviews):
                available_candidates.append(specialist)
        
        return available_candidates
    
    def _score_candidates(self, candidates: List[SpecialistProfile], request: AssignmentRequest) -> List[Tuple[SpecialistProfile, float]]:
        """Score and rank specialist candidates."""
        scored_candidates = []
        
        for candidate in candidates:
            score = self._calculate_candidate_score(candidate, request)
            scored_candidates.append((candidate, score))
        
        # Sort by score descending (best first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates
    
    def _calculate_candidate_score(self, candidate: SpecialistProfile, request: AssignmentRequest) -> float:
        """Calculate assignment score for a candidate specialist."""
        score = 0.0
        
        # Base score from availability and workload
        workload_factor = 1.0 - (candidate.current_workload / candidate.max_concurrent_reviews)
        score += workload_factor * 0.3
        
        # Response time factor (faster is better)
        max_response_time = 24.0  # Maximum expected response time
        response_factor = 1.0 - (candidate.average_response_time_hours / max_response_time)
        score += max(0, response_factor) * 0.2
        
        # Expertise matching
        expertise_score = self._calculate_expertise_match(candidate, request)
        score += expertise_score * 0.3
        
        # Urgency handling capability
        if request.urgency_level == "critical" and candidate.escalation_level >= 2:
            score += 0.1
        elif request.urgency_level == "high" and candidate.escalation_level >= 1:
            score += 0.05
        
        # Status bonus
        if candidate.status == SpecialistStatus.ON_CALL:
            score += 0.1
        elif candidate.status == SpecialistStatus.AVAILABLE:
            score += 0.05
        
        # Historical performance (simplified)
        historical_bonus = self._get_historical_performance_bonus(candidate)
        score += historical_bonus * 0.1
        
        return min(1.0, max(0.0, score))  # Clamp between 0 and 1
    
    def _calculate_expertise_match(self, candidate: SpecialistProfile, request: AssignmentRequest) -> float:
        """Calculate how well candidate's expertise matches request requirements."""
        if not candidate.expertise_areas:
            return 0.5  # Default moderate match
        
        # Keywords that might indicate needed expertise
        expertise_keywords = []
        
        # Extract keywords from risk factors
        for risk_factor in request.primary_risk_factors:
            expertise_keywords.extend(risk_factor.lower().split())
        
        # Extract keywords from file paths
        for file_path in request.files_changed:
            path_parts = file_path.lower().split('/')
            expertise_keywords.extend(path_parts)
        
        if not expertise_keywords:
            return 0.7  # Default good match if no specific keywords
        
        # Calculate overlap between candidate expertise and request keywords
        matched_areas = 0
        for expertise in candidate.expertise_areas:
            expertise_lower = expertise.lower().replace('_', ' ')
            for keyword in expertise_keywords:
                if keyword in expertise_lower or expertise_lower in keyword:
                    matched_areas += 1
                    break
        
        # Calculate match ratio
        if len(candidate.expertise_areas) == 0:
            return 0.5
        
        match_ratio = matched_areas / len(candidate.expertise_areas)
        return min(1.0, match_ratio + 0.3)  # Add base score + bonus for matches
    
    def _get_historical_performance_bonus(self, candidate: SpecialistProfile) -> float:
        """Get historical performance bonus for candidate (simplified)."""
        # In a real implementation, this would analyze past assignment outcomes
        # For now, return a random-like but deterministic value based on specialist ID
        hash_value = hashlib.md5(candidate.specialist_id.encode()).hexdigest()
        return (int(hash_value, 16) % 100) / 1000.0  # 0.0 to 0.099 bonus
    
    def _calculate_sla_deadline(self, request: AssignmentRequest, specialist: SpecialistProfile) -> datetime:
        """Calculate SLA deadline for the assignment."""
        # Get SLA hours from configuration
        specialist_config = self.config.get('specialist_routing', {}).get(request.specialist_type.value, {})
        base_sla_hours = specialist_config.get('sla_hours', 8)
        
        # Adjust based on urgency
        if request.urgency_level == "critical":
            sla_hours = min(base_sla_hours, 2)  # Max 2 hours for critical
        elif request.urgency_level == "high":
            sla_hours = min(base_sla_hours, 4)  # Max 4 hours for high
        elif request.urgency_level == "urgent":
            sla_hours = base_sla_hours * 0.5  # Half normal SLA for urgent
        else:
            sla_hours = base_sla_hours
        
        from datetime import timezone
        return datetime.now(timezone.utc) + timedelta(hours=sla_hours)
    
    def _generate_evidence_checklist(self, request: AssignmentRequest) -> EvidenceChecklist:
        """Generate specialist-specific evidence checklist."""
        evidence_requirements = self.config.get('evidence_requirements', {})
        specialist_key = f"{request.specialist_type.value}_specialist"
        
        requirements = evidence_requirements.get(specialist_key, {})
        mandatory_items = requirements.get('mandatory', [])
        recommended_items = requirements.get('recommended', [])
        
        # Add request-specific custom items
        custom_items = []
        if request.risk_level == 'critical':
            custom_items.append("Executive approval documentation")
        if 'security' in request.primary_risk_factors:
            custom_items.append("Security impact assessment")
        if request.files_changed and any('database' in f.lower() for f in request.files_changed):
            custom_items.append("Database migration rollback plan")
        
        completion_criteria = f"All mandatory items completed, {len(recommended_items)} recommended items reviewed"
        
        return EvidenceChecklist(
            specialist_type=request.specialist_type,
            mandatory_items=mandatory_items,
            recommended_items=recommended_items,
            custom_items=custom_items,
            completion_criteria=completion_criteria
        )
    
    def _build_escalation_chain(self, specialist_type: SpecialistType, assigned_specialist: SpecialistProfile) -> List[SpecialistProfile]:
        """Build escalation chain for the assignment."""
        chain = [assigned_specialist]
        
        # Get all specialists of the same type
        same_type_specialists = self.specialists.get(specialist_type, [])
        
        # Add higher-level specialists (higher escalation_level)
        for specialist in same_type_specialists:
            if (specialist.escalation_level > assigned_specialist.escalation_level and
                specialist.specialist_id != assigned_specialist.specialist_id):
                chain.append(specialist)
        
        # Sort by escalation level
        chain.sort(key=lambda x: x.escalation_level)
        
        return chain[:4]  # Limit to 4 levels max
    
    def _create_specialist_github_issue(self, request: AssignmentRequest, specialist: SpecialistProfile, evidence_checklist: EvidenceChecklist) -> Optional[int]:
        """Create GitHub issue for specialist review."""
        try:
            # Create issue title
            title = f"🔍 {specialist.specialist_type.value.title()} Review Required: Issue #{request.issue_number}"
            
            # Create issue body with evidence checklist
            body = self._generate_specialist_issue_body(request, specialist, evidence_checklist)
            
            # Create labels
            labels = [
                f"state:blocked",
                f"specialist:{specialist.specialist_type.value}",
                f"risk:{request.risk_level}",
                f"assignee:{specialist.github_username}"
            ]
            
            # Create the GitHub issue
            result = subprocess.run([
                'gh', 'issue', 'create',
                '--title', title,
                '--body', body,
                '--assignee', specialist.github_username,
                '--label', ','.join(labels)
            ], capture_output=True, text=True, check=True)
            
            # Extract issue number from output
            output_lines = result.stderr.split('\n')
            for line in output_lines:
                if 'https://github.com/' in line and '/issues/' in line:
                    issue_url = line.strip()
                    issue_number = int(issue_url.split('/issues/')[-1])
                    self.logger.info(f"Created specialist review issue #{issue_number}")
                    return issue_number
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating specialist GitHub issue: {e}")
            return None
    
    def _generate_specialist_issue_body(self, request: AssignmentRequest, specialist: SpecialistProfile, evidence_checklist: EvidenceChecklist) -> str:
        """Generate body content for specialist review issue."""
        body = f"""# {specialist.specialist_type.value.title()} Specialist Review Required

## Original Issue
- **Issue**: #{request.issue_number}
- **Risk Level**: {request.risk_level.upper()}
- **Risk Score**: {request.risk_score:.2f}

## Risk Factors
{chr(10).join([f"- {factor}" for factor in request.primary_risk_factors])}

## Files Changed
{chr(10).join([f"- `{file}`" for file in request.files_changed[:10]])}
{"- ..." if len(request.files_changed) > 10 else ""}

## Evidence Checklist

### Mandatory Requirements
{chr(10).join([f"- [ ] {item}" for item in evidence_checklist.mandatory_items])}

### Recommended Items
{chr(10).join([f"- [ ] {item}" for item in evidence_checklist.recommended_items])}

{"### Custom Requirements" if evidence_checklist.custom_items else ""}
{chr(10).join([f"- [ ] {item}" for item in evidence_checklist.custom_items])}

## Instructions

1. Review the original issue #{request.issue_number} and associated changes
2. Complete the evidence checklist above
3. Provide your specialist assessment in a comment below
4. **Approve** or **Request Changes** based on your review
5. If approved, remove the `state:blocked` label
6. If changes needed, provide specific remediation guidance

## SLA Information
- **Assigned**: {datetime.now().isoformat()}
- **Response Required By**: {self._calculate_sla_deadline(request, specialist).isoformat()}
- **Specialist**: @{specialist.github_username}

## Escalation Chain
If this review is not completed within SLA, it will be escalated automatically.

---
*This issue was automatically created by the Risk-Based Manual Intervention Framework.*
"""
        return body
    
    def _generate_assignment_reasoning(self, specialist: SpecialistProfile, request: AssignmentRequest, score: float) -> List[str]:
        """Generate human-readable reasoning for the assignment."""
        reasoning = []
        
        reasoning.append(f"🎯 Assigned {specialist.name} ({specialist.specialist_type.value}) with confidence {score:.2f}")
        reasoning.append(f"📊 Current workload: {specialist.current_workload}/{specialist.max_concurrent_reviews} reviews")
        reasoning.append(f"⏱️ Average response time: {specialist.average_response_time_hours:.1f} hours")
        
        if specialist.status == SpecialistStatus.ON_CALL:
            reasoning.append("🚨 Specialist is currently on-call for urgent issues")
        
        if request.urgency_level in ['critical', 'high'] and specialist.escalation_level >= 2:
            reasoning.append(f"⚡ Senior specialist assigned for {request.urgency_level} priority")
        
        # Expertise matching
        matching_areas = []
        for expertise in specialist.expertise_areas:
            for risk_factor in request.primary_risk_factors:
                if expertise.lower().replace('_', ' ') in risk_factor.lower():
                    matching_areas.append(expertise)
        
        if matching_areas:
            reasoning.append(f"🎓 Expertise match: {', '.join(matching_areas[:3])}")
        
        return reasoning
    
    def _update_specialist_workload(self, specialist: SpecialistProfile, request: AssignmentRequest):
        """Update specialist workload after assignment."""
        specialist.current_workload += 1
        # In a real implementation, this would persist to database
        self.logger.debug(f"Updated {specialist.name} workload to {specialist.current_workload}")
    
    def _record_assignment(self, request: AssignmentRequest, result: AssignmentResult):
        """Record assignment in history for tracking and analytics."""
        assignment_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'issue_number': request.issue_number,
            'specialist_id': result.assigned_specialist.specialist_id if result.assigned_specialist else None,
            'specialist_type': request.specialist_type.value,
            'risk_score': request.risk_score,
            'risk_level': request.risk_level,
            'assignment_confidence': result.assignment_confidence,
            'sla_deadline': result.sla_deadline.isoformat() if result.sla_deadline else None,
            'github_issue_number': result.github_issue_number
        }
        
        self.assignment_history['assignments'].append(assignment_record)
        
        # Update metrics
        if 'total_assignments' not in self.assignment_history['metrics']:
            self.assignment_history['metrics']['total_assignments'] = 0
        self.assignment_history['metrics']['total_assignments'] += 1
        
        # In a real implementation, this would persist to file or database
        # self._save_assignment_history()
    
    def _create_no_specialist_result(self, request: AssignmentRequest) -> AssignmentResult:
        """Create result when no specialist is available."""
        return AssignmentResult(
            assigned_specialist=None,
            assignment_confidence=0.0,
            estimated_response_time=24.0,  # Default 24 hour fallback
            sla_deadline=datetime.now(timezone.utc) + timedelta(hours=24),
            evidence_checklist=self._generate_evidence_checklist(request),
            escalation_chain=[],
            github_issue_number=None,
            assignment_reasoning=["❌ No available specialists found", "🚨 Manual escalation required"],
            backup_specialists=[]
        )
    
    def _create_error_result(self, request: AssignmentRequest, error: str) -> AssignmentResult:
        """Create result when assignment fails with error."""
        return AssignmentResult(
            assigned_specialist=None,
            assignment_confidence=0.0,
            estimated_response_time=24.0,
            sla_deadline=datetime.now(timezone.utc) + timedelta(hours=24),
            evidence_checklist=self._generate_evidence_checklist(request),
            escalation_chain=[],
            github_issue_number=None,
            assignment_reasoning=[f"❌ Assignment error: {error}"],
            backup_specialists=[]
        )
    
    def get_specialist_workload_report(self) -> Dict[str, Any]:
        """Generate specialist workload and availability report."""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'specialists_by_type': {},
            'workload_summary': {},
            'availability_summary': {}
        }
        
        for specialist_type, specialists in self.specialists.items():
            type_data = []
            total_workload = 0
            total_capacity = 0
            available_count = 0
            
            for specialist in specialists:
                type_data.append({
                    'name': specialist.name,
                    'workload': f"{specialist.current_workload}/{specialist.max_concurrent_reviews}",
                    'status': specialist.status.value,
                    'avg_response_time': specialist.average_response_time_hours,
                    'utilization': specialist.current_workload / specialist.max_concurrent_reviews
                })
                
                total_workload += specialist.current_workload
                total_capacity += specialist.max_concurrent_reviews
                if specialist.status == SpecialistStatus.AVAILABLE:
                    available_count += 1
            
            report['specialists_by_type'][specialist_type.value] = type_data
            report['workload_summary'][specialist_type.value] = {
                'total_workload': total_workload,
                'total_capacity': total_capacity,
                'utilization': total_workload / total_capacity if total_capacity > 0 else 0,
                'available_specialists': available_count,
                'total_specialists': len(specialists)
            }
        
        return report

def main():
    """Command line interface for specialist assignment engine."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python specialist_assignment_engine.py <command> [args]")
        print("Commands:")
        print("  assign <issue_number> <specialist_type> <risk_level>  - Assign specialist")
        print("  workload-report                                      - Show specialist workload")
        print("  list-specialists [type]                              - List available specialists")
        print("  test-assignment                                      - Test assignment logic")
        return
    
    command = sys.argv[1]
    engine = SpecialistAssignmentEngine()
    
    if command == "assign" and len(sys.argv) >= 5:
        issue_num = int(sys.argv[2])
        specialist_type_str = sys.argv[3]
        risk_level = sys.argv[4]
        
        try:
            specialist_type = SpecialistType(specialist_type_str)
        except ValueError:
            print(f"Invalid specialist type: {specialist_type_str}")
            print(f"Valid types: {[t.value for t in SpecialistType]}")
            return 1
        
        # Create mock assignment request
        request = AssignmentRequest(
            issue_number=issue_num,
            risk_score=0.7,
            risk_level=risk_level,
            primary_risk_factors=["security_changes", "auth_modifications"],
            specialist_type=specialist_type,
            urgency_level="high",
            files_changed=["auth/login.py", "security/permissions.py"],
            estimated_review_time=4.0,
            special_requirements=[]
        )
        
        result = engine.assign_specialist(request)
        
        # Output assignment result
        output = {
            'assignment_successful': result.assigned_specialist is not None,
            'assigned_specialist': {
                'name': result.assigned_specialist.name if result.assigned_specialist else None,
                'email': result.assigned_specialist.email if result.assigned_specialist else None,
                'github_username': result.assigned_specialist.github_username if result.assigned_specialist else None
            } if result.assigned_specialist else None,
            'assignment_confidence': result.assignment_confidence,
            'estimated_response_time': result.estimated_response_time,
            'sla_deadline': result.sla_deadline.isoformat() if result.sla_deadline else None,
            'github_issue_number': result.github_issue_number,
            'reasoning': result.assignment_reasoning,
            'evidence_checklist': {
                'mandatory_items': result.evidence_checklist.mandatory_items,
                'recommended_items': result.evidence_checklist.recommended_items,
                'custom_items': result.evidence_checklist.custom_items
            }
        }
        
        print(json.dumps(output, indent=2))
        
    elif command == "workload-report":
        report = engine.get_specialist_workload_report()
        print(json.dumps(report, indent=2))
        
    elif command == "list-specialists":
        specialist_type_filter = sys.argv[2] if len(sys.argv) >= 3 else None
        
        if specialist_type_filter:
            try:
                filter_type = SpecialistType(specialist_type_filter)
                specialists_to_show = {filter_type: engine.specialists[filter_type]}
            except ValueError:
                print(f"Invalid specialist type: {specialist_type_filter}")
                return 1
        else:
            specialists_to_show = engine.specialists
        
        output = {}
        for spec_type, specialists in specialists_to_show.items():
            output[spec_type.value] = [
                {
                    'name': s.name,
                    'email': s.email,
                    'github_username': s.github_username,
                    'expertise_areas': s.expertise_areas,
                    'workload': f"{s.current_workload}/{s.max_concurrent_reviews}",
                    'status': s.status.value,
                    'avg_response_time': s.average_response_time_hours
                }
                for s in specialists
            ]
        
        print(json.dumps(output, indent=2))
        
    elif command == "test-assignment":
        # Test assignment with mock data
        request = AssignmentRequest(
            issue_number=999,
            risk_score=0.85,
            risk_level="high",
            primary_risk_factors=["security_changes", "authentication_bypass", "privilege_escalation"],
            specialist_type=SpecialistType.SECURITY,
            urgency_level="critical",
            files_changed=["auth/oauth.py", "security/jwt_handler.py", "config/security.yaml"],
            estimated_review_time=6.0,
            special_requirements=["executive_approval"]
        )
        
        result = engine.assign_specialist(request)
        
        print("🧪 Test Assignment Result:")
        print(f"Assigned: {result.assigned_specialist.name if result.assigned_specialist else 'None'}")
        print(f"Confidence: {result.assignment_confidence:.2f}")
        print(f"SLA Deadline: {result.sla_deadline}")
        print(f"GitHub Issue: #{result.github_issue_number}" if result.github_issue_number else "GitHub Issue: Not created")
        print("\nReasoning:")
        for reason in result.assignment_reasoning:
            print(f"  {reason}")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())