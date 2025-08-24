#!/usr/bin/env python3
"""
Adversarial Issue Generation Engine - Issue #146 Implementation
Layer 6 of 8-Layer Adversarial Validation Architecture

Architecture: Automated GitHub Issue Creation and Management System
Purpose: Generate GitHub issues for validation failures, improvements, and systematic tracking
Integration: Takes input from Quality Orchestration and generates actionable GitHub issues
"""

import os
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict

# Import our validation system components
try:
    from adversarial_quality_orchestration_layer import QualityWorkflow, QualityDecision, QualityMetrics
    from adversarial_knowledge_integration_layer import KnowledgeRecommendation, ValidationPattern
    from adversarial_validation_execution_engine import ValidationExecution, ValidationResult
except ImportError:
    # Fallbacks for standalone execution
    QualityWorkflow = None
    QualityDecision = None
    QualityMetrics = None
    KnowledgeRecommendation = None
    ValidationPattern = None
    ValidationExecution = None
    ValidationResult = None

class IssueType(Enum):
    VALIDATION_FAILURE = "validation_failure"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    RELIABILITY_CONCERN = "reliability_concern"
    IMPROVEMENT_SUGGESTION = "improvement_suggestion"
    INVESTIGATION_REQUIRED = "investigation_required"
    PATTERN_IMPLEMENTATION = "pattern_implementation"
    KNOWLEDGE_UPDATE = "knowledge_update"

class IssuePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IssueStatus(Enum):
    PENDING = "pending"
    CREATED = "created"
    FAILED = "failed"
    DUPLICATE = "duplicate"

@dataclass
class ValidationIssue:
    """GitHub issue generated from validation results"""
    issue_id: str
    feature_id: str
    issue_type: IssueType
    priority: IssuePriority
    title: str
    description: str
    labels: List[str]
    assignees: List[str]
    milestone: Optional[str]
    evidence_references: List[str]
    validation_data: Dict[str, Any]
    recommendation_references: List[str]
    implementation_guidance: Dict[str, Any]
    acceptance_criteria: List[str]
    related_issues: List[str]
    created_at: str
    github_issue_number: Optional[int]
    github_issue_url: Optional[str]
    issue_status: IssueStatus
    generation_metadata: Dict[str, Any]

class AdversarialIssueGenerator:
    """
    Automated GitHub issue generation system for validation results
    
    Capabilities:
    1. Generate issues from validation failures
    2. Create security vulnerability issues
    3. Generate performance improvement issues
    4. Create investigation issues for unclear results
    5. Generate pattern implementation issues
    6. Create knowledge update issues
    7. Manage issue priorities and labels
    8. Avoid duplicate issue creation
    9. Link related issues and evidence
    10. Provide implementation guidance in issues
    """
    
    def __init__(self, rif_root: str = None, github_repo: str = None):
        self.rif_root = rif_root or os.getcwd()
        self.github_repo = github_repo or self._detect_github_repo()
        self.issue_store = os.path.join(self.rif_root, "knowledge", "generated_issues")
        self.issue_db = os.path.join(self.issue_store, "issue_generation.db")
        self.generation_log = os.path.join(self.issue_store, "generation.log")
        
        # Issue generation templates
        self.issue_templates = self._load_issue_templates()
        
        # Label mappings
        self.label_mappings = {
            IssueType.VALIDATION_FAILURE: ["validation", "bug", "rif-generated"],
            IssueType.SECURITY_VULNERABILITY: ["security", "vulnerability", "critical", "rif-generated"],
            IssueType.PERFORMANCE_ISSUE: ["performance", "optimization", "rif-generated"],
            IssueType.RELIABILITY_CONCERN: ["reliability", "stability", "rif-generated"],
            IssueType.IMPROVEMENT_SUGGESTION: ["enhancement", "suggestion", "rif-generated"],
            IssueType.INVESTIGATION_REQUIRED: ["investigation", "analysis", "rif-generated"],
            IssueType.PATTERN_IMPLEMENTATION: ["pattern", "implementation", "rif-generated"],
            IssueType.KNOWLEDGE_UPDATE: ["knowledge", "documentation", "rif-generated"]
        }
        
        # Priority to GitHub priority mapping
        self.github_priority_labels = {
            IssuePriority.CRITICAL: "priority:critical",
            IssuePriority.HIGH: "priority:high", 
            IssuePriority.MEDIUM: "priority:medium",
            IssuePriority.LOW: "priority:low"
        }
        
        # Generated issue tracking
        self.generated_issues = {}
        self.issue_creation_stats = defaultdict(int)
        
        self._init_issue_store()
        self._init_database()
    
    def _init_issue_store(self):
        """Initialize issue generation storage"""
        directories = [
            self.issue_store,
            os.path.join(self.issue_store, "validation_issues"),
            os.path.join(self.issue_store, "security_issues"),
            os.path.join(self.issue_store, "performance_issues"),
            os.path.join(self.issue_store, "investigation_issues"),
            os.path.join(self.issue_store, "improvement_issues"),
            os.path.join(self.issue_store, "templates")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _init_database(self):
        """Initialize issue generation database"""
        conn = sqlite3.connect(self.issue_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_issues (
                issue_id TEXT PRIMARY KEY,
                feature_id TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                labels TEXT,
                assignees TEXT,
                milestone TEXT,
                evidence_references TEXT,
                validation_data TEXT,
                recommendation_references TEXT,
                implementation_guidance TEXT,
                acceptance_criteria TEXT,
                related_issues TEXT,
                created_at TEXT NOT NULL,
                github_issue_number INTEGER,
                github_issue_url TEXT,
                issue_status TEXT NOT NULL,
                generation_metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS issue_generation_sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT NOT NULL,
                input_workflows INTEGER,
                issues_generated INTEGER,
                issues_created INTEGER,
                issues_failed INTEGER,
                session_metadata TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_detection (
                detection_id TEXT PRIMARY KEY,
                feature_id TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                issue_hash TEXT NOT NULL,
                original_issue_id TEXT NOT NULL,
                duplicate_issue_id TEXT NOT NULL,
                detected_at TEXT NOT NULL
            )
        ''')
        
        # Performance indexes
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_issues_feature ON validation_issues(feature_id)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_issues_type ON validation_issues(issue_type)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_issues_status ON validation_issues(issue_status)''')
        
        conn.commit()
        conn.close()
    
    def generate_issues_from_workflows(self, workflows: List[QualityWorkflow],
                                     quality_metrics: List[QualityMetrics] = None,
                                     recommendations: List[KnowledgeRecommendation] = None) -> Dict[str, List[str]]:
        """
        Generate GitHub issues from completed quality workflows
        
        Args:
            workflows: Completed quality workflows
            quality_metrics: Associated quality metrics
            recommendations: Knowledge-based recommendations
        
        Returns:
            Dictionary mapping issue_type to list of generated issue_ids
        """
        session_id = f"issue_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()
        
        self._log(f"Starting issue generation session: {session_id}")
        
        generated_issues = defaultdict(list)
        issues_created = 0
        issues_failed = 0
        
        # Process each workflow for issue generation
        for workflow in workflows:
            try:
                workflow_issues = self._generate_workflow_issues(
                    workflow, quality_metrics, recommendations
                )
                
                for issue in workflow_issues:
                    # Check for duplicates
                    if self._is_duplicate_issue(issue):
                        self._log(f"Duplicate issue detected for feature {issue.feature_id}")
                        issue.issue_status = IssueStatus.DUPLICATE
                        self._store_validation_issue(issue)
                        continue
                    
                    # Generate GitHub issue
                    github_success = self._create_github_issue(issue)
                    
                    if github_success:
                        issue.issue_status = IssueStatus.CREATED
                        issues_created += 1
                        self.issue_creation_stats[issue.issue_type.value] += 1
                    else:
                        issue.issue_status = IssueStatus.FAILED
                        issues_failed += 1
                    
                    # Store issue
                    self._store_validation_issue(issue)
                    generated_issues[issue.issue_type.value].append(issue.issue_id)
                    self.generated_issues[issue.issue_id] = issue
                
            except Exception as e:
                self._log(f"Error generating issues for workflow {workflow.workflow_id}: {str(e)}")
                issues_failed += 1
        
        # Store session results
        self._store_generation_session(
            session_id, started_at, datetime.now(), len(workflows),
            sum(len(issues) for issues in generated_issues.values()),
            issues_created, issues_failed
        )
        
        self._log(f"Issue generation complete. Generated: {len(generated_issues)}, Created: {issues_created}, Failed: {issues_failed}")
        return dict(generated_issues)
    
    def _generate_workflow_issues(self, workflow: QualityWorkflow,
                                 quality_metrics: List[QualityMetrics] = None,
                                 recommendations: List[KnowledgeRecommendation] = None) -> List[ValidationIssue]:
        """Generate specific issues for a workflow"""
        issues = []
        
        # Get quality metrics for this workflow
        feature_metrics = None
        if quality_metrics:
            feature_metrics = next((m for m in quality_metrics if m.feature_id == workflow.feature_id), None)
        
        # Get recommendations for this workflow
        feature_recommendations = []
        if recommendations:
            feature_recommendations = [r for r in recommendations if r.target_feature_id == workflow.feature_id]
        
        # Generate issues based on workflow decision
        if workflow.final_decision == QualityDecision.REJECT:
            issues.extend(self._generate_rejection_issues(workflow, feature_metrics))
        
        elif workflow.final_decision == QualityDecision.INVESTIGATE:
            issues.extend(self._generate_investigation_issues(workflow, feature_metrics))
        
        elif workflow.final_decision == QualityDecision.CONDITIONAL:
            issues.extend(self._generate_conditional_issues(workflow, feature_metrics))
        
        # Generate recommendation-based issues
        if feature_recommendations:
            issues.extend(self._generate_recommendation_issues(workflow, feature_recommendations))
        
        # Generate security-specific issues
        if feature_metrics and feature_metrics.security_score < 60:
            issues.extend(self._generate_security_issues(workflow, feature_metrics))
        
        # Generate performance issues
        if feature_metrics and feature_metrics.performance_score < 70:
            issues.extend(self._generate_performance_issues(workflow, feature_metrics))
        
        return issues
    
    def _generate_rejection_issues(self, workflow: QualityWorkflow, 
                                  metrics: QualityMetrics = None) -> List[ValidationIssue]:
        """Generate issues for rejected features"""
        issues = []
        
        # Primary validation failure issue
        issue = self._create_validation_issue(
            workflow.feature_id,
            IssueType.VALIDATION_FAILURE,
            IssuePriority.HIGH,
            f"Feature validation failed: {workflow.feature_id}",
            self._generate_failure_description(workflow, metrics),
            workflow,
            metrics
        )
        issues.append(issue)
        
        # Additional specific issues based on failure reasons
        if metrics:
            if metrics.security_score < 40:
                security_issue = self._create_validation_issue(
                    workflow.feature_id,
                    IssueType.SECURITY_VULNERABILITY,
                    IssuePriority.CRITICAL,
                    f"Critical security vulnerabilities in {workflow.feature_id}",
                    self._generate_security_description(workflow, metrics),
                    workflow,
                    metrics
                )
                issues.append(security_issue)
            
            if metrics.reliability_score < 50:
                reliability_issue = self._create_validation_issue(
                    workflow.feature_id,
                    IssueType.RELIABILITY_CONCERN,
                    IssuePriority.HIGH,
                    f"Reliability concerns in {workflow.feature_id}",
                    self._generate_reliability_description(workflow, metrics),
                    workflow,
                    metrics
                )
                issues.append(reliability_issue)
        
        return issues
    
    def _generate_investigation_issues(self, workflow: QualityWorkflow,
                                     metrics: QualityMetrics = None) -> List[ValidationIssue]:
        """Generate issues for features requiring investigation"""
        issues = []
        
        investigation_issue = self._create_validation_issue(
            workflow.feature_id,
            IssueType.INVESTIGATION_REQUIRED,
            IssuePriority.MEDIUM,
            f"Investigation required for {workflow.feature_id}",
            self._generate_investigation_description(workflow, metrics),
            workflow,
            metrics
        )
        issues.append(investigation_issue)
        
        return issues
    
    def _generate_conditional_issues(self, workflow: QualityWorkflow,
                                   metrics: QualityMetrics = None) -> List[ValidationIssue]:
        """Generate issues for conditionally accepted features"""
        issues = []
        
        # Monitoring/improvement issue for conditional acceptance
        improvement_issue = self._create_validation_issue(
            workflow.feature_id,
            IssueType.IMPROVEMENT_SUGGESTION,
            IssuePriority.MEDIUM,
            f"Improvement needed for {workflow.feature_id}",
            self._generate_improvement_description(workflow, metrics),
            workflow,
            metrics
        )
        issues.append(improvement_issue)
        
        return issues
    
    def _generate_recommendation_issues(self, workflow: QualityWorkflow,
                                      recommendations: List[KnowledgeRecommendation]) -> List[ValidationIssue]:
        """Generate issues from knowledge recommendations"""
        issues = []
        
        for recommendation in recommendations[:2]:  # Limit to top 2 recommendations
            if recommendation.recommendation_type == "failure_avoidance":
                priority = IssuePriority.HIGH
            elif recommendation.recommendation_type == "risk_mitigation":
                priority = IssuePriority.MEDIUM
            else:
                priority = IssuePriority.LOW
            
            rec_issue = self._create_validation_issue(
                workflow.feature_id,
                IssueType.PATTERN_IMPLEMENTATION,
                priority,
                f"Implement pattern recommendation for {workflow.feature_id}",
                self._generate_recommendation_description(recommendation),
                workflow,
                None,
                [recommendation.recommendation_id]
            )
            issues.append(rec_issue)
        
        return issues
    
    def _generate_security_issues(self, workflow: QualityWorkflow,
                                metrics: QualityMetrics) -> List[ValidationIssue]:
        """Generate specific security issues"""
        issues = []
        
        if metrics.security_score < 40:
            priority = IssuePriority.CRITICAL
        elif metrics.security_score < 60:
            priority = IssuePriority.HIGH
        else:
            priority = IssuePriority.MEDIUM
        
        security_issue = self._create_validation_issue(
            workflow.feature_id,
            IssueType.SECURITY_VULNERABILITY,
            priority,
            f"Security vulnerabilities detected in {workflow.feature_id}",
            self._generate_detailed_security_description(workflow, metrics),
            workflow,
            metrics
        )
        issues.append(security_issue)
        
        return issues
    
    def _generate_performance_issues(self, workflow: QualityWorkflow,
                                   metrics: QualityMetrics) -> List[ValidationIssue]:
        """Generate specific performance issues"""
        issues = []
        
        priority = IssuePriority.HIGH if metrics.performance_score < 50 else IssuePriority.MEDIUM
        
        performance_issue = self._create_validation_issue(
            workflow.feature_id,
            IssueType.PERFORMANCE_ISSUE,
            priority,
            f"Performance optimization needed for {workflow.feature_id}",
            self._generate_performance_description(workflow, metrics),
            workflow,
            metrics
        )
        issues.append(performance_issue)
        
        return issues
    
    def _create_validation_issue(self, feature_id: str, issue_type: IssueType, priority: IssuePriority,
                               title: str, description: str, workflow: QualityWorkflow,
                               metrics: QualityMetrics = None, 
                               recommendation_refs: List[str] = None) -> ValidationIssue:
        """Create a standardized validation issue"""
        issue_id = f"{issue_type.value}_{feature_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate labels
        labels = self.label_mappings[issue_type].copy()
        labels.append(self.github_priority_labels[priority])
        
        # Generate acceptance criteria
        acceptance_criteria = self._generate_acceptance_criteria(issue_type, workflow, metrics)
        
        # Generate implementation guidance
        implementation_guidance = self._generate_implementation_guidance(issue_type, workflow, metrics)
        
        # Extract evidence references
        evidence_references = []
        if workflow.phase_results.get("evidence_collection"):
            evidence_data = workflow.phase_results["evidence_collection"]
            evidence_references = evidence_data.get("artifacts", [])
        
        return ValidationIssue(
            issue_id=issue_id,
            feature_id=feature_id,
            issue_type=issue_type,
            priority=priority,
            title=title,
            description=description,
            labels=labels,
            assignees=self._determine_assignees(issue_type, feature_id),
            milestone=self._determine_milestone(issue_type, priority),
            evidence_references=evidence_references,
            validation_data={
                "workflow_id": workflow.workflow_id,
                "final_decision": workflow.final_decision.value if workflow.final_decision else None,
                "decision_rationale": workflow.decision_rationale,
                "quality_metrics": asdict(metrics) if metrics else None
            },
            recommendation_references=recommendation_refs or [],
            implementation_guidance=implementation_guidance,
            acceptance_criteria=acceptance_criteria,
            related_issues=[],
            created_at=datetime.now().isoformat(),
            github_issue_number=None,
            github_issue_url=None,
            issue_status=IssueStatus.PENDING,
            generation_metadata={
                "generator": "AdversarialIssueGenerator",
                "workflow_type": workflow.workflow_type,
                "validation_level": workflow.requested_validation_level.value
            }
        )
    
    # Issue description generators
    def _generate_failure_description(self, workflow: QualityWorkflow, metrics: QualityMetrics = None) -> str:
        """Generate description for validation failure issues"""
        description = f"""# Feature Validation Failure Report

## Feature Information
- **Feature ID**: {workflow.feature_id}
- **Workflow Type**: {workflow.workflow_type}
- **Validation Level**: {workflow.requested_validation_level.value}
- **Final Decision**: {workflow.final_decision.value if workflow.final_decision else 'Unknown'}

## Failure Summary
{workflow.decision_rationale or 'Feature failed to meet validation criteria'}

"""
        
        if metrics:
            description += f"""## Quality Metrics
- **Overall Score**: {metrics.overall_score:.1f}/100
- **Functionality Score**: {metrics.functionality_score:.1f}/100
- **Reliability Score**: {metrics.reliability_score:.1f}/100  
- **Security Score**: {metrics.security_score:.1f}/100
- **Performance Score**: {metrics.performance_score:.1f}/100
- **Risk Assessment**: {metrics.risk_assessment}

"""
        
        description += f"""## Workflow Details
- **Created**: {workflow.created_at}
- **Completed**: {workflow.completed_at or 'In Progress'}
- **Phases Completed**: {len(workflow.phases_completed)}/7

## Next Steps
1. Review the validation failures detailed in the workflow results
2. Address the specific issues identified in the quality metrics
3. Implement necessary fixes and improvements
4. Re-run validation to verify fixes

## Evidence References
See validation artifacts and evidence collected during the workflow execution.

---
*This issue was automatically generated by the Adversarial Feature Validation System*
"""
        
        return description
    
    def _generate_security_description(self, workflow: QualityWorkflow, metrics: QualityMetrics) -> str:
        """Generate description for security vulnerability issues"""
        return f"""# Critical Security Vulnerabilities Detected

## Security Assessment Summary
- **Feature**: {workflow.feature_id}
- **Security Score**: {metrics.security_score:.1f}/100 (Critical threshold: 60)
- **Risk Level**: {metrics.risk_assessment}
- **Confidence**: {metrics.confidence_level:.1f}%

## Identified Vulnerabilities
The adversarial security validation has identified critical security vulnerabilities that must be addressed before this feature can be deployed.

## Immediate Actions Required
1. **Security Review**: Conduct immediate security review of the feature
2. **Vulnerability Assessment**: Detailed vulnerability assessment and penetration testing
3. **Security Hardening**: Implement security hardening measures
4. **Re-validation**: Full security re-validation required

## Security Validation Details
{workflow.decision_rationale or 'Critical security vulnerabilities detected during adversarial testing'}

---
*This critical security issue was automatically generated by the Adversarial Security Validation System*
"""
    
    def _generate_reliability_description(self, workflow: QualityWorkflow, metrics: QualityMetrics) -> str:
        """Generate description for reliability concern issues"""
        return f"""# Reliability Concerns Identified

## Reliability Assessment
- **Feature**: {workflow.feature_id}
- **Reliability Score**: {metrics.reliability_score:.1f}/100 (Minimum threshold: 75)
- **Overall Risk**: {metrics.risk_assessment}

## Reliability Issues
The feature has demonstrated reliability concerns that need to be addressed:

{workflow.decision_rationale or 'Reliability issues detected during validation'}

## Recommended Actions
1. **Stability Testing**: Extended stability and stress testing
2. **Error Handling**: Review and improve error handling mechanisms  
3. **Recovery Procedures**: Implement robust recovery procedures
4. **Monitoring**: Add comprehensive reliability monitoring

---
*This reliability issue was automatically generated by the Adversarial Validation System*
"""
    
    def _generate_investigation_description(self, workflow: QualityWorkflow, metrics: QualityMetrics = None) -> str:
        """Generate description for investigation required issues"""
        return f"""# Investigation Required

## Investigation Summary
- **Feature**: {workflow.feature_id}
- **Reason**: {workflow.decision_rationale or 'Unclear validation results require investigation'}

## Investigation Areas
The following areas require detailed investigation:

1. **Validation Results Analysis**
2. **Evidence Review**
3. **Risk Assessment**
4. **Decision Clarification**

"""
        
        if metrics:
            description = f"""## Quality Metrics for Review
- **Overall Score**: {metrics.overall_score:.1f}/100
- **Confidence Level**: {metrics.confidence_level:.1f}%
- **Evidence Completeness**: {metrics.evidence_completeness:.1f}%

"""
        
        description += """## Investigation Tasks
- [ ] Review all validation evidence
- [ ] Analyze quality metrics in detail
- [ ] Determine if additional testing is needed
- [ ] Make final validation decision

---
*This investigation request was automatically generated by the Adversarial Validation System*
"""
        
        return description
    
    def _generate_improvement_description(self, workflow: QualityWorkflow, metrics: QualityMetrics = None) -> str:
        """Generate description for improvement suggestion issues"""
        return f"""# Feature Improvement Recommendations

## Feature Status
- **Feature**: {workflow.feature_id} 
- **Status**: Conditionally Accepted
- **Decision**: {workflow.decision_rationale or 'Feature requires improvements for full acceptance'}

## Improvement Areas
The following improvements are recommended:

"""
        
        if metrics:
            improvements = []
            if metrics.functionality_score < 85:
                improvements.append(f"- **Functionality**: Score {metrics.functionality_score:.1f}/100 - enhance core functionality")
            if metrics.reliability_score < 85:
                improvements.append(f"- **Reliability**: Score {metrics.reliability_score:.1f}/100 - improve stability")
            if metrics.security_score < 85:
                improvements.append(f"- **Security**: Score {metrics.security_score:.1f}/100 - strengthen security")
            if metrics.performance_score < 85:
                improvements.append(f"- **Performance**: Score {metrics.performance_score:.1f}/100 - optimize performance")
            
            if improvements:
                description += "\n".join(improvements) + "\n\n"
        
        description += """## Implementation Plan
1. **Assessment**: Review current implementation
2. **Planning**: Create improvement implementation plan
3. **Implementation**: Execute improvements
4. **Validation**: Re-validate with improvements
5. **Documentation**: Update documentation

---
*This improvement suggestion was automatically generated by the Adversarial Validation System*
"""
        
        return description
    
    def _generate_recommendation_description(self, recommendation: KnowledgeRecommendation) -> str:
        """Generate description for recommendation-based issues"""
        return f"""# Knowledge-Based Pattern Recommendation

## Recommendation Summary
- **Type**: {recommendation.recommendation_type.replace('_', ' ').title()}
- **Confidence**: {recommendation.confidence_level.value}
- **Risk Assessment**: {recommendation.risk_assessment}
- **Priority**: {recommendation.priority}

## Recommendation Details
{recommendation.recommendation_text}

## Implementation Guidance
"""
        
        if recommendation.implementation_guidance:
            for step_num, step in enumerate(recommendation.implementation_guidance.get("implementation_steps", []), 1):
                description += f"{step_num}. {step}\n"
        
        description += f"""
## Validation Requirements
"""
        for req in recommendation.validation_requirements:
            description += f"- {req}\n"
        
        description += """
## Pattern Evidence
This recommendation is based on learned patterns from similar feature validations.

---
*This pattern recommendation was automatically generated by the Knowledge Integration System*
"""
        
        return description
    
    def _generate_detailed_security_description(self, workflow: QualityWorkflow, metrics: QualityMetrics) -> str:
        """Generate detailed security issue description"""
        return f"""# Security Vulnerability Assessment

## Vulnerability Summary
- **Feature**: {workflow.feature_id}
- **Security Score**: {metrics.security_score:.1f}/100
- **Risk Level**: {metrics.risk_assessment.upper()}
- **Validation Level**: {workflow.requested_validation_level.value}

## Security Analysis
The adversarial security validation has identified multiple security concerns:

{workflow.decision_rationale or 'Security vulnerabilities detected during adversarial testing'}

## Security Requirements
- [ ] Input validation and sanitization
- [ ] Authentication and authorization checks
- [ ] Data encryption and protection
- [ ] Access control implementation
- [ ] Security logging and monitoring

## Remediation Plan
1. **Security Audit**: Conduct comprehensive security audit
2. **Vulnerability Fix**: Address identified vulnerabilities
3. **Security Testing**: Perform security-focused testing
4. **Documentation**: Update security documentation
5. **Re-validation**: Complete security re-validation

---
*This security assessment was automatically generated by the Adversarial Security Validation System*
"""
    
    def _generate_performance_description(self, workflow: QualityWorkflow, metrics: QualityMetrics) -> str:
        """Generate performance issue description"""
        return f"""# Performance Optimization Required

## Performance Analysis
- **Feature**: {workflow.feature_id}
- **Performance Score**: {metrics.performance_score:.1f}/100
- **Overall Score Impact**: {metrics.overall_score:.1f}/100

## Performance Issues
The feature demonstrates performance issues that require optimization:

{workflow.decision_rationale or 'Performance issues identified during validation'}

## Optimization Areas
- [ ] Execution time optimization
- [ ] Memory usage optimization  
- [ ] Resource utilization efficiency
- [ ] Scalability improvements
- [ ] Caching and optimization strategies

## Performance Requirements
1. **Benchmarking**: Establish performance baselines
2. **Profiling**: Identify performance bottlenecks
3. **Optimization**: Implement performance improvements
4. **Testing**: Validate performance improvements
5. **Monitoring**: Implement performance monitoring

---
*This performance issue was automatically generated by the Adversarial Performance Validation System*
"""
    
    # Utility methods
    def _generate_acceptance_criteria(self, issue_type: IssueType, workflow: QualityWorkflow,
                                    metrics: QualityMetrics = None) -> List[str]:
        """Generate acceptance criteria for issue resolution"""
        criteria = []
        
        if issue_type == IssueType.VALIDATION_FAILURE:
            criteria = [
                "All validation tests must pass",
                "Overall quality score must be ≥75",
                "No critical security vulnerabilities",
                "Reliability score must be ≥75"
            ]
        elif issue_type == IssueType.SECURITY_VULNERABILITY:
            criteria = [
                "Security score must be ≥80",
                "All identified vulnerabilities must be fixed",
                "Security audit must pass",
                "Penetration testing must pass"
            ]
        elif issue_type == IssueType.PERFORMANCE_ISSUE:
            criteria = [
                "Performance score must be ≥75",
                "Performance benchmarks must be met",
                "Resource usage must be optimized"
            ]
        else:
            criteria = [
                "Issue must be investigated and resolved",
                "Documentation must be updated",
                "Re-validation must pass"
            ]
        
        return criteria
    
    def _generate_implementation_guidance(self, issue_type: IssueType, workflow: QualityWorkflow,
                                        metrics: QualityMetrics = None) -> Dict[str, Any]:
        """Generate implementation guidance for issue resolution"""
        guidance = {
            "priority_actions": [],
            "technical_steps": [],
            "validation_requirements": [],
            "success_criteria": []
        }
        
        if issue_type == IssueType.VALIDATION_FAILURE:
            guidance["priority_actions"] = [
                "Review validation failure details",
                "Identify root cause of failures",
                "Plan remediation strategy"
            ]
            guidance["technical_steps"] = [
                "Fix identified issues in code",
                "Update tests and documentation",
                "Run validation suite"
            ]
        elif issue_type == IssueType.SECURITY_VULNERABILITY:
            guidance["priority_actions"] = [
                "Immediate security review",
                "Vulnerability assessment",
                "Security hardening"
            ]
            guidance["technical_steps"] = [
                "Implement security fixes",
                "Add security tests",
                "Update security documentation"
            ]
        
        return guidance
    
    def _determine_assignees(self, issue_type: IssueType, feature_id: str) -> List[str]:
        """Determine appropriate assignees for issue type"""
        # This would integrate with team configuration
        assignee_mapping = {
            IssueType.SECURITY_VULNERABILITY: ["security-team"],
            IssueType.PERFORMANCE_ISSUE: ["performance-team"],
            IssueType.VALIDATION_FAILURE: ["development-team"],
            IssueType.INVESTIGATION_REQUIRED: ["architecture-team"]
        }
        
        return assignee_mapping.get(issue_type, ["development-team"])
    
    def _determine_milestone(self, issue_type: IssueType, priority: IssuePriority) -> Optional[str]:
        """Determine appropriate milestone for issue"""
        if priority == IssuePriority.CRITICAL:
            return "Critical Security & Reliability"
        elif issue_type == IssueType.SECURITY_VULNERABILITY:
            return "Security Hardening"
        elif issue_type == IssueType.PERFORMANCE_ISSUE:
            return "Performance Optimization"
        else:
            return "Quality Improvements"
    
    def _is_duplicate_issue(self, issue: ValidationIssue) -> bool:
        """Check if issue is a duplicate of existing issue"""
        # Generate hash for issue content
        issue_hash = self._generate_issue_hash(issue)
        
        # Check database for existing issues with same hash
        conn = sqlite3.connect(self.issue_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT issue_id FROM validation_issues 
            WHERE feature_id = ? AND issue_type = ? 
            AND issue_status != 'duplicate'
            AND created_at > datetime('now', '-7 days')
        ''', (issue.feature_id, issue.issue_type.value))
        
        existing_issues = cursor.fetchall()
        conn.close()
        
        # Simple duplicate detection - same feature + issue type within 7 days
        return len(existing_issues) > 0
    
    def _generate_issue_hash(self, issue: ValidationIssue) -> str:
        """Generate hash for duplicate detection"""
        content = f"{issue.feature_id}_{issue.issue_type.value}_{issue.title}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_github_issue(self, issue: ValidationIssue) -> bool:
        """Create actual GitHub issue using gh CLI"""
        try:
            if not self.github_repo:
                self._log("No GitHub repository configured, skipping issue creation")
                return False
            
            # Prepare gh CLI command
            cmd = [
                'gh', 'issue', 'create',
                '--repo', self.github_repo,
                '--title', issue.title,
                '--body', issue.description
            ]
            
            # Add labels
            if issue.labels:
                cmd.extend(['--label', ','.join(issue.labels)])
            
            # Add assignees
            if issue.assignees:
                cmd.extend(['--assignee', ','.join(issue.assignees)])
            
            # Add milestone
            if issue.milestone:
                cmd.extend(['--milestone', issue.milestone])
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Extract issue URL from output
                issue_url = result.stdout.strip()
                issue.github_issue_url = issue_url
                
                # Extract issue number from URL
                if '/issues/' in issue_url:
                    issue_number = issue_url.split('/issues/')[-1]
                    try:
                        issue.github_issue_number = int(issue_number)
                    except ValueError:
                        pass
                
                self._log(f"Created GitHub issue: {issue_url}")
                return True
            else:
                self._log(f"Failed to create GitHub issue: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self._log("GitHub issue creation timed out")
            return False
        except Exception as e:
            self._log(f"Error creating GitHub issue: {str(e)}")
            return False
    
    def _detect_github_repo(self) -> Optional[str]:
        """Detect GitHub repository from git remote"""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True, text=True, cwd=self.rif_root
            )
            
            if result.returncode == 0:
                remote_url = result.stdout.strip()
                
                # Parse GitHub repo from URL
                if 'github.com' in remote_url:
                    if remote_url.startswith('https://'):
                        # https://github.com/user/repo.git
                        repo_part = remote_url.replace('https://github.com/', '').replace('.git', '')
                    elif remote_url.startswith('git@'):
                        # git@github.com:user/repo.git
                        repo_part = remote_url.replace('git@github.com:', '').replace('.git', '')
                    else:
                        return None
                    
                    return repo_part
            
        except Exception as e:
            self._log(f"Error detecting GitHub repo: {str(e)}")
        
        return None
    
    def _load_issue_templates(self) -> Dict[str, str]:
        """Load issue templates from templates directory"""
        templates = {}
        templates_dir = os.path.join(self.issue_store, "templates")
        
        # Create default templates if they don't exist
        default_templates = {
            "validation_failure.md": "# Validation Failure\n\n{description}",
            "security_vulnerability.md": "# Security Vulnerability\n\n{description}",
            "performance_issue.md": "# Performance Issue\n\n{description}"
        }
        
        for template_name, template_content in default_templates.items():
            template_path = os.path.join(templates_dir, template_name)
            if not os.path.exists(template_path):
                with open(template_path, 'w') as f:
                    f.write(template_content)
            
            templates[template_name.replace('.md', '')] = template_content
        
        return templates
    
    # Database operations
    def _store_validation_issue(self, issue: ValidationIssue):
        """Store validation issue in database"""
        conn = sqlite3.connect(self.issue_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO validation_issues (
                issue_id, feature_id, issue_type, priority, title, description,
                labels, assignees, milestone, evidence_references, validation_data,
                recommendation_references, implementation_guidance, acceptance_criteria,
                related_issues, created_at, github_issue_number, github_issue_url,
                issue_status, generation_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            issue.issue_id, issue.feature_id, issue.issue_type.value, issue.priority.value,
            issue.title, issue.description, json.dumps(issue.labels),
            json.dumps(issue.assignees), issue.milestone, json.dumps(issue.evidence_references),
            json.dumps(issue.validation_data), json.dumps(issue.recommendation_references),
            json.dumps(issue.implementation_guidance), json.dumps(issue.acceptance_criteria),
            json.dumps(issue.related_issues), issue.created_at, issue.github_issue_number,
            issue.github_issue_url, issue.issue_status.value, json.dumps(issue.generation_metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_generation_session(self, session_id: str, started_at: datetime, completed_at: datetime,
                                 input_workflows: int, issues_generated: int, issues_created: int,
                                 issues_failed: int):
        """Store issue generation session metadata"""
        conn = sqlite3.connect(self.issue_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO issue_generation_sessions (
                session_id, session_type, input_workflows, issues_generated,
                issues_created, issues_failed, session_metadata,
                started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, "workflow_issues", input_workflows, issues_generated,
            issues_created, issues_failed, json.dumps({}),
            started_at.isoformat(), completed_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _log(self, message: str):
        """Log issue generation events"""
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp}: {message}\n"
        
        with open(self.generation_log, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get issue generation statistics"""
        return {
            "total_issues_generated": len(self.generated_issues),
            "issues_by_type": dict(self.issue_creation_stats),
            "github_repo": self.github_repo,
            "generation_capabilities": {
                "validation_failures": True,
                "security_vulnerabilities": True,
                "performance_issues": True,
                "investigation_issues": True,
                "recommendation_issues": True,
                "duplicate_detection": True
            }
        }

def main():
    """Main execution for issue generation testing"""
    generator = AdversarialIssueGenerator()
    
    print("Testing adversarial issue generation engine...")
    
    # Test generation stats
    stats = generator.get_generation_stats()
    print(f"Generation stats: {stats}")
    
    # Test with mock data (since we don't have real workflow results)
    mock_workflows = []  # Would be populated with real QualityWorkflow objects
    
    if mock_workflows:
        generated_issues = generator.generate_issues_from_workflows(mock_workflows)
        print(f"Generated issues: {generated_issues}")
    else:
        print("No workflow results to generate issues from (expected for testing)")

if __name__ == "__main__":
    main()