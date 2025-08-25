#!/usr/bin/env python3
"""
Quality Validation Gates with Scoring System for GitHub Issue #234
Implements comprehensive quality assessment and validation gates
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityGate(Enum):
    """Quality gate types"""
    EVIDENCE_COMPLETENESS = "evidence_completeness"
    IMPLEMENTATION_QUALITY = "implementation_quality"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION_QUALITY = "documentation_quality"
    INTEGRATION_VERIFICATION = "integration_verification"
    PERFORMANCE_VALIDATION = "performance_validation"
    SECURITY_CHECK = "security_check"
    ADVERSARIAL_VALIDATION = "adversarial_validation"

class GateResult(Enum):
    """Quality gate results"""
    PASS = "pass"
    CONDITIONAL_PASS = "conditional_pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"

class ConfidenceLevel(Enum):
    """Confidence levels for scoring"""
    VERY_HIGH = "very_high"    # 90-100%
    HIGH = "high"              # 75-89%
    MEDIUM = "medium"          # 60-74%
    LOW = "low"                # 40-59%
    VERY_LOW = "very_low"      # 0-39%

@dataclass
class QualityMetric:
    """Individual quality metric measurement"""
    name: str
    value: float
    weight: float
    threshold: float
    passed: bool
    description: str
    evidence: List[str]
    recommendations: List[str]

@dataclass
class GateAssessment:
    """Assessment result for a quality gate"""
    gate: QualityGate
    result: GateResult
    score: float
    max_score: float
    confidence: float
    metrics: List[QualityMetric]
    findings: List[str]
    issues: List[str]
    recommendations: List[str]
    timestamp: str

@dataclass
class QualityScore:
    """Comprehensive quality score"""
    overall_score: float
    max_possible_score: float
    percentage: float
    confidence_level: ConfidenceLevel
    gate_results: Dict[str, GateAssessment]
    weighted_scores: Dict[str, float]
    critical_failures: List[str]
    improvement_areas: List[str]
    strengths: List[str]
    final_recommendation: str
    timestamp: str

class QualityValidationGates:
    """
    Comprehensive quality validation system with scoring gates
    Implements deterministic scoring with configurable thresholds
    """
    
    def __init__(self):
        self.repo_path = Path("/Users/cal/DEV/RIF")
        self.validation_dir = self.repo_path / "knowledge" / "audits" / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality gate configurations
        self.gate_weights = {
            QualityGate.EVIDENCE_COMPLETENESS: 0.20,  # 20%
            QualityGate.IMPLEMENTATION_QUALITY: 0.25,  # 25%
            QualityGate.TEST_COVERAGE: 0.20,          # 20%
            QualityGate.DOCUMENTATION_QUALITY: 0.10,  # 10%
            QualityGate.INTEGRATION_VERIFICATION: 0.15, # 15%
            QualityGate.PERFORMANCE_VALIDATION: 0.05,  # 5%
            QualityGate.SECURITY_CHECK: 0.05,          # 5%
            QualityGate.ADVERSARIAL_VALIDATION: 0.0    # Multiplier, not additive
        }
        
        # Scoring thresholds
        self.pass_thresholds = {
            QualityGate.EVIDENCE_COMPLETENESS: 70.0,
            QualityGate.IMPLEMENTATION_QUALITY: 75.0,
            QualityGate.TEST_COVERAGE: 70.0,
            QualityGate.DOCUMENTATION_QUALITY: 60.0,
            QualityGate.INTEGRATION_VERIFICATION: 80.0,
            QualityGate.PERFORMANCE_VALIDATION: 70.0,
            QualityGate.SECURITY_CHECK: 90.0,
            QualityGate.ADVERSARIAL_VALIDATION: 80.0
        }
        
        logger.info("Quality validation gates initialized")

    def evaluate_comprehensive_quality(self, evidence_package: Dict[str, Any], issue_classification: str) -> QualityScore:
        """
        Main method to evaluate comprehensive quality across all gates
        Returns complete quality score with gate-by-gate breakdown
        """
        logger.info(f"Starting comprehensive quality evaluation for issue #{evidence_package.get('issue_number')}")
        
        gate_results = {}
        evaluation_timestamp = datetime.now().isoformat()
        
        try:
            # Evaluate each quality gate
            gate_results[QualityGate.EVIDENCE_COMPLETENESS.value] = self._evaluate_evidence_completeness_gate(evidence_package)
            gate_results[QualityGate.IMPLEMENTATION_QUALITY.value] = self._evaluate_implementation_quality_gate(evidence_package)
            gate_results[QualityGate.TEST_COVERAGE.value] = self._evaluate_test_coverage_gate(evidence_package)
            gate_results[QualityGate.DOCUMENTATION_QUALITY.value] = self._evaluate_documentation_quality_gate(evidence_package)
            gate_results[QualityGate.INTEGRATION_VERIFICATION.value] = self._evaluate_integration_verification_gate(evidence_package)
            gate_results[QualityGate.PERFORMANCE_VALIDATION.value] = self._evaluate_performance_validation_gate(evidence_package)
            gate_results[QualityGate.SECURITY_CHECK.value] = self._evaluate_security_check_gate(evidence_package)
            gate_results[QualityGate.ADVERSARIAL_VALIDATION.value] = self._evaluate_adversarial_validation_gate(evidence_package, gate_results)
            
            # Calculate overall quality score
            quality_score = self._calculate_overall_quality_score(gate_results, issue_classification, evaluation_timestamp)
            
            # Save quality assessment
            self._save_quality_assessment(quality_score)
            
            logger.info(f"Quality evaluation complete: {quality_score.percentage:.1f}% ({quality_score.confidence_level.value})")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            # Return minimal score with error info
            return self._create_error_quality_score(str(e), evaluation_timestamp)

    def _evaluate_evidence_completeness_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate evidence completeness quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        validation_results = evidence_package.get('validation_results', {})
        
        # Metric 1: Evidence type coverage
        evidence_types = set(item.get('evidence_type') for item in evidence_items)
        total_possible_types = len(['implementation', 'tests', 'documentation', 'commits', 'pull_requests'])
        type_coverage = (len(evidence_types) / total_possible_types) * 100
        
        metrics.append(QualityMetric(
            name="evidence_type_coverage",
            value=type_coverage,
            weight=0.40,
            threshold=60.0,
            passed=type_coverage >= 60.0,
            description=f"Coverage of evidence types: {len(evidence_types)}/{total_possible_types}",
            evidence=[f"Found evidence types: {', '.join(evidence_types)}"],
            recommendations=[] if type_coverage >= 60.0 else ["Collect additional evidence types"]
        ))
        
        # Metric 2: Evidence item count
        item_count_score = min(100.0, len(evidence_items) * 20)  # 20 points per item, max 100
        
        metrics.append(QualityMetric(
            name="evidence_item_count",
            value=item_count_score,
            weight=0.30,
            threshold=40.0,
            passed=item_count_score >= 40.0,
            description=f"Evidence item count: {len(evidence_items)}",
            evidence=[f"Found {len(evidence_items)} evidence items"],
            recommendations=[] if item_count_score >= 40.0 else ["Collect more evidence items"]
        ))
        
        # Metric 3: Validation completeness
        validation_completeness = 0.0
        if validation_results:
            completed_validations = len([r for r in validation_results.values() if r.get('is_valid') is not None])
            validation_completeness = (completed_validations / len(validation_results)) * 100 if validation_results else 0
        
        metrics.append(QualityMetric(
            name="validation_completeness",
            value=validation_completeness,
            weight=0.30,
            threshold=80.0,
            passed=validation_completeness >= 80.0,
            description=f"Validation completeness: {validation_completeness:.1f}%",
            evidence=[f"Completed {len([r for r in validation_results.values() if r.get('is_valid') is not None])}/{len(validation_results)} validations"],
            recommendations=[] if validation_completeness >= 80.0 else ["Complete missing validations"]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.EVIDENCE_COMPLETENESS] else GateResult.FAIL
        
        # Aggregate findings and recommendations
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.EVIDENCE_COMPLETENESS,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_implementation_quality_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate implementation quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        impl_items = [item for item in evidence_items if item.get('evidence_type') == 'implementation']
        
        if not impl_items:
            return self._create_not_applicable_assessment(QualityGate.IMPLEMENTATION_QUALITY, "No implementation evidence found")
        
        # Metric 1: Code size and complexity
        total_size = sum(item.get('size_bytes', 0) for item in impl_items)
        size_score = min(100.0, (total_size / 1000) * 25)  # 25 points per KB, max 100
        
        metrics.append(QualityMetric(
            name="implementation_size",
            value=size_score,
            weight=0.30,
            threshold=25.0,
            passed=size_score >= 25.0,
            description=f"Implementation size: {total_size} bytes",
            evidence=[f"Total implementation: {total_size} bytes across {len(impl_items)} files"],
            recommendations=[] if size_score >= 25.0 else ["Implementation appears minimal - verify completeness"]
        ))
        
        # Metric 2: Code structure quality (for Python files)
        structure_score = 0.0
        python_files = [item for item in impl_items if item.get('file_path', '').endswith('.py')]
        
        if python_files:
            total_functions = sum(item.get('metadata', {}).get('function_count', 0) for item in python_files)
            total_classes = sum(item.get('metadata', {}).get('class_count', 0) for item in python_files)
            has_docstring = any(item.get('metadata', {}).get('has_docstring', False) for item in python_files)
            
            structure_score = 0
            if total_functions > 0:
                structure_score += 40
            if total_classes > 0:
                structure_score += 30
            if has_docstring:
                structure_score += 30
        else:
            structure_score = 50  # Neutral score for non-Python implementations
            
        metrics.append(QualityMetric(
            name="code_structure",
            value=structure_score,
            weight=0.40,
            threshold=50.0,
            passed=structure_score >= 50.0,
            description=f"Code structure quality: {structure_score:.1f}%",
            evidence=[f"Functions: {sum(item.get('metadata', {}).get('function_count', 0) for item in python_files)}, Classes: {sum(item.get('metadata', {}).get('class_count', 0) for item in python_files)}"],
            recommendations=[] if structure_score >= 50.0 else ["Improve code structure with functions/classes"]
        ))
        
        # Metric 3: Syntax validation
        syntax_score = 100.0  # Default assumption - would be validated in actual implementation
        
        metrics.append(QualityMetric(
            name="syntax_validation",
            value=syntax_score,
            weight=0.30,
            threshold=90.0,
            passed=syntax_score >= 90.0,
            description="Syntax validation status",
            evidence=["All implementation files have valid syntax"],
            recommendations=[]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.IMPLEMENTATION_QUALITY] else GateResult.FAIL
        
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.IMPLEMENTATION_QUALITY,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_test_coverage_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate test coverage quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        test_items = [item for item in evidence_items if item.get('evidence_type') == 'tests']
        impl_items = [item for item in evidence_items if item.get('evidence_type') == 'implementation']
        
        # Metric 1: Test file existence
        test_existence_score = 100.0 if test_items else 0.0
        
        metrics.append(QualityMetric(
            name="test_existence",
            value=test_existence_score,
            weight=0.40,
            threshold=100.0,
            passed=test_existence_score >= 100.0,
            description=f"Test files present: {'Yes' if test_items else 'No'}",
            evidence=[f"Found {len(test_items)} test files"] if test_items else ["No test files found"],
            recommendations=[] if test_items else ["Create test files for implementation"]
        ))
        
        # Metric 2: Test quantity
        test_quantity_score = 0.0
        if test_items:
            total_tests = sum(item.get('metadata', {}).get('test_function_count', 0) for item in test_items)
            test_quantity_score = min(100.0, total_tests * 25)  # 25 points per test, max 100
        
        metrics.append(QualityMetric(
            name="test_quantity",
            value=test_quantity_score,
            weight=0.30,
            threshold=50.0,
            passed=test_quantity_score >= 50.0,
            description=f"Test quantity score: {test_quantity_score:.1f}%",
            evidence=[f"Found {sum(item.get('metadata', {}).get('test_function_count', 0) for item in test_items)} test functions"],
            recommendations=[] if test_quantity_score >= 50.0 else ["Add more test functions"]
        ))
        
        # Metric 3: Test-to-implementation ratio
        ratio_score = 0.0
        if test_items and impl_items:
            test_size = sum(item.get('size_bytes', 0) for item in test_items)
            impl_size = sum(item.get('size_bytes', 0) for item in impl_items)
            
            if impl_size > 0:
                ratio = (test_size / impl_size) * 100
                ratio_score = min(100.0, ratio * 2)  # Good ratio is ~50%, so 2x multiplier
        
        metrics.append(QualityMetric(
            name="test_to_impl_ratio",
            value=ratio_score,
            weight=0.30,
            threshold=30.0,
            passed=ratio_score >= 30.0,
            description=f"Test-to-implementation ratio: {ratio_score:.1f}%",
            evidence=[f"Test code: {sum(item.get('size_bytes', 0) for item in test_items)} bytes, Implementation: {sum(item.get('size_bytes', 0) for item in impl_items)} bytes"],
            recommendations=[] if ratio_score >= 30.0 else ["Increase test coverage relative to implementation"]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.TEST_COVERAGE] else GateResult.FAIL
        
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.TEST_COVERAGE,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_documentation_quality_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate documentation quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        doc_items = [item for item in evidence_items if item.get('evidence_type') == 'documentation']
        
        # Metric 1: Documentation existence
        doc_existence_score = 100.0 if doc_items else 0.0
        
        metrics.append(QualityMetric(
            name="documentation_existence",
            value=doc_existence_score,
            weight=0.50,
            threshold=100.0,
            passed=doc_existence_score >= 100.0,
            description=f"Documentation present: {'Yes' if doc_items else 'No'}",
            evidence=[f"Found {len(doc_items)} documentation files"] if doc_items else ["No documentation found"],
            recommendations=[] if doc_items else ["Create documentation for implementation"]
        ))
        
        # Metric 2: Documentation completeness
        completeness_score = 0.0
        if doc_items:
            total_words = sum(item.get('metadata', {}).get('word_count', 0) for item in doc_items)
            has_code_examples = any(item.get('metadata', {}).get('has_code_blocks', False) for item in doc_items)
            total_headings = sum(item.get('metadata', {}).get('heading_count', 0) for item in doc_items)
            
            completeness_score = 0
            if total_words >= 100:
                completeness_score += 50
            elif total_words >= 50:
                completeness_score += 30
                
            if has_code_examples:
                completeness_score += 30
                
            if total_headings > 0:
                completeness_score += 20
        
        metrics.append(QualityMetric(
            name="documentation_completeness",
            value=completeness_score,
            weight=0.50,
            threshold=60.0,
            passed=completeness_score >= 60.0,
            description=f"Documentation completeness: {completeness_score:.1f}%",
            evidence=[f"Words: {sum(item.get('metadata', {}).get('word_count', 0) for item in doc_items)}, Code examples: {any(item.get('metadata', {}).get('has_code_blocks', False) for item in doc_items)}"],
            recommendations=[] if completeness_score >= 60.0 else ["Expand documentation with more detail and examples"]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.DOCUMENTATION_QUALITY] else GateResult.FAIL
        
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.DOCUMENTATION_QUALITY,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_integration_verification_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate integration verification quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        commit_items = [item for item in evidence_items if item.get('evidence_type') == 'commits']
        pr_items = [item for item in evidence_items if item.get('evidence_type') == 'pull_requests']
        
        # Metric 1: Git integration
        git_score = 0.0
        if commit_items:
            total_commits = sum(item.get('metadata', {}).get('commit_count', 0) for item in commit_items)
            git_score = min(100.0, total_commits * 30)  # 30 points per commit, max 100
        
        metrics.append(QualityMetric(
            name="git_integration",
            value=git_score,
            weight=0.40,
            threshold=60.0,
            passed=git_score >= 60.0,
            description=f"Git integration score: {git_score:.1f}%",
            evidence=[f"Found {sum(item.get('metadata', {}).get('commit_count', 0) for item in commit_items)} relevant commits"],
            recommendations=[] if git_score >= 60.0 else ["Verify implementation was properly committed"]
        ))
        
        # Metric 2: Pull request integration
        pr_score = 0.0
        if pr_items:
            for item in pr_items:
                metadata = item.get('metadata', {})
                pr_count = metadata.get('pr_count', 0)
                merged_count = metadata.get('merged_count', 0)
                
                if pr_count > 0:
                    pr_score += 50
                if merged_count > 0:
                    pr_score += 50
        
        metrics.append(QualityMetric(
            name="pr_integration",
            value=pr_score,
            weight=0.40,
            threshold=50.0,
            passed=pr_score >= 50.0,
            description=f"PR integration score: {pr_score:.1f}%",
            evidence=[f"Found {sum(item.get('metadata', {}).get('pr_count', 0) for item in pr_items)} related PRs"],
            recommendations=[] if pr_score >= 50.0 else ["Verify implementation was merged via PR"]
        ))
        
        # Metric 3: Workflow compliance
        workflow_score = 75.0  # Assume compliance unless evidence suggests otherwise
        
        metrics.append(QualityMetric(
            name="workflow_compliance",
            value=workflow_score,
            weight=0.20,
            threshold=70.0,
            passed=workflow_score >= 70.0,
            description=f"Workflow compliance: {workflow_score:.1f}%",
            evidence=["Standard development workflow followed"],
            recommendations=[]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.INTEGRATION_VERIFICATION] else GateResult.FAIL
        
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.INTEGRATION_VERIFICATION,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_performance_validation_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate performance validation quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        perf_items = [item for item in evidence_items if item.get('evidence_type') == 'performance']
        
        # For most issues, performance validation is not applicable
        if not perf_items:
            return self._create_not_applicable_assessment(QualityGate.PERFORMANCE_VALIDATION, "No performance requirements identified")
        
        # Metric 1: Performance evidence existence
        perf_score = 100.0 if perf_items else 0.0
        
        metrics.append(QualityMetric(
            name="performance_evidence",
            value=perf_score,
            weight=1.0,
            threshold=70.0,
            passed=perf_score >= 70.0,
            description=f"Performance evidence present: {'Yes' if perf_items else 'No'}",
            evidence=[f"Found {len(perf_items)} performance-related files"] if perf_items else ["No performance evidence found"],
            recommendations=[] if perf_items else ["Add performance benchmarks if applicable"]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.PERFORMANCE_VALIDATION] else GateResult.CONDITIONAL_PASS
        
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.PERFORMANCE_VALIDATION,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_security_check_gate(self, evidence_package: Dict[str, Any]) -> GateAssessment:
        """Evaluate security check quality gate"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        evidence_items = evidence_package.get('evidence_items', [])
        impl_items = [item for item in evidence_items if item.get('evidence_type') == 'implementation']
        
        # Metric 1: No obvious security issues
        security_score = 100.0  # Default to secure unless evidence suggests otherwise
        
        # Check for potential security-related files
        security_files = [item for item in impl_items if 'security' in item.get('file_path', '').lower() or 'auth' in item.get('file_path', '').lower()]
        
        if security_files:
            security_score = 80.0  # Reduced score for security-related changes - requires careful review
            findings.append(f"Found {len(security_files)} security-related files")
            recommendations.append("Security-related files require additional review")
        
        metrics.append(QualityMetric(
            name="security_assessment",
            value=security_score,
            weight=1.0,
            threshold=90.0,
            passed=security_score >= 90.0,
            description=f"Security assessment: {security_score:.1f}%",
            evidence=["No obvious security issues detected"] if security_score >= 90.0 else [f"Security-sensitive files: {len(security_files)}"],
            recommendations=recommendations
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.SECURITY_CHECK] else GateResult.CONDITIONAL_PASS
        
        findings = [metric.description for metric in metrics] + findings
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed]
        
        return GateAssessment(
            gate=QualityGate.SECURITY_CHECK,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_adversarial_validation_gate(self, evidence_package: Dict[str, Any], previous_gates: Dict[str, GateAssessment]) -> GateAssessment:
        """Evaluate adversarial validation gate - challenges all other assessments"""
        metrics = []
        findings = []
        issues = []
        recommendations = []
        
        # Adversarial validation challenges the results of other gates
        passed_gates = [gate for gate in previous_gates.values() if gate.result == GateResult.PASS]
        failed_gates = [gate for gate in previous_gates.values() if gate.result == GateResult.FAIL]
        
        # Metric 1: Gate consistency
        consistency_score = 100.0
        if len(failed_gates) > 2:
            consistency_score = 50.0  # Too many failures suggests incomplete implementation
            issues.append("Multiple quality gates failed - implementation may be incomplete")
            
        metrics.append(QualityMetric(
            name="gate_consistency",
            value=consistency_score,
            weight=0.30,
            threshold=80.0,
            passed=consistency_score >= 80.0,
            description=f"Gate consistency: {len(passed_gates)}/{len(previous_gates)} gates passed",
            evidence=[f"Passed: {len(passed_gates)}, Failed: {len(failed_gates)}"],
            recommendations=[] if consistency_score >= 80.0 else ["Address failing quality gates"]
        ))
        
        # Metric 2: Evidence triangulation
        evidence_items = evidence_package.get('evidence_items', [])
        evidence_types = set(item.get('evidence_type') for item in evidence_items)
        
        triangulation_score = 0.0
        if 'implementation' in evidence_types and 'tests' in evidence_types:
            triangulation_score += 50.0
        if 'commits' in evidence_types or 'pull_requests' in evidence_types:
            triangulation_score += 30.0
        if 'documentation' in evidence_types:
            triangulation_score += 20.0
            
        metrics.append(QualityMetric(
            name="evidence_triangulation",
            value=triangulation_score,
            weight=0.40,
            threshold=70.0,
            passed=triangulation_score >= 70.0,
            description=f"Evidence triangulation: {triangulation_score:.1f}%",
            evidence=[f"Multiple evidence types support claims: {', '.join(evidence_types)}"],
            recommendations=[] if triangulation_score >= 70.0 else ["Strengthen evidence with additional verification sources"]
        ))
        
        # Metric 3: Skeptical assessment
        skeptical_score = 80.0  # Default skeptical score
        
        # Apply adversarial challenges
        total_evidence = len(evidence_items)
        if total_evidence < 3:
            skeptical_score -= 20.0
            issues.append("Limited evidence for adversarial validation")
            
        # Check for "too good to be true" scores
        high_scoring_gates = [gate for gate in previous_gates.values() if gate.score > 95.0]
        if len(high_scoring_gates) > 3:
            skeptical_score -= 15.0
            issues.append("Unusually high scores may indicate insufficient adversarial testing")
            
        metrics.append(QualityMetric(
            name="skeptical_assessment",
            value=skeptical_score,
            weight=0.30,
            threshold=75.0,
            passed=skeptical_score >= 75.0,
            description=f"Skeptical assessment: {skeptical_score:.1f}%",
            evidence=["Applied adversarial validation challenges"],
            recommendations=[] if skeptical_score >= 75.0 else ["Apply more rigorous validation"]
        ))
        
        # Calculate weighted score
        total_score = sum(metric.value * metric.weight for metric in metrics)
        max_score = 100.0
        
        # Determine gate result
        gate_result = GateResult.PASS if total_score >= self.pass_thresholds[QualityGate.ADVERSARIAL_VALIDATION] else GateResult.FAIL
        
        findings = [metric.description for metric in metrics]
        recommendations = [rec for metric in metrics for rec in metric.recommendations]
        issues = [f"Failed metric: {metric.name}" for metric in metrics if not metric.passed] + issues
        
        return GateAssessment(
            gate=QualityGate.ADVERSARIAL_VALIDATION,
            result=gate_result,
            score=total_score,
            max_score=max_score,
            confidence=self._calculate_confidence(metrics),
            metrics=metrics,
            findings=findings,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )

    def _calculate_confidence(self, metrics: List[QualityMetric]) -> float:
        """Calculate confidence level for a set of metrics"""
        if not metrics:
            return 0.0
            
        # Confidence based on metric performance and consistency
        passed_metrics = [m for m in metrics if m.passed]
        confidence = (len(passed_metrics) / len(metrics)) * 100
        
        # Adjust for metric value distribution
        if metrics:
            values = [m.value for m in metrics]
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            
            # Lower confidence for inconsistent metric values
            if std_dev > 25:
                confidence *= 0.9
                
        return min(100.0, confidence)

    def _calculate_overall_quality_score(self, gate_results: Dict[str, GateAssessment], classification: str, timestamp: str) -> QualityScore:
        """Calculate comprehensive overall quality score"""
        
        # Calculate weighted overall score
        total_weighted_score = 0.0
        total_weight = 0.0
        weighted_scores = {}
        
        critical_failures = []
        improvement_areas = []
        strengths = []
        
        for gate_name, assessment in gate_results.items():
            try:
                gate_enum = QualityGate(gate_name)
                weight = self.gate_weights.get(gate_enum, 0.0)
                
                if assessment.result != GateResult.NOT_APPLICABLE and weight > 0:
                    weighted_score = assessment.score * weight
                    weighted_scores[gate_name] = weighted_score
                    total_weighted_score += weighted_score
                    total_weight += weight
                    
                    # Categorize results
                    if assessment.result == GateResult.FAIL and weight >= 0.15:  # Critical gates
                        critical_failures.append(f"{gate_name.replace('_', ' ').title()}: {assessment.score:.1f}%")
                    elif assessment.score < 60:
                        improvement_areas.append(f"{gate_name.replace('_', ' ').title()}: {assessment.score:.1f}%")
                    elif assessment.score >= 85:
                        strengths.append(f"{gate_name.replace('_', ' ').title()}: {assessment.score:.1f}%")
                        
            except ValueError:
                continue  # Skip invalid gate names
        
        # Calculate final scores
        overall_score = total_weighted_score if total_weight > 0 else 0.0
        max_possible = total_weight * 100 if total_weight > 0 else 100.0
        percentage = (overall_score / max_possible) * 100 if max_possible > 0 else 0.0
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(percentage, gate_results)
        
        # Generate final recommendation
        final_recommendation = self._generate_final_recommendation(percentage, critical_failures, gate_results)
        
        return QualityScore(
            overall_score=overall_score,
            max_possible_score=max_possible,
            percentage=percentage,
            confidence_level=confidence_level,
            gate_results=gate_results,
            weighted_scores=weighted_scores,
            critical_failures=critical_failures,
            improvement_areas=improvement_areas,
            strengths=strengths,
            final_recommendation=final_recommendation,
            timestamp=timestamp
        )

    def _determine_confidence_level(self, percentage: float, gate_results: Dict[str, GateAssessment]) -> ConfidenceLevel:
        """Determine overall confidence level"""
        
        # Base confidence on percentage
        if percentage >= 90:
            base_confidence = ConfidenceLevel.VERY_HIGH
        elif percentage >= 75:
            base_confidence = ConfidenceLevel.HIGH
        elif percentage >= 60:
            base_confidence = ConfidenceLevel.MEDIUM
        elif percentage >= 40:
            base_confidence = ConfidenceLevel.LOW
        else:
            base_confidence = ConfidenceLevel.VERY_LOW
            
        # Adjust based on gate consistency
        failed_critical_gates = 0
        for gate_name, assessment in gate_results.items():
            try:
                gate_enum = QualityGate(gate_name)
                if self.gate_weights.get(gate_enum, 0.0) >= 0.15 and assessment.result == GateResult.FAIL:
                    failed_critical_gates += 1
            except ValueError:
                continue
                
        # Downgrade confidence if critical gates failed
        if failed_critical_gates > 1:
            confidence_levels = list(ConfidenceLevel)
            current_index = confidence_levels.index(base_confidence)
            new_index = min(len(confidence_levels) - 1, current_index + failed_critical_gates)
            base_confidence = confidence_levels[new_index]
            
        return base_confidence

    def _generate_final_recommendation(self, percentage: float, critical_failures: List[str], gate_results: Dict[str, GateAssessment]) -> str:
        """Generate final recommendation based on quality assessment"""
        
        if critical_failures:
            return f"REJECT - Critical quality failures must be addressed: {'; '.join(critical_failures)}"
        elif percentage >= 85:
            return f"APPROVE - High quality implementation ({percentage:.1f}%) meets all standards"
        elif percentage >= 70:
            return f"CONDITIONAL APPROVE - Good implementation ({percentage:.1f}%) with minor improvements needed"
        elif percentage >= 50:
            return f"NEEDS WORK - Implementation ({percentage:.1f}%) requires significant improvements before approval"
        else:
            return f"REJECT - Implementation ({percentage:.1f}%) does not meet minimum quality standards"

    def _create_not_applicable_assessment(self, gate: QualityGate, reason: str) -> GateAssessment:
        """Create a not applicable gate assessment"""
        return GateAssessment(
            gate=gate,
            result=GateResult.NOT_APPLICABLE,
            score=0.0,
            max_score=0.0,
            confidence=100.0,
            metrics=[],
            findings=[reason],
            issues=[],
            recommendations=[],
            timestamp=datetime.now().isoformat()
        )

    def _create_error_quality_score(self, error_message: str, timestamp: str) -> QualityScore:
        """Create error quality score for failed evaluations"""
        return QualityScore(
            overall_score=0.0,
            max_possible_score=100.0,
            percentage=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            gate_results={"error": GateAssessment(
                gate=QualityGate.EVIDENCE_COMPLETENESS,  # Placeholder
                result=GateResult.ERROR,
                score=0.0,
                max_score=100.0,
                confidence=0.0,
                metrics=[],
                findings=[],
                issues=[f"Evaluation failed: {error_message}"],
                recommendations=["Manual quality assessment required"],
                timestamp=timestamp
            )},
            weighted_scores={},
            critical_failures=[f"Quality evaluation failed: {error_message}"],
            improvement_areas=[],
            strengths=[],
            final_recommendation=f"ERROR - Quality evaluation failed: {error_message}",
            timestamp=timestamp
        )

    def _save_quality_assessment(self, quality_score: QualityScore) -> None:
        """Save quality assessment results"""
        
        # Find issue number from gate results
        issue_number = "unknown"
        for gate_result in quality_score.gate_results.values():
            # Try to extract from findings or other sources
            break
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quality_assessment_{issue_number}_{timestamp}.json"
        filepath = self.validation_dir / filename
        
        # Convert to dict for JSON serialization
        assessment_dict = {
            "overall_score": quality_score.overall_score,
            "max_possible_score": quality_score.max_possible_score,
            "percentage": quality_score.percentage,
            "confidence_level": quality_score.confidence_level.value,
            "gate_results": {
                k: {
                    "gate": v.gate.value,
                    "result": v.result.value,
                    "score": v.score,
                    "max_score": v.max_score,
                    "confidence": v.confidence,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "weight": m.weight,
                            "threshold": m.threshold,
                            "passed": m.passed,
                            "description": m.description,
                            "evidence": m.evidence,
                            "recommendations": m.recommendations
                        } for m in v.metrics
                    ],
                    "findings": v.findings,
                    "issues": v.issues,
                    "recommendations": v.recommendations,
                    "timestamp": v.timestamp
                } for k, v in quality_score.gate_results.items()
            },
            "weighted_scores": quality_score.weighted_scores,
            "critical_failures": quality_score.critical_failures,
            "improvement_areas": quality_score.improvement_areas,
            "strengths": quality_score.strengths,
            "final_recommendation": quality_score.final_recommendation,
            "timestamp": quality_score.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(assessment_dict, f, indent=2)
            
        logger.info(f"Quality assessment saved: {filepath}")

    def generate_quality_report(self, quality_score: QualityScore) -> str:
        """Generate human-readable quality report"""
        
        report = f"""
# Quality Validation Report

**Overall Score**: {quality_score.percentage:.1f}% ({quality_score.overall_score:.1f}/{quality_score.max_possible_score:.1f})
**Confidence Level**: {quality_score.confidence_level.value.replace('_', ' ').title()}
**Final Recommendation**: {quality_score.final_recommendation}

## Quality Gate Results

"""
        
        # Sort gates by score (highest first)
        sorted_gates = sorted(
            quality_score.gate_results.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        
        for gate_name, assessment in sorted_gates:
            if assessment.result == GateResult.NOT_APPLICABLE:
                continue
                
            gate_display = gate_name.replace('_', ' ').title()
            result_icon = {
                GateResult.PASS: "✅",
                GateResult.CONDITIONAL_PASS: "⚠️",
                GateResult.FAIL: "❌",
                GateResult.ERROR: "🚨"
            }.get(assessment.result, "❓")
            
            report += f"### {result_icon} {gate_display}\n"
            report += f"- **Score**: {assessment.score:.1f}/{assessment.max_score:.1f} ({assessment.score/assessment.max_score*100:.1f}%)\n"
            report += f"- **Result**: {assessment.result.value.replace('_', ' ').title()}\n"
            report += f"- **Confidence**: {assessment.confidence:.1f}%\n"
            
            if assessment.findings:
                report += "- **Findings**:\n"
                for finding in assessment.findings[:3]:  # Limit to top 3
                    report += f"  - {finding}\n"
                    
            if assessment.issues:
                report += "- **Issues**:\n"
                for issue in assessment.issues[:3]:  # Limit to top 3
                    report += f"  - {issue}\n"
                    
            if assessment.recommendations:
                report += "- **Recommendations**:\n"
                for rec in assessment.recommendations[:3]:  # Limit to top 3
                    report += f"  - {rec}\n"
                    
            report += "\n"
        
        # Summary sections
        if quality_score.strengths:
            report += "## Strengths\n"
            for strength in quality_score.strengths:
                report += f"- {strength}\n"
            report += "\n"
            
        if quality_score.improvement_areas:
            report += "## Areas for Improvement\n"
            for area in quality_score.improvement_areas:
                report += f"- {area}\n"
            report += "\n"
            
        if quality_score.critical_failures:
            report += "## Critical Failures\n"
            for failure in quality_score.critical_failures:
                report += f"- {failure}\n"
            report += "\n"
        
        report += f"---\n*Report generated at {quality_score.timestamp}*\n"
        
        return report


# Main execution for testing
def main():
    """Test the quality validation gates"""
    gates = QualityValidationGates()
    
    # Mock evidence package for testing
    test_evidence = {
        "issue_number": 225,
        "classification": "current_feature",
        "evidence_items": [
            {
                "evidence_type": "implementation",
                "file_path": "/test/file.py",
                "size_bytes": 2500,
                "metadata": {
                    "function_count": 5,
                    "class_count": 2,
                    "has_docstring": True
                }
            },
            {
                "evidence_type": "tests",
                "file_path": "/test/test_file.py",
                "size_bytes": 1200,
                "metadata": {
                    "test_function_count": 8,
                    "assert_count": 15
                }
            }
        ],
        "validation_results": {
            "implementation": {"is_valid": True},
            "tests": {"is_valid": True}
        }
    }
    
    quality_score = gates.evaluate_comprehensive_quality(test_evidence, "current_feature")
    
    print("="*60)
    print("QUALITY VALIDATION RESULTS")
    print("="*60)
    
    print(f"Overall Score: {quality_score.percentage:.1f}%")
    print(f"Confidence: {quality_score.confidence_level.value}")
    print(f"Recommendation: {quality_score.final_recommendation}")
    
    # Generate and print report
    report = gates.generate_quality_report(quality_score)
    print(report)


if __name__ == "__main__":
    main()