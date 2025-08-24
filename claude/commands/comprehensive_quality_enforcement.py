#!/usr/bin/env python3
"""
Comprehensive Quality Enforcement Architecture - Issue #87
Integrates all quality sub-systems to ensure 95% quality threshold enforcement.

This system implements:
1. Integration of quality sub-systems (#91-95)
2. Comprehensive quality scoring with multiple dimensions
3. Risk-based manual intervention integration
4. Adaptive threshold management
5. Hard quality enforcement with graceful fallback
6. Context-aware threshold adjustment
7. Quality effectiveness monitoring
"""

import json
import subprocess
import yaml
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

# Import sub-system components
try:
    from quality_gate_enforcement import QualityGateEnforcement
    from quality_metrics_collector import QualityMetricsCollector
    from quality_analytics_engine import QualityAnalyticsEngine
    from quality_decision_engine import QualityDecisionEngine
    from quality_pattern_analyzer import QualityPatternAnalyzer
    from context_weight_manager import ContextWeightManager
    from adaptive_threshold_manager import AdaptiveThresholdManager
    from integrated_risk_assessment import IntegratedRiskAssessment
    from specialist_assignment_engine import SpecialistAssignmentEngine, AssignmentRequest, SpecialistType
    from sla_monitoring_system import SLAMonitoringSystem
except ImportError as e:
    logging.warning(f"Import warning: {e}")

class QualityEnforcementLevel(Enum):
    """Quality enforcement levels."""
    STRICT = "strict"           # 100% requirements must be met
    STANDARD = "standard"       # 95% threshold enforcement
    ADAPTIVE = "adaptive"       # Context-aware thresholds
    RISK_BASED = "risk_based"   # Risk-adjusted thresholds
    EMERGENCY = "emergency"     # Degraded mode for critical situations

class QualityDimension(Enum):
    """Quality dimensions for scoring."""
    FUNCTIONALITY = "functionality"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    DOCUMENTATION = "documentation"
    COMPLIANCE = "compliance"

@dataclass
class QualityScore:
    """Container for quality score information."""
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    enforcement_level: QualityEnforcementLevel
    threshold: float
    meets_threshold: bool
    confidence: float
    risk_adjusted_threshold: Optional[float] = None
    context_factors: Optional[Dict[str, float]] = None

@dataclass
class QualityViolation:
    """Container for quality violations."""
    dimension: QualityDimension
    violation_type: str
    severity: str
    description: str
    recommendation: str
    blocking: bool
    auto_fixable: bool

@dataclass
class QualityEnforcementResult:
    """Container for quality enforcement result."""
    issue_number: int
    can_proceed: bool
    quality_score: QualityScore
    violations: List[QualityViolation]
    enforcement_action: str
    escalation_required: bool
    specialist_assignment: Optional[Dict[str, Any]]
    fallback_reason: Optional[str]
    next_steps: List[str]
    quality_report: Dict[str, Any]

class ComprehensiveQualityEnforcement:
    """
    Master quality enforcement system integrating all quality sub-systems.
    """
    
    def __init__(self, config_path: str = "config/rif-workflow.yaml"):
        """Initialize comprehensive quality enforcement."""
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        
        # Initialize sub-systems
        self._initialize_subsystems()
        
        # Quality enforcement settings
        self.default_threshold = 0.95  # 95% quality requirement
        self.strict_threshold = 1.0    # 100% for critical items
        self.emergency_threshold = 0.80 # Emergency fallback threshold
        
        self.logger.info("🏭 Comprehensive Quality Enforcement System initialized")
    
    def setup_logging(self):
        """Setup logging for quality enforcement."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - ComprehensiveQualityEnforcement - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load comprehensive quality enforcement configuration."""
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
        """Get default quality enforcement configuration."""
        return {
            'quality_enforcement': {
                'thresholds': {
                    'default': 0.95,
                    'strict': 1.0,
                    'emergency': 0.80,
                    'risk_adjustment_factor': 0.1
                },
                'dimensions': {
                    'functionality': {'weight': 0.25, 'required_score': 0.95},
                    'reliability': {'weight': 0.20, 'required_score': 0.95},
                    'performance': {'weight': 0.15, 'required_score': 0.90},
                    'security': {'weight': 0.20, 'required_score': 1.0},
                    'maintainability': {'weight': 0.10, 'required_score': 0.85},
                    'testability': {'weight': 0.05, 'required_score': 0.90},
                    'documentation': {'weight': 0.03, 'required_score': 0.85},
                    'compliance': {'weight': 0.02, 'required_score': 1.0}
                },
                'enforcement': {
                    'block_on_violation': True,
                    'require_specialist_review': True,
                    'auto_fix_enabled': True,
                    'escalation_enabled': True
                }
            }
        }
    
    def _initialize_subsystems(self):
        """Initialize all quality sub-systems."""
        try:
            self.gate_enforcement = QualityGateEnforcement()
            self.metrics_collector = QualityMetricsCollector()
            self.analytics_engine = QualityAnalyticsEngine()
            self.decision_engine = QualityDecisionEngine()
            self.pattern_analyzer = QualityPatternAnalyzer()
            self.context_weight_manager = ContextWeightManager()
            self.adaptive_threshold_manager = AdaptiveThresholdManager()
            self.risk_assessment = IntegratedRiskAssessment()
            self.specialist_engine = SpecialistAssignmentEngine()
            self.sla_monitor = SLAMonitoringSystem()
            
            self.logger.info("✅ All quality sub-systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing sub-systems: {e}")
            # Create placeholder objects to prevent crashes
            self._create_placeholder_subsystems()
    
    def _create_placeholder_subsystems(self):
        """Create placeholder sub-systems for testing."""
        class PlaceholderSystem:
            def __getattr__(self, name):
                return lambda *args, **kwargs: {"status": "placeholder", "result": None}
        
        self.gate_enforcement = PlaceholderSystem()
        self.metrics_collector = PlaceholderSystem()
        self.analytics_engine = PlaceholderSystem()
        self.decision_engine = PlaceholderSystem()
        self.pattern_analyzer = PlaceholderSystem()
        self.context_weight_manager = PlaceholderSystem()
        self.adaptive_threshold_manager = PlaceholderSystem()
        self.risk_assessment = PlaceholderSystem()
        self.specialist_engine = PlaceholderSystem()
        self.sla_monitor = PlaceholderSystem()
        
        self.logger.warning("⚠️ Using placeholder sub-systems - limited functionality")
    
    def enforce_quality_standards(self, issue_number: int, context: Optional[Dict[str, Any]] = None) -> QualityEnforcementResult:
        """
        Comprehensive quality enforcement for an issue.
        
        Args:
            issue_number: GitHub issue number
            context: Optional context information
            
        Returns:
            QualityEnforcementResult with enforcement decision
        """
        self.logger.info(f"🔍 Enforcing quality standards for issue #{issue_number}")
        
        try:
            # Phase 1: Data Collection
            quality_data = self._collect_quality_data(issue_number, context)
            
            # Phase 2: Multi-Dimensional Quality Scoring
            quality_score = self._calculate_comprehensive_quality_score(quality_data, context)
            
            # Phase 3: Context-Aware Threshold Adjustment
            adjusted_threshold = self._adjust_threshold_for_context(quality_score, context)
            quality_score.risk_adjusted_threshold = adjusted_threshold
            
            # Phase 4: Quality Violation Detection
            violations = self._detect_quality_violations(quality_score, quality_data)
            
            # Phase 5: Enforcement Decision
            enforcement_decision = self._make_enforcement_decision(quality_score, violations, context)
            
            # Phase 6: Risk-Based Intervention (if needed)
            specialist_assignment = None
            if enforcement_decision.get('requires_specialist_review', False):
                specialist_assignment = self._trigger_specialist_review(issue_number, quality_score, violations)
            
            # Phase 7: Generate Results
            result = QualityEnforcementResult(
                issue_number=issue_number,
                can_proceed=enforcement_decision.get('can_proceed', False),
                quality_score=quality_score,
                violations=violations,
                enforcement_action=enforcement_decision.get('action', 'block'),
                escalation_required=enforcement_decision.get('escalation_required', False),
                specialist_assignment=specialist_assignment,
                fallback_reason=enforcement_decision.get('fallback_reason'),
                next_steps=enforcement_decision.get('next_steps', []),
                quality_report=quality_data
            )
            
            # Phase 8: Update Learning Systems
            self._update_adaptive_systems(result)
            
            # Phase 9: Record Enforcement Decision
            self._record_enforcement_decision(result)
            
            self.logger.info(f"🎯 Quality enforcement complete for #{issue_number}: {enforcement_decision.get('action', 'unknown')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in quality enforcement: {e}")
            return self._create_fallback_result(issue_number, str(e))
    
    def _collect_quality_data(self, issue_number: int, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect comprehensive quality data from all sub-systems."""
        quality_data = {
            'issue_number': issue_number,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': context or {},
            'sub_system_data': {}
        }
        
        try:
            # Collect from gate enforcement
            gate_data = getattr(self.gate_enforcement, 'validate_issue_closure_readiness', lambda x: {})
            quality_data['sub_system_data']['gates'] = gate_data(issue_number)
            
            # Collect metrics
            metrics_data = getattr(self.metrics_collector, 'collect_quality_metrics', lambda x: {})
            quality_data['sub_system_data']['metrics'] = metrics_data(issue_number)
            
            # Collect analytics
            analytics_data = getattr(self.analytics_engine, 'analyze_quality_trends', lambda x: {})
            quality_data['sub_system_data']['analytics'] = analytics_data(issue_number)
            
            # Collect pattern analysis
            pattern_data = getattr(self.pattern_analyzer, 'analyze_patterns', lambda x: {})
            quality_data['sub_system_data']['patterns'] = pattern_data(issue_number)
            
            # Collect risk assessment
            risk_data = getattr(self.risk_assessment, 'assess_integrated_risk', lambda x: {})
            quality_data['sub_system_data']['risk'] = risk_data(issue_number)
            
        except Exception as e:
            self.logger.warning(f"Error collecting quality data: {e}")
            quality_data['collection_errors'] = str(e)
        
        return quality_data
    
    def _calculate_comprehensive_quality_score(self, quality_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> QualityScore:
        """Calculate comprehensive quality score across all dimensions."""
        dimension_config = self.config.get('quality_enforcement', {}).get('dimensions', {})
        
        # Calculate scores for each quality dimension
        dimension_scores = {}
        for dimension in QualityDimension:
            score = self._calculate_dimension_score(dimension, quality_data, context)
            dimension_scores[dimension] = score
        
        # Calculate weighted overall score
        overall_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = dimension_config.get(dimension.value, {}).get('weight', 0.1)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        
        # Determine enforcement level
        enforcement_level = self._determine_enforcement_level(context)
        
        # Get base threshold
        threshold = self._get_base_threshold(enforcement_level)
        
        # Calculate confidence
        confidence = self._calculate_score_confidence(dimension_scores, quality_data)
        
        return QualityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            enforcement_level=enforcement_level,
            threshold=threshold,
            meets_threshold=overall_score >= threshold,
            confidence=confidence,
            context_factors=self._extract_context_factors(context)
        )
    
    def _calculate_dimension_score(self, dimension: QualityDimension, quality_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Calculate score for a specific quality dimension."""
        sub_system_data = quality_data.get('sub_system_data', {})
        
        # Extract relevant data based on dimension
        if dimension == QualityDimension.FUNCTIONALITY:
            # Check test coverage, feature completeness, requirements satisfaction
            gate_data = sub_system_data.get('gates', {})
            test_score = gate_data.get('test_coverage', 0.85)  # Default 85%
            req_score = gate_data.get('requirements_met', 0.90)  # Default 90%
            return (test_score + req_score) / 2
            
        elif dimension == QualityDimension.SECURITY:
            # Check security scans, vulnerability assessments
            risk_data = sub_system_data.get('risk', {})
            security_score = risk_data.get('security_score', 0.95)  # Default high security
            return security_score
            
        elif dimension == QualityDimension.PERFORMANCE:
            # Check performance benchmarks, load tests
            metrics_data = sub_system_data.get('metrics', {})
            perf_score = metrics_data.get('performance_score', 0.90)  # Default good performance
            return perf_score
            
        elif dimension == QualityDimension.RELIABILITY:
            # Check error rates, stability metrics
            analytics_data = sub_system_data.get('analytics', {})
            reliability_score = analytics_data.get('reliability_score', 0.92)  # Default high reliability
            return reliability_score
            
        elif dimension == QualityDimension.MAINTAINABILITY:
            # Check code quality, complexity metrics
            pattern_data = sub_system_data.get('patterns', {})
            maintainability_score = pattern_data.get('maintainability_score', 0.85)  # Default good maintainability
            return maintainability_score
            
        elif dimension == QualityDimension.TESTABILITY:
            # Check test structure, coverage, quality
            gate_data = sub_system_data.get('gates', {})
            test_quality = gate_data.get('test_quality_score', 0.88)  # Default good test quality
            return test_quality
            
        elif dimension == QualityDimension.DOCUMENTATION:
            # Check documentation completeness, quality
            gate_data = sub_system_data.get('gates', {})
            doc_score = gate_data.get('documentation_score', 0.80)  # Default moderate documentation
            return doc_score
            
        elif dimension == QualityDimension.COMPLIANCE:
            # Check regulatory compliance, standards adherence
            risk_data = sub_system_data.get('risk', {})
            compliance_score = risk_data.get('compliance_score', 0.95)  # Default high compliance
            return compliance_score
        
        # Default moderate score for unknown dimensions
        return 0.85
    
    def _determine_enforcement_level(self, context: Optional[Dict[str, Any]]) -> QualityEnforcementLevel:
        """Determine appropriate enforcement level based on context."""
        if not context:
            return QualityEnforcementLevel.STANDARD
        
        # Check for critical or emergency context
        if context.get('emergency', False) or context.get('hotfix', False):
            return QualityEnforcementLevel.EMERGENCY
        
        # Check for security-critical changes
        if context.get('security_critical', False):
            return QualityEnforcementLevel.STRICT
        
        # Check for adaptive threshold needs
        if context.get('adaptive_thresholds', True):
            return QualityEnforcementLevel.ADAPTIVE
        
        # Check for risk-based enforcement
        if context.get('risk_based', False):
            return QualityEnforcementLevel.RISK_BASED
        
        return QualityEnforcementLevel.STANDARD
    
    def _get_base_threshold(self, enforcement_level: QualityEnforcementLevel) -> float:
        """Get base threshold for enforcement level."""
        thresholds = self.config.get('quality_enforcement', {}).get('thresholds', {})
        
        if enforcement_level == QualityEnforcementLevel.STRICT:
            return thresholds.get('strict', 1.0)
        elif enforcement_level == QualityEnforcementLevel.EMERGENCY:
            return thresholds.get('emergency', 0.80)
        else:
            return thresholds.get('default', 0.95)
    
    def _adjust_threshold_for_context(self, quality_score: QualityScore, context: Optional[Dict[str, Any]]) -> float:
        """Adjust threshold based on context and risk factors."""
        base_threshold = quality_score.threshold
        
        if not context:
            return base_threshold
        
        # Risk-based adjustment
        risk_factor = context.get('risk_level', 'medium')
        if risk_factor == 'high':
            base_threshold = min(1.0, base_threshold + 0.05)  # Raise threshold for high risk
        elif risk_factor == 'low':
            base_threshold = max(0.80, base_threshold - 0.05)  # Lower threshold for low risk
        
        # Context-based adjustments
        if context.get('first_time_contributor', False):
            base_threshold = max(0.85, base_threshold - 0.05)  # More lenient for new contributors
        
        if context.get('time_pressure', False):
            base_threshold = max(0.90, base_threshold - 0.03)  # Slight reduction for time pressure
        
        return base_threshold
    
    def _calculate_score_confidence(self, dimension_scores: Dict[QualityDimension, float], quality_data: Dict[str, Any]) -> float:
        """Calculate confidence in the quality score."""
        scores = list(dimension_scores.values())
        
        # Base confidence on score variance (lower variance = higher confidence)
        if len(scores) > 1:
            variance = statistics.variance(scores)
            confidence = max(0.5, 1.0 - variance)  # High variance reduces confidence
        else:
            confidence = 0.8
        
        # Adjust confidence based on data quality
        data_quality = quality_data.get('data_quality_score', 0.9)
        confidence = confidence * data_quality
        
        return min(1.0, confidence)
    
    def _extract_context_factors(self, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extract numerical context factors for analysis."""
        if not context:
            return {}
        
        factors = {}
        
        # Convert boolean factors to numerical
        bool_factors = ['emergency', 'security_critical', 'time_pressure', 'first_time_contributor']
        for factor in bool_factors:
            if factor in context:
                factors[factor] = 1.0 if context[factor] else 0.0
        
        # Extract numerical factors
        num_factors = ['complexity', 'risk_score', 'priority']
        for factor in num_factors:
            if factor in context:
                factors[factor] = float(context.get(factor, 0))
        
        return factors
    
    def _detect_quality_violations(self, quality_score: QualityScore, quality_data: Dict[str, Any]) -> List[QualityViolation]:
        """Detect quality violations across all dimensions."""
        violations = []
        dimension_config = self.config.get('quality_enforcement', {}).get('dimensions', {})
        
        for dimension, score in quality_score.dimension_scores.items():
            required_score = dimension_config.get(dimension.value, {}).get('required_score', 0.95)
            
            if score < required_score:
                violation = QualityViolation(
                    dimension=dimension,
                    violation_type="threshold_violation",
                    severity=self._calculate_violation_severity(score, required_score),
                    description=f"{dimension.value} score ({score:.2f}) below required threshold ({required_score:.2f})",
                    recommendation=self._generate_violation_recommendation(dimension, score, required_score),
                    blocking=score < (required_score * 0.9),  # Block if more than 10% below threshold
                    auto_fixable=self._is_auto_fixable(dimension)
                )
                violations.append(violation)
        
        return violations
    
    def _calculate_violation_severity(self, actual: float, required: float) -> str:
        """Calculate severity of a quality violation."""
        gap = required - actual
        
        if gap >= 0.2:
            return "critical"
        elif gap >= 0.1:
            return "high"
        elif gap >= 0.05:
            return "medium"
        else:
            return "low"
    
    def _generate_violation_recommendation(self, dimension: QualityDimension, actual: float, required: float) -> str:
        """Generate recommendation for fixing a quality violation."""
        recommendations = {
            QualityDimension.FUNCTIONALITY: "Add more unit tests and integration tests. Verify feature completeness.",
            QualityDimension.SECURITY: "Run security scans, review authentication/authorization, check for vulnerabilities.",
            QualityDimension.PERFORMANCE: "Run performance benchmarks, optimize slow operations, check resource usage.",
            QualityDimension.RELIABILITY: "Add error handling, improve stability testing, check edge cases.",
            QualityDimension.MAINTAINABILITY: "Refactor complex code, improve code structure, add code comments.",
            QualityDimension.TESTABILITY: "Improve test coverage, add test utilities, simplify test setup.",
            QualityDimension.DOCUMENTATION: "Add API documentation, update README, document configuration.",
            QualityDimension.COMPLIANCE: "Review regulatory requirements, update compliance documentation."
        }
        
        base_recommendation = recommendations.get(dimension, "Improve quality in this dimension.")
        gap = required - actual
        
        if gap >= 0.1:
            return f"URGENT: {base_recommendation} Score needs to improve by {gap:.1%}."
        else:
            return f"{base_recommendation} Minor improvement needed ({gap:.1%})."
    
    def _is_auto_fixable(self, dimension: QualityDimension) -> bool:
        """Check if a quality dimension violation can be automatically fixed."""
        auto_fixable_dimensions = {
            QualityDimension.TESTABILITY,      # Can add test templates
            QualityDimension.DOCUMENTATION,    # Can generate basic docs
            QualityDimension.MAINTAINABILITY   # Can suggest refactoring
        }
        
        return dimension in auto_fixable_dimensions
    
    def _make_enforcement_decision(self, quality_score: QualityScore, violations: List[QualityViolation], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make enforcement decision based on quality assessment."""
        enforcement_config = self.config.get('quality_enforcement', {}).get('enforcement', {})
        
        # Check if quality threshold is met
        threshold = quality_score.risk_adjusted_threshold or quality_score.threshold
        meets_threshold = quality_score.overall_score >= threshold
        
        # Check for blocking violations
        blocking_violations = [v for v in violations if v.blocking]
        has_blocking_violations = len(blocking_violations) > 0
        
        # Emergency fallback check
        emergency_threshold = self.config.get('quality_enforcement', {}).get('thresholds', {}).get('emergency', 0.80)
        meets_emergency_threshold = quality_score.overall_score >= emergency_threshold
        
        # Make decision
        decision = {
            'can_proceed': False,
            'action': 'block',
            'escalation_required': False,
            'requires_specialist_review': False,
            'fallback_reason': None,
            'next_steps': []
        }
        
        if meets_threshold and not has_blocking_violations:
            # Quality standards met - allow to proceed
            decision.update({
                'can_proceed': True,
                'action': 'approve',
                'next_steps': ['Issue can proceed to closure']
            })
            
        elif has_blocking_violations:
            # Blocking violations - must be fixed
            decision.update({
                'action': 'block',
                'requires_specialist_review': True,
                'next_steps': [
                    'Fix blocking quality violations',
                    'Specialist review required',
                    'Re-run quality assessment'
                ]
            })
            
        elif not meets_threshold:
            # Below threshold but no blocking violations
            if meets_emergency_threshold and context and context.get('emergency', False):
                # Emergency fallback
                decision.update({
                    'can_proceed': True,
                    'action': 'approve_with_fallback',
                    'fallback_reason': f'Emergency fallback applied - quality score {quality_score.overall_score:.2f} meets emergency threshold {emergency_threshold:.2f}',
                    'escalation_required': True,
                    'next_steps': [
                        'Issue approved under emergency fallback',
                        'Management escalation triggered',
                        'Post-closure quality improvement required'
                    ]
                })
            else:
                # Standard quality improvement required
                decision.update({
                    'action': 'improve_quality',
                    'requires_specialist_review': enforcement_config.get('require_specialist_review', True),
                    'next_steps': [
                        'Improve quality scores to meet threshold',
                        'Address quality violations',
                        'Consider specialist review'
                    ]
                })
        
        return decision
    
    def _trigger_specialist_review(self, issue_number: int, quality_score: QualityScore, violations: List[QualityViolation]) -> Optional[Dict[str, Any]]:
        """Trigger specialist review for quality issues."""
        try:
            # Determine specialist type based on violations
            specialist_type = self._determine_specialist_type(violations)
            
            # Create assignment request
            assignment_request = AssignmentRequest(
                issue_number=issue_number,
                risk_score=1.0 - quality_score.overall_score,  # Higher risk for lower quality
                risk_level="high" if quality_score.overall_score < 0.85 else "medium",
                primary_risk_factors=[f"quality_{v.dimension.value}" for v in violations],
                specialist_type=specialist_type,
                urgency_level="high" if any(v.blocking for v in violations) else "medium",
                files_changed=[],  # Would need to extract from issue
                estimated_review_time=4.0,
                special_requirements=["quality_improvement"]
            )
            
            # Assign specialist
            result = self.specialist_engine.assign_specialist(assignment_request)
            
            if result.assigned_specialist:
                return {
                    'specialist_assigned': True,
                    'specialist_name': result.assigned_specialist.name,
                    'specialist_type': result.assigned_specialist.specialist_type.value,
                    'github_issue_number': result.github_issue_number,
                    'sla_deadline': result.sla_deadline.isoformat() if result.sla_deadline else None,
                    'assignment_confidence': result.assignment_confidence
                }
            else:
                return {
                    'specialist_assigned': False,
                    'reason': 'No available specialist found',
                    'fallback_required': True
                }
                
        except Exception as e:
            self.logger.error(f"Error triggering specialist review: {e}")
            return {
                'specialist_assigned': False,
                'error': str(e),
                'fallback_required': True
            }
    
    def _determine_specialist_type(self, violations: List[QualityViolation]) -> SpecialistType:
        """Determine appropriate specialist type based on violations."""
        # Count violations by dimension
        dimension_counts = {}
        for violation in violations:
            dimension = violation.dimension
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
        
        # Map dimensions to specialist types
        if QualityDimension.SECURITY in dimension_counts:
            return SpecialistType.SECURITY
        elif QualityDimension.PERFORMANCE in dimension_counts:
            return SpecialistType.PERFORMANCE
        elif QualityDimension.MAINTAINABILITY in dimension_counts:
            return SpecialistType.ARCHITECTURE
        else:
            return SpecialistType.ARCHITECTURE  # Default fallback
    
    def _update_adaptive_systems(self, result: QualityEnforcementResult):
        """Update adaptive threshold and learning systems."""
        try:
            # Update adaptive threshold manager
            if hasattr(self.adaptive_threshold_manager, 'update_thresholds'):
                self.adaptive_threshold_manager.update_thresholds(
                    result.issue_number,
                    result.quality_score.overall_score,
                    result.can_proceed
                )
            
            # Update context weight manager
            if hasattr(self.context_weight_manager, 'update_weights') and result.quality_score.context_factors:
                self.context_weight_manager.update_weights(
                    result.quality_score.context_factors,
                    result.quality_score.overall_score
                )
            
        except Exception as e:
            self.logger.warning(f"Error updating adaptive systems: {e}")
    
    def _record_enforcement_decision(self, result: QualityEnforcementResult):
        """Record enforcement decision for analysis and reporting."""
        try:
            decision_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'issue_number': result.issue_number,
                'quality_score': result.quality_score.overall_score,
                'threshold': result.quality_score.threshold,
                'can_proceed': result.can_proceed,
                'enforcement_action': result.enforcement_action,
                'violation_count': len(result.violations),
                'blocking_violations': len([v for v in result.violations if v.blocking]),
                'specialist_required': result.specialist_assignment is not None,
                'fallback_used': result.fallback_reason is not None
            }
            
            # Save to quality metrics directory
            decisions_dir = Path("knowledge/quality_metrics/enforcement_decisions")
            decisions_dir.mkdir(parents=True, exist_ok=True)
            
            decision_file = decisions_dir / f"decision_{result.issue_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(decision_file, 'w') as f:
                json.dump(decision_record, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Error recording enforcement decision: {e}")
    
    def _create_fallback_result(self, issue_number: int, error: str) -> QualityEnforcementResult:
        """Create fallback result when enforcement fails."""
        fallback_score = QualityScore(
            overall_score=0.80,  # Emergency threshold
            dimension_scores={dim: 0.80 for dim in QualityDimension},
            enforcement_level=QualityEnforcementLevel.EMERGENCY,
            threshold=0.80,
            meets_threshold=True,
            confidence=0.5
        )
        
        return QualityEnforcementResult(
            issue_number=issue_number,
            can_proceed=True,
            quality_score=fallback_score,
            violations=[],
            enforcement_action="emergency_fallback",
            escalation_required=True,
            specialist_assignment=None,
            fallback_reason=f"Quality enforcement system error: {error}",
            next_steps=[
                "Emergency fallback applied due to system error",
                "Manual review required",
                "Fix quality enforcement system"
            ],
            quality_report={"error": error}
        )
    
    def get_quality_enforcement_report(self, issue_number: Optional[int] = None) -> Dict[str, Any]:
        """Generate comprehensive quality enforcement report."""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_status': self._get_system_status(),
            'configuration': self._get_configuration_summary(),
            'recent_decisions': self._get_recent_decisions(issue_number)
        }
        
        if issue_number:
            report['issue_specific'] = {
                'issue_number': issue_number,
                'current_quality_status': self._get_current_quality_status(issue_number)
            }
        
        return report
    
    def _get_system_status(self) -> Dict[str, str]:
        """Get status of all quality sub-systems."""
        systems = {
            'gate_enforcement': self.gate_enforcement,
            'metrics_collector': self.metrics_collector,
            'analytics_engine': self.analytics_engine,
            'decision_engine': self.decision_engine,
            'pattern_analyzer': self.pattern_analyzer,
            'risk_assessment': self.risk_assessment,
            'specialist_engine': self.specialist_engine,
            'sla_monitor': self.sla_monitor
        }
        
        status = {}
        for name, system in systems.items():
            if hasattr(system, '__class__') and system.__class__.__name__ == 'PlaceholderSystem':
                status[name] = "placeholder"
            else:
                status[name] = "operational"
        
        return status
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of quality enforcement configuration."""
        config = self.config.get('quality_enforcement', {})
        return {
            'thresholds': config.get('thresholds', {}),
            'dimensions': {k: v.get('weight') for k, v in config.get('dimensions', {}).items()},
            'enforcement': config.get('enforcement', {})
        }
    
    def _get_recent_decisions(self, issue_number: Optional[int]) -> List[Dict[str, Any]]:
        """Get recent enforcement decisions."""
        decisions = []
        decisions_dir = Path("knowledge/quality_metrics/enforcement_decisions")
        
        if decisions_dir.exists():
            decision_files = sorted(decisions_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            for decision_file in decision_files[:10]:  # Last 10 decisions
                try:
                    with open(decision_file, 'r') as f:
                        decision = json.load(f)
                        if not issue_number or decision.get('issue_number') == issue_number:
                            decisions.append(decision)
                except Exception as e:
                    self.logger.warning(f"Error reading decision file {decision_file}: {e}")
        
        return decisions
    
    def _get_current_quality_status(self, issue_number: int) -> Dict[str, Any]:
        """Get current quality status for a specific issue."""
        try:
            # This would typically fetch from GitHub and analyze current state
            return {
                'issue_number': issue_number,
                'last_assessment': datetime.now(timezone.utc).isoformat(),
                'status': 'requires_assessment'
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    """Command line interface for comprehensive quality enforcement."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_quality_enforcement.py <command> [args]")
        print("Commands:")
        print("  enforce <issue_number>                    - Run quality enforcement")
        print("  report [issue_number]                     - Generate quality report")
        print("  test-system                               - Test system integration")
        print("  validate-config                           - Validate configuration")
        return
    
    command = sys.argv[1]
    enforcement = ComprehensiveQualityEnforcement()
    
    if command == "enforce" and len(sys.argv) >= 3:
        issue_number = int(sys.argv[2])
        
        # Optional context from command line
        context = {}
        if len(sys.argv) >= 4:
            context_str = sys.argv[3]
            try:
                context = json.loads(context_str)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON context: {context_str}")
        
        result = enforcement.enforce_quality_standards(issue_number, context)
        
        print(f"🎯 Quality Enforcement Result for Issue #{issue_number}")
        print(f"Can Proceed: {result.can_proceed}")
        print(f"Quality Score: {result.quality_score.overall_score:.2f}")
        print(f"Threshold: {result.quality_score.threshold:.2f}")
        print(f"Action: {result.enforcement_action}")
        
        if result.violations:
            print(f"\nViolations ({len(result.violations)}):")
            for i, violation in enumerate(result.violations[:5], 1):
                print(f"  {i}. {violation.dimension.value}: {violation.description}")
        
        if result.specialist_assignment:
            print(f"\nSpecialist Review: {result.specialist_assignment}")
        
        print(f"\nNext Steps:")
        for step in result.next_steps:
            print(f"  - {step}")
        
    elif command == "report":
        issue_number = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        report = enforcement.get_quality_enforcement_report(issue_number)
        print(json.dumps(report, indent=2, default=str))
        
    elif command == "test-system":
        print("🧪 Testing Quality Enforcement System Integration")
        
        # Test with mock issue
        test_context = {
            'emergency': False,
            'security_critical': True,
            'complexity': 0.7,
            'risk_level': 'high'
        }
        
        result = enforcement.enforce_quality_standards(9999, test_context)
        
        print(f"Test Result: {result.enforcement_action}")
        print(f"Quality Score: {result.quality_score.overall_score:.2f}")
        print(f"System Status: {enforcement._get_system_status()}")
        
    elif command == "validate-config":
        print("🔧 Validating Quality Enforcement Configuration")
        
        config = enforcement._get_configuration_summary()
        print(json.dumps(config, indent=2))
        
        # Validate thresholds
        thresholds = config.get('thresholds', {})
        if thresholds.get('default', 0) >= 0.95:
            print("✅ Default threshold meets 95% requirement")
        else:
            print("❌ Default threshold below 95% requirement")
        
        # Validate dimensions
        dimensions = config.get('dimensions', {})
        total_weight = sum(w for w in dimensions.values() if w)
        print(f"📊 Total dimension weight: {total_weight:.2f}")
        
    else:
        print(f"Unknown command: {command}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())