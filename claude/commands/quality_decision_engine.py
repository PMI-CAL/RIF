#!/usr/bin/env python3
"""
Multi-Dimensional Quality Decision Engine - Issue #93 Phase 1
Enhanced quality scoring with risk weighting and context awareness.

This engine replaces simple pass/fail decisions with nuanced quality assessment:
- Multi-dimensional scoring across coverage, security, performance, code quality
- Risk adjustment based on change characteristics
- Context weighting per component type
- Clear decision explanations and reasoning
"""

import json
import yaml
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class QualityDecision(Enum):
    """Quality decision types with clear semantics."""
    PASS = "PASS"
    CONCERNS = "CONCERNS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"

@dataclass
class QualityMetrics:
    """Container for raw quality metrics."""
    test_coverage: float = 0.0
    security_score: Dict[str, int] = None
    performance_regression: float = 0.0
    code_quality_score: float = 100.0
    
    def __post_init__(self):
        if self.security_score is None:
            self.security_score = {
                'critical': 0,
                'high': 0, 
                'medium': 0,
                'low': 0
            }

@dataclass 
class RiskFactors:
    """Container for risk assessment factors."""
    large_change: bool = False
    security_files: bool = False
    critical_paths: bool = False
    external_dependencies: bool = False
    previous_failures: int = 0
    no_tests: bool = False
    low_coverage_area: bool = False
    change_size_loc: int = 0
    files_modified: List[str] = None
    
    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []

@dataclass
class QualityResult:
    """Complete quality assessment result."""
    risk_adjusted_score: float
    base_quality_score: float
    risk_multiplier: float
    context_weight: float
    decision: QualityDecision
    dimension_scores: Dict[str, float]
    risk_factors: Dict[str, Any]
    decision_explanation: str
    recommendations: List[str]
    evidence: List[str]
    performance_metrics: Dict[str, float]

class QualityDecisionEngine:
    """
    Multi-dimensional quality scoring engine with risk adjustment and context awareness.
    
    Enhanced formula: Risk_Adjusted_Score = Base_Quality × (1 - Risk_Multiplier) × Context_Weight
    
    Base_Quality = Weighted(
        Coverage(30%) + Security(40%) + Performance(20%) + Code_Quality(10%)
    )
    """
    
    def __init__(self, config_path: str = "config/quality-dimensions.yaml"):
        """Initialize the quality decision engine."""
        self.config_path = config_path
        
        # Setup logging first
        self.setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        self.dimensions = self.config.get('quality_dimensions', {})
        self.risk_config = self.config.get('risk_adjustment', {})
        self.context_weights = self.config.get('context_weights', {})
        self.decision_matrix = self.config.get('decision_matrix', {})
        self.performance_config = self.config.get('performance', {})
        
        # Performance tracking
        self.calculation_times = []
        
        # Validate configuration
        self._validate_configuration()
    
    def setup_logging(self):
        """Setup logging for the quality decision engine."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - QualityDecisionEngine - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load quality dimensions configuration."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.error(f"Config file {self.config_path} not found")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback configuration for backward compatibility."""
        return {
            'quality_dimensions': {
                'test_coverage': {'weight': 30, 'threshold': 80},
                'security_validation': {'weight': 40, 'critical_vulnerabilities': 0},
                'performance_impact': {'weight': 20, 'max_regression': 5.0},
                'code_quality': {'weight': 10, 'max_critical_issues': 0}
            },
            'risk_adjustment': {'max_risk_multiplier': 0.3},
            'context_weights': {'default': {'weight': 1.0}},
            'decision_matrix': {
                'pass': {'conditions': ['risk_adjusted_score >= 80']},
                'concerns': {'conditions': ['risk_adjusted_score >= 60']},
                'fail': {'conditions': ['risk_adjusted_score < 60']},
                'blocked': {'conditions': ['risk_level == critical']}
            },
            'performance': {'calculation_time_limit_ms': 100}
        }
    
    def _validate_configuration(self):
        """Validate configuration completeness and consistency."""
        required_sections = ['quality_dimensions', 'risk_adjustment', 'context_weights', 'decision_matrix']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Missing configuration section: {section}")
        
        # Validate dimension weights sum to reasonable value
        total_weight = sum(dim.get('weight', 0) for dim in self.dimensions.values())
        if abs(total_weight - 100) > 5:  # Allow 5% tolerance
            self.logger.warning(f"Dimension weights sum to {total_weight}%, expected ~100%")
    
    def calculate_risk_adjusted_score(
        self, 
        base_metrics: QualityMetrics, 
        risk_factors: RiskFactors, 
        context: str
    ) -> float:
        """
        Calculate the risk-adjusted quality score.
        
        Args:
            base_metrics: Raw quality metrics from validation
            risk_factors: Risk assessment factors
            context: Component context type
            
        Returns:
            Risk-adjusted quality score (0-100)
        """
        start_time = time.time()
        
        try:
            # Step 1: Calculate base quality score from dimensions
            base_quality = self._calculate_base_quality_score(base_metrics)
            
            # Step 2: Calculate risk multiplier 
            risk_multiplier = self._calculate_risk_multiplier(risk_factors)
            
            # Step 3: Get context weight
            context_weight = self._get_context_weight(context)
            
            # Step 4: Apply formula
            # Risk_Adjusted_Score = Base_Quality × (1 - Risk_Multiplier) × Context_Weight
            risk_adjusted_score = base_quality * (1 - risk_multiplier) * context_weight
            
            # Clamp between 0 and 100
            risk_adjusted_score = max(0.0, min(100.0, risk_adjusted_score))
            
            # Track performance
            calculation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.calculation_times.append(calculation_time)
            
            if calculation_time > self.performance_config.get('calculation_time_limit_ms', 100):
                self.logger.warning(f"Quality calculation took {calculation_time:.1f}ms (limit: {self.performance_config.get('calculation_time_limit_ms', 100)}ms)")
            
            self.logger.debug(f"Quality score calculated: base={base_quality:.1f}, risk={risk_multiplier:.3f}, context={context_weight:.2f}, final={risk_adjusted_score:.1f}")
            
            return risk_adjusted_score
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted score: {e}")
            # Fallback to legacy calculation
            return self._legacy_quality_calculation(base_metrics)
    
    def _calculate_base_quality_score(self, metrics: QualityMetrics) -> float:
        """Calculate base quality score using weighted dimensions."""
        dimension_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Test Coverage dimension (30% weight)
        if 'test_coverage' in self.dimensions:
            coverage_config = self.dimensions['test_coverage']
            weight = coverage_config.get('weight', 30)
            
            # Convert coverage percentage to score (0-100)
            coverage_score = min(100.0, metrics.test_coverage)
            dimension_scores['test_coverage'] = coverage_score
            total_weighted_score += coverage_score * weight
            total_weight += weight
        
        # Security Validation dimension (40% weight)
        if 'security_validation' in self.dimensions:
            security_config = self.dimensions['security_validation']
            weight = security_config.get('weight', 40)
            
            # Security score based on vulnerability count (inverse scoring)
            critical = metrics.security_score.get('critical', 0)
            high = metrics.security_score.get('high', 0)
            medium = metrics.security_score.get('medium', 0)
            
            # Scoring: 100 - (50*critical + 20*high + 5*medium)
            security_score = max(0.0, 100 - (50 * critical + 20 * high + 5 * medium))
            dimension_scores['security_validation'] = security_score
            total_weighted_score += security_score * weight
            total_weight += weight
        
        # Performance Impact dimension (20% weight)
        if 'performance_impact' in self.dimensions:
            perf_config = self.dimensions['performance_impact']
            weight = perf_config.get('weight', 20)
            
            # Performance score based on regression (inverse scoring)
            regression = abs(metrics.performance_regression)
            max_regression = perf_config.get('max_regression', 5.0)
            
            if regression <= max_regression:
                perf_score = 100.0 - (regression / max_regression) * 20  # Linear penalty up to 20 points
            else:
                perf_score = max(0.0, 80.0 - (regression - max_regression) * 10)  # Harsher penalty beyond limit
            
            dimension_scores['performance_impact'] = perf_score
            total_weighted_score += perf_score * weight
            total_weight += weight
        
        # Code Quality dimension (10% weight)
        if 'code_quality' in self.dimensions:
            quality_config = self.dimensions['code_quality']
            weight = quality_config.get('weight', 10)
            
            # Use maintainability score directly
            quality_score = max(0.0, min(100.0, metrics.code_quality_score))
            dimension_scores['code_quality'] = quality_score
            total_weighted_score += quality_score * weight
            total_weight += weight
        
        # Calculate final weighted average
        if total_weight > 0:
            base_score = total_weighted_score / total_weight
        else:
            base_score = 0.0
        
        self.logger.debug(f"Base quality score: {base_score:.1f} from dimensions: {dimension_scores}")
        return base_score
    
    def _calculate_risk_multiplier(self, risk_factors: RiskFactors) -> float:
        """Calculate risk multiplier based on change characteristics."""
        risk_multiplier = 0.0
        max_multiplier = self.risk_config.get('max_risk_multiplier', 0.3)
        
        risk_config = self.risk_config.get('risk_factors', {})
        
        # Large change risk
        if risk_factors.large_change:
            risk_multiplier += risk_config.get('large_change', {}).get('multiplier', 0.1)
        
        # Security files risk
        if risk_factors.security_files:
            risk_multiplier += risk_config.get('security_files', {}).get('multiplier', 0.15)
        
        # Critical paths risk
        if risk_factors.critical_paths:
            risk_multiplier += risk_config.get('critical_paths', {}).get('multiplier', 0.1)
        
        # External dependencies risk
        if risk_factors.external_dependencies:
            risk_multiplier += risk_config.get('external_dependencies', {}).get('multiplier', 0.05)
        
        # Previous failures risk
        if risk_factors.previous_failures > 0:
            failure_multiplier = risk_config.get('previous_failures', {}).get('multiplier', 0.1)
            risk_multiplier += min(failure_multiplier * risk_factors.previous_failures, 0.2)  # Cap at 0.2
        
        # No tests risk
        if risk_factors.no_tests:
            risk_multiplier += risk_config.get('no_tests', {}).get('multiplier', 0.2)
        
        # Low coverage area risk
        if risk_factors.low_coverage_area:
            risk_multiplier += risk_config.get('low_coverage_area', {}).get('multiplier', 0.1)
        
        # Cap at maximum risk multiplier
        risk_multiplier = min(risk_multiplier, max_multiplier)
        
        self.logger.debug(f"Risk multiplier calculated: {risk_multiplier:.3f} (max: {max_multiplier})")
        return risk_multiplier
    
    def _get_context_weight(self, context: str) -> float:
        """Get context weight based on component type."""
        if context in self.context_weights:
            weight = self.context_weights[context].get('weight', 1.0)
        else:
            # Try to classify context automatically
            weight = self._classify_context_automatically(context)
        
        self.logger.debug(f"Context weight for '{context}': {weight}")
        return weight
    
    def _classify_context_automatically(self, context: str) -> float:
        """Automatically classify context based on patterns."""
        context_lower = context.lower()
        
        # Check against known patterns
        for context_type, config in self.context_weights.items():
            patterns = config.get('patterns', [])
            for pattern in patterns:
                # Simple pattern matching (could be enhanced with regex)
                if pattern.replace('*', '') in context_lower:
                    return config.get('weight', 1.0)
        
        # Default weight if no match found
        return 1.0
    
    def make_quality_decision(
        self, 
        score: float, 
        critical_issues: List[str],
        risk_level: str = "low"
    ) -> QualityDecision:
        """
        Make quality decision based on score and critical issues.
        
        Args:
            score: Risk-adjusted quality score
            critical_issues: List of critical issues found
            risk_level: Overall risk assessment level
            
        Returns:
            Quality decision enum
        """
        try:
            # Check for BLOCKED conditions first
            if risk_level == "critical":
                return QualityDecision.BLOCKED
            
            if len(critical_issues) >= 3:  # Multiple gate failures
                return QualityDecision.BLOCKED
            
            # Check for critical security issues
            has_critical_security = any('security' in issue.lower() and 'critical' in issue.lower() 
                                     for issue in critical_issues)
            
            if has_critical_security:
                return QualityDecision.FAIL
            
            # Apply decision matrix based on score
            if score >= 80:  # Default threshold, should be made configurable per context
                if len(critical_issues) == 0:
                    return QualityDecision.PASS
                else:
                    return QualityDecision.CONCERNS
            elif score >= 60:
                return QualityDecision.CONCERNS
            else:
                return QualityDecision.FAIL
        
        except Exception as e:
            self.logger.error(f"Error making quality decision: {e}")
            return QualityDecision.FAIL  # Fail-safe default
    
    def generate_decision_explanation(
        self, 
        result: QualityResult
    ) -> str:
        """
        Generate clear explanation for quality decision.
        
        Args:
            result: Complete quality assessment result
            
        Returns:
            Detailed explanation string
        """
        try:
            explanation_parts = []
            
            # Header with decision
            explanation_parts.append(f"🎯 **Quality Decision: {result.decision.value}**")
            
            # Score breakdown
            explanation_parts.append(f"📊 **Risk-Adjusted Score: {result.risk_adjusted_score:.1f}/100**")
            explanation_parts.append(f"   • Base Quality Score: {result.base_quality_score:.1f}")
            explanation_parts.append(f"   • Risk Adjustment: -{result.risk_multiplier:.1%}")
            explanation_parts.append(f"   • Context Weight: {result.context_weight:.2f}x")
            
            # Dimension breakdown
            explanation_parts.append(f"\n📋 **Dimension Scores:**")
            for dimension, score in result.dimension_scores.items():
                weight = self.dimensions.get(dimension, {}).get('weight', 0)
                explanation_parts.append(f"   • {dimension.replace('_', ' ').title()}: {score:.1f}/100 (weight: {weight}%)")
            
            # Risk factors if any
            if any(result.risk_factors.values()):
                explanation_parts.append(f"\n⚠️ **Risk Factors Detected:**")
                for factor, value in result.risk_factors.items():
                    if value:
                        explanation_parts.append(f"   • {factor.replace('_', ' ').title()}")
            
            # Decision rationale
            explanation_parts.append(f"\n🤔 **Decision Rationale:**")
            if result.decision == QualityDecision.PASS:
                explanation_parts.append("   ✅ All quality requirements met")
                explanation_parts.append("   ✅ Risk-adjusted score exceeds threshold")
                explanation_parts.append("   ✅ No critical security issues found")
            elif result.decision == QualityDecision.CONCERNS:
                explanation_parts.append("   ⚠️ Generally acceptable quality with minor concerns")
                explanation_parts.append("   ⚠️ Score meets minimum threshold but has improvement areas")
            elif result.decision == QualityDecision.FAIL:
                explanation_parts.append("   ❌ Quality score below acceptable threshold")
                explanation_parts.append("   ❌ Significant issues requiring attention")
            elif result.decision == QualityDecision.BLOCKED:
                explanation_parts.append("   🚫 Critical risk factors require manual review")
                explanation_parts.append("   🚫 Multiple quality gate failures detected")
            
            # Recommendations
            if result.recommendations:
                explanation_parts.append(f"\n💡 **Recommendations:**")
                for rec in result.recommendations:
                    explanation_parts.append(f"   • {rec}")
            
            # Evidence
            if result.evidence:
                explanation_parts.append(f"\n📝 **Supporting Evidence:**")
                for evidence in result.evidence:
                    explanation_parts.append(f"   • {evidence}")
            
            # Performance metrics
            if result.performance_metrics.get('calculation_time_ms'):
                explanation_parts.append(f"\n⏱️ **Performance:** Calculated in {result.performance_metrics['calculation_time_ms']:.1f}ms")
            
            return "\n".join(explanation_parts)
        
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return f"Quality Decision: {result.decision.value} (Score: {result.risk_adjusted_score:.1f})"
    
    def _legacy_quality_calculation(self, metrics: QualityMetrics) -> float:
        """Fallback to legacy quality calculation for compatibility."""
        # Legacy formula: 100 - (20 × FAILs) - (10 × CONCERNs)
        # Approximate using available metrics
        fails = 0
        concerns = 0
        
        # Convert metrics to fail/concern counts
        if metrics.test_coverage < 60:
            fails += 1
        elif metrics.test_coverage < 80:
            concerns += 1
        
        if metrics.security_score.get('critical', 0) > 0:
            fails += 1
        elif metrics.security_score.get('high', 0) > 0:
            concerns += 1
        
        if abs(metrics.performance_regression) > 10:
            fails += 1
        elif abs(metrics.performance_regression) > 5:
            concerns += 1
        
        if metrics.code_quality_score < 50:
            fails += 1
        elif metrics.code_quality_score < 70:
            concerns += 1
        
        legacy_score = max(0, 100 - (20 * fails) - (10 * concerns))
        self.logger.info(f"Using legacy quality calculation: {legacy_score} (fails={fails}, concerns={concerns})")
        return legacy_score
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the engine."""
        if not self.calculation_times:
            return {}
        
        return {
            'average_calculation_time_ms': sum(self.calculation_times) / len(self.calculation_times),
            'max_calculation_time_ms': max(self.calculation_times),
            'min_calculation_time_ms': min(self.calculation_times),
            'total_calculations': len(self.calculation_times),
            'performance_target_ms': self.performance_config.get('calculation_time_limit_ms', 100)
        }

def main():
    """Command line interface for the quality decision engine."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Dimensional Quality Decision Engine')
    parser.add_argument('--config', default='config/quality-dimensions.yaml', help='Configuration file path')
    parser.add_argument('--test-coverage', type=float, default=85.0, help='Test coverage percentage')
    parser.add_argument('--critical-vulns', type=int, default=0, help='Critical vulnerabilities count')
    parser.add_argument('--high-vulns', type=int, default=0, help='High vulnerabilities count')
    parser.add_argument('--performance-regression', type=float, default=0.0, help='Performance regression percentage')
    parser.add_argument('--code-quality', type=float, default=80.0, help='Code quality score')
    parser.add_argument('--context', default='business_logic', help='Component context type')
    parser.add_argument('--large-change', action='store_true', help='Large change flag')
    parser.add_argument('--security-files', action='store_true', help='Security files modified')
    parser.add_argument('--output', choices=['score', 'decision', 'full'], default='full', help='Output format')
    
    args = parser.parse_args()
    
    # Create engine
    engine = QualityDecisionEngine(config_path=args.config)
    
    # Create metrics
    metrics = QualityMetrics(
        test_coverage=args.test_coverage,
        security_score={
            'critical': args.critical_vulns,
            'high': args.high_vulns,
            'medium': 0,
            'low': 0
        },
        performance_regression=args.performance_regression,
        code_quality_score=args.code_quality
    )
    
    # Create risk factors
    risk_factors = RiskFactors(
        large_change=args.large_change,
        security_files=args.security_files,
        critical_paths=False,
        external_dependencies=False,
        previous_failures=0,
        no_tests=False,
        low_coverage_area=args.test_coverage < 50
    )
    
    # Calculate score
    score = engine.calculate_risk_adjusted_score(metrics, risk_factors, args.context)
    
    # Make decision
    decision = engine.make_quality_decision(score, [], "low")
    
    # Generate result
    result = QualityResult(
        risk_adjusted_score=score,
        base_quality_score=engine._calculate_base_quality_score(metrics),
        risk_multiplier=engine._calculate_risk_multiplier(risk_factors),
        context_weight=engine._get_context_weight(args.context),
        decision=decision,
        dimension_scores={"test_coverage": metrics.test_coverage, "security": 100 - args.critical_vulns * 50},
        risk_factors={"large_change": args.large_change, "security_files": args.security_files},
        decision_explanation="",
        recommendations=[],
        evidence=[],
        performance_metrics=engine.get_performance_metrics()
    )
    
    # Output based on format
    if args.output == 'score':
        print(f"{score:.1f}")
    elif args.output == 'decision':
        print(decision.value)
    else:
        result.decision_explanation = engine.generate_decision_explanation(result)
        print(json.dumps({
            'risk_adjusted_score': result.risk_adjusted_score,
            'decision': result.decision.value,
            'explanation': result.decision_explanation,
            'performance_metrics': result.performance_metrics
        }, indent=2))
    
    return 0 if decision != QualityDecision.FAIL else 1

if __name__ == "__main__":
    exit(main())