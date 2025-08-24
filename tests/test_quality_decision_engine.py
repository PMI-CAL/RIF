#!/usr/bin/env python3
"""
Test suite for Quality Decision Engine
Issue #93: Multi-Dimensional Quality Scoring System
"""

import os
import sys
import tempfile
import yaml
import pytest
from pathlib import Path

# Add the claude/commands directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'claude', 'commands'))

from quality_decision_engine import (
    QualityDecisionEngine,
    QualityMetrics,
    RiskFactors,
    QualityResult,
    QualityDecision
)

class TestQualityDecisionEngine:
    """Test suite for QualityDecisionEngine class."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_quality_config.yaml"
        
        # Create test configuration
        self._create_test_config()
        
        # Initialize engine with test config
        self.engine = QualityDecisionEngine(config_path=str(self.config_path))
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """Create test configuration file."""
        config = {
            'quality_dimensions': {
                'test_coverage': {'weight': 30, 'threshold': 80},
                'security_validation': {'weight': 40, 'critical_vulnerabilities': 0},
                'performance_impact': {'weight': 20, 'max_regression': 5.0},
                'code_quality': {'weight': 10, 'max_critical_issues': 0}
            },
            'risk_adjustment': {
                'max_risk_multiplier': 0.3,
                'risk_factors': {
                    'large_change': {'multiplier': 0.1},
                    'security_files': {'multiplier': 0.15},
                    'critical_paths': {'multiplier': 0.1},
                    'external_dependencies': {'multiplier': 0.05},
                    'previous_failures': {'multiplier': 0.1},
                    'no_tests': {'multiplier': 0.2},
                    'low_coverage_area': {'multiplier': 0.1}
                }
            },
            'context_weights': {
                'critical_algorithms': {'weight': 1.2},
                'public_apis': {'weight': 1.1},
                'business_logic': {'weight': 1.0},
                'ui_components': {'weight': 0.8}
            },
            'decision_matrix': {
                'pass': {'conditions': ['risk_adjusted_score >= 80']},
                'concerns': {'conditions': ['risk_adjusted_score >= 60']},
                'fail': {'conditions': ['risk_adjusted_score < 60']},
                'blocked': {'conditions': ['risk_level == critical']}
            },
            'performance': {'calculation_time_limit_ms': 100}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine.config is not None
        assert self.engine.dimensions is not None
        assert self.engine.risk_config is not None
        assert self.engine.context_weights is not None
        assert self.engine.decision_matrix is not None
        assert len(self.engine.calculation_times) == 0
    
    def test_base_quality_score_calculation(self):
        """Test base quality score calculation with different metrics."""
        # Test high-quality metrics
        high_quality_metrics = QualityMetrics(
            test_coverage=95.0,
            security_score={'critical': 0, 'high': 0, 'medium': 1, 'low': 2},
            performance_regression=1.0,
            code_quality_score=90.0
        )
        
        base_score = self.engine._calculate_base_quality_score(high_quality_metrics)
        assert 85 <= base_score <= 100  # Should be high score
        
        # Test low-quality metrics
        low_quality_metrics = QualityMetrics(
            test_coverage=40.0,
            security_score={'critical': 2, 'high': 3, 'medium': 5, 'low': 10},
            performance_regression=15.0,
            code_quality_score=30.0
        )
        
        base_score = self.engine._calculate_base_quality_score(low_quality_metrics)
        assert 0 <= base_score <= 30  # Should be low score
        
        # Test medium-quality metrics
        medium_quality_metrics = QualityMetrics(
            test_coverage=75.0,
            security_score={'critical': 0, 'high': 1, 'medium': 3, 'low': 5},
            performance_regression=3.0,
            code_quality_score=70.0
        )
        
        base_score = self.engine._calculate_base_quality_score(medium_quality_metrics)
        assert 60 <= base_score <= 85  # Should be medium score
    
    def test_risk_multiplier_calculation(self):
        """Test risk multiplier calculation with various risk factors."""
        # Test no risk factors
        no_risk = RiskFactors()
        risk_multiplier = self.engine._calculate_risk_multiplier(no_risk)
        assert risk_multiplier == 0.0
        
        # Test single risk factor
        single_risk = RiskFactors(large_change=True)
        risk_multiplier = self.engine._calculate_risk_multiplier(single_risk)
        assert 0.05 <= risk_multiplier <= 0.15
        
        # Test multiple risk factors
        multiple_risks = RiskFactors(
            large_change=True,
            security_files=True,
            critical_paths=True,
            no_tests=True
        )
        risk_multiplier = self.engine._calculate_risk_multiplier(multiple_risks)
        assert 0.3 <= risk_multiplier <= 0.5  # Should be capped at max
        
        # Test previous failures with different counts
        failure_risk = RiskFactors(previous_failures=3)
        risk_multiplier = self.engine._calculate_risk_multiplier(failure_risk)
        assert 0.1 <= risk_multiplier <= 0.3
    
    def test_context_weight_application(self):
        """Test context weight application for different component types."""
        # Test critical algorithms (high weight)
        critical_weight = self.engine._get_context_weight('critical_algorithms')
        assert critical_weight == 1.2
        
        # Test public APIs (elevated weight)
        api_weight = self.engine._get_context_weight('public_apis')
        assert api_weight == 1.1
        
        # Test business logic (standard weight)
        business_weight = self.engine._get_context_weight('business_logic')
        assert business_weight == 1.0
        
        # Test UI components (reduced weight)
        ui_weight = self.engine._get_context_weight('ui_components')
        assert ui_weight == 0.8
        
        # Test unknown context (default weight)
        unknown_weight = self.engine._get_context_weight('unknown_component')
        assert unknown_weight == 1.0
    
    def test_risk_adjusted_score_calculation(self):
        """Test complete risk-adjusted score calculation."""
        # Test high-quality, low-risk scenario
        high_quality_metrics = QualityMetrics(
            test_coverage=90.0,
            security_score={'critical': 0, 'high': 0, 'medium': 1, 'low': 2},
            performance_regression=1.0,
            code_quality_score=85.0
        )
        
        low_risk_factors = RiskFactors(large_change=False, security_files=False)
        
        score = self.engine.calculate_risk_adjusted_score(
            high_quality_metrics, low_risk_factors, 'business_logic'
        )
        
        assert 80 <= score <= 100  # Should be high score
        
        # Test same metrics but high risk
        high_risk_factors = RiskFactors(
            large_change=True,
            security_files=True,
            critical_paths=True,
            no_tests=True
        )
        
        risky_score = self.engine.calculate_risk_adjusted_score(
            high_quality_metrics, high_risk_factors, 'business_logic'
        )
        
        assert risky_score < score  # Risk should reduce score
        assert 50 <= risky_score <= score - 10  # Significant reduction
        
        # Test context impact
        critical_score = self.engine.calculate_risk_adjusted_score(
            high_quality_metrics, low_risk_factors, 'critical_algorithms'
        )
        
        assert critical_score >= score  # Critical context should increase score
    
    def test_quality_decisions(self):
        """Test quality decision making based on scores."""
        # Test PASS decision
        pass_decision = self.engine.make_quality_decision(85.0, [], "low")
        assert pass_decision == QualityDecision.PASS
        
        # Test CONCERNS decision
        concerns_decision = self.engine.make_quality_decision(70.0, ["Minor linting issues"], "low")
        assert concerns_decision == QualityDecision.CONCERNS
        
        # Test FAIL decision (low score)
        fail_decision = self.engine.make_quality_decision(45.0, [], "low")
        assert fail_decision == QualityDecision.FAIL
        
        # Test FAIL decision (critical security)
        security_fail = self.engine.make_quality_decision(85.0, ["Critical security vulnerability detected"], "low")
        assert security_fail == QualityDecision.FAIL
        
        # Test BLOCKED decision (critical risk)
        blocked_decision = self.engine.make_quality_decision(75.0, [], "critical")
        assert blocked_decision == QualityDecision.BLOCKED
        
        # Test BLOCKED decision (multiple failures)
        multiple_issues = ["Issue 1", "Issue 2", "Issue 3", "Issue 4"]
        blocked_multiple = self.engine.make_quality_decision(80.0, multiple_issues, "low")
        assert blocked_multiple == QualityDecision.BLOCKED
    
    def test_decision_explanation_generation(self):
        """Test decision explanation generation."""
        # Create sample quality result
        metrics = QualityMetrics(
            test_coverage=85.0,
            security_score={'critical': 0, 'high': 1, 'medium': 2, 'low': 3},
            performance_regression=2.5,
            code_quality_score=75.0
        )
        
        risk_factors = RiskFactors(large_change=True, security_files=False)
        context = 'business_logic'
        
        # Calculate scores
        base_score = self.engine._calculate_base_quality_score(metrics)
        risk_multiplier = self.engine._calculate_risk_multiplier(risk_factors)
        context_weight = self.engine._get_context_weight(context)
        adjusted_score = base_score * (1 - risk_multiplier) * context_weight
        
        # Create quality result
        result = QualityResult(
            risk_adjusted_score=adjusted_score,
            base_quality_score=base_score,
            risk_multiplier=risk_multiplier,
            context_weight=context_weight,
            decision=self.engine.make_quality_decision(adjusted_score, [], "low"),
            dimension_scores={'test_coverage': 85.0, 'security_validation': 85.0},
            risk_factors={'large_change': True, 'security_files': False},
            decision_explanation="",
            recommendations=["Improve test coverage", "Address security vulnerabilities"],
            evidence=["Coverage report shows 85%", "Security scan found 1 high issue"],
            performance_metrics={}
        )
        
        # Generate explanation
        explanation = self.engine.generate_decision_explanation(result)
        
        # Check explanation contains key elements
        assert "Quality Decision:" in explanation
        assert "Risk-Adjusted Score:" in explanation
        assert "Base Quality Score:" in explanation
        assert "Risk Adjustment:" in explanation
        assert "Context Weight:" in explanation
        assert "Dimension Scores:" in explanation
        assert "Decision Rationale:" in explanation
        
        # Check specific values are present
        assert f"{adjusted_score:.1f}" in explanation
        assert f"{base_score:.1f}" in explanation
        assert f"{risk_multiplier:.1%}" in explanation
        
        # Check recommendations and evidence
        assert "Improve test coverage" in explanation
        assert "Coverage report shows 85%" in explanation
    
    def test_performance_tracking(self):
        """Test performance tracking during calculations."""
        metrics = QualityMetrics(
            test_coverage=80.0,
            security_score={'critical': 0, 'high': 0, 'medium': 1, 'low': 2},
            performance_regression=3.0,
            code_quality_score=70.0
        )
        
        risk_factors = RiskFactors(large_change=True)
        
        # Perform multiple calculations to build performance data
        for _ in range(5):
            self.engine.calculate_risk_adjusted_score(metrics, risk_factors, 'business_logic')
        
        # Check performance metrics
        perf_metrics = self.engine.get_performance_metrics()
        assert 'average_calculation_time_ms' in perf_metrics
        assert 'max_calculation_time_ms' in perf_metrics
        assert 'min_calculation_time_ms' in perf_metrics
        assert 'total_calculations' in perf_metrics
        
        assert perf_metrics['total_calculations'] == 5
        assert perf_metrics['average_calculation_time_ms'] > 0
        assert perf_metrics['max_calculation_time_ms'] >= perf_metrics['min_calculation_time_ms']
        
        # Performance should be within acceptable limits
        assert perf_metrics['average_calculation_time_ms'] < 100  # Under 100ms average
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test with empty metrics
        empty_metrics = QualityMetrics()
        empty_risks = RiskFactors()
        
        # Should handle gracefully without crashing
        score = self.engine.calculate_risk_adjusted_score(empty_metrics, empty_risks, 'unknown')
        assert 0 <= score <= 100
        
        # Test with extreme values
        extreme_metrics = QualityMetrics(
            test_coverage=150.0,  # Over 100%
            security_score={'critical': -1, 'high': 1000, 'medium': -5, 'low': 0},
            performance_regression=-50.0,  # Negative regression
            code_quality_score=200.0  # Over 100%
        )
        
        extreme_score = self.engine.calculate_risk_adjusted_score(extreme_metrics, empty_risks, 'business_logic')
        assert 0 <= extreme_score <= 100  # Should be clamped to valid range
        
        # Test with None values
        try:
            self.engine.calculate_risk_adjusted_score(None, None, None)
        except:
            pass  # Should handle gracefully or raise appropriate exception
    
    def test_configuration_validation(self):
        """Test configuration validation and fallbacks."""
        # Test with missing config file
        missing_config_engine = QualityDecisionEngine(config_path="nonexistent.yaml")
        
        # Should use default configuration
        assert missing_config_engine.config is not None
        assert 'quality_dimensions' in missing_config_engine.config
        
        # Test dimension weight validation
        total_weight = sum(dim.get('weight', 0) for dim in self.engine.dimensions.values())
        assert 95 <= total_weight <= 105  # Should be approximately 100
    
    def test_legacy_compatibility(self):
        """Test backward compatibility with legacy quality calculation."""
        metrics = QualityMetrics(
            test_coverage=60.0,  # Would be 1 fail in legacy
            security_score={'critical': 1, 'high': 0, 'medium': 0, 'low': 0},  # 1 fail
            performance_regression=15.0,  # Would be 1 fail in legacy
            code_quality_score=40.0  # Would be 1 fail in legacy
        )
        
        # Test legacy fallback calculation
        legacy_score = self.engine._legacy_quality_calculation(metrics)
        
        # Should follow legacy formula: 100 - (20 × 3 FAILs) = 40
        # (Only 3 FAILs because one condition might not trigger)
        assert 25 <= legacy_score <= 45
    
    def test_full_workflow_integration(self):
        """Test complete workflow from metrics to decision explanation."""
        # Realistic scenario: medium-quality change with some risk
        metrics = QualityMetrics(
            test_coverage=82.0,
            security_score={'critical': 0, 'high': 1, 'medium': 2, 'low': 5},
            performance_regression=4.0,
            code_quality_score=78.0
        )
        
        risk_factors = RiskFactors(
            large_change=True,
            security_files=False,
            critical_paths=False,
            external_dependencies=True,
            previous_failures=1,
            no_tests=False,
            low_coverage_area=False
        )
        
        context = 'public_apis'
        
        # Calculate risk-adjusted score
        adjusted_score = self.engine.calculate_risk_adjusted_score(metrics, risk_factors, context)
        
        # Make decision
        decision = self.engine.make_quality_decision(adjusted_score, ["High security vulnerability in dependency"], "medium")
        
        # Create complete result
        result = QualityResult(
            risk_adjusted_score=adjusted_score,
            base_quality_score=self.engine._calculate_base_quality_score(metrics),
            risk_multiplier=self.engine._calculate_risk_multiplier(risk_factors),
            context_weight=self.engine._get_context_weight(context),
            decision=decision,
            dimension_scores={
                'test_coverage': metrics.test_coverage,
                'security_validation': 80.0,  # Calculated security score
                'performance_impact': 80.0,   # Calculated performance score
                'code_quality': metrics.code_quality_score
            },
            risk_factors={'large_change': True, 'external_dependencies': True, 'previous_failures': 1},
            decision_explanation="",
            recommendations=["Address high security vulnerability", "Improve test coverage to 90%+"],
            evidence=["Security scan report", "Test coverage report"],
            performance_metrics=self.engine.get_performance_metrics()
        )
        
        # Generate explanation
        explanation = self.engine.generate_decision_explanation(result)
        
        # Verify complete workflow
        assert isinstance(adjusted_score, float)
        assert 0 <= adjusted_score <= 100
        assert isinstance(decision, QualityDecision)
        assert len(explanation) > 100  # Should be detailed
        assert "public_apis" not in explanation.lower()  # Context should be translated
        assert "Quality Decision:" in explanation
        assert str(decision.value) in explanation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])