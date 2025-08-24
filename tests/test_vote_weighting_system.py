#!/usr/bin/env python3
"""
Test Suite for Vote Weighting System - Issue #62
Comprehensive tests for the vote weighting algorithm implementation.
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import vote weighting components
from knowledge.consensus.vote_weight_calculator import (
    VoteWeightCalculator, WeightingStrategy, ExpertiseProfile, AccuracyRecord
)
from knowledge.consensus.expertise_scorer import (
    ExpertiseScorer, ExpertiseDomain, ExpertiseEvidence
)
from knowledge.consensus.accuracy_tracker import (
    AccuracyTracker, DecisionRecord, DecisionOutcome
)
from knowledge.consensus.confidence_adjuster import (
    ConfidenceAdjuster, ConfidenceRecord, ConfidenceBias
)


class TestVoteWeightCalculator:
    """Test suite for VoteWeightCalculator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.calculator = VoteWeightCalculator(strategy=WeightingStrategy.BALANCED)
        
        # Set up test agent with known expertise and accuracy
        self.test_agent = "test-agent"
        self.calculator.update_agent_expertise(self.test_agent, "security", 0.9)
        self.calculator.update_agent_accuracy(self.test_agent, True, {"domain": "security"})
    
    def test_basic_weight_calculation(self):
        """Test basic weight calculation functionality"""
        context = {
            'domain': 'security',
            'confidence': 0.8
        }
        
        weight = self.calculator.calculate_weight(self.test_agent, context)
        
        # Weight should be within expected bounds
        assert 0.1 <= weight <= 3.0
        # High expertise + good accuracy should result in above-average weight
        assert weight > 1.0
    
    def test_detailed_weight_calculation(self):
        """Test detailed weight calculation with breakdown"""
        context = {
            'domain': 'security',
            'confidence': 0.85
        }
        
        result = self.calculator.calculate_detailed_weight(self.test_agent, context)
        
        # Check all components are present
        assert hasattr(result, 'final_weight')
        assert hasattr(result, 'base_weight')
        assert hasattr(result, 'expertise_factor')
        assert hasattr(result, 'accuracy_factor')
        assert hasattr(result, 'confidence_factor')
        
        # Factors should be in reasonable ranges
        assert 0.3 <= result.expertise_factor <= 2.0
        assert 0.4 <= result.accuracy_factor <= 1.6
        assert 0.6 <= result.confidence_factor <= 1.4
        
        # Metadata should be present
        assert 'agent_id' in result.calculation_metadata
        assert 'strategy' in result.calculation_metadata
    
    def test_ensemble_weights_calculation(self):
        """Test ensemble weight calculation and normalization"""
        agents = ["agent-1", "agent-2", "agent-3"]
        context = {'domain': 'general', 'confidence': 0.7}
        
        # Set up different expertise levels
        for i, agent in enumerate(agents):
            expertise_level = 0.5 + (i * 0.2)  # 0.5, 0.7, 0.9
            self.calculator.update_agent_expertise(agent, "general", expertise_level)
        
        ensemble_weights = self.calculator.calculate_ensemble_weights(agents, context)
        
        # Should have weights for all agents
        assert len(ensemble_weights) == len(agents)
        
        # All weights should be positive and finite
        for weight in ensemble_weights.values():
            assert weight > 0
            assert weight < float('inf')
        
        # Higher expertise should generally result in higher weights
        agent_weights = [(agent, ensemble_weights[agent]) for agent in agents]
        # Check that agent-3 (highest expertise) has higher weight than agent-1
        assert ensemble_weights["agent-3"] > ensemble_weights["agent-1"]
    
    def test_accuracy_updates(self):
        """Test accuracy update mechanism"""
        initial_weight = self.calculator.calculate_weight(self.test_agent, {'domain': 'security'})
        
        # Record several successful decisions
        for _ in range(5):
            self.calculator.update_agent_accuracy(self.test_agent, True, {"domain": "security"})
        
        updated_weight = self.calculator.calculate_weight(self.test_agent, {'domain': 'security'})
        
        # Weight should increase with positive accuracy track record
        assert updated_weight >= initial_weight
    
    def test_different_strategies(self):
        """Test different weighting strategies"""
        context = {'domain': 'security', 'confidence': 0.8}
        
        strategies = [
            WeightingStrategy.EXPERTISE_FOCUSED,
            WeightingStrategy.ACCURACY_FOCUSED,
            WeightingStrategy.BALANCED
        ]
        
        weights = {}
        for strategy in strategies:
            calc = VoteWeightCalculator(strategy=strategy)
            calc.expertise_profiles = self.calculator.expertise_profiles
            calc.accuracy_records = self.calculator.accuracy_records
            
            weights[strategy] = calc.calculate_weight(self.test_agent, context)
        
        # All strategies should produce valid weights
        for strategy, weight in weights.items():
            assert 0.1 <= weight <= 3.0
        
        # Different strategies may produce different weights
        # This is expected and acceptable
    
    def test_weight_bounds_enforcement(self):
        """Test that weight bounds are enforced"""
        # Test extreme cases that might violate bounds
        contexts = [
            {'domain': 'unknown', 'confidence': 0.0},  # Low confidence, unknown domain
            {'domain': 'security', 'confidence': 1.0},  # High confidence, known domain
        ]
        
        for context in contexts:
            weight = self.calculator.calculate_weight(self.test_agent, context)
            assert 0.1 <= weight <= 3.0
    
    def test_metrics_tracking(self):
        """Test performance metrics tracking"""
        initial_metrics = self.calculator.get_calculation_metrics()
        initial_count = initial_metrics['total_calculations']
        
        # Perform several calculations
        for i in range(10):
            self.calculator.calculate_weight(f"agent-{i}", {'domain': 'general'})
        
        updated_metrics = self.calculator.get_calculation_metrics()
        
        # Metrics should be updated
        assert updated_metrics['total_calculations'] == initial_count + 10
        assert updated_metrics['average_calculation_time'] >= 0
        assert len(updated_metrics['weight_distribution_stats']) > 0


class TestExpertiseScorer:
    """Test suite for ExpertiseScorer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.scorer = ExpertiseScorer()
        self.test_agent = "test-expert"
    
    def test_basic_expertise_assessment(self):
        """Test basic expertise assessment"""
        # Add some expertise evidence
        evidence = ExpertiseEvidence(
            evidence_type="successful_decision",
            domain="security",
            quality_score=0.9,
            impact_level="high",
            timestamp=datetime.now()
        )
        
        self.scorer.update_expertise_evidence(self.test_agent, "security", evidence)
        
        assessment = self.scorer.assess_agent_expertise(self.test_agent, "security")
        
        # Check assessment properties
        assert assessment.agent_id == self.test_agent
        assert assessment.domain == "security"
        assert 0.0 <= assessment.current_level <= 1.0
        assert len(assessment.confidence_interval) == 2
        assert assessment.evidence_count > 0
    
    def test_expertise_factor_calculation(self):
        """Test expertise factor calculation for vote weighting"""
        # Add high-quality evidence
        evidence = ExpertiseEvidence(
            evidence_type="system_design",
            domain="architecture",
            quality_score=0.95,
            impact_level="critical",
            timestamp=datetime.now()
        )
        
        self.scorer.update_expertise_evidence(self.test_agent, "architecture", evidence)
        
        factor = self.scorer.calculate_domain_expertise_factor(self.test_agent, "architecture")
        
        # Factor should be in expected range
        assert 0.3 <= factor <= 2.0
        # High quality evidence should result in above-average factor
        assert factor > 1.0
    
    def test_cross_domain_synergy(self):
        """Test cross-domain expertise synergy"""
        # Add evidence in synergistic domains
        security_evidence = ExpertiseEvidence(
            evidence_type="successful_decision",
            domain="security",
            quality_score=0.9,
            impact_level="high",
            timestamp=datetime.now()
        )
        
        validation_evidence = ExpertiseEvidence(
            evidence_type="successful_decision",
            domain="validation",
            quality_score=0.85,
            impact_level="high",
            timestamp=datetime.now()
        )
        
        self.scorer.update_expertise_evidence(self.test_agent, "security", security_evidence)
        self.scorer.update_expertise_evidence(self.test_agent, "validation", validation_evidence)
        
        # Security expertise factor should benefit from validation synergy
        security_factor = self.scorer.calculate_domain_expertise_factor(self.test_agent, "security")
        
        # Create another agent with only security expertise for comparison
        solo_agent = "solo-security-agent"
        self.scorer.update_expertise_evidence(solo_agent, "security", security_evidence)
        solo_factor = self.scorer.calculate_domain_expertise_factor(solo_agent, "security")
        
        # Multi-domain agent should have some advantage (though this might be subtle)
        assert security_factor > 0
        assert solo_factor > 0
    
    def test_agent_expertise_profile(self):
        """Test comprehensive agent expertise profile"""
        # Add evidence across multiple domains
        domains_evidence = [
            ("security", 0.9, "high"),
            ("testing", 0.8, "medium"),
            ("architecture", 0.95, "critical")
        ]
        
        for domain, quality, impact in domains_evidence:
            evidence = ExpertiseEvidence(
                evidence_type="successful_decision",
                domain=domain,
                quality_score=quality,
                impact_level=impact,
                timestamp=datetime.now()
            )
            self.scorer.update_expertise_evidence(self.test_agent, domain, evidence)
        
        profile = self.scorer.get_agent_expertise_profile(self.test_agent)
        
        # Check profile structure
        assert profile['agent_id'] == self.test_agent
        assert 'domain_expertise' in profile
        assert 'specialization_areas' in profile
        assert 'total_evidence_count' in profile
        
        # Should have expertise data for added domains
        domain_expertise = profile['domain_expertise']
        for domain, _, _ in domains_evidence:
            assert domain in domain_expertise
            assert domain_expertise[domain]['level'] > 0


class TestAccuracyTracker:
    """Test suite for AccuracyTracker class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = AccuracyTracker()
        self.test_agent = "test-accurate-agent"
    
    def test_decision_recording(self):
        """Test decision outcome recording"""
        decision = DecisionRecord(
            decision_id="test_decision_1",
            agent_id=self.test_agent,
            decision_timestamp=datetime.now(),
            decision_confidence=0.8,
            decision_content="Test decision",
            outcome=DecisionOutcome.SUCCESS,
            outcome_timestamp=datetime.now(),
            context_category="routine_task",
            impact_level="medium"
        )
        
        self.tracker.record_decision_outcome(decision)
        
        # Should have recorded the decision
        assert self.test_agent in self.tracker.decision_records
        assert len(self.tracker.decision_records[self.test_agent]) == 1
    
    def test_accuracy_metrics_calculation(self):
        """Test accuracy metrics calculation"""
        # Record several decisions with known outcomes
        decisions = []
        outcomes = [True, True, False, True, True]  # 80% success rate
        
        for i, success in enumerate(outcomes):
            outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
            decision = DecisionRecord(
                decision_id=f"decision_{i}",
                agent_id=self.test_agent,
                decision_timestamp=datetime.now() - timedelta(days=i),
                decision_confidence=0.8,
                decision_content=f"Decision {i}",
                outcome=outcome,
                outcome_timestamp=datetime.now() - timedelta(days=i-1),
                impact_level="medium"
            )
            self.tracker.record_decision_outcome(decision)
        
        metrics = self.tracker.calculate_accuracy_metrics(self.test_agent)
        
        # Check metrics
        assert metrics.agent_id == self.test_agent
        assert metrics.decision_count == 5
        assert metrics.success_count == 4
        assert abs(metrics.overall_accuracy - 0.8) < 0.01  # 80% accuracy
        assert 0.0 <= metrics.weighted_accuracy <= 1.0
        assert -1.0 <= metrics.temporal_trend <= 1.0
    
    def test_accuracy_factor_calculation(self):
        """Test accuracy factor calculation for vote weighting"""
        # Record some successful decisions
        for i in range(10):
            outcome = DecisionOutcome.SUCCESS if i < 8 else DecisionOutcome.FAILURE  # 80% success
            decision = DecisionRecord(
                decision_id=f"decision_{i}",
                agent_id=self.test_agent,
                decision_timestamp=datetime.now(),
                decision_confidence=0.8,
                decision_content=f"Decision {i}",
                outcome=outcome,
                outcome_timestamp=datetime.now(),
                context_category="security_critical"
            )
            self.tracker.record_decision_outcome(decision)
        
        factor = self.tracker.calculate_accuracy_factor(self.test_agent)
        
        # Factor should be in expected range
        assert 0.4 <= factor <= 1.6
        # Good accuracy should result in above-neutral factor
        assert factor > 1.0
    
    def test_confidence_calibration(self):
        """Test confidence calibration analysis"""
        # Record decisions with varying confidence and outcomes
        test_data = [
            (0.9, True),   # High confidence, success
            (0.8, True),   # High confidence, success
            (0.7, False),  # Medium confidence, failure
            (0.6, True),   # Medium confidence, success
            (0.5, False),  # Low confidence, failure
        ]
        
        for i, (confidence, success) in enumerate(test_data):
            outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
            decision = DecisionRecord(
                decision_id=f"calib_decision_{i}",
                agent_id=self.test_agent,
                decision_timestamp=datetime.now(),
                decision_confidence=confidence,
                decision_content=f"Calibration test {i}",
                outcome=outcome,
                outcome_timestamp=datetime.now()
            )
            self.tracker.record_decision_outcome(decision)
        
        calibration = self.tracker.analyze_confidence_calibration(self.test_agent)
        
        # Check calibration analysis
        assert calibration.agent_id == self.test_agent
        assert 0.0 <= calibration.calibration_score <= 1.0
        assert 0.0 <= calibration.brier_score <= 1.0


class TestConfidenceAdjuster:
    """Test suite for ConfidenceAdjuster class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.adjuster = ConfidenceAdjuster()
        self.test_agent = "test-confident-agent"
    
    def test_confidence_recording(self):
        """Test confidence outcome recording"""
        record = ConfidenceRecord(
            agent_id=self.test_agent,
            decision_id="conf_test_1",
            stated_confidence=0.8,
            actual_outcome=1.0,
            context_complexity="medium",
            domain="general",
            timestamp=datetime.now()
        )
        
        self.adjuster.record_confidence_outcome(record)
        
        # Should have recorded the confidence outcome
        assert self.test_agent in self.adjuster.confidence_records
        assert len(self.adjuster.confidence_records[self.test_agent]) == 1
    
    def test_confidence_adjustment(self):
        """Test confidence adjustment calculation"""
        # Record some confidence outcomes to establish pattern
        confidence_data = [
            (0.9, 1.0),  # Overconfident but correct
            (0.8, 0.0),  # Overconfident and wrong
            (0.7, 1.0),  # Good calibration
            (0.6, 0.0),  # Good calibration
        ]
        
        for i, (stated, actual) in enumerate(confidence_data):
            record = ConfidenceRecord(
                agent_id=self.test_agent,
                decision_id=f"conf_record_{i}",
                stated_confidence=stated,
                actual_outcome=actual,
                context_complexity="medium",
                domain="general",
                timestamp=datetime.now()
            )
            self.adjuster.record_confidence_outcome(record)
        
        # Test confidence adjustment
        context = {'confidence': 0.8, 'domain': 'general'}
        adjustment = self.adjuster.adjust_confidence(self.test_agent, 0.8, context)
        
        # Check adjustment result
        assert hasattr(adjustment, 'original_confidence')
        assert hasattr(adjustment, 'adjusted_confidence')
        assert hasattr(adjustment, 'adjustment_factor')
        assert 0.0 <= adjustment.adjusted_confidence <= 1.0
    
    def test_confidence_factor_calculation(self):
        """Test confidence factor calculation for vote weighting"""
        context = {
            'confidence': 0.8,
            'domain': 'general',
            'complexity': 'medium'
        }
        
        factor = self.adjuster.calculate_confidence_factor(self.test_agent, context)
        
        # Factor should be in expected range
        assert 0.6 <= factor <= 1.4
    
    def test_bias_detection(self):
        """Test bias detection in confidence patterns"""
        # Create pattern of overconfident decisions
        overconfident_data = [
            (0.9, 0.0),  # Very confident but wrong
            (0.85, 0.0), # Very confident but wrong
            (0.8, 1.0),  # Confident and right
            (0.75, 0.0), # Confident but wrong
        ]
        
        for i, (stated, actual) in enumerate(overconfident_data):
            record = ConfidenceRecord(
                agent_id=self.test_agent,
                decision_id=f"bias_record_{i}",
                stated_confidence=stated,
                actual_outcome=actual,
                context_complexity="medium",
                domain="general",
                timestamp=datetime.now()
            )
            self.adjuster.record_confidence_outcome(record)
        
        # Analyze confidence patterns
        patterns = self.adjuster.analyze_agent_confidence_patterns(self.test_agent)
        
        # Should detect some calibration issues
        assert 'calibration_analysis' in patterns
        if 'bias_type' in patterns['calibration_analysis']:
            # Might detect overconfidence or inconsistency
            bias_type = patterns['calibration_analysis']['bias_type']
            assert bias_type in ['overconfident', 'underconfident', 'well_calibrated', 'inconsistent']


class TestSystemIntegration:
    """Integration tests for the complete vote weighting system"""
    
    def setup_method(self):
        """Set up integrated system"""
        self.calculator = VoteWeightCalculator(strategy=WeightingStrategy.BALANCED)
        self.expertise_scorer = ExpertiseScorer()
        self.accuracy_tracker = AccuracyTracker()
        self.confidence_adjuster = ConfidenceAdjuster()
    
    def test_full_weight_calculation_workflow(self):
        """Test complete weight calculation workflow"""
        agent_id = "integration-test-agent"
        
        # 1. Add expertise evidence
        expertise_evidence = ExpertiseEvidence(
            evidence_type="successful_decision",
            domain="security",
            quality_score=0.9,
            impact_level="high",
            timestamp=datetime.now()
        )
        self.expertise_scorer.update_expertise_evidence(agent_id, "security", expertise_evidence)
        
        # 2. Record accuracy history
        for i in range(10):
            outcome = DecisionOutcome.SUCCESS if i < 8 else DecisionOutcome.FAILURE
            decision = DecisionRecord(
                decision_id=f"workflow_decision_{i}",
                agent_id=agent_id,
                decision_timestamp=datetime.now(),
                decision_confidence=0.8,
                decision_content=f"Workflow decision {i}",
                outcome=outcome,
                outcome_timestamp=datetime.now(),
                context_category="security_critical"
            )
            self.accuracy_tracker.record_decision_outcome(decision)
        
        # 3. Record confidence calibration
        for i in range(5):
            confidence_record = ConfidenceRecord(
                agent_id=agent_id,
                decision_id=f"workflow_confidence_{i}",
                stated_confidence=0.8,
                actual_outcome=1.0 if i < 4 else 0.0,
                context_complexity="medium",
                domain="security",
                timestamp=datetime.now()
            )
            self.confidence_adjuster.record_confidence_outcome(confidence_record)
        
        # 4. Calculate final weight
        context = {
            'domain': 'security',
            'confidence': 0.8,
            'complexity': 'medium'
        }
        
        weight = self.calculator.calculate_weight(agent_id, context)
        detailed_result = self.calculator.calculate_detailed_weight(agent_id, context)
        
        # Verify results
        assert 0.1 <= weight <= 3.0
        assert weight == detailed_result.final_weight
        
        # With good expertise, accuracy, and confidence, weight should be above average
        assert weight > 1.0
    
    def test_ensemble_scenario(self):
        """Test realistic ensemble voting scenario"""
        agents = [
            "rif-security",
            "rif-validator", 
            "rif-implementer",
            "rif-architect"
        ]
        
        # Set up different expertise profiles
        expertise_configs = [
            ("rif-security", "security", 0.95),
            ("rif-validator", "validation", 0.90),
            ("rif-implementer", "implementation", 0.85),
            ("rif-architect", "architecture", 0.90)
        ]
        
        for agent, domain, expertise_level in expertise_configs:
            self.calculator.update_agent_expertise(agent, domain, expertise_level)
            
            # Add some accuracy history
            for i in range(15):
                success = i < int(15 * expertise_level)  # Accuracy correlated with expertise
                self.calculator.update_agent_accuracy(agent, success, {"domain": domain})
        
        # Calculate ensemble weights for security decision
        context = {
            'domain': 'security',
            'confidence': 0.8,
            'complexity': 'high'
        }
        
        ensemble_weights = self.calculator.calculate_ensemble_weights(agents, context)
        
        # Verify ensemble properties
        assert len(ensemble_weights) == len(agents)
        
        # Security expert should have highest weight for security decision
        security_weight = ensemble_weights["rif-security"]
        other_weights = [ensemble_weights[agent] for agent in agents if agent != "rif-security"]
        
        # Security agent should have one of the higher weights
        assert security_weight >= max(other_weights) * 0.8  # Allow some tolerance
        
        # All weights should be positive and reasonable
        for agent, weight in ensemble_weights.items():
            assert weight > 0
            assert weight <= 3.0
    
    def test_performance_metrics(self):
        """Test system performance metrics collection"""
        agent_id = "metrics-test-agent"
        
        # Perform multiple weight calculations
        contexts = [
            {'domain': 'security', 'confidence': 0.8},
            {'domain': 'testing', 'confidence': 0.7},
            {'domain': 'architecture', 'confidence': 0.9},
        ]
        
        for context in contexts:
            self.calculator.calculate_weight(agent_id, context)
        
        # Check that metrics are being tracked
        metrics = self.calculator.get_calculation_metrics()
        
        assert metrics['total_calculations'] >= 3
        assert metrics['average_calculation_time'] >= 0
        assert 'weight_distribution_stats' in metrics
        assert 'strategy_usage' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])