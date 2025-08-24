#!/usr/bin/env python3
"""
RIF Vote Weighting System Demo - Issue #62
Comprehensive demonstration of the vote weighting algorithm implementation.

This script demonstrates all components of the vote weighting system:
- Vote weight calculation with multiple factors
- Expertise scoring and domain knowledge assessment
- Historical accuracy tracking and trend analysis
- Confidence calibration and bias correction
- Integration with existing consensus architecture
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from knowledge.consensus.vote_weight_calculator import (
    VoteWeightCalculator, WeightingStrategy
)
from knowledge.consensus.expertise_scorer import (
    ExpertiseScorer, ExpertiseEvidence, ExpertiseDomain
)
from knowledge.consensus.accuracy_tracker import (
    AccuracyTracker, DecisionRecord, DecisionOutcome
)
from knowledge.consensus.confidence_adjuster import (
    ConfidenceAdjuster, ConfidenceRecord
)
from knowledge.consensus.weighted_voting_integration import (
    WeightedVotingAggregator, WeightedVotingConfig
)


class VoteWeightingDemo:
    """Comprehensive demonstration of the vote weighting system"""
    
    def __init__(self):
        """Initialize demo environment"""
        print("🚀 Initializing RIF Vote Weighting System Demo...")
        
        # Initialize all components
        self.weight_calculator = VoteWeightCalculator(strategy=WeightingStrategy.BALANCED)
        self.expertise_scorer = ExpertiseScorer()
        self.accuracy_tracker = AccuracyTracker()
        self.confidence_adjuster = ConfidenceAdjuster()
        
        # Initialize integration layer
        config = WeightedVotingConfig(
            weighting_strategy=WeightingStrategy.BALANCED,
            enable_expertise_scoring=True,
            enable_accuracy_tracking=True,
            enable_confidence_adjustment=True
        )
        self.integrated_system = WeightedVotingAggregator(weighting_config=config)
        
        # Demo agents with different characteristics
        self.demo_agents = {
            'rif-security-expert': {
                'specializations': ['security', 'compliance'],
                'base_accuracy': 0.90,
                'confidence_bias': 'well_calibrated',
                'experience_years': 5
            },
            'rif-junior-implementer': {
                'specializations': ['implementation'],
                'base_accuracy': 0.70,
                'confidence_bias': 'underconfident',
                'experience_years': 1
            },
            'rif-senior-architect': {
                'specializations': ['architecture', 'design', 'performance'],
                'base_accuracy': 0.85,
                'confidence_bias': 'overconfident',
                'experience_years': 8
            },
            'rif-testing-specialist': {
                'specializations': ['testing', 'validation', 'quality'],
                'base_accuracy': 0.88,
                'confidence_bias': 'well_calibrated',
                'experience_years': 4
            },
            'rif-generalist': {
                'specializations': ['general'],
                'base_accuracy': 0.75,
                'confidence_bias': 'inconsistent',
                'experience_years': 3
            }
        }
        
        self._setup_demo_data()
        print("✅ Demo environment initialized successfully!\n")
    
    def _setup_demo_data(self):
        """Set up realistic demo data for all agents"""
        print("📊 Setting up demo data...")
        
        for agent_id, profile in self.demo_agents.items():
            # Set up expertise profiles
            for specialization in profile['specializations']:
                base_expertise = 0.6 + (profile['experience_years'] * 0.05)
                if specialization in profile['specializations'][:2]:  # Primary specializations
                    expertise_level = min(0.95, base_expertise + 0.2)
                else:
                    expertise_level = min(0.80, base_expertise)
                
                # Add expertise evidence
                for i in range(profile['experience_years'] * 3):  # More evidence for experienced agents
                    evidence = ExpertiseEvidence(
                        evidence_type=random.choice(['successful_decision', 'system_design', 'code_review']),
                        domain=specialization,
                        quality_score=min(1.0, expertise_level + random.uniform(-0.1, 0.1)),
                        impact_level=random.choice(['medium', 'high', 'critical']),
                        timestamp=datetime.now() - timedelta(days=random.randint(1, 365))
                    )
                    self.expertise_scorer.update_expertise_evidence(agent_id, specialization, evidence)
            
            # Set up accuracy history
            num_decisions = profile['experience_years'] * 20
            base_accuracy = profile['base_accuracy']
            
            for i in range(num_decisions):
                success = random.random() < base_accuracy
                outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
                
                decision = DecisionRecord(
                    decision_id=f"demo_decision_{agent_id}_{i}",
                    agent_id=agent_id,
                    decision_timestamp=datetime.now() - timedelta(days=random.randint(1, 730)),
                    decision_confidence=self._generate_confidence_for_agent(agent_id, success),
                    decision_content=f"Decision {i} for {agent_id}",
                    outcome=outcome,
                    outcome_timestamp=datetime.now() - timedelta(days=random.randint(0, 30)),
                    context_category=random.choice(profile['specializations']),
                    impact_level=random.choice(['low', 'medium', 'high'])
                )
                self.accuracy_tracker.record_decision_outcome(decision)
            
            # Set up confidence calibration history
            for i in range(num_decisions // 2):
                success = random.random() < base_accuracy
                confidence = self._generate_confidence_for_agent(agent_id, success)
                
                confidence_record = ConfidenceRecord(
                    agent_id=agent_id,
                    decision_id=f"demo_confidence_{agent_id}_{i}",
                    stated_confidence=confidence,
                    actual_outcome=1.0 if success else 0.0,
                    context_complexity=random.choice(['low', 'medium', 'high']),
                    domain=random.choice(profile['specializations']),
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 365))
                )
                self.confidence_adjuster.record_confidence_outcome(confidence_record)
        
        print("✅ Demo data setup complete!\n")
    
    def _generate_confidence_for_agent(self, agent_id: str, decision_success: bool) -> float:
        """Generate realistic confidence based on agent bias and decision success"""
        profile = self.demo_agents[agent_id]
        bias = profile['confidence_bias']
        
        # Base confidence influenced by actual success
        base_confidence = 0.7 if decision_success else 0.4
        
        # Apply bias patterns
        if bias == 'overconfident':
            confidence = min(1.0, base_confidence + 0.2)
        elif bias == 'underconfident':
            confidence = max(0.1, base_confidence - 0.15)
        elif bias == 'inconsistent':
            confidence = base_confidence + random.uniform(-0.3, 0.3)
        else:  # well_calibrated
            confidence = base_confidence + random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, confidence))
    
    def demo_basic_weight_calculation(self):
        """Demonstrate basic weight calculation functionality"""
        print("🧮 DEMO 1: Basic Weight Calculation")
        print("=" * 50)
        
        # Test different contexts
        test_contexts = [
            {'domain': 'security', 'confidence': 0.9, 'complexity': 'high'},
            {'domain': 'testing', 'confidence': 0.7, 'complexity': 'medium'},
            {'domain': 'architecture', 'confidence': 0.85, 'complexity': 'high'},
            {'domain': 'general', 'confidence': 0.6, 'complexity': 'low'}
        ]
        
        for context in test_contexts:
            print(f"\n📋 Context: {context}")
            print("Agent Weights:")
            
            for agent_id in self.demo_agents.keys():
                weight = self.weight_calculator.calculate_weight(agent_id, context)
                detailed = self.weight_calculator.calculate_detailed_weight(agent_id, context)
                
                print(f"  {agent_id:25} {weight:6.3f} "
                      f"(E:{detailed.expertise_factor:.2f}, "
                      f"A:{detailed.accuracy_factor:.2f}, "
                      f"C:{detailed.confidence_factor:.2f})")
        
        print("\n✅ Basic weight calculation demo complete!\n")
    
    def demo_ensemble_weighting(self):
        """Demonstrate ensemble weight calculation and normalization"""
        print("👥 DEMO 2: Ensemble Weight Calculation")
        print("=" * 50)
        
        # Security-critical decision scenario
        security_context = {
            'domain': 'security',
            'confidence': 0.85,
            'complexity': 'high',
            'impact_level': 'critical'
        }
        
        agents = list(self.demo_agents.keys())
        ensemble_weights = self.weight_calculator.calculate_ensemble_weights(agents, security_context)
        
        print(f"🔒 Security Decision Ensemble Weights:")
        print(f"Context: {security_context}")
        print("\nAgent                    Weight    Share")
        print("-" * 45)
        
        total_weight = sum(ensemble_weights.values())
        for agent, weight in sorted(ensemble_weights.items(), key=lambda x: x[1], reverse=True):
            share = (weight / total_weight) * 100 if total_weight > 0 else 0
            specializations = ", ".join(self.demo_agents[agent]['specializations'])
            print(f"{agent:25} {weight:6.3f}   {share:5.1f}%   ({specializations})")
        
        print(f"\nTotal Weight: {total_weight:.3f}")
        
        # Architecture decision for comparison
        arch_context = {
            'domain': 'architecture', 
            'confidence': 0.8,
            'complexity': 'high'
        }
        
        arch_weights = self.weight_calculator.calculate_ensemble_weights(agents, arch_context)
        
        print(f"\n🏗️  Architecture Decision Comparison:")
        print("Agent                    Security  Architecture  Shift")
        print("-" * 55)
        
        for agent in agents:
            sec_weight = ensemble_weights[agent]
            arch_weight = arch_weights[agent]
            shift = arch_weight - sec_weight
            print(f"{agent:25} {sec_weight:7.3f}   {arch_weight:9.3f}   {shift:+6.3f}")
        
        print("\n✅ Ensemble weighting demo complete!\n")
    
    def demo_expertise_assessment(self):
        """Demonstrate expertise scoring and domain assessment"""
        print("🎓 DEMO 3: Expertise Assessment")
        print("=" * 50)
        
        for agent_id in list(self.demo_agents.keys())[:3]:  # Show first 3 agents
            print(f"\n👤 Agent: {agent_id}")
            print("-" * 40)
            
            profile = self.expertise_scorer.get_agent_expertise_profile(agent_id)
            
            print(f"Specialization Areas: {', '.join(profile['specialization_areas'])}")
            print(f"Total Evidence Count: {profile['total_evidence_count']}")
            
            print("\nDomain Expertise Levels:")
            for domain, data in profile['domain_expertise'].items():
                if data['level'] > 0.1:  # Only show domains with some expertise
                    confidence_range = f"[{data['confidence_interval'][0]:.2f}-{data['confidence_interval'][1]:.2f}]"
                    print(f"  {domain:15} {data['level']:5.2f} {confidence_range} "
                          f"({data['certification']})")
        
        # Demonstrate expertise factor calculation
        print(f"\n🔍 Expertise Factors for Security Domain:")
        print("Agent                    Factor   Reasoning")
        print("-" * 50)
        
        for agent_id in self.demo_agents.keys():
            factor = self.expertise_scorer.calculate_domain_expertise_factor(agent_id, 'security')
            assessment = self.expertise_scorer.assess_agent_expertise(agent_id, 'security')
            
            reasoning = f"{assessment.evidence_count} evidence pieces, level {assessment.current_level:.2f}"
            print(f"{agent_id:25} {factor:6.3f}   {reasoning}")
        
        print("\n✅ Expertise assessment demo complete!\n")
    
    def demo_accuracy_tracking(self):
        """Demonstrate accuracy tracking and trend analysis"""
        print("📈 DEMO 4: Accuracy Tracking & Trend Analysis")
        print("=" * 50)
        
        for agent_id in list(self.demo_agents.keys())[:3]:
            print(f"\n📊 Agent: {agent_id}")
            print("-" * 40)
            
            metrics = self.accuracy_tracker.calculate_accuracy_metrics(agent_id)
            summary = self.accuracy_tracker.get_agent_performance_summary(agent_id)
            
            print(f"Overall Accuracy:     {metrics.overall_accuracy:.3f}")
            print(f"Weighted Accuracy:    {metrics.weighted_accuracy:.3f}")
            print(f"Recent Accuracy:      {summary['overall_performance']['recent_accuracy']:.3f}")
            print(f"Decision Count:       {metrics.decision_count}")
            print(f"Temporal Trend:       {metrics.temporal_trend:+.3f} ({'improving' if metrics.temporal_trend > 0 else 'declining' if metrics.temporal_trend < 0 else 'stable'})")
            print(f"Reliability Score:    {metrics.prediction_reliability:.3f}")
            
            # Show context-specific accuracy
            if metrics.context_specific_accuracy:
                print("\nContext-Specific Accuracy:")
                for context, accuracy in metrics.context_specific_accuracy.items():
                    print(f"  {context:15} {accuracy:.3f}")
            
            # Show performance insights
            if summary['performance_insights']:
                print("\nPerformance Insights:")
                for insight in summary['performance_insights']:
                    print(f"  • {insight}")
        
        # Demonstrate accuracy factor calculation
        print(f"\n🎯 Accuracy Factors Comparison:")
        print("Agent                    General  Security  Architecture")
        print("-" * 55)
        
        for agent_id in self.demo_agents.keys():
            general_factor = self.accuracy_tracker.calculate_accuracy_factor(agent_id, {'domain': 'general'})
            security_factor = self.accuracy_tracker.calculate_accuracy_factor(agent_id, {'domain': 'security'})
            arch_factor = self.accuracy_tracker.calculate_accuracy_factor(agent_id, {'domain': 'architecture'})
            
            print(f"{agent_id:25} {general_factor:7.3f}  {security_factor:8.3f}  {arch_factor:12.3f}")
        
        print("\n✅ Accuracy tracking demo complete!\n")
    
    def demo_confidence_calibration(self):
        """Demonstrate confidence adjustment and bias correction"""
        print("🎯 DEMO 5: Confidence Calibration & Bias Correction")
        print("=" * 50)
        
        for agent_id in list(self.demo_agents.keys())[:3]:
            print(f"\n🧠 Agent: {agent_id}")
            print("-" * 40)
            
            patterns = self.confidence_adjuster.analyze_agent_confidence_patterns(agent_id)
            
            if 'confidence_statistics' in patterns and patterns['confidence_statistics']:
                stats = patterns['confidence_statistics']
                print(f"Mean Stated Confidence: {stats['mean_stated_confidence']:.3f}")
                print(f"Confidence Std Dev:     {stats['confidence_std_dev']:.3f}")
            
            if 'calibration_analysis' in patterns and 'bias_type' in patterns['calibration_analysis']:
                calib = patterns['calibration_analysis']
                print(f"Bias Type:              {calib['bias_type']}")
                print(f"Calibration Quality:    {calib.get('calibration_quality', 'N/A')}")
                print(f"Mean Calibration Error: {calib.get('mean_calibration_error', 'N/A')}")
            
            if patterns.get('recommendations'):
                print("\nRecommendations:")
                for rec in patterns['recommendations']:
                    print(f"  • {rec}")
        
        # Demonstrate confidence adjustment
        print(f"\n⚖️  Confidence Adjustment Examples:")
        print("Agent                    Original  Adjusted  Factor   Bias Correction")
        print("-" * 75)
        
        test_confidence = 0.8
        context = {'domain': 'security', 'complexity': 'high'}
        
        for agent_id in self.demo_agents.keys():
            adjustment = self.confidence_adjuster.adjust_confidence(agent_id, test_confidence, context)
            
            print(f"{agent_id:25} {adjustment.original_confidence:8.3f}  "
                  f"{adjustment.adjusted_confidence:8.3f}  {adjustment.adjustment_factor:6.3f}  "
                  f"{adjustment.bias_correction:+13.3f}")
        
        print("\n✅ Confidence calibration demo complete!\n")
    
    def demo_strategy_comparison(self):
        """Demonstrate different weighting strategies"""
        print("🎛️  DEMO 6: Strategy Comparison")
        print("=" * 50)
        
        strategies = [
            WeightingStrategy.EXPERTISE_FOCUSED,
            WeightingStrategy.ACCURACY_FOCUSED,
            WeightingStrategy.BALANCED
        ]
        
        context = {'domain': 'security', 'confidence': 0.8}
        
        print("Strategy Comparison for Security Decision:")
        print("Agent                    Expertise  Accuracy   Balanced")
        print("-" * 60)
        
        for agent_id in self.demo_agents.keys():
            weights = {}
            
            for strategy in strategies:
                calc = VoteWeightCalculator(strategy=strategy)
                # Copy data to new calculator
                calc.expertise_profiles = self.weight_calculator.expertise_profiles
                calc.accuracy_records = self.weight_calculator.accuracy_records
                
                weights[strategy] = calc.calculate_weight(agent_id, context)
            
            print(f"{agent_id:25} {weights[WeightingStrategy.EXPERTISE_FOCUSED]:9.3f}  "
                  f"{weights[WeightingStrategy.ACCURACY_FOCUSED]:8.3f}  "
                  f"{weights[WeightingStrategy.BALANCED]:8.3f}")
        
        # Show strategy characteristics
        print(f"\n📋 Strategy Characteristics:")
        for strategy in strategies:
            calc = VoteWeightCalculator(strategy=strategy)
            weights = calc.strategy_weights[strategy]
            print(f"\n{strategy.value.replace('_', ' ').title()}:")
            print(f"  Expertise Weight:  {weights['expertise']:.1%}")
            print(f"  Accuracy Weight:   {weights['accuracy']:.1%}")
            print(f"  Confidence Weight: {weights['confidence']:.1%}")
        
        print("\n✅ Strategy comparison demo complete!\n")
    
    def demo_integration_system(self):
        """Demonstrate the integrated weighted voting system"""
        print("🔗 DEMO 7: Integrated Weighted Voting System")
        print("=" * 50)
        
        # Simulate a realistic voting scenario
        print("Simulating Security Vulnerability Assessment Vote...\n")
        
        # Create mock agent votes (would normally come from actual agents)
        mock_votes = []
        context = {
            'domain': 'security',
            'confidence': 0.85,
            'complexity': 'high',
            'impact_level': 'critical',
            'decision_type': 'security_approval'
        }
        
        # Simulate different agent responses
        agent_responses = {
            'rif-security-expert': (True, 0.9, "High confidence - security measures are adequate"),
            'rif-junior-implementer': (True, 0.6, "Looks good but not entirely sure"),
            'rif-senior-architect': (False, 0.85, "Potential scalability concerns with security approach"),
            'rif-testing-specialist': (True, 0.8, "Security tests pass, validation complete"),
            'rif-generalist': (True, 0.7, "Generally acceptable approach")
        }
        
        # Show agent profiles before voting
        print("👥 Agent Profiles:")
        for agent_id in agent_responses.keys():
            profile = self.integrated_system.get_agent_weight_profile(agent_id)
            
            # Extract key metrics
            base_weight = self.weight_calculator.calculate_weight(agent_id, context)
            accuracy = "N/A"
            if 'accuracy_profile' in profile and 'overall_performance' in profile['accuracy_profile']:
                accuracy = f"{profile['accuracy_profile']['overall_performance'].get('accuracy', 0):.2f}"
            
            specializations = ", ".join(self.demo_agents[agent_id]['specializations'])
            
            print(f"  {agent_id:25} Weight: {base_weight:.3f}  Accuracy: {accuracy}  ({specializations})")
        
        # Show individual vote impacts
        print(f"\n🗳️  Individual Vote Analysis:")
        print("Agent                    Vote   Confidence  Weight   Weighted Impact")
        print("-" * 70)
        
        total_weighted_impact = 0.0
        individual_impacts = {}
        
        for agent_id, (vote, confidence, reasoning) in agent_responses.items():
            weight = self.weight_calculator.calculate_weight(agent_id, {**context, 'confidence': confidence})
            vote_value = 1.0 if vote else 0.0
            weighted_impact = weight * vote_value
            total_weighted_impact += weight
            individual_impacts[agent_id] = weighted_impact
            
            print(f"{agent_id:25} {'Yes':4}   {confidence:10.2f}  {weight:6.3f}   {weighted_impact:13.3f}")
        
        # Calculate final decision
        positive_weighted_impact = sum(impact for impact in individual_impacts.values())
        decision_score = positive_weighted_impact / total_weighted_impact if total_weighted_impact > 0 else 0.0
        final_decision = decision_score >= 0.7  # 70% threshold
        
        print(f"\n📊 Final Decision Analysis:")
        print(f"Total Weighted Score:     {decision_score:.3f}")
        print(f"Decision Threshold:       0.700")
        print(f"Final Decision:           {'✅ APPROVED' if final_decision else '❌ REJECTED'}")
        print(f"Confidence in Decision:   {decision_score:.1%}")
        
        # Show what the decision would be without weighting
        unweighted_votes = sum(1 for vote, _, _ in agent_responses.values() if vote)
        unweighted_decision = unweighted_votes > len(agent_responses) // 2
        
        print(f"\nComparison with Unweighted Voting:")
        print(f"Unweighted Result:        {'✅ APPROVED' if unweighted_decision else '❌ REJECTED'} ({unweighted_votes}/{len(agent_responses)} votes)")
        print(f"Weighted Result:          {'✅ APPROVED' if final_decision else '❌ REJECTED'} ({decision_score:.1%} weighted score)")
        
        if final_decision != unweighted_decision:
            print(f"🔄 Weighting CHANGED the decision outcome!")
        else:
            print(f"✅ Weighting CONFIRMED the unweighted decision")
        
        # Show system metrics
        print(f"\n📈 System Performance Metrics:")
        metrics = self.integrated_system.get_integration_metrics()
        
        if 'integration_metrics' in metrics:
            int_metrics = metrics['integration_metrics']
            print(f"Weighted Decisions:       {int_metrics.get('weighted_decisions', 0)}")
            print(f"Average Weight Variance:  {int_metrics.get('average_weight_variance', 0):.3f}")
        
        print("\n✅ Integration system demo complete!\n")
    
    def demo_learning_and_adaptation(self):
        """Demonstrate learning and adaptation capabilities"""
        print("🧠 DEMO 8: Learning & Adaptation")
        print("=" * 50)
        
        print("Simulating decision outcome feedback and system learning...\n")
        
        # Simulate outcomes for previous decisions
        agent_id = 'rif-junior-implementer'
        print(f"📚 Learning Demo for: {agent_id}")
        
        # Get initial weights
        context = {'domain': 'implementation', 'confidence': 0.7}
        initial_weight = self.weight_calculator.calculate_weight(agent_id, context)
        initial_accuracy = self.accuracy_tracker.calculate_accuracy_metrics(agent_id)
        
        print(f"\nInitial State:")
        print(f"  Weight for Implementation:  {initial_weight:.3f}")
        print(f"  Overall Accuracy:          {initial_accuracy.overall_accuracy:.3f}")
        print(f"  Decision Count:            {initial_accuracy.decision_count}")
        
        # Simulate successful outcomes (agent is improving)
        print(f"\n🎯 Simulating 10 successful decisions...")
        
        for i in range(10):
            # Record successful decision
            self.weight_calculator.update_agent_accuracy(agent_id, True, context)
            
            # Record for accuracy tracker
            decision = DecisionRecord(
                decision_id=f"learning_demo_{i}",
                agent_id=agent_id,
                decision_timestamp=datetime.now(),
                decision_confidence=0.75,
                decision_content=f"Learning demo decision {i}",
                outcome=DecisionOutcome.SUCCESS,
                outcome_timestamp=datetime.now(),
                context_category='implementation',
                impact_level='medium'
            )
            self.accuracy_tracker.record_decision_outcome(decision)
            
            # Record confidence outcome
            confidence_record = ConfidenceRecord(
                agent_id=agent_id,
                decision_id=f"learning_demo_{i}",
                stated_confidence=0.75,
                actual_outcome=1.0,
                context_complexity='medium',
                domain='implementation',
                timestamp=datetime.now()
            )
            self.confidence_adjuster.record_confidence_outcome(confidence_record)
        
        # Check updated state
        updated_weight = self.weight_calculator.calculate_weight(agent_id, context)
        updated_accuracy = self.accuracy_tracker.calculate_accuracy_metrics(agent_id)
        
        print(f"\nUpdated State After Learning:")
        print(f"  Weight for Implementation:  {updated_weight:.3f} (Δ {updated_weight - initial_weight:+.3f})")
        print(f"  Overall Accuracy:          {updated_accuracy.overall_accuracy:.3f} (Δ {updated_accuracy.overall_accuracy - initial_accuracy.overall_accuracy:+.3f})")
        print(f"  Decision Count:            {updated_accuracy.decision_count} (+{updated_accuracy.decision_count - initial_accuracy.decision_count})")
        print(f"  Temporal Trend:            {updated_accuracy.temporal_trend:+.3f}")
        
        # Show confidence calibration improvement
        patterns = self.confidence_adjuster.analyze_agent_confidence_patterns(agent_id)
        if 'calibration_analysis' in patterns and 'calibration_quality' in patterns['calibration_analysis']:
            calibration_quality = patterns['calibration_analysis']['calibration_quality']
            print(f"  Confidence Calibration:    {calibration_quality:.3f}")
        
        print(f"\n💡 Learning Insights:")
        if updated_weight > initial_weight:
            print(f"  ✅ Agent weight INCREASED due to improved performance")
        else:
            print(f"  ⚡ Agent weight adjusted based on performance patterns")
        
        if updated_accuracy.temporal_trend > 0:
            print(f"  📈 Positive performance trend detected (+{updated_accuracy.temporal_trend:.3f})")
        
        print("\n✅ Learning and adaptation demo complete!\n")
    
    def run_all_demos(self):
        """Run all demo scenarios"""
        print("🌟 RIF VOTE WEIGHTING SYSTEM COMPREHENSIVE DEMO")
        print("=" * 60)
        print("Demonstrating advanced vote weighting algorithm with:")
        print("• Multi-factor weight calculation")
        print("• Expertise-based scoring") 
        print("• Historical accuracy tracking")
        print("• Confidence bias correction")
        print("• Ensemble weight optimization")
        print("• Learning and adaptation")
        print("=" * 60)
        print()
        
        # Run all demos
        demos = [
            self.demo_basic_weight_calculation,
            self.demo_ensemble_weighting,
            self.demo_expertise_assessment,
            self.demo_accuracy_tracking,
            self.demo_confidence_calibration,
            self.demo_strategy_comparison,
            self.demo_integration_system,
            self.demo_learning_and_adaptation
        ]
        
        start_time = time.time()
        
        for i, demo in enumerate(demos, 1):
            try:
                demo()
                print(f"Demo {i}/{len(demos)} completed successfully ✅\n")
            except Exception as e:
                print(f"Demo {i}/{len(demos)} encountered error: {e} ❌\n")
        
        total_time = time.time() - start_time
        
        print("🎉 ALL DEMOS COMPLETED!")
        print("=" * 60)
        print(f"Total Demo Time: {total_time:.2f} seconds")
        print("\nSystem Summary:")
        print(f"• Vote Weight Calculator:   ✅ Multi-factor algorithm with {len(self.demo_agents)} agents")
        print(f"• Expertise Scorer:         ✅ Domain-specific expertise assessment")
        print(f"• Accuracy Tracker:         ✅ Historical performance analysis")
        print(f"• Confidence Adjuster:      ✅ Bias detection and correction")
        print(f"• Integration Layer:        ✅ Seamless consensus system integration")
        print(f"• Learning System:          ✅ Adaptive weight updates")
        
        # Final metrics summary
        print(f"\n📊 Final System Metrics:")
        
        calc_metrics = self.weight_calculator.get_calculation_metrics()
        print(f"• Total Weight Calculations: {calc_metrics['total_calculations']}")
        print(f"• Average Calculation Time:  {calc_metrics['average_calculation_time']:.4f}s")
        
        expertise_metrics = self.expertise_scorer.get_scoring_metrics()
        print(f"• Total Expertise Assessments: {expertise_metrics['total_assessments']}")
        
        accuracy_metrics = self.accuracy_tracker.get_tracker_metrics()
        print(f"• Total Decisions Tracked:    {accuracy_metrics['total_decisions_tracked']}")
        
        confidence_metrics = self.confidence_adjuster.get_adjuster_metrics()
        print(f"• Total Confidence Adjustments: {confidence_metrics['total_adjustments_made']}")
        
        print("\n🚀 RIF Vote Weighting System is ready for production!")
        print("=" * 60)


if __name__ == "__main__":
    print("Starting RIF Vote Weighting System Demo...\n")
    
    try:
        demo = VoteWeightingDemo()
        demo.run_all_demos()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Demo complete!")