#!/usr/bin/env python3
"""
RIF Arbitrator Agent
Specialized agent for complex conflict resolution and arbitration.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import arbitration components
from .arbitration_system import ArbitrationDecision, ArbitrationType
from .conflict_detector import ConflictAnalysis, ConflictPattern
from .escalation_engine import EscalationResult

# Import consensus components
import sys
sys.path.insert(0, '/Users/cal/DEV/RIF/claude/commands')

from consensus_architecture import AgentVote, ConfidenceLevel, ConsensusArchitecture
from voting_aggregator import VoteConflict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitratorMode(Enum):
    """Modes of arbitrator operation"""
    ANALYTICAL = "analytical"           # Deep analysis of evidence and reasoning
    CONSENSUS_BUILDING = "consensus_building"  # Focus on finding common ground
    DECISIVE = "decisive"               # Quick decisive resolution
    INVESTIGATIVE = "investigative"     # Thorough investigation of discrepancies
    BALANCED = "balanced"               # Balanced approach considering all factors

class ArbitratorSkill(Enum):
    """Specialized arbitrator skills"""
    EVIDENCE_ANALYSIS = "evidence_analysis"
    REASONING_EVALUATION = "reasoning_evaluation"
    BIAS_DETECTION = "bias_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    CONFLICT_MEDIATION = "conflict_mediation"
    TECHNICAL_EXPERTISE = "technical_expertise"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class ArbitratorConfig:
    """Configuration for arbitrator agent"""
    arbitrator_id: str
    mode: ArbitratorMode
    skills: List[ArbitratorSkill]
    expertise_domains: List[str]
    confidence_threshold: float
    max_analysis_time_minutes: int
    evidence_weight: float = 0.4
    reasoning_weight: float = 0.3
    expertise_weight: float = 0.3
    bias_detection_enabled: bool = True
    deep_analysis_enabled: bool = True

@dataclass
class ArbitratorAnalysis:
    """Detailed analysis performed by arbitrator"""
    analysis_id: str
    arbitrator_id: str
    timestamp: datetime
    conflict_analysis: ConflictAnalysis
    evidence_evaluation: Dict[str, float]
    reasoning_evaluation: Dict[str, Dict[str, Any]]
    bias_assessment: Dict[str, Any]
    consensus_opportunities: List[str]
    decision_factors: Dict[str, float]
    confidence_breakdown: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class ArbitratorAgent:
    """
    Specialized agent for sophisticated conflict arbitration and resolution
    """
    
    def __init__(self, config: ArbitratorConfig, 
                 consensus: Optional[ConsensusArchitecture] = None):
        """Initialize arbitrator agent"""
        self.config = config
        self.consensus = consensus or ConsensusArchitecture()
        
        # Analysis techniques based on skills
        self.analysis_techniques = {
            ArbitratorSkill.EVIDENCE_ANALYSIS: self._analyze_evidence_quality,
            ArbitratorSkill.REASONING_EVALUATION: self._evaluate_reasoning_depth,
            ArbitratorSkill.BIAS_DETECTION: self._detect_cognitive_biases,
            ArbitratorSkill.PATTERN_RECOGNITION: self._recognize_decision_patterns,
            ArbitratorSkill.CONFLICT_MEDIATION: self._identify_mediation_opportunities,
            ArbitratorSkill.TECHNICAL_EXPERTISE: self._apply_technical_expertise,
            ArbitratorSkill.RISK_ASSESSMENT: self._assess_decision_risks
        }
        
        # Performance tracking
        self.arbitration_history = []
        self.success_metrics = {
            "total_arbitrations": 0,
            "successful_resolutions": 0,
            "confidence_accuracy": 0.0,
            "average_analysis_time": 0.0,
            "skill_effectiveness": {skill.value: 0.0 for skill in self.config.skills}
        }

    def arbitrate_conflict(self, votes: List[AgentVote], conflict_analysis: ConflictAnalysis,
                          context: Dict[str, Any]) -> ArbitrationDecision:
        """
        Main arbitration method - performs sophisticated analysis and makes decision
        
        Args:
            votes: Conflicting agent votes
            conflict_analysis: Detailed conflict analysis
            context: Decision context
            
        Returns:
            ArbitrationDecision: Sophisticated arbitration decision
        """
        start_time = time.time()
        analysis_id = f"arb-{self.config.arbitrator_id}-{int(start_time)}"
        
        logger.info(f"Arbitrator {self.config.arbitrator_id} starting analysis {analysis_id}")
        
        # Perform comprehensive analysis
        arbitrator_analysis = self._perform_comprehensive_analysis(
            votes, conflict_analysis, context, analysis_id
        )
        
        # Generate arbitration decision
        decision = self._generate_arbitration_decision(
            votes, conflict_analysis, arbitrator_analysis, context
        )
        
        # Update performance metrics
        analysis_time = time.time() - start_time
        self._update_performance_metrics(decision, analysis_time)
        
        # Store arbitration in history
        self.arbitration_history.append({
            "analysis": arbitrator_analysis,
            "decision": decision,
            "processing_time": analysis_time,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Arbitration {analysis_id} completed in {analysis_time:.2f}s with confidence {decision.confidence_score:.2f}")
        
        return decision

    def _perform_comprehensive_analysis(self, votes: List[AgentVote], 
                                      conflict_analysis: ConflictAnalysis,
                                      context: Dict[str, Any],
                                      analysis_id: str) -> ArbitratorAnalysis:
        """Perform comprehensive analysis using all available skills"""
        
        # Initialize analysis
        analysis = ArbitratorAnalysis(
            analysis_id=analysis_id,
            arbitrator_id=self.config.arbitrator_id,
            timestamp=datetime.now(),
            conflict_analysis=conflict_analysis,
            evidence_evaluation={},
            reasoning_evaluation={},
            bias_assessment={},
            consensus_opportunities=[],
            decision_factors={},
            confidence_breakdown={},
            recommendations=[]
        )
        
        # Apply each configured skill
        for skill in self.config.skills:
            if skill in self.analysis_techniques:
                try:
                    technique_result = self.analysis_techniques[skill](votes, conflict_analysis, context)
                    self._integrate_technique_result(analysis, skill, technique_result)
                except Exception as e:
                    logger.error(f"Skill {skill.value} analysis failed: {str(e)}")
                    continue
        
        # Synthesize overall analysis
        analysis = self._synthesize_analysis(analysis, votes, context)
        
        return analysis

    def _analyze_evidence_quality(self, votes: List[AgentVote], 
                                conflict_analysis: ConflictAnalysis,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality and consistency of evidence"""
        evidence_analysis = {
            "individual_quality": {},
            "consistency_score": 0.0,
            "reliability_factors": {},
            "quality_distribution": {},
            "outliers": []
        }
        
        # Analyze individual evidence quality
        evidence_scores = []
        for vote in votes:
            agent_id = vote.agent_id
            quality = vote.evidence_quality
            evidence_scores.append(quality)
            
            evidence_analysis["individual_quality"][agent_id] = {
                "quality_score": quality,
                "reasoning_length": len(vote.reasoning),
                "confidence_alignment": self._assess_confidence_evidence_alignment(vote),
                "domain_relevance": self._assess_domain_relevance(vote, context)
            }
        
        # Calculate consistency score
        if len(evidence_scores) > 1:
            import statistics
            evidence_variance = statistics.variance(evidence_scores)
            evidence_analysis["consistency_score"] = max(0.0, 1.0 - evidence_variance)
        else:
            evidence_analysis["consistency_score"] = 1.0
        
        # Identify quality outliers
        if len(evidence_scores) > 2:
            mean_quality = statistics.mean(evidence_scores)
            std_quality = statistics.stdev(evidence_scores) if len(evidence_scores) > 1 else 0
            
            for vote in votes:
                if std_quality > 0 and abs(vote.evidence_quality - mean_quality) > 2 * std_quality:
                    evidence_analysis["outliers"].append({
                        "agent_id": vote.agent_id,
                        "quality_score": vote.evidence_quality,
                        "deviation": abs(vote.evidence_quality - mean_quality)
                    })
        
        # Quality distribution analysis
        quality_ranges = {"low": 0, "medium": 0, "high": 0}
        for score in evidence_scores:
            if score < 0.4:
                quality_ranges["low"] += 1
            elif score < 0.7:
                quality_ranges["medium"] += 1
            else:
                quality_ranges["high"] += 1
        
        evidence_analysis["quality_distribution"] = quality_ranges
        
        return evidence_analysis

    def _evaluate_reasoning_depth(self, votes: List[AgentVote],
                                conflict_analysis: ConflictAnalysis,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate depth and quality of reasoning"""
        reasoning_analysis = {
            "depth_scores": {},
            "logical_consistency": {},
            "argument_strength": {},
            "reasoning_patterns": {},
            "critical_thinking": {}
        }
        
        for vote in votes:
            agent_id = vote.agent_id
            reasoning = vote.reasoning
            
            # Analyze reasoning depth (simplified - would use NLP in production)
            depth_score = self._calculate_reasoning_depth_score(reasoning)
            reasoning_analysis["depth_scores"][agent_id] = depth_score
            
            # Assess logical consistency
            consistency_score = self._assess_logical_consistency(reasoning, vote.vote)
            reasoning_analysis["logical_consistency"][agent_id] = consistency_score
            
            # Evaluate argument strength
            strength_score = self._evaluate_argument_strength(reasoning, vote.evidence_quality)
            reasoning_analysis["argument_strength"][agent_id] = strength_score
            
            # Identify reasoning patterns
            patterns = self._identify_reasoning_patterns(reasoning)
            reasoning_analysis["reasoning_patterns"][agent_id] = patterns
            
            # Assess critical thinking indicators
            critical_thinking = self._assess_critical_thinking_indicators(reasoning)
            reasoning_analysis["critical_thinking"][agent_id] = critical_thinking
        
        return reasoning_analysis

    def _detect_cognitive_biases(self, votes: List[AgentVote],
                               conflict_analysis: ConflictAnalysis,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential cognitive biases in voting patterns"""
        bias_analysis = {
            "potential_biases": {},
            "bias_indicators": {},
            "agent_bias_scores": {},
            "systematic_biases": []
        }
        
        # Check for common biases
        bias_checks = {
            "confirmation_bias": self._check_confirmation_bias,
            "anchoring_bias": self._check_anchoring_bias,
            "availability_bias": self._check_availability_bias,
            "groupthink": self._check_groupthink_patterns
        }
        
        for bias_type, check_function in bias_checks.items():
            try:
                bias_result = check_function(votes, context)
                if bias_result["detected"]:
                    bias_analysis["potential_biases"][bias_type] = bias_result
            except Exception as e:
                logger.warning(f"Bias check {bias_type} failed: {str(e)}")
        
        # Calculate individual bias scores
        for vote in votes:
            agent_bias_score = self._calculate_individual_bias_score(vote, votes, context)
            bias_analysis["agent_bias_scores"][vote.agent_id] = agent_bias_score
        
        return bias_analysis

    def _recognize_decision_patterns(self, votes: List[AgentVote],
                                   conflict_analysis: ConflictAnalysis,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize patterns in decision-making approaches"""
        pattern_analysis = {
            "decision_styles": {},
            "consistency_patterns": {},
            "expertise_patterns": {},
            "temporal_patterns": {}
        }
        
        for vote in votes:
            agent_id = vote.agent_id
            
            # Analyze decision style
            decision_style = self._analyze_decision_style(vote, context)
            pattern_analysis["decision_styles"][agent_id] = decision_style
            
            # Check historical consistency (would require agent history in production)
            consistency = {"score": 0.8, "pattern": "stable"}  # Placeholder
            pattern_analysis["consistency_patterns"][agent_id] = consistency
            
            # Analyze expertise alignment
            domain = context.get("domain", "general")
            expertise_level = self.consensus.agent_expertise.get(agent_id, {}).get(domain, 0.5)
            expertise_pattern = self._analyze_expertise_pattern(vote, expertise_level)
            pattern_analysis["expertise_patterns"][agent_id] = expertise_pattern
        
        return pattern_analysis

    def _identify_mediation_opportunities(self, votes: List[AgentVote],
                                        conflict_analysis: ConflictAnalysis,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify opportunities for consensus building and mediation"""
        mediation_analysis = {
            "common_ground": [],
            "bridging_opportunities": [],
            "compromise_solutions": [],
            "mediation_strategies": []
        }
        
        # Find common ground in reasoning
        common_themes = self._find_common_reasoning_themes(votes)
        mediation_analysis["common_ground"] = common_themes
        
        # Identify potential bridges between positions
        bridges = self._identify_position_bridges(votes)
        mediation_analysis["bridging_opportunities"] = bridges
        
        # Generate compromise solutions
        compromises = self._generate_compromise_solutions(votes, context)
        mediation_analysis["compromise_solutions"] = compromises
        
        # Suggest mediation strategies
        strategies = self._suggest_mediation_strategies(conflict_analysis, votes)
        mediation_analysis["mediation_strategies"] = strategies
        
        return mediation_analysis

    def _apply_technical_expertise(self, votes: List[AgentVote],
                                 conflict_analysis: ConflictAnalysis,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply domain-specific technical expertise"""
        technical_analysis = {
            "domain_assessment": {},
            "technical_accuracy": {},
            "implementation_feasibility": {},
            "best_practices": {}
        }
        
        domain = context.get("domain", "general")
        
        # Assess technical accuracy of each position
        for vote in votes:
            agent_id = vote.agent_id
            
            # Check technical accuracy (simplified)
            accuracy_score = self._assess_technical_accuracy(vote, domain, context)
            technical_analysis["technical_accuracy"][agent_id] = accuracy_score
            
            # Evaluate implementation feasibility
            feasibility_score = self._evaluate_implementation_feasibility(vote, context)
            technical_analysis["implementation_feasibility"][agent_id] = feasibility_score
            
            # Check alignment with best practices
            best_practice_score = self._check_best_practice_alignment(vote, domain)
            technical_analysis["best_practices"][agent_id] = best_practice_score
        
        # Overall domain assessment
        technical_analysis["domain_assessment"] = {
            "complexity": context.get("complexity", "medium"),
            "risk_level": context.get("risk_level", "medium"),
            "technical_consensus_possible": len(set(vote.vote for vote in votes)) <= 2
        }
        
        return technical_analysis

    def _assess_decision_risks(self, votes: List[AgentVote],
                             conflict_analysis: ConflictAnalysis,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with different decision options"""
        risk_analysis = {
            "decision_risks": {},
            "risk_factors": {},
            "mitigation_strategies": {},
            "risk_tolerance_alignment": {}
        }
        
        # Analyze risks for each decision option
        decision_options = list(set(vote.vote for vote in votes))
        
        for option in decision_options:
            option_risks = self._analyze_decision_option_risks(option, context)
            risk_analysis["decision_risks"][str(option)] = option_risks
        
        # Identify general risk factors
        risk_factors = self._identify_risk_factors(conflict_analysis, context)
        risk_analysis["risk_factors"] = risk_factors
        
        # Generate mitigation strategies
        mitigations = self._generate_risk_mitigation_strategies(risk_analysis["decision_risks"], context)
        risk_analysis["mitigation_strategies"] = mitigations
        
        return risk_analysis

    def _generate_arbitration_decision(self, votes: List[AgentVote],
                                     conflict_analysis: ConflictAnalysis,
                                     arbitrator_analysis: ArbitratorAnalysis,
                                     context: Dict[str, Any]) -> ArbitrationDecision:
        """Generate final arbitration decision based on comprehensive analysis"""
        
        # Calculate decision factors based on mode and analysis
        decision_factors = self._calculate_decision_factors(arbitrator_analysis, context)
        
        # Determine final decision using weighted scoring
        final_decision = self._determine_final_decision(votes, decision_factors, arbitrator_analysis)
        
        # Calculate confidence score
        confidence_score = self._calculate_arbitrator_confidence(
            final_decision, arbitrator_analysis, decision_factors
        )
        
        # Generate comprehensive reasoning
        reasoning = self._generate_arbitrator_reasoning(
            final_decision, arbitrator_analysis, decision_factors
        )
        
        # Create supporting evidence summary
        supporting_evidence = self._create_supporting_evidence_summary(
            arbitrator_analysis, decision_factors
        )
        
        # Identify dissenting opinions
        dissenting_opinions = [
            f"{vote.agent_id}: {vote.reasoning}"
            for vote in votes
            if vote.vote != final_decision
        ]
        
        decision = ArbitrationDecision(
            decision_id=f"arbitrator-{arbitrator_analysis.analysis_id}",
            final_decision=final_decision,
            confidence_score=confidence_score,
            arbitration_type=ArbitrationType.ARBITRATOR_AGENT,
            reasoning=reasoning,
            supporting_evidence=supporting_evidence,
            dissenting_opinions=dissenting_opinions,
            resolution_method="arbitrator_agent"
        )
        
        return decision

    # Helper methods for analysis techniques
    def _assess_confidence_evidence_alignment(self, vote: AgentVote) -> float:
        """Assess alignment between confidence level and evidence quality"""
        confidence_score = self._confidence_to_score(vote.confidence)
        evidence_score = vote.evidence_quality
        
        # Good alignment means confidence matches evidence quality
        alignment = 1.0 - abs(confidence_score - evidence_score)
        return max(0.0, alignment)

    def _assess_domain_relevance(self, vote: AgentVote, context: Dict[str, Any]) -> float:
        """Assess relevance of vote to domain context"""
        domain = context.get("domain", "general")
        agent_expertise = self.consensus.agent_expertise.get(vote.agent_id, {})
        domain_expertise = agent_expertise.get(domain, 0.5)
        
        # Relevance is based on domain expertise
        return domain_expertise

    def _calculate_reasoning_depth_score(self, reasoning: str) -> float:
        """Calculate depth score for reasoning (simplified)"""
        # Simple heuristics - would use NLP in production
        length_score = min(1.0, len(reasoning) / 200)  # Normalize by 200 characters
        complexity_indicators = ["because", "therefore", "however", "although", "considering"]
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in reasoning.lower()) / len(complexity_indicators)
        
        return (length_score + complexity_score) / 2

    def _assess_logical_consistency(self, reasoning: str, vote_value: Any) -> float:
        """Assess logical consistency of reasoning with vote"""
        # Simplified consistency check
        positive_indicators = ["good", "correct", "should", "yes", "approve"]
        negative_indicators = ["bad", "incorrect", "shouldn't", "no", "reject"]
        
        positive_count = sum(1 for word in positive_indicators if word in reasoning.lower())
        negative_count = sum(1 for word in negative_indicators if word in reasoning.lower())
        
        if isinstance(vote_value, bool):
            if vote_value and positive_count > negative_count:
                return 0.8
            elif not vote_value and negative_count > positive_count:
                return 0.8
            else:
                return 0.4
        
        return 0.6  # Default consistency score

    def _evaluate_argument_strength(self, reasoning: str, evidence_quality: float) -> float:
        """Evaluate strength of argument"""
        reasoning_depth = self._calculate_reasoning_depth_score(reasoning)
        
        # Combine reasoning depth with evidence quality
        strength = (reasoning_depth * 0.6) + (evidence_quality * 0.4)
        return strength

    def _identify_reasoning_patterns(self, reasoning: str) -> Dict[str, bool]:
        """Identify patterns in reasoning approach"""
        patterns = {
            "analytical": any(word in reasoning.lower() for word in ["analyze", "data", "evidence", "facts"]),
            "intuitive": any(word in reasoning.lower() for word in ["feel", "sense", "intuition", "gut"]),
            "risk_focused": any(word in reasoning.lower() for word in ["risk", "danger", "safe", "secure"]),
            "process_focused": any(word in reasoning.lower() for word in ["process", "procedure", "standard", "protocol"]),
            "outcome_focused": any(word in reasoning.lower() for word in ["result", "outcome", "impact", "consequence"])
        }
        
        return patterns

    def _assess_critical_thinking_indicators(self, reasoning: str) -> Dict[str, float]:
        """Assess indicators of critical thinking"""
        indicators = {
            "questions_assumptions": 0.5,  # Would analyze for assumption challenging
            "considers_alternatives": 0.5,  # Would look for alternative consideration
            "evaluates_evidence": 0.5,     # Would assess evidence evaluation
            "acknowledges_uncertainty": 0.5  # Would look for uncertainty acknowledgment
        }
        
        # Simplified analysis - would use NLP in production
        reasoning_lower = reasoning.lower()
        
        if any(word in reasoning_lower for word in ["assume", "question", "consider", "alternative"]):
            indicators["considers_alternatives"] = 0.8
        
        if any(word in reasoning_lower for word in ["evidence", "proof", "support", "data"]):
            indicators["evaluates_evidence"] = 0.8
        
        if any(word in reasoning_lower for word in ["uncertain", "might", "could", "perhaps"]):
            indicators["acknowledges_uncertainty"] = 0.8
        
        return indicators

    def _check_confirmation_bias(self, votes: List[AgentVote], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for confirmation bias patterns"""
        # Simplified confirmation bias detection
        return {
            "detected": False,
            "confidence": 0.0,
            "indicators": [],
            "affected_agents": []
        }

    def _check_anchoring_bias(self, votes: List[AgentVote], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for anchoring bias patterns"""
        return {
            "detected": False,
            "confidence": 0.0,
            "indicators": [],
            "affected_agents": []
        }

    def _check_availability_bias(self, votes: List[AgentVote], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for availability bias patterns"""
        return {
            "detected": False,
            "confidence": 0.0,
            "indicators": [],
            "affected_agents": []
        }

    def _check_groupthink_patterns(self, votes: List[AgentVote], context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for groupthink patterns"""
        # Look for unusual consensus that might indicate groupthink
        unique_votes = len(set(vote.vote for vote in votes))
        
        if unique_votes == 1 and len(votes) > 3:
            return {
                "detected": True,
                "confidence": 0.6,
                "indicators": ["Unanimous decision in complex scenario"],
                "affected_agents": [vote.agent_id for vote in votes]
            }
        
        return {
            "detected": False,
            "confidence": 0.0,
            "indicators": [],
            "affected_agents": []
        }

    def _calculate_individual_bias_score(self, vote: AgentVote, all_votes: List[AgentVote], 
                                       context: Dict[str, Any]) -> float:
        """Calculate bias score for individual agent"""
        # Simplified bias scoring - would be more sophisticated in production
        bias_score = 0.0
        
        # Check if vote is extreme compared to others
        if len(all_votes) > 1:
            other_votes = [v for v in all_votes if v.agent_id != vote.agent_id]
            # Simple extremeness check would go here
        
        return bias_score

    def _analyze_decision_style(self, vote: AgentVote, context: Dict[str, Any]) -> Dict[str, str]:
        """Analyze decision-making style"""
        reasoning = vote.reasoning.lower()
        
        style = {
            "primary_style": "balanced",
            "risk_tolerance": "medium",
            "information_processing": "thorough"
        }
        
        # Analyze risk tolerance
        if any(word in reasoning for word in ["risk", "danger", "safe", "secure"]):
            if any(word in reasoning for word in ["avoid", "prevent", "careful"]):
                style["risk_tolerance"] = "conservative"
            else:
                style["risk_tolerance"] = "risk_aware"
        
        # Analyze information processing
        if len(reasoning) > 100:
            style["information_processing"] = "thorough"
        elif len(reasoning) < 50:
            style["information_processing"] = "concise"
        
        return style

    def _analyze_expertise_pattern(self, vote: AgentVote, expertise_level: float) -> Dict[str, Any]:
        """Analyze how expertise level affects decision pattern"""
        return {
            "expertise_level": expertise_level,
            "confidence_expertise_alignment": abs(self._confidence_to_score(vote.confidence) - expertise_level),
            "expertise_appropriate": expertise_level > 0.7 if vote.confidence.value in ["high", "very_high"] else True
        }

    def _find_common_reasoning_themes(self, votes: List[AgentVote]) -> List[str]:
        """Find common themes in reasoning across votes"""
        # Simplified theme identification - would use NLP in production
        all_reasoning = " ".join(vote.reasoning.lower() for vote in votes)
        
        common_themes = []
        theme_keywords = {
            "quality": ["quality", "standard", "excellence"],
            "risk": ["risk", "danger", "safe", "security"],
            "efficiency": ["efficient", "fast", "quick", "performance"],
            "cost": ["cost", "expensive", "budget", "resource"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_reasoning for keyword in keywords):
                common_themes.append(theme)
        
        return common_themes

    def _identify_position_bridges(self, votes: List[AgentVote]) -> List[str]:
        """Identify potential bridges between different positions"""
        bridges = []
        
        # Look for partial agreements or conditional statements
        for vote in votes:
            reasoning = vote.reasoning.lower()
            if any(word in reasoning for word in ["if", "unless", "provided", "assuming"]):
                bridges.append(f"Conditional agreement from {vote.agent_id}: {vote.reasoning[:100]}...")
        
        return bridges

    def _generate_compromise_solutions(self, votes: List[AgentVote], context: Dict[str, Any]) -> List[str]:
        """Generate potential compromise solutions"""
        compromises = []
        
        # Analyze different vote values
        vote_values = list(set(vote.vote for vote in votes))
        
        if len(vote_values) == 2 and all(isinstance(v, bool) for v in vote_values):
            compromises.append("Conditional approval with monitoring")
            compromises.append("Phased implementation with review points")
            compromises.append("Pilot program to gather more evidence")
        
        return compromises

    def _suggest_mediation_strategies(self, conflict_analysis: ConflictAnalysis, 
                                    votes: List[AgentVote]) -> List[str]:
        """Suggest strategies for mediating the conflict"""
        strategies = []
        
        # Based on conflict patterns
        if ConflictPattern.EXPERTISE_GAP in conflict_analysis.patterns_detected:
            strategies.append("Bring in additional domain experts")
        
        if ConflictPattern.EVIDENCE_QUALITY in conflict_analysis.patterns_detected:
            strategies.append("Require higher evidence standards")
        
        if ConflictPattern.UNCERTAIN in conflict_analysis.patterns_detected:
            strategies.append("Gather additional information before deciding")
        
        return strategies

    def _assess_technical_accuracy(self, vote: AgentVote, domain: str, context: Dict[str, Any]) -> float:
        """Assess technical accuracy of vote in domain context"""
        # Simplified technical accuracy assessment
        # In production, would use domain-specific validation
        
        agent_expertise = self.consensus.agent_expertise.get(vote.agent_id, {}).get(domain, 0.5)
        evidence_quality = vote.evidence_quality
        
        # Technical accuracy correlates with expertise and evidence
        accuracy = (agent_expertise * 0.6) + (evidence_quality * 0.4)
        return accuracy

    def _evaluate_implementation_feasibility(self, vote: AgentVote, context: Dict[str, Any]) -> float:
        """Evaluate feasibility of implementing the vote's position"""
        # Simplified feasibility assessment
        # Would consider resource constraints, technical complexity, etc.
        
        complexity = context.get("complexity", "medium")
        complexity_scores = {"low": 0.9, "medium": 0.7, "high": 0.5, "very_high": 0.3}
        
        return complexity_scores.get(complexity, 0.7)

    def _check_best_practice_alignment(self, vote: AgentVote, domain: str) -> float:
        """Check alignment with domain best practices"""
        # Simplified best practice checking
        # Would use domain-specific best practice rules
        
        reasoning = vote.reasoning.lower()
        best_practice_indicators = ["standard", "best practice", "guideline", "protocol", "established"]
        
        indicator_count = sum(1 for indicator in best_practice_indicators if indicator in reasoning)
        return min(1.0, indicator_count / 2.0)

    def _analyze_decision_option_risks(self, option: Any, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze risks for specific decision option"""
        risks = {
            "technical_risk": 0.5,
            "business_risk": 0.5,
            "security_risk": 0.5,
            "compliance_risk": 0.5
        }
        
        # Adjust based on context
        if context.get("security_critical", False):
            if isinstance(option, bool) and option:  # Approving security-critical decision
                risks["security_risk"] = 0.3  # Lower risk if approved after security review
            else:
                risks["security_risk"] = 0.7  # Higher risk if rejected
        
        risk_level = context.get("risk_level", "medium")
        risk_multipliers = {"low": 0.8, "medium": 1.0, "high": 1.3, "critical": 1.6}
        multiplier = risk_multipliers.get(risk_level, 1.0)
        
        for risk_type in risks:
            risks[risk_type] = min(1.0, risks[risk_type] * multiplier)
        
        return risks

    def _identify_risk_factors(self, conflict_analysis: ConflictAnalysis, context: Dict[str, Any]) -> List[str]:
        """Identify general risk factors in the decision"""
        risk_factors = []
        
        if conflict_analysis.severity.value in ["high", "critical"]:
            risk_factors.append("High-severity conflict increases decision risk")
        
        if ConflictPattern.EVIDENCE_QUALITY in conflict_analysis.patterns_detected:
            risk_factors.append("Poor evidence quality increases uncertainty")
        
        if context.get("security_critical", False):
            risk_factors.append("Security-critical decision requires high confidence")
        
        return risk_factors

    def _generate_risk_mitigation_strategies(self, decision_risks: Dict[str, Dict[str, float]], 
                                           context: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        # General mitigation strategies
        strategies.append("Implement monitoring and review mechanisms")
        strategies.append("Define clear rollback procedures")
        strategies.append("Establish success criteria and checkpoints")
        
        # Context-specific strategies
        if context.get("security_critical", False):
            strategies.append("Implement additional security validation")
        
        return strategies

    def _integrate_technique_result(self, analysis: ArbitratorAnalysis, 
                                  skill: ArbitratorSkill, result: Dict[str, Any]):
        """Integrate technique result into overall analysis"""
        if skill == ArbitratorSkill.EVIDENCE_ANALYSIS:
            analysis.evidence_evaluation = result
        elif skill == ArbitratorSkill.REASONING_EVALUATION:
            analysis.reasoning_evaluation = result
        elif skill == ArbitratorSkill.BIAS_DETECTION:
            analysis.bias_assessment = result
        elif skill == ArbitratorSkill.CONFLICT_MEDIATION:
            analysis.consensus_opportunities = result.get("bridging_opportunities", [])
        # Add other integrations as needed

    def _synthesize_analysis(self, analysis: ArbitratorAnalysis, 
                           votes: List[AgentVote], context: Dict[str, Any]) -> ArbitratorAnalysis:
        """Synthesize all analysis components into cohesive assessment"""
        
        # Calculate overall decision factors
        decision_factors = {}
        
        # Evidence factor
        if analysis.evidence_evaluation:
            evidence_scores = analysis.evidence_evaluation.get("individual_quality", {})
            avg_evidence = sum(item.get("quality_score", 0.5) for item in evidence_scores.values()) / len(evidence_scores) if evidence_scores else 0.5
            decision_factors["evidence_quality"] = avg_evidence
        
        # Reasoning factor
        if analysis.reasoning_evaluation:
            depth_scores = analysis.reasoning_evaluation.get("depth_scores", {})
            avg_reasoning = sum(depth_scores.values()) / len(depth_scores) if depth_scores else 0.5
            decision_factors["reasoning_depth"] = avg_reasoning
        
        # Bias factor (inverse - less bias is better)
        if analysis.bias_assessment:
            bias_scores = analysis.bias_assessment.get("agent_bias_scores", {})
            avg_bias = sum(bias_scores.values()) / len(bias_scores) if bias_scores else 0.0
            decision_factors["bias_absence"] = max(0.0, 1.0 - avg_bias)
        
        analysis.decision_factors = decision_factors
        
        # Generate overall recommendations
        recommendations = []
        
        if decision_factors.get("evidence_quality", 0.5) < 0.6:
            recommendations.append("Consider gathering additional evidence")
        
        if decision_factors.get("reasoning_depth", 0.5) < 0.5:
            recommendations.append("Request more detailed reasoning from agents")
        
        if analysis.bias_assessment.get("potential_biases"):
            recommendations.append("Address potential cognitive biases before finalizing")
        
        analysis.recommendations = recommendations
        
        return analysis

    def _calculate_decision_factors(self, analysis: ArbitratorAnalysis, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate decision factors based on comprehensive analysis"""
        factors = analysis.decision_factors.copy()
        
        # Apply configuration weights
        weighted_factors = {}
        
        if "evidence_quality" in factors:
            weighted_factors["evidence"] = factors["evidence_quality"] * self.config.evidence_weight
        
        if "reasoning_depth" in factors:
            weighted_factors["reasoning"] = factors["reasoning_depth"] * self.config.reasoning_weight
        
        # Add expertise consideration
        domain = context.get("domain", "general")
        expertise_factor = 0.0
        vote_count = 0
        
        # Would calculate based on participating agents' expertise
        # Simplified here
        weighted_factors["expertise"] = 0.7 * self.config.expertise_weight
        
        return weighted_factors

    def _determine_final_decision(self, votes: List[AgentVote], decision_factors: Dict[str, float],
                                analysis: ArbitratorAnalysis) -> Any:
        """Determine final decision using weighted analysis"""
        
        # Score each unique vote option
        vote_options = list(set(vote.vote for vote in votes))
        option_scores = {}
        
        for option in vote_options:
            score = 0.0
            supporting_votes = [v for v in votes if v.vote == option]
            
            # Base score from vote count
            score += len(supporting_votes) / len(votes) * 0.3
            
            # Evidence quality score
            if analysis.evidence_evaluation:
                evidence_scores = analysis.evidence_evaluation.get("individual_quality", {})
                avg_evidence = sum(
                    evidence_scores.get(v.agent_id, {}).get("quality_score", 0.5) 
                    for v in supporting_votes
                ) / len(supporting_votes) if supporting_votes else 0.5
                score += avg_evidence * decision_factors.get("evidence", 0.0)
            
            # Reasoning quality score
            if analysis.reasoning_evaluation:
                reasoning_scores = analysis.reasoning_evaluation.get("depth_scores", {})
                avg_reasoning = sum(
                    reasoning_scores.get(v.agent_id, 0.5) 
                    for v in supporting_votes
                ) / len(supporting_votes) if supporting_votes else 0.5
                score += avg_reasoning * decision_factors.get("reasoning", 0.0)
            
            # Expertise alignment score
            domain = "general"  # Would get from context
            expertise_scores = [
                self.consensus.agent_expertise.get(v.agent_id, {}).get(domain, 0.5)
                for v in supporting_votes
            ]
            avg_expertise = sum(expertise_scores) / len(expertise_scores) if expertise_scores else 0.5
            score += avg_expertise * decision_factors.get("expertise", 0.0)
            
            option_scores[option] = score
        
        # Return option with highest score
        return max(option_scores.keys(), key=lambda k: option_scores[k])

    def _calculate_arbitrator_confidence(self, final_decision: Any, 
                                       analysis: ArbitratorAnalysis,
                                       decision_factors: Dict[str, float]) -> float:
        """Calculate confidence in arbitration decision"""
        
        # Base confidence from decision factors
        factor_confidence = sum(decision_factors.values()) / max(1, len(decision_factors))
        
        # Adjust based on analysis quality
        analysis_quality = 0.8  # Base quality score
        
        if analysis.evidence_evaluation:
            consistency = analysis.evidence_evaluation.get("consistency_score", 0.7)
            analysis_quality = (analysis_quality + consistency) / 2
        
        # Adjust for bias concerns
        if analysis.bias_assessment.get("potential_biases"):
            bias_penalty = len(analysis.bias_assessment["potential_biases"]) * 0.1
            analysis_quality = max(0.0, analysis_quality - bias_penalty)
        
        # Combine factors
        confidence = (factor_confidence * 0.7) + (analysis_quality * 0.3)
        
        # Apply mode-specific adjustments
        if self.config.mode == ArbitratorMode.DECISIVE:
            confidence = min(1.0, confidence * 1.1)  # Boost confidence for decisive mode
        elif self.config.mode == ArbitratorMode.ANALYTICAL:
            confidence = max(0.0, confidence - 0.05)  # Slight penalty for high analytical standards
        
        return min(1.0, max(0.0, confidence))

    def _generate_arbitrator_reasoning(self, final_decision: Any,
                                     analysis: ArbitratorAnalysis,
                                     decision_factors: Dict[str, float]) -> str:
        """Generate comprehensive reasoning for arbitration decision"""
        
        reasoning_parts = []
        
        # Introduction
        reasoning_parts.append(
            f"After comprehensive analysis using {len(self.config.skills)} specialized arbitration skills, "
            f"I have determined that the optimal decision is: {final_decision}"
        )
        
        # Evidence analysis
        if analysis.evidence_evaluation:
            evidence_quality = decision_factors.get("evidence", 0.0)
            reasoning_parts.append(
                f"Evidence analysis (weight: {evidence_quality:.2f}) indicates "
                f"{'strong' if evidence_quality > 0.7 else 'moderate' if evidence_quality > 0.4 else 'weak'} "
                f"evidentiary support for this position."
            )
        
        # Reasoning evaluation
        if analysis.reasoning_evaluation:
            reasoning_quality = decision_factors.get("reasoning", 0.0)
            reasoning_parts.append(
                f"Reasoning evaluation shows "
                f"{'high-quality' if reasoning_quality > 0.7 else 'adequate' if reasoning_quality > 0.4 else 'limited'} "
                f"logical argumentation supporting this decision."
            )
        
        # Bias considerations
        if analysis.bias_assessment.get("potential_biases"):
            bias_count = len(analysis.bias_assessment["potential_biases"])
            reasoning_parts.append(
                f"Bias analysis identified {bias_count} potential cognitive bias(es), "
                f"which have been factored into the decision process."
            )
        
        # Risk considerations
        risk_factors = analysis.metadata.get("risk_factors", [])
        if risk_factors:
            reasoning_parts.append(
                f"Risk assessment considered {len(risk_factors)} key factors, "
                f"and appropriate mitigation strategies are recommended."
            )
        
        # Mode-specific reasoning
        if self.config.mode == ArbitratorMode.CONSENSUS_BUILDING:
            reasoning_parts.append(
                "Special attention was given to identifying common ground and "
                "opportunities for consensus building among the conflicting positions."
            )
        elif self.config.mode == ArbitratorMode.ANALYTICAL:
            reasoning_parts.append(
                "Deep analytical evaluation was performed across multiple dimensions "
                "to ensure the most technically sound decision."
            )
        
        return " ".join(reasoning_parts)

    def _create_supporting_evidence_summary(self, analysis: ArbitratorAnalysis,
                                          decision_factors: Dict[str, float]) -> Dict[str, Any]:
        """Create summary of supporting evidence for decision"""
        
        evidence_summary = {
            "arbitrator_config": {
                "arbitrator_id": self.config.arbitrator_id,
                "mode": self.config.mode.value,
                "skills_applied": [skill.value for skill in self.config.skills]
            },
            "analysis_summary": {
                "decision_factors": decision_factors,
                "evidence_evaluation": analysis.evidence_evaluation,
                "reasoning_evaluation": analysis.reasoning_evaluation,
                "bias_assessment": analysis.bias_assessment
            },
            "confidence_breakdown": analysis.confidence_breakdown,
            "recommendations": analysis.recommendations
        }
        
        return evidence_summary

    def _confidence_to_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numerical score"""
        confidence_map = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 1.0
        }
        return confidence_map.get(confidence, 0.5)

    def _update_performance_metrics(self, decision: ArbitrationDecision, analysis_time: float):
        """Update arbitrator performance metrics"""
        self.success_metrics["total_arbitrations"] += 1
        
        # Update average analysis time
        total_arbitrations = self.success_metrics["total_arbitrations"]
        current_avg = self.success_metrics["average_analysis_time"]
        self.success_metrics["average_analysis_time"] = (
            (current_avg * (total_arbitrations - 1) + analysis_time) / total_arbitrations
        )
        
        # Assess success (would need feedback mechanism in production)
        if decision.confidence_score >= self.config.confidence_threshold:
            self.success_metrics["successful_resolutions"] += 1

    def get_arbitrator_metrics(self) -> Dict[str, Any]:
        """Get arbitrator performance metrics"""
        return {
            **self.success_metrics,
            "configuration": {
                "arbitrator_id": self.config.arbitrator_id,
                "mode": self.config.mode.value,
                "skills": [skill.value for skill in self.config.skills],
                "confidence_threshold": self.config.confidence_threshold
            },
            "arbitration_history_count": len(self.arbitration_history)
        }


def main():
    """Demonstration of arbitrator agent functionality"""
    print("=== RIF Arbitrator Agent Demo ===\n")
    
    # Initialize components
    from consensus_architecture import ConsensusArchitecture
    from .conflict_detector import ConflictDetector
    
    consensus = ConsensusArchitecture()
    detector = ConflictDetector(consensus.agent_expertise)
    
    # Create arbitrator configuration
    config = ArbitratorConfig(
        arbitrator_id="arb-demo-001",
        mode=ArbitratorMode.BALANCED,
        skills=[
            ArbitratorSkill.EVIDENCE_ANALYSIS,
            ArbitratorSkill.REASONING_EVALUATION,
            ArbitratorSkill.BIAS_DETECTION,
            ArbitratorSkill.CONFLICT_MEDIATION
        ],
        expertise_domains=["implementation", "security", "quality"],
        confidence_threshold=0.75,
        max_analysis_time_minutes=30
    )
    
    # Initialize arbitrator
    arbitrator = ArbitratorAgent(config, consensus)
    
    print(f"Arbitrator {config.arbitrator_id} initialized")
    print(f"Mode: {config.mode.value}")
    print(f"Skills: {[skill.value for skill in config.skills]}")
    print()
    
    # Example arbitration scenario
    print("Example: Complex Technical Conflict")
    votes = [
        consensus.create_vote("rif-implementer", True, ConfidenceLevel.HIGH, 
                            "Implementation is technically sound and follows best practices", 0.8),
        consensus.create_vote("rif-security", False, ConfidenceLevel.VERY_HIGH,
                            "Critical security vulnerability detected in authentication module", 0.95),
        consensus.create_vote("rif-validator", True, ConfidenceLevel.MEDIUM,
                            "Basic functionality tests pass, but security tests not comprehensive", 0.6)
    ]
    
    context = {
        "domain": "security",
        "security_critical": True,
        "risk_level": "high",
        "complexity": "medium"
    }
    
    # Analyze conflicts
    conflict_analysis = detector.analyze_conflicts(votes, context)
    print(f"Conflict detected: {conflict_analysis.severity.value} severity")
    print(f"Patterns: {[p.value for p in conflict_analysis.patterns_detected]}")
    
    # Perform arbitration
    decision = arbitrator.arbitrate_conflict(votes, conflict_analysis, context)
    
    print(f"\nArbitration Decision:")
    print(f"Final Decision: {decision.final_decision}")
    print(f"Confidence: {decision.confidence_score:.2f}")
    print(f"Reasoning: {decision.reasoning[:200]}...")
    print(f"Supporting Evidence Keys: {list(decision.supporting_evidence.keys())}")
    
    # Show arbitrator metrics
    print(f"\n=== Arbitrator Metrics ===")
    metrics = arbitrator.get_arbitrator_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict) and len(value) <= 10:
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif not isinstance(value, dict):
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()