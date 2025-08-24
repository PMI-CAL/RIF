#!/usr/bin/env python3
"""
RIF Expertise Scorer - Agent Domain Expertise Evaluation
Part of Issue #62 implementation for vote weighting algorithm.

This module provides sophisticated expertise scoring that considers:
- Domain-specific knowledge assessment
- Multi-disciplinary expertise bonuses
- Expertise aging and decay over time
- Dynamic learning and skill updates
- Cross-domain knowledge transfer
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertiseDomain(Enum):
    """Standard expertise domains in RIF system"""
    SECURITY = "security"
    TESTING = "testing" 
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    GENERAL = "general"


@dataclass
class ExpertiseEvidence:
    """Evidence supporting expertise level in a domain"""
    evidence_type: str  # 'successful_decision', 'code_review', 'system_design', etc.
    domain: str
    quality_score: float  # 0.0 - 1.0
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    context_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertiseAssessment:
    """Comprehensive expertise assessment for an agent"""
    agent_id: str
    domain: str
    current_level: float  # 0.0 - 1.0
    confidence_interval: Tuple[float, float]
    evidence_count: int
    last_assessment: datetime
    learning_trajectory: float  # Rate of expertise change
    peer_validation_score: float = 0.0
    certification_level: str = "none"


@dataclass
class CrossDomainSynergy:
    """Represents synergy between different expertise domains"""
    primary_domain: str
    secondary_domain: str
    synergy_coefficient: float  # Multiplier for combined expertise
    evidence_examples: List[str] = field(default_factory=list)


class ExpertiseScorer:
    """
    Advanced expertise scoring system for RIF agents.
    
    This system evaluates agent expertise across multiple dimensions:
    - Direct domain competency assessment
    - Evidence-based scoring with quality weighting
    - Time-decay modeling for skill currency
    - Cross-domain synergy identification
    - Peer validation and consensus scoring
    - Learning trajectory analysis
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize expertise scorer"""
        self.knowledge_system = knowledge_system
        
        # Agent expertise data
        self.agent_expertise: Dict[str, Dict[str, float]] = {}
        self.expertise_evidence: Dict[str, List[ExpertiseEvidence]] = {}
        self.assessment_history: Dict[str, List[ExpertiseAssessment]] = {}
        
        # Domain relationship mappings
        self.domain_synergies = self._initialize_domain_synergies()
        
        # Scoring parameters
        self.scoring_params = {
            'evidence_decay_rate': 0.95,  # Monthly decay for evidence relevance
            'learning_momentum': 0.1,     # Weight given to learning trends
            'peer_validation_weight': 0.15,  # Weight of peer validation
            'min_evidence_threshold': 3,  # Minimum evidence pieces for reliable score
            'expertise_ceiling': 1.0,     # Maximum possible expertise level
            'expertise_floor': 0.0,       # Minimum expertise level
            'cross_domain_bonus_cap': 0.3  # Maximum bonus from cross-domain synergy
        }
        
        # Performance tracking
        self.scoring_metrics = {
            'total_assessments': 0,
            'average_confidence_interval_width': 0.0,
            'domain_coverage_stats': {},
            'evidence_quality_distribution': {}
        }
        
        # Load existing data
        self._load_expertise_history()
        
        logger.info("Expertise Scorer initialized")
    
    def assess_agent_expertise(self, agent_id: str, domain: str) -> ExpertiseAssessment:
        """
        Assess an agent's expertise level in a specific domain.
        
        Args:
            agent_id: Agent identifier
            domain: Domain to assess (e.g., 'security', 'testing')
            
        Returns:
            ExpertiseAssessment with detailed evaluation
        """
        try:
            # Gather evidence for this agent-domain combination
            evidence_pieces = self._gather_domain_evidence(agent_id, domain)
            
            # Calculate base expertise level
            base_level = self._calculate_base_expertise(evidence_pieces)
            
            # Apply time decay for evidence currency
            decayed_level = self._apply_time_decay(base_level, evidence_pieces)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                decayed_level, evidence_pieces
            )
            
            # Assess learning trajectory
            learning_trajectory = self._assess_learning_trajectory(agent_id, domain)
            
            # Apply learning momentum
            adjusted_level = self._apply_learning_momentum(
                decayed_level, learning_trajectory
            )
            
            # Get peer validation score
            peer_validation = self._get_peer_validation_score(agent_id, domain)
            
            # Final expertise level calculation
            final_level = self._calculate_final_expertise_level(
                adjusted_level, peer_validation, agent_id, domain
            )
            
            # Create assessment
            assessment = ExpertiseAssessment(
                agent_id=agent_id,
                domain=domain,
                current_level=final_level,
                confidence_interval=confidence_interval,
                evidence_count=len(evidence_pieces),
                last_assessment=datetime.now(),
                learning_trajectory=learning_trajectory,
                peer_validation_score=peer_validation,
                certification_level=self._determine_certification_level(final_level)
            )
            
            # Store assessment
            self._store_assessment(assessment)
            
            # Update metrics
            self._update_scoring_metrics(assessment)
            
            logger.debug(f"Assessed {agent_id} expertise in {domain}: {final_level:.3f}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing expertise for {agent_id} in {domain}: {str(e)}")
            return self._create_fallback_assessment(agent_id, domain)
    
    def calculate_domain_expertise_factor(self, agent_id: str, domain: str) -> float:
        """
        Calculate expertise factor for vote weighting.
        
        This is the main interface used by VoteWeightCalculator.
        
        Args:
            agent_id: Agent identifier
            domain: Domain of expertise
            
        Returns:
            Expertise factor (0.0 - 2.0 range for weight calculation)
        """
        # Get current assessment
        assessment = self.assess_agent_expertise(agent_id, domain)
        
        # Convert expertise level to factor
        base_factor = assessment.current_level
        
        # Apply cross-domain bonuses
        cross_domain_bonus = self._calculate_cross_domain_bonus(agent_id, domain)
        
        # Apply confidence weighting
        confidence_weight = self._calculate_confidence_weight(assessment.confidence_interval)
        
        # Calculate final factor
        expertise_factor = (base_factor + cross_domain_bonus) * confidence_weight
        
        # Scale to appropriate range (0.3 - 2.0)
        scaled_factor = 0.3 + (expertise_factor * 1.7)
        
        return min(2.0, max(0.3, scaled_factor))
    
    def update_expertise_evidence(self, agent_id: str, domain: str, 
                                evidence: ExpertiseEvidence) -> None:
        """
        Add new evidence for an agent's expertise in a domain.
        
        Args:
            agent_id: Agent identifier
            domain: Expertise domain
            evidence: Evidence supporting expertise level
        """
        if agent_id not in self.expertise_evidence:
            self.expertise_evidence[agent_id] = []
        
        # Add domain to evidence if not present
        evidence.domain = domain
        evidence.timestamp = datetime.now()
        
        self.expertise_evidence[agent_id].append(evidence)
        
        # Maintain reasonable evidence history (last 100 pieces per agent)
        if len(self.expertise_evidence[agent_id]) > 100:
            # Keep most recent and highest quality evidence
            evidence_list = self.expertise_evidence[agent_id]
            evidence_list.sort(key=lambda e: (e.timestamp, e.quality_score), reverse=True)
            self.expertise_evidence[agent_id] = evidence_list[:100]
        
        logger.debug(f"Added expertise evidence for {agent_id} in {domain}")
    
    def get_agent_expertise_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive expertise profile for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with expertise levels across all domains
        """
        profile = {
            'agent_id': agent_id,
            'domain_expertise': {},
            'total_evidence_count': 0,
            'specialization_areas': [],
            'learning_trends': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Assess expertise in all relevant domains
        for domain_enum in ExpertiseDomain:
            domain = domain_enum.value
            assessment = self.assess_agent_expertise(agent_id, domain)
            
            profile['domain_expertise'][domain] = {
                'level': assessment.current_level,
                'confidence_interval': assessment.confidence_interval,
                'evidence_count': assessment.evidence_count,
                'learning_trajectory': assessment.learning_trajectory,
                'certification': assessment.certification_level
            }
            
            profile['total_evidence_count'] += assessment.evidence_count
        
        # Identify specialization areas (>0.8 expertise)
        profile['specialization_areas'] = [
            domain for domain, data in profile['domain_expertise'].items()
            if data['level'] > 0.8
        ]
        
        # Calculate learning trends
        profile['learning_trends'] = self._calculate_learning_trends(agent_id)
        
        return profile
    
    def _gather_domain_evidence(self, agent_id: str, domain: str) -> List[ExpertiseEvidence]:
        """Gather all evidence for an agent's expertise in a domain"""
        if agent_id not in self.expertise_evidence:
            return []
        
        # Filter evidence for the specific domain
        domain_evidence = [
            evidence for evidence in self.expertise_evidence[agent_id]
            if evidence.domain == domain
        ]
        
        # Sort by recency and quality
        domain_evidence.sort(key=lambda e: (e.timestamp, e.quality_score), reverse=True)
        
        return domain_evidence
    
    def _calculate_base_expertise(self, evidence_pieces: List[ExpertiseEvidence]) -> float:
        """Calculate base expertise level from evidence"""
        if not evidence_pieces:
            return 0.5  # Neutral baseline for agents with no evidence
        
        # Weight evidence by quality and impact
        weighted_scores = []
        
        for evidence in evidence_pieces:
            # Base score from evidence quality
            base_score = evidence.quality_score
            
            # Impact level multiplier
            impact_multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5,
                'critical': 2.0
            }
            impact_multiplier = impact_multipliers.get(evidence.impact_level, 1.0)
            
            # Evidence type weighting
            type_weights = {
                'successful_decision': 1.2,
                'code_review': 1.0,
                'system_design': 1.3,
                'problem_solving': 1.1,
                'knowledge_sharing': 0.8,
                'certification': 1.5
            }
            type_weight = type_weights.get(evidence.evidence_type, 1.0)
            
            weighted_score = base_score * impact_multiplier * type_weight
            weighted_scores.append(min(1.0, weighted_score))  # Cap at 1.0
        
        # Use weighted average with diminishing returns for quantity
        if len(weighted_scores) <= 5:
            base_expertise = statistics.mean(weighted_scores)
        else:
            # Apply diminishing returns for excessive evidence
            top_scores = sorted(weighted_scores, reverse=True)[:5]
            remaining_scores = weighted_scores[5:]
            
            primary_score = statistics.mean(top_scores)
            secondary_score = statistics.mean(remaining_scores) if remaining_scores else 0
            
            base_expertise = primary_score * 0.8 + secondary_score * 0.2
        
        return min(1.0, max(0.0, base_expertise))
    
    def _apply_time_decay(self, base_level: float, 
                         evidence_pieces: List[ExpertiseEvidence]) -> float:
        """Apply time-based decay to expertise based on evidence currency"""
        if not evidence_pieces:
            return base_level
        
        current_time = datetime.now()
        decay_rate = self.scoring_params['evidence_decay_rate']
        
        # Calculate weighted expertise considering evidence age
        weighted_expertise = 0.0
        total_weight = 0.0
        
        for evidence in evidence_pieces:
            months_old = (current_time - evidence.timestamp).days / 30.0
            decay_factor = decay_rate ** months_old
            
            evidence_weight = evidence.quality_score * decay_factor
            weighted_expertise += base_level * evidence_weight
            total_weight += evidence_weight
        
        if total_weight == 0:
            return base_level * (decay_rate ** 6)  # 6-month default decay
        
        return weighted_expertise / total_weight
    
    def _calculate_confidence_interval(self, expertise_level: float, 
                                     evidence_pieces: List[ExpertiseEvidence]) -> Tuple[float, float]:
        """Calculate confidence interval for expertise assessment"""
        if len(evidence_pieces) < self.scoring_params['min_evidence_threshold']:
            # Wide interval for insufficient evidence
            margin = 0.3
        else:
            # Calculate margin based on evidence consistency
            quality_scores = [e.quality_score for e in evidence_pieces]
            quality_variance = statistics.variance(quality_scores) if len(quality_scores) > 1 else 0.1
            
            # More consistent evidence = narrower confidence interval
            margin = min(0.3, max(0.05, quality_variance * 0.5))
        
        lower_bound = max(0.0, expertise_level - margin)
        upper_bound = min(1.0, expertise_level + margin)
        
        return (lower_bound, upper_bound)
    
    def _assess_learning_trajectory(self, agent_id: str, domain: str) -> float:
        """Assess agent's learning trajectory in a domain"""
        if agent_id not in self.assessment_history:
            return 0.0
        
        # Get historical assessments for this domain
        domain_assessments = [
            assessment for assessment in self.assessment_history[agent_id]
            if assessment.domain == domain
        ]
        
        if len(domain_assessments) < 2:
            return 0.0
        
        # Sort by date
        domain_assessments.sort(key=lambda a: a.last_assessment)
        
        # Calculate trajectory from recent assessments
        recent_assessments = domain_assessments[-5:]  # Last 5 assessments
        
        if len(recent_assessments) < 2:
            return 0.0
        
        # Linear regression on expertise levels over time
        x_values = [(a.last_assessment - recent_assessments[0].last_assessment).days 
                   for a in recent_assessments]
        y_values = [a.current_level for a in recent_assessments]
        
        if len(set(x_values)) < 2:  # All assessments on same day
            return 0.0
        
        # Simple linear regression
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize slope to reasonable range (-0.5 to 0.5)
        return max(-0.5, min(0.5, slope))
    
    def _apply_learning_momentum(self, current_level: float, trajectory: float) -> float:
        """Apply learning trajectory to current expertise level"""
        momentum_weight = self.scoring_params['learning_momentum']
        adjustment = trajectory * momentum_weight
        
        adjusted_level = current_level + adjustment
        return min(1.0, max(0.0, adjusted_level))
    
    def _get_peer_validation_score(self, agent_id: str, domain: str) -> float:
        """Get peer validation score (simulated based on evidence quality)"""
        # In a real implementation, this would aggregate peer reviews
        # For now, simulate based on evidence quality and consistency
        
        evidence = self._gather_domain_evidence(agent_id, domain)
        if not evidence:
            return 0.5
        
        # Use average evidence quality as proxy for peer validation
        avg_quality = statistics.mean([e.quality_score for e in evidence])
        
        # Add some variability based on evidence count
        evidence_bonus = min(0.2, len(evidence) * 0.02)
        
        peer_score = avg_quality + evidence_bonus
        return min(1.0, max(0.0, peer_score))
    
    def _calculate_final_expertise_level(self, adjusted_level: float, 
                                       peer_validation: float,
                                       agent_id: str, domain: str) -> float:
        """Calculate final expertise level incorporating all factors"""
        peer_weight = self.scoring_params['peer_validation_weight']
        
        # Blend adjusted level with peer validation
        final_level = (
            adjusted_level * (1 - peer_weight) + 
            peer_validation * peer_weight
        )
        
        # Apply cross-domain learning bonus
        cross_domain_bonus = self._calculate_cross_domain_learning_bonus(agent_id, domain)
        final_level += cross_domain_bonus
        
        return min(1.0, max(0.0, final_level))
    
    def _calculate_cross_domain_bonus(self, agent_id: str, domain: str) -> float:
        """Calculate bonus from cross-domain expertise synergies"""
        bonus = 0.0
        max_bonus = self.scoring_params['cross_domain_bonus_cap']
        
        # Get agent's expertise in other domains (avoid recursion)
        other_domains = {}
        for domain_enum in ExpertiseDomain:
            other_domain = domain_enum.value
            if other_domain != domain:
                # Get assessment without calling get_agent_expertise_profile to avoid recursion
                evidence = self._gather_domain_evidence(agent_id, other_domain)
                if evidence:
                    level = self._calculate_base_expertise(evidence)
                    other_domains[other_domain] = level
        
        # Find synergistic domains
        for synergy in self.domain_synergies:
            if synergy.primary_domain == domain or synergy.secondary_domain == domain:
                other_domain = (synergy.secondary_domain if synergy.primary_domain == domain 
                              else synergy.primary_domain)
                
                if other_domain in other_domains:
                    other_expertise = other_domains[other_domain]
                    synergy_bonus = other_expertise * synergy.synergy_coefficient * 0.1
                    bonus += synergy_bonus
        
        return min(max_bonus, bonus)
    
    def _calculate_cross_domain_learning_bonus(self, agent_id: str, domain: str) -> float:
        """Calculate learning bonus from related domains"""
        # Similar to cross-domain bonus but focused on learning transfer
        return self._calculate_cross_domain_bonus(agent_id, domain) * 0.3
    
    def _calculate_confidence_weight(self, confidence_interval: Tuple[float, float]) -> float:
        """Convert confidence interval to weight multiplier"""
        lower, upper = confidence_interval
        interval_width = upper - lower
        
        # Narrower confidence interval = higher weight
        # Wide interval (>0.4) gets 0.8x weight, narrow (<0.1) gets 1.1x weight
        if interval_width > 0.4:
            return 0.8
        elif interval_width < 0.1:
            return 1.1
        else:
            return 1.0 - (interval_width - 0.1) * 0.67  # Linear interpolation
    
    def _determine_certification_level(self, expertise_level: float) -> str:
        """Determine certification level based on expertise"""
        if expertise_level >= 0.95:
            return "expert"
        elif expertise_level >= 0.85:
            return "advanced"
        elif expertise_level >= 0.70:
            return "intermediate"
        elif expertise_level >= 0.50:
            return "basic"
        else:
            return "none"
    
    def _store_assessment(self, assessment: ExpertiseAssessment) -> None:
        """Store assessment in history"""
        agent_id = assessment.agent_id
        
        if agent_id not in self.assessment_history:
            self.assessment_history[agent_id] = []
        
        self.assessment_history[agent_id].append(assessment)
        
        # Maintain reasonable history size
        if len(self.assessment_history[agent_id]) > 50:
            self.assessment_history[agent_id] = self.assessment_history[agent_id][-50:]
    
    def _update_scoring_metrics(self, assessment: ExpertiseAssessment) -> None:
        """Update scoring performance metrics"""
        self.scoring_metrics['total_assessments'] += 1
        
        # Update confidence interval width tracking
        interval_width = assessment.confidence_interval[1] - assessment.confidence_interval[0]
        total_assessments = self.scoring_metrics['total_assessments']
        current_avg_width = self.scoring_metrics['average_confidence_interval_width']
        
        self.scoring_metrics['average_confidence_interval_width'] = (
            (current_avg_width * (total_assessments - 1) + interval_width) / total_assessments
        )
        
        # Update domain coverage
        domain = assessment.domain
        if domain not in self.scoring_metrics['domain_coverage_stats']:
            self.scoring_metrics['domain_coverage_stats'][domain] = 0
        self.scoring_metrics['domain_coverage_stats'][domain] += 1
    
    def _create_fallback_assessment(self, agent_id: str, domain: str) -> ExpertiseAssessment:
        """Create fallback assessment when error occurs"""
        return ExpertiseAssessment(
            agent_id=agent_id,
            domain=domain,
            current_level=0.5,
            confidence_interval=(0.2, 0.8),
            evidence_count=0,
            last_assessment=datetime.now(),
            learning_trajectory=0.0,
            certification_level="none"
        )
    
    def _initialize_domain_synergies(self) -> List[CrossDomainSynergy]:
        """Initialize domain synergy relationships"""
        synergies = [
            CrossDomainSynergy("security", "validation", 1.3),
            CrossDomainSynergy("architecture", "performance", 1.2),
            CrossDomainSynergy("implementation", "debugging", 1.4),
            CrossDomainSynergy("testing", "validation", 1.2),
            CrossDomainSynergy("analysis", "planning", 1.3),
            CrossDomainSynergy("security", "compliance", 1.5),
            CrossDomainSynergy("architecture", "integration", 1.2),
            CrossDomainSynergy("implementation", "integration", 1.1)
        ]
        
        return synergies
    
    def _calculate_learning_trends(self, agent_id: str) -> Dict[str, float]:
        """Calculate learning trends across all domains for an agent"""
        trends = {}
        
        for domain_enum in ExpertiseDomain:
            domain = domain_enum.value
            trajectory = self._assess_learning_trajectory(agent_id, domain)
            if trajectory != 0.0:
                trends[domain] = trajectory
        
        return trends
    
    def _load_expertise_history(self) -> None:
        """Load historical expertise data"""
        # This would load from knowledge system or database
        # For now, initialize with empty data
        pass
    
    def get_scoring_metrics(self) -> Dict[str, Any]:
        """Get current scoring performance metrics"""
        return {
            **self.scoring_metrics,
            'active_agents': len(self.expertise_evidence),
            'total_domains': len(ExpertiseDomain),
            'synergy_relationships': len(self.domain_synergies)
        }
    
    def export_expertise_data(self) -> Dict[str, Any]:
        """Export expertise data for persistence"""
        return {
            'agent_expertise': self.agent_expertise,
            'assessment_history': {
                agent: [
                    {
                        'domain': assessment.domain,
                        'current_level': assessment.current_level,
                        'confidence_interval': assessment.confidence_interval,
                        'evidence_count': assessment.evidence_count,
                        'last_assessment': assessment.last_assessment.isoformat(),
                        'learning_trajectory': assessment.learning_trajectory,
                        'certification_level': assessment.certification_level
                    }
                    for assessment in assessments
                ]
                for agent, assessments in self.assessment_history.items()
            },
            'scoring_metrics': self.scoring_metrics,
            'export_timestamp': datetime.now().isoformat()
        }


def main():
    """Demonstration of expertise scoring functionality"""
    print("=== RIF Expertise Scorer Demo ===\n")
    
    # Initialize scorer
    scorer = ExpertiseScorer()
    
    # Example 1: Add expertise evidence
    print("Example 1: Adding Expertise Evidence")
    
    evidence_examples = [
        ExpertiseEvidence(
            evidence_type="successful_decision",
            domain="security",
            quality_score=0.9,
            impact_level="high",
            timestamp=datetime.now(),
            context_metadata={"decision": "vulnerability_assessment", "success_rate": 0.95}
        ),
        ExpertiseEvidence(
            evidence_type="system_design",
            domain="architecture",
            quality_score=0.85,
            impact_level="critical",
            timestamp=datetime.now() - timedelta(days=30),
            context_metadata={"system": "consensus_architecture", "complexity": "high"}
        )
    ]
    
    for evidence in evidence_examples:
        scorer.update_expertise_evidence("rif-security", evidence.domain, evidence)
        print(f"Added {evidence.evidence_type} evidence for rif-security in {evidence.domain}")
    
    print()
    
    # Example 2: Assess expertise
    print("Example 2: Expertise Assessment")
    
    assessment = scorer.assess_agent_expertise("rif-security", "security")
    print(f"Security expertise for rif-security: {assessment.current_level:.3f}")
    print(f"Confidence interval: {assessment.confidence_interval}")
    print(f"Evidence count: {assessment.evidence_count}")
    print(f"Learning trajectory: {assessment.learning_trajectory:.3f}")
    print(f"Certification level: {assessment.certification_level}")
    
    print()
    
    # Example 3: Calculate expertise factor for vote weighting
    print("Example 3: Vote Weighting Factor")
    
    factor = scorer.calculate_domain_expertise_factor("rif-security", "security")
    print(f"Expertise factor for vote weighting: {factor:.3f}")
    
    print()
    
    # Example 4: Full agent profile
    print("Example 4: Agent Expertise Profile")
    
    profile = scorer.get_agent_expertise_profile("rif-security")
    print(f"Specialization areas: {profile['specialization_areas']}")
    print(f"Total evidence count: {profile['total_evidence_count']}")
    
    for domain, data in profile['domain_expertise'].items():
        if data['level'] > 0.1:  # Only show domains with some expertise
            print(f"{domain}: {data['level']:.2f} ({data['certification']})")
    
    print()
    
    # Show metrics
    print("=== Scoring Metrics ===")
    metrics = scorer.get_scoring_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()