"""
Pattern Ranker - Advanced Multi-Criteria Pattern Ranking System

This module implements sophisticated pattern ranking using multiple weighted criteria
to accurately rank patterns by their applicability to specific issue contexts.

Key Features:
- Multi-criteria decision analysis (MCDA)
- Weighted scoring with context adaptation
- Historical performance integration
- Technology stack compatibility ranking
- Complexity-aware ranking adjustments
- Success rate predictions
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import knowledge system interfaces
try:
    from knowledge.interface import get_knowledge_system
    from knowledge.database.database_interface import RIFDatabase
except ImportError:
    def get_knowledge_system():
        raise ImportError("Knowledge system not available")
    
    class RIFDatabase:
        pass

# Import pattern application core components
from knowledge.pattern_application.core import (
    Pattern, IssueContext, TechStack
)


@dataclass
class RankingCriteria:
    """Criteria used for pattern ranking."""
    semantic_similarity: float
    tech_compatibility: float
    complexity_alignment: float
    historical_success: float
    domain_relevance: float
    usage_frequency: float
    pattern_quality: float
    context_specificity: float


@dataclass
class RankingResult:
    """Result of pattern ranking analysis."""
    pattern: Pattern
    overall_score: float
    criteria_scores: RankingCriteria
    ranking_explanation: str
    confidence_level: str
    predicted_success_rate: float
    adaptation_complexity: str


class PatternRanker:
    """
    Advanced pattern ranking system using multi-criteria decision analysis.
    
    This system ranks patterns based on multiple weighted criteria:
    - Semantic similarity to issue description
    - Technology stack compatibility
    - Complexity alignment
    - Historical success rates
    - Domain relevance
    - Usage frequency and pattern maturity
    - Pattern quality indicators
    - Context-specific factors
    """
    
    def __init__(self, knowledge_system=None, database: Optional[RIFDatabase] = None):
        """Initialize the pattern ranker."""
        self.logger = logging.getLogger(__name__)
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self.database = database
        
        # Default ranking weights (can be adjusted based on context)
        self.default_weights = {
            'semantic_similarity': 0.25,
            'tech_compatibility': 0.20,
            'complexity_alignment': 0.15,
            'historical_success': 0.15,
            'domain_relevance': 0.10,
            'usage_frequency': 0.05,
            'pattern_quality': 0.05,
            'context_specificity': 0.05
        }
        
        # Context-based weight adjustments
        self.context_weight_adjustments = {
            'high_tech_specificity': {
                'tech_compatibility': 1.3,
                'semantic_similarity': 0.8
            },
            'high_complexity': {
                'complexity_alignment': 1.4,
                'historical_success': 1.2
            },
            'new_domain': {
                'domain_relevance': 0.7,
                'pattern_quality': 1.3
            },
            'critical_issue': {
                'historical_success': 1.4,
                'pattern_quality': 1.3,
                'usage_frequency': 1.2
            }
        }
        
        # Quality thresholds for filtering
        self.quality_thresholds = {
            'minimum_confidence': 0.3,
            'minimum_relevance': 0.2,
            'minimum_success_rate': 0.1
        }
        
        self.logger.info("Pattern Ranker initialized")
    
    def rank_patterns(self, patterns: List[Pattern], 
                     issue_context: IssueContext) -> List[Pattern]:
        """
        Rank patterns by applicability using multi-criteria analysis.
        
        This implements the core ranking requirement from Issue #76 acceptance
        criteria: "Ranks patterns accurately"
        
        Args:
            patterns: List of patterns to rank
            issue_context: Context for ranking
            
        Returns:
            List of patterns ranked by applicability (highest first)
        """
        self.logger.info(f"Ranking {len(patterns)} patterns for issue {issue_context.issue_id}")
        
        if not patterns:
            return []
        
        try:
            # Calculate ranking results for all patterns
            ranking_results = []
            
            for pattern in patterns:
                ranking_result = self.calculate_pattern_ranking(pattern, issue_context)
                
                # Filter out patterns below quality thresholds
                if self._meets_quality_thresholds(ranking_result):
                    ranking_results.append(ranking_result)
            
            # Sort by overall score (highest first)
            ranking_results.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Log ranking summary
            self.logger.info(f"Ranked {len(ranking_results)} patterns (filtered {len(patterns) - len(ranking_results)} below threshold)")
            
            # Return ranked patterns
            return [result.pattern for result in ranking_results]
            
        except Exception as e:
            self.logger.error(f"Error ranking patterns: {str(e)}")
            # Return original list as fallback
            return patterns
    
    def calculate_pattern_ranking(self, pattern: Pattern, 
                                issue_context: IssueContext) -> RankingResult:
        """
        Calculate comprehensive ranking for a pattern-context pair.
        
        Args:
            pattern: Pattern to rank
            issue_context: Issue context for ranking
            
        Returns:
            RankingResult with detailed scoring breakdown
        """
        # Calculate individual criteria scores
        criteria_scores = self._calculate_all_criteria_scores(pattern, issue_context)
        
        # Get context-adjusted weights
        weights = self._get_context_adjusted_weights(issue_context)
        
        # Calculate overall weighted score
        overall_score = self._calculate_weighted_score(criteria_scores, weights)
        
        # Generate ranking explanation
        explanation = self._generate_ranking_explanation(
            criteria_scores, weights, overall_score
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(criteria_scores, overall_score)
        
        # Predict success rate
        predicted_success_rate = self._predict_success_rate(
            pattern, issue_context, criteria_scores
        )
        
        # Assess adaptation complexity
        adaptation_complexity = self._assess_adaptation_complexity(
            pattern, issue_context, criteria_scores
        )
        
        return RankingResult(
            pattern=pattern,
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            ranking_explanation=explanation,
            confidence_level=confidence_level,
            predicted_success_rate=predicted_success_rate,
            adaptation_complexity=adaptation_complexity
        )
    
    def calculate_ranking_score(self, pattern: Pattern, 
                              issue_context: IssueContext) -> float:
        """
        Calculate simple ranking score for quick comparisons.
        
        Args:
            pattern: Pattern to score
            issue_context: Context for scoring
            
        Returns:
            Ranking score (0.0 to 1.0)
        """
        ranking_result = self.calculate_pattern_ranking(pattern, issue_context)
        return ranking_result.overall_score
    
    def _calculate_all_criteria_scores(self, pattern: Pattern, 
                                     issue_context: IssueContext) -> RankingCriteria:
        """Calculate scores for all ranking criteria."""
        
        # Semantic similarity
        semantic_score = self._calculate_semantic_similarity_score(pattern, issue_context)
        
        # Technology compatibility
        tech_score = self._calculate_tech_compatibility_score(pattern, issue_context)
        
        # Complexity alignment
        complexity_score = self._calculate_complexity_alignment_score(pattern, issue_context)
        
        # Historical success
        historical_score = self._calculate_historical_success_score(pattern, issue_context)
        
        # Domain relevance
        domain_score = self._calculate_domain_relevance_score(pattern, issue_context)
        
        # Usage frequency
        usage_score = self._calculate_usage_frequency_score(pattern)
        
        # Pattern quality
        quality_score = self._calculate_pattern_quality_score(pattern)
        
        # Context specificity
        context_score = self._calculate_context_specificity_score(pattern, issue_context)
        
        return RankingCriteria(
            semantic_similarity=semantic_score,
            tech_compatibility=tech_score,
            complexity_alignment=complexity_score,
            historical_success=historical_score,
            domain_relevance=domain_score,
            usage_frequency=usage_score,
            pattern_quality=quality_score,
            context_specificity=context_score
        )
    
    def _calculate_semantic_similarity_score(self, pattern: Pattern, 
                                           issue_context: IssueContext) -> float:
        """Calculate semantic similarity between pattern and issue."""
        # Import similarity engine for semantic analysis
        try:
            from .similarity_engine import SimilarityEngine
            similarity_engine = SimilarityEngine(self.knowledge_system, self.database)
            
            pattern_text = f"{pattern.name} {pattern.description}"
            issue_text = f"{issue_context.title} {issue_context.description}"
            
            return similarity_engine.calculate_semantic_similarity(pattern_text, issue_text)
        except Exception as e:
            self.logger.debug(f"Semantic similarity calculation failed: {str(e)}")
            return self._fallback_text_similarity(pattern, issue_context)
    
    def _calculate_tech_compatibility_score(self, pattern: Pattern, 
                                          issue_context: IssueContext) -> float:
        """Calculate technology stack compatibility score."""
        if not pattern.tech_stack or not issue_context.tech_stack:
            return 0.5  # Neutral score when no tech info
        
        compatibility_score = 0.0
        
        # Primary language compatibility (50%)
        lang_score = self._compare_languages(
            pattern.tech_stack.primary_language,
            issue_context.tech_stack.primary_language
        )
        compatibility_score += lang_score * 0.5
        
        # Framework compatibility (30%)
        framework_score = self._compare_frameworks(
            pattern.tech_stack.frameworks,
            issue_context.tech_stack.frameworks
        )
        compatibility_score += framework_score * 0.3
        
        # Database compatibility (10%)
        db_score = self._compare_databases(
            pattern.tech_stack.databases,
            issue_context.tech_stack.databases
        )
        compatibility_score += db_score * 0.1
        
        # Architecture compatibility (10%)
        arch_score = self._compare_architectures(
            pattern.tech_stack.architecture_pattern,
            issue_context.tech_stack.architecture_pattern
        )
        compatibility_score += arch_score * 0.1
        
        return compatibility_score
    
    def _calculate_complexity_alignment_score(self, pattern: Pattern, 
                                            issue_context: IssueContext) -> float:
        """Calculate how well pattern complexity aligns with issue complexity."""
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            pattern_idx = complexity_levels.index(pattern.complexity)
            issue_idx = complexity_levels.index(issue_context.complexity)
            
            # Perfect match
            if pattern_idx == issue_idx:
                return 1.0
            
            # Calculate distance-based score
            distance = abs(pattern_idx - issue_idx)
            if distance == 1:
                return 0.8  # Adjacent levels
            elif distance == 2:
                return 0.5  # Two levels apart
            else:
                return 0.2  # Very different complexity
                
        except ValueError:
            return 0.5  # Unknown complexity
    
    def _calculate_historical_success_score(self, pattern: Pattern, 
                                          issue_context: IssueContext) -> float:
        """Calculate score based on historical success of the pattern."""
        # Base score from pattern's success rate
        base_score = pattern.success_rate
        
        # Boost based on usage count (indicates proven track record)
        usage_boost = min(0.2, pattern.usage_count * 0.02)  # Max 20% boost
        
        # Boost based on confidence
        confidence_boost = pattern.confidence * 0.1  # Max 10% boost
        
        # Penalty for patterns with no validation criteria
        validation_penalty = 0.0 if pattern.validation_criteria else -0.1
        
        final_score = base_score + usage_boost + confidence_boost + validation_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_domain_relevance_score(self, pattern: Pattern, 
                                        issue_context: IssueContext) -> float:
        """Calculate domain relevance score."""
        if pattern.domain == issue_context.domain:
            return 1.0
        elif pattern.domain == 'general':
            return 0.7  # General patterns are broadly applicable
        elif issue_context.domain == 'general':
            return 0.6  # Issue in general domain can use specific patterns
        else:
            # Check for related domains
            related_domains = self._get_related_domains(pattern.domain, issue_context.domain)
            return 0.3 if related_domains else 0.0
    
    def _calculate_usage_frequency_score(self, pattern: Pattern) -> float:
        """Calculate score based on pattern usage frequency."""
        if pattern.usage_count == 0:
            return 0.0
        
        # Logarithmic scaling to avoid over-weighting very popular patterns
        import math
        normalized_score = math.log(pattern.usage_count + 1) / math.log(100)  # Log base 100
        
        return min(1.0, normalized_score)
    
    def _calculate_pattern_quality_score(self, pattern: Pattern) -> float:
        """Calculate pattern quality score based on completeness and reliability."""
        quality_score = 0.0
        
        # Implementation steps completeness (25%)
        if pattern.implementation_steps:
            steps_score = min(1.0, len(pattern.implementation_steps) / 5)  # Normalize to 5 steps
            quality_score += steps_score * 0.25
        
        # Code examples availability (25%)
        if pattern.code_examples:
            examples_score = min(1.0, len(pattern.code_examples) / 3)  # Normalize to 3 examples
            quality_score += examples_score * 0.25
        
        # Validation criteria availability (20%)
        if pattern.validation_criteria:
            validation_score = min(1.0, len(pattern.validation_criteria) / 3)
            quality_score += validation_score * 0.20
        
        # Pattern confidence (30%)
        quality_score += pattern.confidence * 0.30
        
        return quality_score
    
    def _calculate_context_specificity_score(self, pattern: Pattern, 
                                           issue_context: IssueContext) -> float:
        """Calculate how specifically the pattern matches the context."""
        specificity_score = 0.0
        
        # Tag/label overlap (40%)
        if pattern.tags and issue_context.labels:
            pattern_tags = set(tag.lower() for tag in pattern.tags)
            issue_labels = set(label.lower() for label in issue_context.labels)
            
            overlap = len(pattern_tags.intersection(issue_labels))
            total = len(pattern_tags.union(issue_labels))
            
            if total > 0:
                specificity_score += (overlap / total) * 0.4
        
        # Constraint compatibility (30%)
        constraint_score = self._calculate_constraint_compatibility(
            pattern, issue_context
        )
        specificity_score += constraint_score * 0.3
        
        # Similar issue precedents (30%)
        precedent_score = self._calculate_precedent_score(pattern, issue_context)
        specificity_score += precedent_score * 0.3
        
        return specificity_score
    
    def _get_context_adjusted_weights(self, issue_context: IssueContext) -> Dict[str, float]:
        """Get weights adjusted for the specific context."""
        weights = self.default_weights.copy()
        
        # Analyze context characteristics
        context_characteristics = self._analyze_context_characteristics(issue_context)
        
        # Apply weight adjustments
        for characteristic in context_characteristics:
            if characteristic in self.context_weight_adjustments:
                adjustments = self.context_weight_adjustments[characteristic]
                for criterion, multiplier in adjustments.items():
                    if criterion in weights:
                        weights[criterion] *= multiplier
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_weighted_score(self, criteria_scores: RankingCriteria, 
                                weights: Dict[str, float]) -> float:
        """Calculate overall weighted score."""
        score = 0.0
        
        score += criteria_scores.semantic_similarity * weights.get('semantic_similarity', 0)
        score += criteria_scores.tech_compatibility * weights.get('tech_compatibility', 0)
        score += criteria_scores.complexity_alignment * weights.get('complexity_alignment', 0)
        score += criteria_scores.historical_success * weights.get('historical_success', 0)
        score += criteria_scores.domain_relevance * weights.get('domain_relevance', 0)
        score += criteria_scores.usage_frequency * weights.get('usage_frequency', 0)
        score += criteria_scores.pattern_quality * weights.get('pattern_quality', 0)
        score += criteria_scores.context_specificity * weights.get('context_specificity', 0)
        
        return min(1.0, max(0.0, score))
    
    def _meets_quality_thresholds(self, ranking_result: RankingResult) -> bool:
        """Check if ranking result meets minimum quality thresholds."""
        criteria = ranking_result.criteria_scores
        
        # Check minimum confidence
        if ranking_result.pattern.confidence < self.quality_thresholds['minimum_confidence']:
            return False
        
        # Check minimum relevance (semantic similarity)
        if criteria.semantic_similarity < self.quality_thresholds['minimum_relevance']:
            return False
        
        # Check minimum success rate prediction
        if ranking_result.predicted_success_rate < self.quality_thresholds['minimum_success_rate']:
            return False
        
        return True
    
    def _analyze_context_characteristics(self, issue_context: IssueContext) -> List[str]:
        """Analyze issue context to identify characteristics affecting ranking."""
        characteristics = []
        
        # High tech specificity
        if (issue_context.tech_stack and 
            issue_context.tech_stack.frameworks and 
            len(issue_context.tech_stack.frameworks) > 2):
            characteristics.append('high_tech_specificity')
        
        # High complexity
        if issue_context.complexity in ['high', 'very-high']:
            characteristics.append('high_complexity')
        
        # New or uncommon domain
        common_domains = ['web', 'api', 'database', 'frontend', 'backend', 'general']
        if issue_context.domain not in common_domains:
            characteristics.append('new_domain')
        
        # Critical issue (based on labels)
        critical_labels = ['critical', 'urgent', 'blocker', 'production']
        if any(label.lower() in critical_labels for label in issue_context.labels):
            characteristics.append('critical_issue')
        
        return characteristics
    
    def _fallback_text_similarity(self, pattern: Pattern, 
                                 issue_context: IssueContext) -> float:
        """Fallback text similarity calculation."""
        import re
        
        pattern_text = f"{pattern.name} {pattern.description}".lower()
        issue_text = f"{issue_context.title} {issue_context.description}".lower()
        
        pattern_words = set(re.findall(r'\w+', pattern_text))
        issue_words = set(re.findall(r'\w+', issue_text))
        
        if not pattern_words or not issue_words:
            return 0.0
        
        intersection = len(pattern_words.intersection(issue_words))
        union = len(pattern_words.union(issue_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_languages(self, lang1: Optional[str], lang2: Optional[str]) -> float:
        """Compare programming language compatibility."""
        if not lang1 or not lang2:
            return 0.5
        
        if lang1.lower() == lang2.lower():
            return 1.0
        
        # Language families
        language_families = {
            'javascript': ['typescript', 'node.js'],
            'python': ['python3'],
            'c': ['c++'],
            'java': ['kotlin', 'scala'],
            'c#': ['f#', 'vb.net']
        }
        
        lang1_lower = lang1.lower()
        lang2_lower = lang2.lower()
        
        for family, related in language_families.items():
            if lang1_lower == family and lang2_lower in related:
                return 0.8
            if lang2_lower == family and lang1_lower in related:
                return 0.8
        
        return 0.0
    
    def _compare_frameworks(self, frameworks1: List[str], frameworks2: List[str]) -> float:
        """Compare framework compatibility."""
        if not frameworks1 or not frameworks2:
            return 0.5
        
        set1 = set(fw.lower() for fw in frameworks1)
        set2 = set(fw.lower() for fw in frameworks2)
        
        overlap = len(set1.intersection(set2))
        total = len(set1.union(set2))
        
        return overlap / total if total > 0 else 0.0
    
    def _compare_databases(self, databases1: List[str], databases2: List[str]) -> float:
        """Compare database compatibility."""
        if not databases1 or not databases2:
            return 0.5
        
        set1 = set(db.lower() for db in databases1)
        set2 = set(db.lower() for db in databases2)
        
        overlap = len(set1.intersection(set2))
        total = len(set1.union(set2))
        
        return overlap / total if total > 0 else 0.0
    
    def _compare_architectures(self, arch1: Optional[str], arch2: Optional[str]) -> float:
        """Compare architecture pattern compatibility."""
        if not arch1 or not arch2:
            return 0.5
        
        if arch1.lower() == arch2.lower():
            return 1.0
        
        return 0.0
    
    def _get_related_domains(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are related."""
        related_domain_groups = [
            ['web', 'frontend', 'backend', 'api'],
            ['database', 'data', 'analytics'],
            ['mobile', 'android', 'ios'],
            ['devops', 'deployment', 'infrastructure']
        ]
        
        domain1_lower = domain1.lower()
        domain2_lower = domain2.lower()
        
        for group in related_domain_groups:
            if domain1_lower in group and domain2_lower in group:
                return True
        
        return False
    
    def _calculate_constraint_compatibility(self, pattern: Pattern, 
                                          issue_context: IssueContext) -> float:
        """Calculate compatibility with issue constraints."""
        # This would analyze issue constraints vs pattern requirements
        # For now, return neutral score
        return 0.5
    
    def _calculate_precedent_score(self, pattern: Pattern, 
                                 issue_context: IssueContext) -> float:
        """Calculate score based on similar issue precedents."""
        # This would look for similar issues where this pattern was used successfully
        # For now, use usage count as proxy
        if pattern.usage_count > 0:
            return min(1.0, pattern.usage_count / 20)  # Normalize to 20 uses
        return 0.0
    
    def _generate_ranking_explanation(self, criteria_scores: RankingCriteria,
                                    weights: Dict[str, float],
                                    overall_score: float) -> str:
        """Generate human-readable explanation of ranking."""
        explanations = []
        
        # Find top contributing factors
        contributions = []
        contributions.append(('Semantic similarity', criteria_scores.semantic_similarity * weights.get('semantic_similarity', 0)))
        contributions.append(('Tech compatibility', criteria_scores.tech_compatibility * weights.get('tech_compatibility', 0)))
        contributions.append(('Complexity alignment', criteria_scores.complexity_alignment * weights.get('complexity_alignment', 0)))
        contributions.append(('Historical success', criteria_scores.historical_success * weights.get('historical_success', 0)))
        
        # Sort by contribution
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Create explanation
        top_factors = contributions[:3]
        explanations.append(f"Overall score: {overall_score:.2f}")
        explanations.append("Top factors: " + ", ".join([f"{factor} ({contribution:.2f})" for factor, contribution in top_factors]))
        
        return "; ".join(explanations)
    
    def _determine_confidence_level(self, criteria_scores: RankingCriteria, 
                                  overall_score: float) -> str:
        """Determine confidence level in ranking."""
        if overall_score >= 0.8:
            return 'high'
        elif overall_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _predict_success_rate(self, pattern: Pattern, issue_context: IssueContext,
                            criteria_scores: RankingCriteria) -> float:
        """Predict success rate for pattern application."""
        # Combine multiple factors for prediction
        base_rate = pattern.success_rate
        relevance_factor = criteria_scores.semantic_similarity
        tech_factor = criteria_scores.tech_compatibility
        complexity_factor = criteria_scores.complexity_alignment
        
        # Weighted prediction
        predicted_rate = (
            base_rate * 0.4 +
            relevance_factor * 0.3 +
            tech_factor * 0.2 +
            complexity_factor * 0.1
        )
        
        return min(1.0, max(0.0, predicted_rate))
    
    def _assess_adaptation_complexity(self, pattern: Pattern, issue_context: IssueContext,
                                    criteria_scores: RankingCriteria) -> str:
        """Assess complexity of adapting pattern to context."""
        adaptation_score = (
            criteria_scores.tech_compatibility * 0.4 +
            criteria_scores.complexity_alignment * 0.3 +
            criteria_scores.domain_relevance * 0.3
        )
        
        if adaptation_score >= 0.8:
            return 'minimal'
        elif adaptation_score >= 0.6:
            return 'moderate'
        elif adaptation_score >= 0.4:
            return 'significant'
        else:
            return 'extensive'