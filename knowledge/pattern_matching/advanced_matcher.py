"""
Advanced Pattern Matching System - Issue #76

This is the main implementation of the sophisticated pattern matching system that finds
and ranks applicable patterns for new issues using multiple advanced techniques.

Features:
- Multi-dimensional similarity detection
- Context-aware pattern ranking
- Machine learning-enhanced scoring
- Historical success rate integration
- Technology stack compatibility analysis
- Semantic understanding of issue descriptions
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

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
    Pattern, IssueContext, TechStack, PatternMatchingInterface
)

# Import pattern matching components
from .similarity_engine import SimilarityEngine
from .pattern_ranker import PatternRanker
from .recommendation_generator import RecommendationGenerator
from .confidence_scorer import ConfidenceScorer


@dataclass
class PatternMatchResult:
    """Result of pattern matching operation."""
    pattern: Pattern
    relevance_score: float
    confidence_score: float
    similarity_factors: Dict[str, float]
    recommendation_strength: str  # 'strong', 'moderate', 'weak'
    adaptation_requirements: List[str]
    estimated_success_rate: float
    historical_precedents: List[str]


class AdvancedPatternMatcher(PatternMatchingInterface):
    """
    Advanced pattern matching system implementing Issue #76 requirements.
    
    This system provides sophisticated pattern matching using multiple techniques:
    - Vector similarity search for semantic matching
    - Multi-criteria pattern ranking
    - Historical success rate analysis
    - Technology stack compatibility scoring
    - Context-aware recommendation generation
    """
    
    def __init__(self, knowledge_system=None, database: Optional[RIFDatabase] = None):
        """Initialize the advanced pattern matcher."""
        self.logger = logging.getLogger(__name__)
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self.database = database
        
        # Initialize core components
        self.similarity_engine = SimilarityEngine(knowledge_system, database)
        self.pattern_ranker = PatternRanker(knowledge_system, database)
        self.recommendation_generator = RecommendationGenerator(knowledge_system, database)
        self.confidence_scorer = ConfidenceScorer(knowledge_system, database)
        
        # Cache for performance optimization
        self._pattern_cache = {}
        self._similarity_cache = {}
        self._cache_ttl = timedelta(hours=1)
        self._last_cache_cleanup = datetime.utcnow()
        
        self.logger.info("Advanced Pattern Matcher initialized")
    
    def find_applicable_patterns(self, issue_context: IssueContext, 
                               limit: int = 10) -> List[Pattern]:
        """
        Find patterns applicable to the given issue context using advanced techniques.
        
        This is the main entry point that implements the requirements from Issue #76:
        1. Finds relevant similar issues
        2. Ranks patterns accurately
        3. Generates useful recommendations
        4. Provides confidence scores
        
        Args:
            issue_context: Context information for the issue
            limit: Maximum number of patterns to return
            
        Returns:
            List of applicable patterns ranked by relevance
        """
        self.logger.info(f"Finding applicable patterns for issue {issue_context.issue_id}")
        
        try:
            # Step 1: Find similar issues (Acceptance Criteria #1)
            similar_issues = self.similarity_engine.find_similar_issues(
                issue_context, 
                similarity_threshold=0.7,
                limit=50  # Get more candidates for better ranking
            )
            
            self.logger.debug(f"Found {len(similar_issues)} similar issues")
            
            # Step 2: Extract candidate patterns from similar issues
            candidate_patterns = self._extract_patterns_from_issues(similar_issues)
            
            if not candidate_patterns:
                self.logger.warning("No candidate patterns found")
                return []
            
            # Step 3: Rank patterns accurately (Acceptance Criteria #2)
            ranked_patterns = self.pattern_ranker.rank_patterns(
                candidate_patterns, 
                issue_context
            )
            
            # Step 4: Apply advanced scoring and filtering
            scored_patterns = []
            for pattern in ranked_patterns:
                # Calculate comprehensive relevance score
                relevance_score = self.calculate_pattern_relevance(pattern, issue_context)
                
                # Calculate confidence score (Acceptance Criteria #4)
                confidence_score = self.confidence_scorer.calculate_confidence(
                    pattern, issue_context
                )
                
                # Only include patterns with sufficient relevance and confidence
                if relevance_score > 0.3 and confidence_score > 0.4:
                    match_result = PatternMatchResult(
                        pattern=pattern,
                        relevance_score=relevance_score,
                        confidence_score=confidence_score,
                        similarity_factors=self._get_similarity_factors(pattern, issue_context),
                        recommendation_strength=self._determine_recommendation_strength(
                            relevance_score, confidence_score
                        ),
                        adaptation_requirements=self._identify_adaptation_requirements(
                            pattern, issue_context
                        ),
                        estimated_success_rate=self._estimate_success_rate(
                            pattern, issue_context, similar_issues
                        ),
                        historical_precedents=self._find_historical_precedents(
                            pattern, similar_issues
                        )
                    )
                    scored_patterns.append(match_result)
            
            # Sort by combined score (relevance * confidence)
            scored_patterns.sort(
                key=lambda x: x.relevance_score * x.confidence_score, 
                reverse=True
            )
            
            # Return top patterns
            result_patterns = [result.pattern for result in scored_patterns[:limit]]
            
            self.logger.info(f"Returning {len(result_patterns)} applicable patterns")
            return result_patterns
            
        except Exception as e:
            self.logger.error(f"Error finding applicable patterns: {str(e)}")
            return []
    
    def rank_patterns(self, patterns: List[Pattern], 
                     issue_context: IssueContext) -> List[Pattern]:
        """
        Rank patterns by applicability using advanced multi-criteria ranking.
        
        Args:
            patterns: List of patterns to rank
            issue_context: Context for ranking
            
        Returns:
            List of patterns ranked by relevance (highest first)
        """
        return self.pattern_ranker.rank_patterns(patterns, issue_context)
    
    def calculate_pattern_relevance(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """
        Calculate comprehensive pattern relevance score using multiple factors.
        
        This method implements advanced relevance calculation that considers:
        - Semantic similarity
        - Technology stack compatibility
        - Complexity alignment
        - Domain matching
        - Historical success rate
        - Context compatibility
        
        Args:
            pattern: Pattern to evaluate
            issue_context: Context to match against
            
        Returns:
            Relevance score (0.0 to 1.0, higher is more relevant)
        """
        try:
            # Use similarity engine for semantic analysis
            semantic_score = self.similarity_engine.calculate_semantic_similarity(
                pattern.description,
                f"{issue_context.title} {issue_context.description}"
            )
            
            # Use pattern ranker for multi-criteria scoring
            ranking_score = self.pattern_ranker.calculate_ranking_score(
                pattern, issue_context
            )
            
            # Technology compatibility from similarity engine
            tech_score = self.similarity_engine.calculate_tech_compatibility(
                pattern.tech_stack, issue_context.tech_stack
            )
            
            # Historical success factor
            success_score = self._calculate_historical_success_factor(pattern, issue_context)
            
            # Context alignment score
            context_score = self._calculate_context_alignment(pattern, issue_context)
            
            # Combine scores with weights
            weights = {
                'semantic': 0.25,
                'ranking': 0.25,
                'tech': 0.20,
                'success': 0.15,
                'context': 0.15
            }
            
            final_score = (
                semantic_score * weights['semantic'] +
                ranking_score * weights['ranking'] +
                tech_score * weights['tech'] +
                success_score * weights['success'] +
                context_score * weights['context']
            )
            
            # Apply pattern quality multiplier
            quality_multiplier = 0.5 + (pattern.confidence * 0.5)
            final_score *= quality_multiplier
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern relevance: {str(e)}")
            return 0.0
    
    def generate_recommendations(self, patterns: List[Pattern], 
                               issue_context: IssueContext) -> List[Dict[str, Any]]:
        """
        Generate useful recommendations from applicable patterns (Acceptance Criteria #3).
        
        Args:
            patterns: List of applicable patterns
            issue_context: Context for recommendations
            
        Returns:
            List of recommendation dictionaries with detailed guidance
        """
        return self.recommendation_generator.generate_recommendations(
            patterns, issue_context
        )
    
    def _extract_patterns_from_issues(self, similar_issues: List[Dict[str, Any]]) -> List[Pattern]:
        """Extract patterns associated with similar issues."""
        candidate_patterns = []
        seen_pattern_ids = set()
        
        for issue in similar_issues:
            try:
                # Get patterns associated with this issue from knowledge system
                patterns = self._get_patterns_for_issue(issue.get('issue_id', ''))
                
                for pattern in patterns:
                    if pattern.pattern_id not in seen_pattern_ids:
                        candidate_patterns.append(pattern)
                        seen_pattern_ids.add(pattern.pattern_id)
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract patterns from issue {issue}: {str(e)}")
                continue
        
        return candidate_patterns
    
    def _get_patterns_for_issue(self, issue_id: str) -> List[Pattern]:
        """Get patterns that have been successfully applied to an issue."""
        try:
            # Query knowledge system for patterns used in this issue
            results = self.knowledge_system.retrieve_knowledge(
                query=f"issue_id:{issue_id}",
                collection="patterns",
                n_results=20
            )
            
            patterns = []
            for result in results:
                try:
                    pattern = self._convert_result_to_pattern(result)
                    patterns.append(pattern)
                except Exception as e:
                    self.logger.debug(f"Failed to convert result to pattern: {str(e)}")
                    continue
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Failed to get patterns for issue {issue_id}: {str(e)}")
            return []
    
    def _convert_result_to_pattern(self, result: Dict[str, Any]) -> Pattern:
        """Convert knowledge system result to Pattern object."""
        from knowledge.pattern_application.core import TechStack
        
        # Handle nested content
        content = result.get('content', {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"description": content}
        
        # Extract tech stack
        tech_stack = None
        if 'tech_stack' in content and content['tech_stack']:
            ts_data = content['tech_stack']
            tech_stack = TechStack(
                primary_language=ts_data.get('primary_language', ''),
                frameworks=ts_data.get('frameworks', []),
                databases=ts_data.get('databases', []),
                tools=ts_data.get('tools', []),
                architecture_pattern=ts_data.get('architecture_pattern'),
                deployment_target=ts_data.get('deployment_target')
            )
        
        return Pattern(
            pattern_id=content.get('pattern_id', result.get('id', f'pattern_{hash(str(content))}')),
            name=content.get('pattern_name', content.get('name', 'Unknown Pattern')),
            description=content.get('description', ''),
            complexity=content.get('complexity', 'medium'),
            tech_stack=tech_stack,
            domain=content.get('domain', 'general'),
            tags=content.get('tags', []),
            confidence=content.get('confidence', 0.7),
            success_rate=content.get('success_rate', 0.0),
            usage_count=content.get('usage_count', 0),
            implementation_steps=content.get('implementation_steps', []),
            code_examples=content.get('code_examples', []),
            validation_criteria=content.get('validation_criteria', [])
        )
    
    def _get_similarity_factors(self, pattern: Pattern, 
                              issue_context: IssueContext) -> Dict[str, float]:
        """Get detailed similarity factor breakdown."""
        return {
            'semantic_similarity': self.similarity_engine.calculate_semantic_similarity(
                pattern.description, f"{issue_context.title} {issue_context.description}"
            ),
            'tech_compatibility': self.similarity_engine.calculate_tech_compatibility(
                pattern.tech_stack, issue_context.tech_stack
            ),
            'complexity_alignment': self._calculate_complexity_alignment(
                pattern.complexity, issue_context.complexity
            ),
            'domain_match': self._calculate_domain_match(
                pattern.domain, issue_context.domain
            )
        }
    
    def _determine_recommendation_strength(self, relevance_score: float, 
                                         confidence_score: float) -> str:
        """Determine recommendation strength based on scores."""
        combined_score = relevance_score * confidence_score
        
        if combined_score >= 0.8:
            return 'strong'
        elif combined_score >= 0.6:
            return 'moderate'
        else:
            return 'weak'
    
    def _identify_adaptation_requirements(self, pattern: Pattern, 
                                        issue_context: IssueContext) -> List[str]:
        """Identify what adaptations are needed for the pattern."""
        adaptations = []
        
        # Technology stack adaptations
        if pattern.tech_stack and issue_context.tech_stack:
            if pattern.tech_stack.primary_language != issue_context.tech_stack.primary_language:
                adaptations.append(f"Language adaptation: {pattern.tech_stack.primary_language} → {issue_context.tech_stack.primary_language}")
            
            pattern_frameworks = set(pattern.tech_stack.frameworks)
            context_frameworks = set(issue_context.tech_stack.frameworks)
            if not pattern_frameworks.intersection(context_frameworks):
                adaptations.append("Framework adaptation required")
        
        # Complexity adaptations
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        if pattern.complexity in complexity_levels and issue_context.complexity in complexity_levels:
            pattern_idx = complexity_levels.index(pattern.complexity)
            context_idx = complexity_levels.index(issue_context.complexity)
            if abs(pattern_idx - context_idx) > 1:
                adaptations.append(f"Complexity scaling: {pattern.complexity} → {issue_context.complexity}")
        
        # Domain adaptations
        if pattern.domain != issue_context.domain and pattern.domain != 'general':
            adaptations.append(f"Domain adaptation: {pattern.domain} → {issue_context.domain}")
        
        return adaptations
    
    def _estimate_success_rate(self, pattern: Pattern, issue_context: IssueContext,
                             similar_issues: List[Dict[str, Any]]) -> float:
        """Estimate success rate for this pattern in this context."""
        # Start with pattern's historical success rate
        base_rate = pattern.success_rate
        
        # Adjust based on context similarity
        context_similarity = self.calculate_pattern_relevance(pattern, issue_context)
        
        # Adjust based on similar issue outcomes
        similar_success_rate = self._calculate_similar_issue_success_rate(
            pattern, similar_issues
        )
        
        # Combine with weights
        estimated_rate = (
            base_rate * 0.4 +
            context_similarity * 0.3 +
            similar_success_rate * 0.3
        )
        
        return min(1.0, max(0.0, estimated_rate))
    
    def _find_historical_precedents(self, pattern: Pattern, 
                                  similar_issues: List[Dict[str, Any]]) -> List[str]:
        """Find historical precedents where this pattern was used successfully."""
        precedents = []
        
        for issue in similar_issues:
            # Check if this pattern was used in the similar issue
            issue_id = issue.get('issue_id', '')
            if self._was_pattern_used_in_issue(pattern.pattern_id, issue_id):
                issue_title = issue.get('title', f'Issue {issue_id}')
                precedents.append(f"{issue_title} (#{issue_id})")
        
        return precedents[:5]  # Return top 5 precedents
    
    def _calculate_historical_success_factor(self, pattern: Pattern, 
                                           issue_context: IssueContext) -> float:
        """Calculate historical success factor for pattern in similar contexts."""
        # This would analyze historical usage of this pattern in similar contexts
        # For now, use pattern's success rate as baseline
        base_success = pattern.success_rate
        
        # Boost for patterns with high usage count (proven track record)
        if pattern.usage_count > 10:
            base_success += 0.1
        elif pattern.usage_count > 5:
            base_success += 0.05
        
        # Boost for patterns with validation criteria (more reliable)
        if pattern.validation_criteria:
            base_success += 0.05
        
        # Boost for patterns with code examples (easier to implement)
        if pattern.code_examples:
            base_success += 0.05
        
        return min(1.0, base_success)
    
    def _calculate_context_alignment(self, pattern: Pattern, 
                                   issue_context: IssueContext) -> float:
        """Calculate how well the pattern aligns with the issue context."""
        alignment_score = 0.0
        
        # Tag/label alignment
        if pattern.tags and issue_context.labels:
            pattern_tags = set(tag.lower() for tag in pattern.tags)
            context_labels = set(label.lower() for label in issue_context.labels)
            if pattern_tags.intersection(context_labels):
                alignment_score += 0.3
        
        # Complexity alignment
        alignment_score += self._calculate_complexity_alignment(
            pattern.complexity, issue_context.complexity
        ) * 0.3
        
        # Domain alignment
        alignment_score += self._calculate_domain_match(
            pattern.domain, issue_context.domain
        ) * 0.4
        
        return alignment_score
    
    def _calculate_complexity_alignment(self, pattern_complexity: str, 
                                      context_complexity: str) -> float:
        """Calculate complexity alignment score."""
        complexity_levels = ['low', 'medium', 'high', 'very-high']
        
        try:
            pattern_idx = complexity_levels.index(pattern_complexity)
            context_idx = complexity_levels.index(context_complexity)
            
            # Perfect match gets full score
            if pattern_idx == context_idx:
                return 1.0
            
            # Adjacent levels get partial score
            diff = abs(pattern_idx - context_idx)
            if diff == 1:
                return 0.7
            elif diff == 2:
                return 0.4
            else:
                return 0.1
                
        except ValueError:
            return 0.5  # Unknown complexity
    
    def _calculate_domain_match(self, pattern_domain: str, context_domain: str) -> float:
        """Calculate domain matching score."""
        if pattern_domain == context_domain:
            return 1.0
        elif pattern_domain == 'general' or context_domain == 'general':
            return 0.5
        else:
            return 0.0
    
    def _calculate_similar_issue_success_rate(self, pattern: Pattern, 
                                            similar_issues: List[Dict[str, Any]]) -> float:
        """Calculate success rate based on similar issue outcomes."""
        if not similar_issues:
            return 0.5
        
        successes = 0
        total = 0
        
        for issue in similar_issues:
            if self._was_pattern_used_in_issue(pattern.pattern_id, issue.get('issue_id', '')):
                total += 1
                # Check if the issue was successfully resolved
                if issue.get('state') == 'closed' and issue.get('outcome') == 'success':
                    successes += 1
        
        if total == 0:
            return 0.5  # No data available
        
        return successes / total
    
    def _was_pattern_used_in_issue(self, pattern_id: str, issue_id: str) -> bool:
        """Check if a pattern was used in a specific issue."""
        try:
            # Query knowledge system for pattern usage in issue
            results = self.knowledge_system.retrieve_knowledge(
                query=f"pattern_id:{pattern_id} AND issue_id:{issue_id}",
                collection="applications",
                n_results=1
            )
            return len(results) > 0
        except Exception:
            return False
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        if datetime.utcnow() - self._last_cache_cleanup > self._cache_ttl:
            self._pattern_cache.clear()
            self._similarity_cache.clear()
            self._last_cache_cleanup = datetime.utcnow()