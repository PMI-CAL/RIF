"""
Pattern Matching System - Abstraction Layer

This module provides pattern matching functionality with multiple implementations:
1. BasicPatternMatcher: Simple similarity-based matching
2. InterimPatternMatcher: Enhanced matching while waiting for Issue #76
3. AdvancedPatternMatcher: Integration point for Issue #76 when ready

The system provides abstraction to enable parallel development while maintaining
consistent interfaces for pattern finding and ranking.
"""

from typing import Dict, List, Any, Optional
import re
import logging
from abc import abstractmethod
from datetime import datetime

from .core import (
    Pattern, IssueContext, TechStack, PatternMatchingInterface,
    generate_id, load_pattern_from_json
)

# Import knowledge interface
try:
    from knowledge.interface import get_knowledge_system
except ImportError:
    def get_knowledge_system():
        raise ImportError("Knowledge system not available")

logger = logging.getLogger(__name__)


class BasicPatternMatcher(PatternMatchingInterface):
    """
    Basic pattern matching implementation using simple text similarity.
    
    This implementation provides basic pattern matching functionality
    for immediate use while more sophisticated implementations are developed.
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize basic pattern matcher."""
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self._load_patterns()
    
    def find_applicable_patterns(self, issue_context: IssueContext, 
                               limit: int = 10) -> List[Pattern]:
        """
        Find patterns applicable to the given issue context.
        
        Uses basic text similarity and metadata matching to find
        relevant patterns.
        
        Args:
            issue_context: Context information for the issue
            limit: Maximum number of patterns to return
            
        Returns:
            List of applicable patterns ranked by relevance
        """
        logger.info(f"Finding applicable patterns for issue {issue_context.issue_id}")
        
        # Get all patterns from knowledge system
        all_patterns = self._get_all_patterns()
        
        if not all_patterns:
            logger.warning("No patterns found in knowledge system")
            return []
        
        # Score and rank patterns
        scored_patterns = []
        for pattern in all_patterns:
            relevance_score = self.calculate_pattern_relevance(pattern, issue_context)
            if relevance_score > 0.1:  # Only include patterns with some relevance
                scored_patterns.append((pattern, relevance_score))
        
        # Sort by relevance score (descending)
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Return top patterns
        result_patterns = [pattern for pattern, score in scored_patterns[:limit]]
        
        logger.info(f"Found {len(result_patterns)} applicable patterns")
        return result_patterns
    
    def rank_patterns(self, patterns: List[Pattern], 
                     issue_context: IssueContext) -> List[Pattern]:
        """
        Rank patterns by applicability to the given context.
        
        Args:
            patterns: List of patterns to rank
            issue_context: Context for ranking
            
        Returns:
            List of patterns ranked by relevance (highest first)
        """
        scored_patterns = []
        
        for pattern in patterns:
            relevance_score = self.calculate_pattern_relevance(pattern, issue_context)
            scored_patterns.append((pattern, relevance_score))
        
        # Sort by relevance score (descending)
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, score in scored_patterns]
    
    def calculate_pattern_relevance(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """
        Calculate relevance score for a pattern-context pair.
        
        Uses multiple factors:
        - Text similarity between descriptions
        - Technology stack compatibility
        - Complexity alignment
        - Domain matching
        - Tag overlap
        
        Args:
            pattern: Pattern to evaluate
            issue_context: Context to match against
            
        Returns:
            Relevance score (0.0 to 1.0, higher is more relevant)
        """
        score = 0.0
        
        # Text similarity (30% of score)
        text_similarity = self._calculate_text_similarity(
            pattern.description, 
            f"{issue_context.title} {issue_context.description}"
        )
        score += text_similarity * 0.3
        
        # Technology stack compatibility (25% of score)
        tech_score = self._calculate_tech_compatibility(
            pattern.tech_stack, issue_context.tech_stack)
        score += tech_score * 0.25
        
        # Complexity alignment (20% of score)
        complexity_score = self._calculate_complexity_alignment(
            pattern.complexity, issue_context.complexity)
        score += complexity_score * 0.2
        
        # Domain matching (15% of score)
        domain_score = self._calculate_domain_match(
            pattern.domain, issue_context.domain)
        score += domain_score * 0.15
        
        # Tag/label overlap (10% of score)
        tag_score = self._calculate_tag_overlap(
            pattern.tags, issue_context.labels)
        score += tag_score * 0.1
        
        # Pattern quality bonus (based on confidence and success rate)
        quality_bonus = (pattern.confidence + pattern.success_rate) / 2 * 0.1
        score += quality_bonus
        
        return min(1.0, score)  # Cap at 1.0
    
    def _get_all_patterns(self) -> List[Pattern]:
        """Get all patterns from knowledge system."""
        try:
            # Search for all patterns
            results = self.knowledge_system.retrieve_knowledge(
                query="*", collection="patterns", n_results=100)
            
            patterns = []
            for result in results:
                try:
                    pattern = self._convert_result_to_pattern(result)
                    patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Failed to convert result to pattern: {str(e)}")
                    continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {str(e)}")
            return []
    
    def _convert_result_to_pattern(self, result: Dict[str, Any]) -> Pattern:
        """Convert knowledge system result to Pattern object."""
        from .core import TechStack
        
        # Handle nested content
        content = result.get('content', {})
        if isinstance(content, str):
            import json
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
            pattern_id=content.get('pattern_id', result.get('id', generate_id('pattern_'))),
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_tech_compatibility(self, pattern_tech: Optional[TechStack], 
                                    context_tech: Optional[TechStack]) -> float:
        """Calculate technology stack compatibility score."""
        if not pattern_tech or not context_tech:
            return 0.5  # Neutral score if no tech stack info
        
        score = 0.0
        
        # Primary language match (50% of tech score)
        if pattern_tech.primary_language == context_tech.primary_language:
            score += 0.5
        elif pattern_tech.primary_language == 'unknown' or context_tech.primary_language == 'unknown':
            score += 0.25
        
        # Framework overlap (30% of tech score)
        if pattern_tech.frameworks and context_tech.frameworks:
            pattern_frameworks = set(pattern_tech.frameworks)
            context_frameworks = set(context_tech.frameworks)
            overlap = len(pattern_frameworks.intersection(context_frameworks))
            total = len(pattern_frameworks.union(context_frameworks))
            if total > 0:
                score += (overlap / total) * 0.3
        
        # Database compatibility (20% of tech score)
        if pattern_tech.databases and context_tech.databases:
            pattern_dbs = set(pattern_tech.databases)
            context_dbs = set(context_tech.databases)
            if pattern_dbs.intersection(context_dbs):
                score += 0.2
        
        return score
    
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
    
    def _calculate_tag_overlap(self, pattern_tags: List[str], 
                             context_labels: List[str]) -> float:
        """Calculate tag/label overlap score."""
        if not pattern_tags or not context_labels:
            return 0.0
        
        pattern_tags_set = set(tag.lower() for tag in pattern_tags)
        context_labels_set = set(label.lower() for label in context_labels)
        
        intersection = len(pattern_tags_set.intersection(context_labels_set))
        union = len(pattern_tags_set.union(context_labels_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _load_patterns(self):
        """Load patterns from knowledge system for caching."""
        # This could implement pattern caching for performance
        pass


class InterimPatternMatcher(BasicPatternMatcher):
    """
    Enhanced pattern matching implementation for use while Issue #76 is developed.
    
    This implementation builds upon BasicPatternMatcher with additional
    sophisticated matching techniques including semantic similarity,
    historical usage patterns, and success rate weighting.
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize interim pattern matcher."""
        super().__init__(knowledge_system)
        self._load_enhanced_features()
    
    def find_applicable_patterns(self, issue_context: IssueContext, 
                               limit: int = 10) -> List[Pattern]:
        """
        Enhanced pattern finding with semantic analysis and historical data.
        """
        logger.info(f"Finding applicable patterns (enhanced) for issue {issue_context.issue_id}")
        
        # Get base patterns from parent implementation
        base_patterns = super().find_applicable_patterns(issue_context, limit * 2)
        
        if not base_patterns:
            return []
        
        # Apply enhanced filtering and ranking
        enhanced_patterns = self._apply_enhanced_ranking(base_patterns, issue_context)
        
        # Apply success rate weighting
        success_weighted_patterns = self._apply_success_weighting(enhanced_patterns, issue_context)
        
        # Consider historical usage patterns
        history_weighted_patterns = self._apply_history_weighting(success_weighted_patterns, issue_context)
        
        logger.info(f"Enhanced matching found {len(history_weighted_patterns[:limit])} top patterns")
        
        return history_weighted_patterns[:limit]
    
    def calculate_pattern_relevance(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """
        Enhanced relevance calculation with additional factors.
        """
        # Get base relevance score
        base_score = super().calculate_pattern_relevance(pattern, issue_context)
        
        # Apply enhancement factors
        enhancement_score = 0.0
        
        # Semantic similarity enhancement (if available)
        semantic_score = self._calculate_semantic_similarity(pattern, issue_context)
        enhancement_score += semantic_score * 0.15
        
        # Historical success enhancement
        historical_score = self._calculate_historical_success(pattern, issue_context)
        enhancement_score += historical_score * 0.1
        
        # Usage frequency enhancement
        usage_score = self._calculate_usage_frequency_score(pattern)
        enhancement_score += usage_score * 0.05
        
        # Recent pattern bonus
        recency_score = self._calculate_recency_score(pattern)
        enhancement_score += recency_score * 0.05
        
        # Combine scores
        final_score = base_score + enhancement_score
        
        return min(1.0, final_score)
    
    def _load_enhanced_features(self):
        """Load enhanced feature configurations."""
        self.semantic_keywords = {
            'crud': ['create', 'read', 'update', 'delete', 'database', 'model'],
            'auth': ['authentication', 'authorization', 'login', 'user', 'session'],
            'api': ['rest', 'endpoint', 'service', 'http', 'request'],
            'ui': ['interface', 'component', 'render', 'display', 'frontend'],
            'test': ['testing', 'unit', 'integration', 'validation', 'quality'],
        }
        
        self.success_factors = {
            'well_documented': 0.1,
            'has_examples': 0.1,
            'recent_usage': 0.05,
            'high_confidence': 0.1,
        }
    
    def _apply_enhanced_ranking(self, patterns: List[Pattern], 
                              issue_context: IssueContext) -> List[Pattern]:
        """Apply enhanced ranking algorithms."""
        # Re-rank using enhanced scoring
        scored_patterns = []
        
        for pattern in patterns:
            enhanced_score = self.calculate_pattern_relevance(pattern, issue_context)
            scored_patterns.append((pattern, enhanced_score))
        
        # Sort by enhanced score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, score in scored_patterns]
    
    def _apply_success_weighting(self, patterns: List[Pattern], 
                               issue_context: IssueContext) -> List[Pattern]:
        """Apply success rate weighting to pattern ranking."""
        weighted_patterns = []
        
        for pattern in patterns:
            # Calculate success weight
            success_weight = 1.0
            
            if pattern.success_rate > 0.8:
                success_weight = 1.2
            elif pattern.success_rate > 0.6:
                success_weight = 1.1
            elif pattern.success_rate < 0.3:
                success_weight = 0.8
            
            # Apply confidence weighting
            confidence_weight = 0.5 + (pattern.confidence * 0.5)
            
            final_weight = success_weight * confidence_weight
            
            weighted_patterns.append((pattern, final_weight))
        
        # Sort by weight
        weighted_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, weight in weighted_patterns]
    
    def _apply_history_weighting(self, patterns: List[Pattern], 
                               issue_context: IssueContext) -> List[Pattern]:
        """Apply historical usage pattern weighting."""
        # For now, use usage count as a simple history indicator
        # This could be enhanced with actual historical analysis
        
        history_weighted = []
        
        for pattern in patterns:
            history_weight = 1.0
            
            # Boost patterns with higher usage
            if pattern.usage_count > 10:
                history_weight = 1.15
            elif pattern.usage_count > 5:
                history_weight = 1.1
            elif pattern.usage_count == 0:
                history_weight = 0.9
            
            history_weighted.append((pattern, history_weight))
        
        # Sort by history weight
        history_weighted.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, weight in history_weighted]
    
    def _calculate_semantic_similarity(self, pattern: Pattern, 
                                     issue_context: IssueContext) -> float:
        """Calculate semantic similarity using keyword clustering."""
        score = 0.0
        
        # Combine pattern and context text
        pattern_text = f"{pattern.name} {pattern.description}".lower()
        context_text = f"{issue_context.title} {issue_context.description}".lower()
        
        # Check semantic keyword clusters
        for concept, keywords in self.semantic_keywords.items():
            pattern_matches = sum(1 for keyword in keywords if keyword in pattern_text)
            context_matches = sum(1 for keyword in keywords if keyword in context_text)
            
            if pattern_matches > 0 and context_matches > 0:
                # Both texts relate to this concept
                concept_score = min(pattern_matches, context_matches) / len(keywords)
                score += concept_score
        
        return min(1.0, score)
    
    def _calculate_historical_success(self, pattern: Pattern, 
                                    issue_context: IssueContext) -> float:
        """Calculate historical success score for similar contexts."""
        # This could analyze historical applications of this pattern
        # For now, use a simple heuristic based on pattern metadata
        
        base_score = pattern.success_rate
        
        # Bonus for patterns with validation criteria
        if pattern.validation_criteria:
            base_score += 0.1
        
        # Bonus for patterns with code examples
        if pattern.code_examples:
            base_score += 0.1
        
        # Bonus for patterns with detailed implementation steps
        if pattern.implementation_steps and len(pattern.implementation_steps) > 3:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_usage_frequency_score(self, pattern: Pattern) -> float:
        """Calculate score based on usage frequency."""
        if pattern.usage_count == 0:
            return 0.0
        
        # Logarithmic scaling to avoid over-weighting very popular patterns
        import math
        return min(1.0, math.log(pattern.usage_count + 1) / 10)
    
    def _calculate_recency_score(self, pattern: Pattern) -> float:
        """Calculate score based on pattern recency."""
        # This would require timestamp information in patterns
        # For now, return a neutral score
        return 0.5


class AdvancedPatternMatcher(PatternMatchingInterface):
    """
    Advanced pattern matcher that integrates with Issue #76 when available.
    
    This class serves as an integration point for the sophisticated pattern
    matching system being developed in Issue #76. It falls back to interim
    matching when the advanced system is not available.
    """
    
    def __init__(self, knowledge_system=None):
        """Initialize advanced pattern matcher."""
        self.knowledge_system = knowledge_system or get_knowledge_system()
        self.fallback_matcher = InterimPatternMatcher(knowledge_system)
        self._check_advanced_system()
    
    def find_applicable_patterns(self, issue_context: IssueContext, 
                               limit: int = 10) -> List[Pattern]:
        """
        Find patterns using advanced system or fallback to interim matcher.
        """
        if self.advanced_available:
            return self._find_patterns_advanced(issue_context, limit)
        else:
            logger.info("Advanced pattern matching not available, using fallback")
            return self.fallback_matcher.find_applicable_patterns(issue_context, limit)
    
    def rank_patterns(self, patterns: List[Pattern], 
                     issue_context: IssueContext) -> List[Pattern]:
        """Rank patterns using advanced system or fallback."""
        if self.advanced_available:
            return self._rank_patterns_advanced(patterns, issue_context)
        else:
            return self.fallback_matcher.rank_patterns(patterns, issue_context)
    
    def calculate_pattern_relevance(self, pattern: Pattern, 
                                  issue_context: IssueContext) -> float:
        """Calculate relevance using advanced system or fallback."""
        if self.advanced_available:
            return self._calculate_relevance_advanced(pattern, issue_context)
        else:
            return self.fallback_matcher.calculate_pattern_relevance(pattern, issue_context)
    
    def _check_advanced_system(self):
        """Check if advanced pattern matching system from Issue #76 is available."""
        try:
            # Try to import Issue #76 pattern matching system
            # This will be updated when Issue #76 is completed
            from knowledge.pattern_matching.advanced_matcher import AdvancedMatcher
            self.advanced_matcher = AdvancedMatcher(self.knowledge_system)
            self.advanced_available = True
            logger.info("Advanced pattern matching system available")
        except ImportError:
            self.advanced_available = False
            logger.info("Advanced pattern matching system not available, using interim implementation")
    
    def _find_patterns_advanced(self, issue_context: IssueContext, limit: int) -> List[Pattern]:
        """Use advanced pattern matching system (when available)."""
        # This will be implemented when Issue #76 is completed
        return self.advanced_matcher.find_applicable_patterns(issue_context, limit)
    
    def _rank_patterns_advanced(self, patterns: List[Pattern], 
                              issue_context: IssueContext) -> List[Pattern]:
        """Use advanced pattern ranking (when available)."""
        return self.advanced_matcher.rank_patterns(patterns, issue_context)
    
    def _calculate_relevance_advanced(self, pattern: Pattern, 
                                    issue_context: IssueContext) -> float:
        """Use advanced relevance calculation (when available)."""
        return self.advanced_matcher.calculate_pattern_relevance(pattern, issue_context)