"""
Result Ranking and Relevance Scoring System - Issue #33
Advanced ranking algorithms for hybrid search results
"""

import re
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, Counter
import json

from .hybrid_search_engine import SearchResult, HybridSearchResults


@dataclass
class RankingContext:
    """Context information for ranking results"""
    query_text: str
    query_intent: str
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Current workspace context  
    active_files: Set[str] = field(default_factory=set)
    recent_entities: List[str] = field(default_factory=list)
    project_languages: Set[str] = field(default_factory=set)
    
    # Query-specific context
    mentioned_entities: List[str] = field(default_factory=list)
    mentioned_concepts: List[str] = field(default_factory=list)
    semantic_keywords: List[str] = field(default_factory=list)


@dataclass
class RelevanceSignals:
    """Individual relevance signals for a result"""
    # Core relevance signals
    semantic_similarity: float = 0.0      # Vector similarity score
    structural_relevance: float = 0.0     # Graph distance/relationship strength
    exact_match: float = 0.0              # Direct name/keyword matching
    
    # Context relevance signals
    file_context: float = 0.0             # Currently active/recent files
    entity_context: float = 0.0           # Recently accessed entities
    language_context: float = 0.0         # Project language preferences
    
    # Quality signals
    entity_importance: float = 0.0        # Entity centrality/usage frequency
    code_quality: float = 0.0             # Code quality metrics if available
    recency: float = 0.0                  # How recently entity was modified
    
    # Diversity signals
    result_novelty: float = 0.0           # How different from other results
    type_diversity: float = 0.0           # Different entity types represented


@dataclass
class RankingWeights:
    """Configurable weights for different ranking signals"""
    # PRD specified weights (40% direct, 30% semantic, 20% structural, 10% temporal)
    semantic_similarity: float = 0.30
    structural_relevance: float = 0.20
    exact_match: float = 0.40
    temporal: float = 0.10
    
    # Additional context weights
    file_context: float = 0.05
    entity_context: float = 0.05
    language_context: float = 0.03
    
    # Quality and diversity weights
    entity_importance: float = 0.08
    code_quality: float = 0.05
    result_novelty: float = 0.02


class RelevanceScorer:
    """Calculates individual relevance signals for search results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common programming keywords for exact matching
        self.programming_keywords = {
            'auth', 'authentication', 'login', 'logout', 'password', 'token',
            'user', 'session', 'security', 'hash', 'encrypt', 'decrypt',
            'database', 'db', 'sql', 'query', 'connection', 'transaction',
            'api', 'rest', 'http', 'request', 'response', 'endpoint',
            'error', 'exception', 'handle', 'catch', 'try', 'throw',
            'config', 'settings', 'environment', 'env', 'variable',
            'test', 'mock', 'assert', 'validate', 'verify'
        }
    
    def calculate_semantic_similarity(self, result: SearchResult, 
                                    context: RankingContext) -> float:
        """Calculate semantic similarity score"""
        if result.source_strategy == "vector":
            # Vector search already provides similarity score
            return min(result.relevance_score, 1.0)
        
        # For non-vector results, calculate text-based similarity
        result_text = self._extract_searchable_text(result)
        query_text = context.query_text.lower()
        
        # Simple token overlap similarity
        result_tokens = set(re.findall(r'\b\w+\b', result_text.lower()))
        query_tokens = set(re.findall(r'\b\w+\b', query_text))
        
        if not result_tokens or not query_tokens:
            return 0.0
        
        intersection = result_tokens & query_tokens
        union = result_tokens | query_tokens
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Boost similarity if programming keywords match
        keyword_matches = intersection & self.programming_keywords
        keyword_boost = min(0.2 * len(keyword_matches), 0.4)
        
        return min(jaccard_similarity + keyword_boost, 1.0)
    
    def calculate_structural_relevance(self, result: SearchResult,
                                     context: RankingContext) -> float:
        """Calculate structural/graph-based relevance"""
        if result.source_strategy == "graph":
            # Graph search provides relationship strength
            return min(result.relevance_score, 1.0)
        
        # For non-graph results, estimate structural relevance
        structural_score = 0.0
        
        # Boost if in metadata we have relationship information
        if 'relationship_count' in result.metadata:
            rel_count = result.metadata['relationship_count']
            # Normalize relationship count (more relationships = more important)
            structural_score += min(rel_count / 20.0, 0.3)  # Max 0.3 boost
        
        # Boost if entity name appears in query context entities
        if result.entity_name in context.mentioned_entities:
            structural_score += 0.4
        
        # Consider file-level relationships
        if result.file_path in context.active_files:
            structural_score += 0.2
        
        return min(structural_score, 1.0)
    
    def calculate_exact_match(self, result: SearchResult, 
                            context: RankingContext) -> float:
        """Calculate exact matching score"""
        exact_score = 0.0
        query_lower = context.query_text.lower()
        
        # Exact entity name match
        if result.entity_name.lower() in query_lower:
            exact_score += 0.6
        
        # Partial entity name match
        entity_words = re.findall(r'\b\w+\b', result.entity_name.lower())
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        for entity_word in entity_words:
            if entity_word in query_words:
                exact_score += 0.1
        
        # File path match
        file_name = result.file_path.split('/')[-1].lower() if result.file_path else ""
        if any(word in file_name for word in query_words):
            exact_score += 0.2
        
        # Entity type match
        if result.entity_type.lower() in query_lower:
            exact_score += 0.1
        
        return min(exact_score, 1.0)
    
    def calculate_file_context(self, result: SearchResult,
                             context: RankingContext) -> float:
        """Calculate file context relevance"""
        if not result.file_path:
            return 0.0
        
        score = 0.0
        
        # Boost if file is currently active
        if result.file_path in context.active_files:
            score += 0.8
        
        # Boost if file is in same directory as active files
        result_dir = '/'.join(result.file_path.split('/')[:-1])
        for active_file in context.active_files:
            active_dir = '/'.join(active_file.split('/')[:-1])
            if result_dir == active_dir:
                score += 0.3
                break
        
        # Language context
        file_ext = result.file_path.split('.')[-1].lower() if '.' in result.file_path else ""
        if file_ext in context.project_languages:
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_entity_context(self, result: SearchResult,
                               context: RankingContext) -> float:
        """Calculate entity context relevance"""
        score = 0.0
        
        # Boost if entity was recently accessed
        if result.entity_name in context.recent_entities:
            recent_index = context.recent_entities.index(result.entity_name)
            # More recent = higher score
            recency_score = 1.0 - (recent_index / len(context.recent_entities))
            score += 0.5 * recency_score
        
        # Boost if mentioned in query context
        if result.entity_name in context.mentioned_entities:
            score += 0.4
        
        return min(score, 1.0)
    
    def calculate_entity_importance(self, result: SearchResult,
                                  context: RankingContext) -> float:
        """Calculate entity importance/centrality"""
        importance_score = 0.5  # Default baseline
        
        # Use relationship count as proxy for importance
        if 'relationship_count' in result.metadata:
            rel_count = result.metadata['relationship_count']
            # Logarithmic scaling for relationship importance
            importance_score = min(0.3 + 0.7 * (np.log(rel_count + 1) / 10), 1.0)
        
        # Boost for certain entity types that tend to be important
        importance_boosts = {
            'class': 0.1,
            'function': 0.05,
            'module': 0.15,
            'interface': 0.1
        }
        
        boost = importance_boosts.get(result.entity_type, 0.0)
        importance_score = min(importance_score + boost, 1.0)
        
        return importance_score
    
    def calculate_recency(self, result: SearchResult,
                        context: RankingContext) -> float:
        """Calculate temporal recency score"""
        # Default to neutral if no timestamp available
        if 'updated_at' not in result.metadata:
            return 0.5
        
        try:
            # Parse timestamp and calculate days since update
            update_timestamp = result.metadata['updated_at']
            # This would need actual timestamp parsing based on format
            # For now, return default
            return 0.5
        except:
            return 0.5
    
    def calculate_result_novelty(self, result: SearchResult, 
                               other_results: List[SearchResult]) -> float:
        """Calculate how novel/different this result is from others"""
        if not other_results:
            return 1.0
        
        # Calculate similarity to other results
        similarities = []
        result_features = self._extract_result_features(result)
        
        for other_result in other_results:
            if other_result.entity_id == result.entity_id:
                continue
            
            other_features = self._extract_result_features(other_result)
            similarity = self._calculate_feature_similarity(result_features, other_features)
            similarities.append(similarity)
        
        # Novelty is inverse of average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return 1.0 - avg_similarity
        
        return 1.0
    
    def _extract_searchable_text(self, result: SearchResult) -> str:
        """Extract searchable text content from result"""
        text_parts = [result.entity_name, result.entity_type]
        
        if result.file_path:
            # Add filename without path
            filename = result.file_path.split('/')[-1]
            text_parts.append(filename)
        
        if result.content:
            text_parts.append(result.content)
        
        # Add metadata text
        for key, value in result.metadata.items():
            if isinstance(value, str):
                text_parts.append(value)
        
        return ' '.join(text_parts)
    
    def _extract_result_features(self, result: SearchResult) -> Dict[str, Any]:
        """Extract features for similarity calculation"""
        return {
            'entity_type': result.entity_type,
            'file_path': result.file_path,
            'source_strategy': result.source_strategy,
            'entity_name_tokens': set(re.findall(r'\b\w+\b', result.entity_name.lower())),
            'file_extension': result.file_path.split('.')[-1] if result.file_path and '.' in result.file_path else None
        }
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], 
                                    features2: Dict[str, Any]) -> float:
        """Calculate similarity between result features"""
        similarity = 0.0
        
        # Entity type similarity
        if features1['entity_type'] == features2['entity_type']:
            similarity += 0.3
        
        # File similarity
        if features1['file_path'] == features2['file_path']:
            similarity += 0.4
        elif features1['file_extension'] == features2['file_extension']:
            similarity += 0.1
        
        # Name token similarity
        tokens1 = features1['entity_name_tokens']
        tokens2 = features2['entity_name_tokens']
        if tokens1 and tokens2:
            token_similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
            similarity += 0.3 * token_similarity
        
        return min(similarity, 1.0)


class ResultRanker:
    """
    Advanced result ranking system combining multiple relevance signals.
    """
    
    def __init__(self, weights: RankingWeights = None):
        self.weights = weights or RankingWeights()
        self.scorer = RelevanceScorer()
        self.logger = logging.getLogger(__name__)
        
        # Learning parameters for weight adjustment
        self.learning_rate = 0.01
        self.feedback_history = []
    
    def rank_results(self, results: List[SearchResult], 
                    context: RankingContext) -> List[SearchResult]:
        """
        Rank search results using multi-signal relevance scoring.
        
        Args:
            results: List of search results to rank
            context: Ranking context with query and user information
            
        Returns:
            Ranked list of results
        """
        if not results:
            return []
        
        start_time = time.time()
        self.logger.debug(f"Ranking {len(results)} results")
        
        # Calculate relevance signals for each result
        scored_results = []
        for result in results:
            signals = self._calculate_relevance_signals(result, context, results)
            final_score = self._combine_relevance_signals(signals)
            
            # Update result with final score
            result.relevance_score = final_score
            result.metadata['relevance_signals'] = signals.__dict__
            
            scored_results.append(result)
        
        # Sort by final relevance score
        ranked_results = sorted(scored_results, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity filtering to avoid too many similar results
        diversified_results = self._apply_diversity_filtering(ranked_results, max_similar=3)
        
        ranking_time_ms = int((time.time() - start_time) * 1000)
        self.logger.debug(f"Ranking completed in {ranking_time_ms}ms")
        
        return diversified_results
    
    def _calculate_relevance_signals(self, result: SearchResult, 
                                   context: RankingContext,
                                   all_results: List[SearchResult]) -> RelevanceSignals:
        """Calculate all relevance signals for a result"""
        signals = RelevanceSignals()
        
        # Core relevance signals
        signals.semantic_similarity = self.scorer.calculate_semantic_similarity(result, context)
        signals.structural_relevance = self.scorer.calculate_structural_relevance(result, context)
        signals.exact_match = self.scorer.calculate_exact_match(result, context)
        
        # Context signals
        signals.file_context = self.scorer.calculate_file_context(result, context)
        signals.entity_context = self.scorer.calculate_entity_context(result, context)
        
        # Quality signals
        signals.entity_importance = self.scorer.calculate_entity_importance(result, context)
        signals.recency = self.scorer.calculate_recency(result, context)
        
        # Diversity signals
        other_results = [r for r in all_results if r.entity_id != result.entity_id]
        signals.result_novelty = self.scorer.calculate_result_novelty(result, other_results)
        
        return signals
    
    def _combine_relevance_signals(self, signals: RelevanceSignals) -> float:
        """Combine individual signals into final relevance score"""
        final_score = (
            self.weights.semantic_similarity * signals.semantic_similarity +
            self.weights.structural_relevance * signals.structural_relevance +
            self.weights.exact_match * signals.exact_match +
            self.weights.temporal * signals.recency +
            self.weights.file_context * signals.file_context +
            self.weights.entity_context * signals.entity_context +
            self.weights.entity_importance * signals.entity_importance +
            self.weights.result_novelty * signals.result_novelty
        )
        
        # Normalize to 0-1 range
        return min(max(final_score, 0.0), 1.0)
    
    def _apply_diversity_filtering(self, ranked_results: List[SearchResult], 
                                 max_similar: int = 3) -> List[SearchResult]:
        """Apply diversity filtering to reduce similar results"""
        if len(ranked_results) <= max_similar:
            return ranked_results
        
        # Group results by similarity
        similarity_groups = defaultdict(list)
        processed_results = []
        
        for result in ranked_results:
            # Find most similar group
            best_group = None
            best_similarity = 0.0
            
            result_features = self.scorer._extract_result_features(result)
            
            for group_key, group_results in similarity_groups.items():
                if not group_results:
                    continue
                
                # Calculate similarity to group representative (first result)
                group_features = self.scorer._extract_result_features(group_results[0])
                similarity = self.scorer._calculate_feature_similarity(result_features, group_features)
                
                if similarity > 0.7 and similarity > best_similarity:  # Similarity threshold
                    best_group = group_key
                    best_similarity = similarity
            
            # Add to existing group or create new group
            if best_group:
                similarity_groups[best_group].append(result)
            else:
                # Create new group with unique key
                new_group_key = f"group_{len(similarity_groups)}"
                similarity_groups[new_group_key].append(result)
        
        # Select top results from each group
        for group_results in similarity_groups.values():
            # Sort group by relevance and take top results
            group_results.sort(key=lambda x: x.relevance_score, reverse=True)
            processed_results.extend(group_results[:max_similar])
        
        # Re-sort all selected results
        processed_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return processed_results
    
    def update_weights_from_feedback(self, query: str, results: List[SearchResult], 
                                   user_selections: List[str]):
        """Update ranking weights based on user feedback"""
        # This would implement learning from user selections
        # For now, just log the feedback
        self.feedback_history.append({
            'query': query,
            'total_results': len(results),
            'selected_results': len(user_selections),
            'timestamp': time.time()
        })
        
        # TODO: Implement weight adjustment based on which results users select
        self.logger.debug(f"Recorded feedback for query: {query}")
    
    def get_ranking_explanation(self, result: SearchResult) -> Dict[str, Any]:
        """Get explanation of why a result was ranked at its position"""
        if 'relevance_signals' not in result.metadata:
            return {"error": "No relevance signals available"}
        
        signals = result.metadata['relevance_signals']
        
        explanation = {
            "final_score": result.relevance_score,
            "signal_breakdown": {
                "semantic_similarity": {
                    "value": signals.get('semantic_similarity', 0.0),
                    "weight": self.weights.semantic_similarity,
                    "contribution": signals.get('semantic_similarity', 0.0) * self.weights.semantic_similarity
                },
                "exact_match": {
                    "value": signals.get('exact_match', 0.0),
                    "weight": self.weights.exact_match,
                    "contribution": signals.get('exact_match', 0.0) * self.weights.exact_match
                },
                "structural_relevance": {
                    "value": signals.get('structural_relevance', 0.0),
                    "weight": self.weights.structural_relevance,
                    "contribution": signals.get('structural_relevance', 0.0) * self.weights.structural_relevance
                }
            }
        }
        
        return explanation


# Convenience functions
def create_ranking_context(query_text: str, query_intent: str = "unknown",
                          active_files: Set[str] = None,
                          recent_entities: List[str] = None) -> RankingContext:
    """Create a ranking context with common defaults"""
    return RankingContext(
        query_text=query_text,
        query_intent=query_intent,
        active_files=active_files or set(),
        recent_entities=recent_entities or []
    )


def rank_search_results(results: List[SearchResult], 
                       query_text: str,
                       query_intent: str = "unknown") -> List[SearchResult]:
    """Convenience function for ranking results"""
    context = create_ranking_context(query_text, query_intent)
    ranker = ResultRanker()
    return ranker.rank_results(results, context)


# Example usage
if __name__ == "__main__":
    # Test ranking with sample results
    sample_results = [
        SearchResult(
            entity_id="1",
            entity_name="authenticateUser", 
            entity_type="function",
            file_path="/src/auth.py",
            relevance_score=0.8,
            source_strategy="vector"
        ),
        SearchResult(
            entity_id="2",
            entity_name="loginHandler",
            entity_type="function", 
            file_path="/src/handlers.py",
            relevance_score=0.6,
            source_strategy="graph"
        )
    ]
    
    ranked = rank_search_results(sample_results, "find authentication functions")
    
    for i, result in enumerate(ranked):
        print(f"{i+1}. {result.entity_name} - {result.relevance_score:.2f}")