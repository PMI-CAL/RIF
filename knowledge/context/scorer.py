"""
Relevance scoring system for context optimization.

Implements multi-factor relevance scoring algorithm to rank query results
based on direct relevance, semantic similarity, structural importance,
and temporal factors.
"""

import math
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Multi-dimensional relevance scoring system for query results.
    
    Scoring Algorithm:
    - Direct relevance (40%): exact matches, keyword overlaps
    - Semantic relevance (30%): embedding similarity scores
    - Structural relevance (20%): graph distance, dependency relationships
    - Temporal relevance (10%): recency and access patterns
    """
    
    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        'direct': 0.40,
        'semantic': 0.30,
        'structural': 0.20,
        'temporal': 0.10
    }
    
    def __init__(self):
        self.query_cache = {}
        self.term_extractor = self._create_term_extractor()
    
    def calculate_relevance_score(self, 
                                query: str,
                                result: Dict[str, Any],
                                query_context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate comprehensive relevance score for a query result.
        
        Args:
            query: Original search query
            result: Query result with content, metadata, and distance
            query_context: Optional context about the query (agent type, issue, etc.)
            
        Returns:
            Relevance score between 0.0 and 1.0 (higher = more relevant)
        """
        try:
            # Extract query terms for scoring
            query_terms = self._extract_query_terms(query)
            
            # Calculate component scores
            direct_score = self._calculate_direct_relevance(query_terms, result)
            semantic_score = self._calculate_semantic_relevance(result)
            structural_score = self._calculate_structural_relevance(result, query_context)
            temporal_score = self._calculate_temporal_relevance(result)
            
            # Weighted combination
            total_score = (
                direct_score * self.WEIGHTS['direct'] +
                semantic_score * self.WEIGHTS['semantic'] +
                structural_score * self.WEIGHTS['structural'] +
                temporal_score * self.WEIGHTS['temporal']
            )
            
            # Clamp to valid range
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.warning(f"Error calculating relevance score: {e}")
            # Fallback to semantic score if available
            return self._calculate_semantic_relevance(result)
    
    def _calculate_direct_relevance(self, query_terms: List[str], result: Dict[str, Any]) -> float:
        """
        Calculate direct relevance based on exact matches and keyword overlaps.
        
        Factors:
        - Exact phrase matches: +0.8
        - Individual keyword matches: +0.4 per term
        - Case-insensitive matches: +0.2 per term
        - Title/metadata matches: +0.6 per term
        """
        content = str(result.get('content', ''))
        metadata = result.get('metadata', {})
        title = str(metadata.get('title', ''))
        
        content_lower = content.lower()
        title_lower = title.lower()
        
        score = 0.0
        total_terms = len(query_terms)
        
        if total_terms == 0:
            return 0.0
        
        # Check for exact phrase matches (high value)
        query_phrase = ' '.join(query_terms).lower()
        if query_phrase in content_lower:
            score += 0.8
        if query_phrase in title_lower:
            score += 0.9  # Title matches even more valuable
        
        # Individual term matching
        matched_terms = 0
        for term in query_terms:
            term_lower = term.lower()
            term_score = 0.0
            
            # Exact matches in content
            if term in content:
                term_score += 0.4
            elif term_lower in content_lower:
                term_score += 0.2
            
            # Title matches (higher weight)
            if term in title:
                term_score += 0.6
            elif term_lower in title_lower:
                term_score += 0.4
            
            # Metadata matches
            for key, value in metadata.items():
                if term_lower in str(value).lower():
                    term_score += 0.3
                    break
            
            if term_score > 0:
                matched_terms += 1
                score += term_score
        
        # Normalize by number of query terms
        normalized_score = score / max(total_terms, 1)
        
        # Boost for high match percentage
        match_percentage = matched_terms / total_terms
        if match_percentage >= 0.8:
            normalized_score *= 1.2
        elif match_percentage >= 0.5:
            normalized_score *= 1.1
        
        return min(1.0, normalized_score)
    
    def _calculate_semantic_relevance(self, result: Dict[str, Any]) -> float:
        """
        Calculate semantic relevance from embedding similarity scores.
        
        Uses the distance field from vector search results.
        Lower distance = higher relevance.
        """
        distance = result.get('distance', 1.0)
        
        # Convert distance to similarity score
        # Most embedding models use cosine distance [0, 2]
        # Convert to relevance score [0, 1] where 0 distance = 1.0 relevance
        if distance <= 0:
            return 1.0
        
        # Use exponential decay for distance
        # This emphasizes the difference between very similar items
        similarity = math.exp(-distance * 2)
        return min(1.0, max(0.0, similarity))
    
    def _calculate_structural_relevance(self, 
                                      result: Dict[str, Any], 
                                      query_context: Optional[Dict[str, Any]]) -> float:
        """
        Calculate structural relevance based on graph relationships and dependencies.
        
        Factors:
        - Direct dependencies: +0.8
        - Same component/module: +0.6
        - Related patterns: +0.4
        - Issue relationships: +0.5
        """
        if not query_context:
            return 0.5  # Neutral score without context
        
        metadata = result.get('metadata', {})
        result_type = metadata.get('type', '')
        result_source = metadata.get('source', '')
        result_tags = set(str(metadata.get('tags', '')).split(','))
        
        context_issue = query_context.get('issue_id')
        context_component = query_context.get('component')
        context_agent = query_context.get('agent_type')
        context_tags = set(query_context.get('tags', []))
        
        score = 0.0
        
        # Same issue relationship (very high relevance)
        if context_issue and result_source == f"issue_{context_issue}":
            score += 0.8
        
        # Related issues (moderate relevance)
        if context_issue and 'issue_' in result_source:
            score += 0.3
        
        # Same component/module
        if context_component:
            if context_component.lower() in str(metadata.get('component', '')).lower():
                score += 0.6
            if context_component.lower() in result.get('content', '').lower():
                score += 0.4
        
        # Tag overlap (indicates related concepts)
        if result_tags and context_tags:
            tag_overlap = len(result_tags & context_tags)
            tag_union = len(result_tags | context_tags)
            if tag_union > 0:
                jaccard_similarity = tag_overlap / tag_union
                score += jaccard_similarity * 0.5
        
        # Agent type relevance
        if context_agent and context_agent.lower() in result.get('content', '').lower():
            score += 0.3
        
        # Pattern type relevance
        if result_type == 'pattern':
            score += 0.2  # Patterns generally useful
        elif result_type == 'decision':
            score += 0.3  # Decisions often crucial
        elif result_type == 'learning':
            score += 0.4  # Learnings help avoid mistakes
        
        return min(1.0, score)
    
    def _calculate_temporal_relevance(self, result: Dict[str, Any]) -> float:
        """
        Calculate temporal relevance based on recency and access patterns.
        
        Factors:
        - Recent items get higher scores
        - Frequently accessed items get boost
        - Stale patterns get penalties
        """
        metadata = result.get('metadata', {})
        
        # Extract timestamps
        created_at = self._parse_timestamp(metadata.get('created_at'))
        updated_at = self._parse_timestamp(metadata.get('updated_at'))
        accessed_at = self._parse_timestamp(metadata.get('last_accessed'))
        
        now = datetime.now()
        score = 0.0
        
        # Recency scoring (exponential decay)
        relevant_time = updated_at or created_at or now
        days_old = (now - relevant_time).days
        
        if days_old <= 1:
            score += 1.0  # Very recent
        elif days_old <= 7:
            score += 0.8  # Recent
        elif days_old <= 30:
            score += 0.6  # Moderately recent
        elif days_old <= 90:
            score += 0.4  # Somewhat old
        else:
            # Older items get exponential decay
            score += max(0.1, 0.4 * math.exp(-days_old / 365))
        
        # Access frequency boost
        access_count = metadata.get('access_count', 0)
        if access_count > 10:
            score *= 1.2
        elif access_count > 5:
            score *= 1.1
        
        # Recent access boost
        if accessed_at and (now - accessed_at).days <= 7:
            score *= 1.15
        
        return min(1.0, score)
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query string."""
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Basic term extraction (can be enhanced with NLP)
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract terms
        terms = re.findall(r'\b\w+\b', query.lower())
        meaningful_terms = [term for term in terms if term not in stop_words and len(term) > 2]
        
        self.query_cache[query] = meaningful_terms
        return meaningful_terms
    
    def _parse_timestamp(self, timestamp_str: Any) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if not timestamp_str:
            return None
        
        timestamp_str = str(timestamp_str)
        
        # Common formats
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _create_term_extractor(self):
        """Create term extraction utilities."""
        # Could be enhanced with NLP libraries like spacy or nltk
        return {
            'patterns': {
                'code_terms': re.compile(r'\b(?:function|class|method|variable|api|endpoint)\b', re.I),
                'technical_terms': re.compile(r'\b(?:bug|error|fix|implement|optimize|refactor)\b', re.I),
                'domain_terms': re.compile(r'\b(?:user|auth|database|frontend|backend|service)\b', re.I)
            }
        }
    
    def get_score_breakdown(self, 
                          query: str,
                          result: Dict[str, Any],
                          query_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get detailed breakdown of relevance score components.
        
        Useful for debugging and explaining scoring decisions.
        """
        query_terms = self._extract_query_terms(query)
        
        breakdown = {
            'direct': self._calculate_direct_relevance(query_terms, result),
            'semantic': self._calculate_semantic_relevance(result),
            'structural': self._calculate_structural_relevance(result, query_context),
            'temporal': self._calculate_temporal_relevance(result)
        }
        
        breakdown['weighted_total'] = sum(
            breakdown[component] * self.WEIGHTS[component]
            for component in breakdown
        )
        
        breakdown['weights'] = self.WEIGHTS.copy()
        
        return breakdown