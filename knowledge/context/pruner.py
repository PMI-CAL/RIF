"""
Context pruning system for optimizing results to fit agent context windows.

Implements intelligent pruning strategies to maintain essential context
while staying within token limits for different agent configurations.
"""

import json
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Context window budget allocation."""
    total_tokens: int
    direct_results: int  # 50% for direct results
    context_preservation: int  # 25% for context
    reserve: int  # 25% reserve for flexibility
    
    @classmethod
    def from_window_size(cls, window_size: int) -> 'ContextBudget':
        """Create budget from total context window size."""
        return cls(
            total_tokens=window_size,
            direct_results=int(window_size * 0.5),
            context_preservation=int(window_size * 0.25),
            reserve=int(window_size * 0.25)
        )


class ContextPruner:
    """
    Intelligent context pruning for agent consumption optimization.
    
    Pruning Strategies:
    1. Keep highest scored items within budget
    2. Preserve essential context connections
    3. Maintain result diversity
    4. Add summarization for overflow
    """
    
    # Predefined context window sizes for different agents
    AGENT_WINDOWS = {
        'rif-analyst': 8000,
        'rif-architect': 12000,
        'rif-implementer': 6000,
        'rif-validator': 8000,
        'rif-learner': 10000,
        'default': 8000
    }
    
    # Token estimation multipliers for different content types
    TOKEN_MULTIPLIERS = {
        'text': 1.3,  # Natural language text
        'code': 1.8,  # Code content (more tokens per character)
        'json': 1.5,  # Structured data
        'metadata': 1.0  # Simple key-value pairs
    }
    
    def __init__(self):
        self.estimation_cache = {}
    
    def prune_results(self,
                     results: List[Dict[str, Any]],
                     agent_type: str = 'default',
                     custom_window: Optional[int] = None,
                     preserve_context: bool = True,
                     min_results: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prune query results to fit within agent context window.
        
        Args:
            results: List of query results with relevance scores
            agent_type: Type of agent (determines context window size)
            custom_window: Override default window size
            preserve_context: Whether to preserve context connections
            min_results: Minimum number of results to include
            
        Returns:
            Tuple of (pruned_results, pruning_info)
        """
        if not results:
            return [], {'pruning_applied': False, 'reason': 'no_results'}
        
        # Determine context budget
        window_size = custom_window or self.AGENT_WINDOWS.get(agent_type, self.AGENT_WINDOWS['default'])
        budget = ContextBudget.from_window_size(window_size)
        
        # Sort results by relevance score (highest first)
        sorted_results = sorted(results, key=lambda r: r.get('relevance_score', 0.0), reverse=True)
        
        # Try progressive pruning strategies
        pruning_info = {
            'original_count': len(results),
            'window_size': window_size,
            'budget': budget.__dict__,
            'strategy_used': None,
            'tokens_saved': 0,
            'pruning_applied': True
        }
        
        # Strategy 1: Basic token-based pruning
        basic_pruned = self._basic_token_pruning(sorted_results, budget, min_results)
        
        if basic_pruned:
            pruning_info['strategy_used'] = 'basic_token_pruning'
            pruning_info['final_count'] = len(basic_pruned)
            pruning_info['tokens_saved'] = self._estimate_tokens(results) - self._estimate_tokens(basic_pruned)
            
            # Strategy 2: Context preservation (if requested and needed)
            if preserve_context and len(basic_pruned) < len(results):
                preserved_results = self._preserve_essential_context(basic_pruned, results, budget)
                if preserved_results:
                    pruning_info['strategy_used'] = 'context_preserving_pruning'
                    pruning_info['final_count'] = len(preserved_results)
                    return preserved_results, pruning_info
            
            return basic_pruned, pruning_info
        
        # Fallback: Keep minimum results even if over budget
        fallback_results = sorted_results[:min_results]
        pruning_info['strategy_used'] = 'fallback_minimum'
        pruning_info['final_count'] = len(fallback_results)
        pruning_info['warning'] = 'Exceeded context window with minimum results'
        
        return fallback_results, pruning_info
    
    def _basic_token_pruning(self,
                           results: List[Dict[str, Any]],
                           budget: ContextBudget,
                           min_results: int) -> List[Dict[str, Any]]:
        """
        Basic pruning strategy based on token budget and relevance scores.
        """
        pruned = []
        current_tokens = 0
        
        for result in results:
            result_tokens = self._estimate_result_tokens(result)
            
            # Always include minimum results, even if over budget
            if len(pruned) < min_results:
                pruned.append(result)
                current_tokens += result_tokens
                continue
            
            # Check if adding this result would exceed budget
            if current_tokens + result_tokens <= budget.direct_results:
                pruned.append(result)
                current_tokens += result_tokens
            else:
                # Try summarization if we have reserve tokens
                if current_tokens + budget.reserve >= budget.total_tokens * 0.8:
                    break
                
                # Consider adding summarized version
                summarized_result = self._create_summarized_result(result, budget.reserve // 2)
                if summarized_result:
                    summary_tokens = self._estimate_result_tokens(summarized_result)
                    if current_tokens + summary_tokens <= budget.direct_results + budget.reserve:
                        pruned.append(summarized_result)
                        current_tokens += summary_tokens
                
                break
        
        return pruned
    
    def _preserve_essential_context(self,
                                  pruned_results: List[Dict[str, Any]],
                                  original_results: List[Dict[str, Any]],
                                  budget: ContextBudget) -> Optional[List[Dict[str, Any]]]:
        """
        Add back essential context items that provide important connections.
        """
        if not pruned_results:
            return None
        
        # Identify context items not in pruned results
        pruned_ids = {r.get('id', str(i)) for i, r in enumerate(pruned_results)}
        context_candidates = [r for r in original_results if r.get('id', str(original_results.index(r))) not in pruned_ids]
        
        if not context_candidates:
            return pruned_results
        
        # Find context items that provide important connections
        essential_context = self._identify_essential_context(pruned_results, context_candidates)
        
        # Check if we can fit essential context within budget
        current_tokens = self._estimate_tokens(pruned_results)
        available_tokens = budget.context_preservation
        
        enhanced_results = pruned_results.copy()
        
        for context_item in essential_context:
            context_tokens = self._estimate_result_tokens(context_item)
            
            if current_tokens + context_tokens <= budget.direct_results + available_tokens:
                enhanced_results.append(context_item)
                current_tokens += context_tokens
                available_tokens -= context_tokens
            else:
                # Try adding summarized version
                summarized = self._create_summarized_result(context_item, available_tokens)
                if summarized:
                    summary_tokens = self._estimate_result_tokens(summarized)
                    if current_tokens + summary_tokens <= budget.direct_results + budget.context_preservation:
                        enhanced_results.append(summarized)
                        current_tokens += summary_tokens
                        break
        
        return enhanced_results
    
    def _identify_essential_context(self,
                                  pruned_results: List[Dict[str, Any]],
                                  candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify context items that provide essential connections to pruned results.
        """
        essential = []
        
        # Extract metadata from pruned results for connection analysis
        pruned_sources = {r.get('metadata', {}).get('source', '') for r in pruned_results}
        pruned_tags = set()
        for result in pruned_results:
            tags = str(result.get('metadata', {}).get('tags', ''))
            pruned_tags.update(tags.split(','))
        
        pruned_types = {r.get('metadata', {}).get('type', '') for r in pruned_results}
        
        for candidate in candidates:
            candidate_metadata = candidate.get('metadata', {})
            candidate_source = candidate_metadata.get('source', '')
            candidate_tags = set(str(candidate_metadata.get('tags', '')).split(','))
            candidate_type = candidate_metadata.get('type', '')
            
            # Connection scoring
            connection_score = 0.0
            
            # Same source connection
            if candidate_source in pruned_sources:
                connection_score += 0.6
            
            # Tag overlap
            tag_overlap = len(candidate_tags & pruned_tags)
            if tag_overlap > 0:
                connection_score += (tag_overlap / max(len(candidate_tags), len(pruned_tags))) * 0.4
            
            # Complementary types (decisions complement patterns, etc.)
            if candidate_type == 'decision' and 'pattern' in pruned_types:
                connection_score += 0.3
            elif candidate_type == 'learning' and ('pattern' in pruned_types or 'decision' in pruned_types):
                connection_score += 0.4
            
            # High connection score means essential context
            if connection_score >= 0.5:
                candidate['connection_score'] = connection_score
                essential.append(candidate)
        
        # Sort by connection score
        essential.sort(key=lambda x: x.get('connection_score', 0.0), reverse=True)
        
        return essential[:3]  # Limit to top 3 essential context items
    
    def _create_summarized_result(self, result: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """
        Create a summarized version of a result to fit token budget.
        """
        if max_tokens < 50:  # Too small to be useful
            return None
        
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        # Create summary based on content type
        if isinstance(content, dict):
            # Structured content - keep essential keys
            summarized_content = self._summarize_structured_content(content, max_tokens)
        else:
            # Text content - truncate intelligently
            summarized_content = self._summarize_text_content(str(content), max_tokens)
        
        # Create summarized result
        summarized_result = {
            'id': result.get('id', ''),
            'content': summarized_content,
            'metadata': metadata.copy(),
            'distance': result.get('distance', 1.0),
            'relevance_score': result.get('relevance_score', 0.0),
            'collection': result.get('collection', ''),
            'summarized': True,
            'original_length': len(str(content))
        }
        
        # Add summary indicator to metadata
        summarized_result['metadata']['summarized'] = True
        summarized_result['metadata']['summary_ratio'] = len(str(summarized_content)) / max(len(str(content)), 1)
        
        return summarized_result
    
    def _summarize_structured_content(self, content: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Summarize structured (JSON) content by keeping essential keys."""
        essential_keys = [
            'title', 'description', 'summary', 'name', 'type', 'status',
            'decision', 'recommendation', 'approach', 'pattern', 'implementation'
        ]
        
        summarized = {}
        current_size = 0
        
        # Add essential keys first
        for key in essential_keys:
            if key in content and current_size < max_tokens * 0.75:
                value = content[key]
                value_size = len(str(value)) * self.TOKEN_MULTIPLIERS['json']
                
                if current_size + value_size <= max_tokens * 0.75:
                    summarized[key] = value
                    current_size += value_size
                else:
                    # Truncate long values
                    max_value_tokens = max_tokens * 0.75 - current_size
                    max_chars = int(max_value_tokens / self.TOKEN_MULTIPLIERS['json'])
                    if max_chars > 50:
                        truncated_value = str(value)[:max_chars] + '...'
                        summarized[key] = truncated_value
                        break
        
        # Add other keys if space allows
        remaining_keys = [k for k in content.keys() if k not in essential_keys and k not in summarized]
        for key in remaining_keys:
            if current_size >= max_tokens * 0.9:
                break
            
            value = content[key]
            value_size = len(str(value)) * self.TOKEN_MULTIPLIERS['json']
            
            if current_size + value_size <= max_tokens * 0.9:
                summarized[key] = value
                current_size += value_size
        
        # Add truncation indicator
        if len(summarized) < len(content):
            summarized['_truncated'] = f"Showing {len(summarized)}/{len(content)} fields"
        
        return summarized
    
    def _summarize_text_content(self, content: str, max_tokens: int) -> str:
        """Summarize text content by intelligent truncation."""
        max_chars = int(max_tokens / self.TOKEN_MULTIPLIERS['text'])
        
        if len(content) <= max_chars:
            return content
        
        # Try to break at sentence boundaries
        sentences = content.split('. ')
        summarized = ""
        
        for sentence in sentences:
            if len(summarized + sentence + '. ') <= max_chars - 20:  # Leave room for ellipsis
                summarized += sentence + '. '
            else:
                break
        
        if not summarized:  # Fallback to character truncation
            summarized = content[:max_chars - 3]
        
        return summarized.rstrip() + '...'
    
    def _estimate_result_tokens(self, result: Dict[str, Any]) -> int:
        """Estimate token count for a single result."""
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        # Content tokens
        if isinstance(content, dict):
            content_tokens = len(json.dumps(content)) * self.TOKEN_MULTIPLIERS['json']
        elif self._is_code_content(content):
            content_tokens = len(str(content)) * self.TOKEN_MULTIPLIERS['code']
        else:
            content_tokens = len(str(content)) * self.TOKEN_MULTIPLIERS['text']
        
        # Metadata tokens
        metadata_tokens = len(json.dumps(metadata)) * self.TOKEN_MULTIPLIERS['metadata']
        
        # Overhead tokens (for structure, formatting, etc.)
        overhead_tokens = 50
        
        return int(content_tokens + metadata_tokens + overhead_tokens)
    
    def _estimate_tokens(self, results: List[Dict[str, Any]]) -> int:
        """Estimate total token count for a list of results."""
        cache_key = str([r.get('id', '') for r in results])
        
        if cache_key in self.estimation_cache:
            return self.estimation_cache[cache_key]
        
        total = sum(self._estimate_result_tokens(result) for result in results)
        self.estimation_cache[cache_key] = total
        
        return total
    
    def _is_code_content(self, content: Any) -> bool:
        """Detect if content contains code."""
        content_str = str(content).lower()
        
        # Simple heuristics for code detection
        code_indicators = [
            'function ', 'class ', 'def ', 'import ', 'require(',
            '{', '}', '()', '=>', '==', '!=', '&&', '||',
            'console.log', 'print(', 'return ', 'if (', 'for ('
        ]
        
        return any(indicator in content_str for indicator in code_indicators)
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get statistics about pruning operations."""
        return {
            'cache_size': len(self.estimation_cache),
            'agent_windows': self.AGENT_WINDOWS,
            'token_multipliers': self.TOKEN_MULTIPLIERS
        }