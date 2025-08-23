"""
Main context optimizer that coordinates scoring and pruning for agent consumption.

This is the primary interface for context window optimization, integrating
relevance scoring and intelligent pruning to deliver optimal query results
within agent context constraints.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .scorer import RelevanceScorer
from .pruner import ContextPruner

logger = logging.getLogger(__name__)


class ContextOptimizer:
    """
    Main context optimization system for RIF knowledge queries.
    
    Coordinates relevance scoring and context pruning to optimize
    query results for agent consumption within context window limits.
    
    Usage:
        optimizer = ContextOptimizer()
        optimized_results = optimizer.optimize_for_agent(
            results=query_results,
            query="find authentication patterns",
            agent_type="rif-implementer",
            context={"issue_id": "34", "component": "auth"}
        )
    """
    
    def __init__(self):
        self.scorer = RelevanceScorer()
        self.pruner = ContextPruner()
        self.optimization_history = []
        self.performance_metrics = {
            'optimizations_performed': 0,
            'avg_relevance_improvement': 0.0,
            'avg_token_reduction': 0.0,
            'context_preservation_rate': 0.0
        }
    
    def optimize_for_agent(self,
                          results: List[Dict[str, Any]],
                          query: str,
                          agent_type: str = 'default',
                          context: Optional[Dict[str, Any]] = None,
                          custom_window: Optional[int] = None,
                          preserve_context: bool = True,
                          min_results: int = 3,
                          explain: bool = False) -> Dict[str, Any]:
        """
        Optimize query results for specific agent consumption.
        
        Args:
            results: Raw query results from knowledge system
            query: Original search query
            agent_type: Target agent type (determines context window)
            context: Query context (issue_id, component, etc.)
            custom_window: Override default context window size
            preserve_context: Whether to preserve context connections
            min_results: Minimum number of results to include
            explain: Include detailed optimization explanation
            
        Returns:
            Dictionary containing:
            {
                'optimized_results': [...],  # Optimized results list
                'optimization_info': {...},  # Optimization metadata
                'performance_stats': {...},  # Performance metrics
                'explanation': {...}         # Detailed explanation (if requested)
            }
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Calculate relevance scores for all results
            scored_results = self._add_relevance_scores(results, query, context)
            
            # Step 2: Apply context pruning
            pruned_results, pruning_info = self.pruner.prune_results(
                scored_results,
                agent_type=agent_type,
                custom_window=custom_window,
                preserve_context=preserve_context,
                min_results=min_results
            )
            
            # Step 3: Final quality adjustments
            final_results = self._apply_final_adjustments(pruned_results, query, context)
            
            # Step 4: Generate optimization info
            optimization_info = self._generate_optimization_info(
                original_results=results,
                scored_results=scored_results,
                final_results=final_results,
                pruning_info=pruning_info,
                query=query,
                agent_type=agent_type,
                context=context
            )
            
            # Step 5: Update performance metrics
            self._update_metrics(optimization_info)
            
            # Step 6: Record optimization history
            self._record_optimization(optimization_info, start_time)
            
            # Prepare return value
            result = {
                'optimized_results': final_results,
                'optimization_info': optimization_info,
                'performance_stats': self.performance_metrics.copy()
            }
            
            if explain:
                result['explanation'] = self._generate_explanation(
                    results, scored_results, final_results, optimization_info
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            # Return original results with error info
            return {
                'optimized_results': results,
                'optimization_info': {
                    'optimization_applied': False,
                    'error': str(e),
                    'fallback_used': True
                },
                'performance_stats': self.performance_metrics.copy()
            }
    
    def _add_relevance_scores(self,
                            results: List[Dict[str, Any]],
                            query: str,
                            context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add relevance scores to all results."""
        scored_results = []
        
        for result in results:
            scored_result = result.copy()
            relevance_score = self.scorer.calculate_relevance_score(query, result, context)
            scored_result['relevance_score'] = relevance_score
            scored_results.append(scored_result)
        
        return scored_results
    
    def _apply_final_adjustments(self,
                               results: List[Dict[str, Any]],
                               query: str,
                               context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply final quality and diversity adjustments."""
        if not results:
            return results
        
        adjusted_results = results.copy()
        
        # Ensure diversity in result types
        adjusted_results = self._ensure_result_diversity(adjusted_results)
        
        # Add quality indicators
        for result in adjusted_results:
            result['optimization_applied'] = True
            result['optimization_timestamp'] = datetime.now().isoformat()
        
        return adjusted_results
    
    def _ensure_result_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity in result types and sources."""
        if len(results) <= 3:
            return results  # Too few results to diversify
        
        # Group by type and source
        type_groups = {}
        source_groups = {}
        
        for i, result in enumerate(results):
            result_type = result.get('metadata', {}).get('type', 'unknown')
            result_source = result.get('metadata', {}).get('source', 'unknown')
            
            type_groups.setdefault(result_type, []).append((i, result))
            source_groups.setdefault(result_source, []).append((i, result))
        
        # If we have too many results from one type/source, rebalance
        max_per_type = max(2, len(results) // max(len(type_groups), 2))
        max_per_source = max(1, len(results) // max(len(source_groups), 3))
        
        balanced_results = []
        used_indices = set()
        
        # First, ensure we have diverse types
        for result_type, type_results in type_groups.items():
            type_count = 0
            for idx, result in type_results:
                if idx not in used_indices and type_count < max_per_type:
                    balanced_results.append(result)
                    used_indices.add(idx)
                    type_count += 1
        
        # Fill remaining slots with highest relevance scores
        remaining_results = [(i, r) for i, r in enumerate(results) if i not in used_indices]
        remaining_results.sort(key=lambda x: x[1].get('relevance_score', 0.0), reverse=True)
        
        for idx, result in remaining_results:
            if len(balanced_results) >= len(results):
                break
            balanced_results.append(result)
        
        return balanced_results
    
    def _generate_optimization_info(self,
                                  original_results: List[Dict[str, Any]],
                                  scored_results: List[Dict[str, Any]],
                                  final_results: List[Dict[str, Any]],
                                  pruning_info: Dict[str, Any],
                                  query: str,
                                  agent_type: str,
                                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive optimization information."""
        
        # Calculate relevance statistics
        original_scores = [r.get('distance', 1.0) for r in original_results]
        final_relevance_scores = [r.get('relevance_score', 0.0) for r in final_results]
        
        avg_original_distance = sum(original_scores) / max(len(original_scores), 1)
        avg_final_relevance = sum(final_relevance_scores) / max(len(final_relevance_scores), 1)
        
        # Token estimation
        original_tokens = self.pruner._estimate_tokens(original_results)
        final_tokens = self.pruner._estimate_tokens(final_results)
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'agent_type': agent_type,
            'context': context,
            'optimization_applied': True,
            
            'count_changes': {
                'original_count': len(original_results),
                'final_count': len(final_results),
                'reduction_percentage': (len(original_results) - len(final_results)) / max(len(original_results), 1) * 100
            },
            
            'relevance_improvements': {
                'avg_original_distance': avg_original_distance,
                'avg_final_relevance': avg_final_relevance,
                'relevance_improvement': avg_final_relevance - (1.0 - avg_original_distance)  # Approximate
            },
            
            'token_optimization': {
                'original_tokens': original_tokens,
                'final_tokens': final_tokens,
                'tokens_saved': original_tokens - final_tokens,
                'reduction_percentage': (original_tokens - final_tokens) / max(original_tokens, 1) * 100
            },
            
            'pruning_details': pruning_info,
            
            'quality_indicators': {
                'diversity_maintained': self._check_diversity(final_results),
                'context_preserved': pruning_info.get('strategy_used') == 'context_preserving_pruning',
                'min_relevance_threshold': min(final_relevance_scores) if final_relevance_scores else 0.0,
                'max_relevance_score': max(final_relevance_scores) if final_relevance_scores else 0.0
            }
        }
        
        return info
    
    def _check_diversity(self, results: List[Dict[str, Any]]) -> bool:
        """Check if results maintain good diversity."""
        if len(results) <= 2:
            return True
        
        types = set()
        sources = set()
        
        for result in results:
            metadata = result.get('metadata', {})
            types.add(metadata.get('type', 'unknown'))
            sources.add(metadata.get('source', 'unknown'))
        
        # Good diversity means at least 2 different types or 3 different sources
        return len(types) >= 2 or len(sources) >= 3
    
    def _update_metrics(self, optimization_info: Dict[str, Any]):
        """Update cumulative performance metrics."""
        self.performance_metrics['optimizations_performed'] += 1
        
        # Running average for relevance improvement
        current_improvement = optimization_info.get('relevance_improvements', {}).get('relevance_improvement', 0.0)
        n = self.performance_metrics['optimizations_performed']
        current_avg = self.performance_metrics['avg_relevance_improvement']
        self.performance_metrics['avg_relevance_improvement'] = (current_avg * (n-1) + current_improvement) / n
        
        # Running average for token reduction
        current_reduction = optimization_info.get('token_optimization', {}).get('reduction_percentage', 0.0)
        current_avg = self.performance_metrics['avg_token_reduction']
        self.performance_metrics['avg_token_reduction'] = (current_avg * (n-1) + current_reduction) / n
        
        # Context preservation rate
        context_preserved = 1.0 if optimization_info.get('quality_indicators', {}).get('context_preserved', False) else 0.0
        current_avg = self.performance_metrics['context_preservation_rate']
        self.performance_metrics['context_preservation_rate'] = (current_avg * (n-1) + context_preserved) / n
    
    def _record_optimization(self, optimization_info: Dict[str, Any], start_time: datetime):
        """Record optimization in history for analysis."""
        record = {
            'timestamp': start_time.isoformat(),
            'duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'optimization_info': optimization_info
        }
        
        self.optimization_history.append(record)
        
        # Keep only last 100 optimizations
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def _generate_explanation(self,
                            original_results: List[Dict[str, Any]],
                            scored_results: List[Dict[str, Any]],
                            final_results: List[Dict[str, Any]],
                            optimization_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanation of optimization decisions."""
        
        explanation = {
            'summary': f"Optimized {len(original_results)} results to {len(final_results)} results for agent consumption",
            
            'scoring_details': {
                'method': 'Multi-factor relevance scoring (direct 40%, semantic 30%, structural 20%, temporal 10%)',
                'score_distribution': self._analyze_score_distribution(scored_results)
            },
            
            'pruning_details': {
                'strategy': optimization_info.get('pruning_details', {}).get('strategy_used', 'unknown'),
                'token_budget': optimization_info.get('pruning_details', {}).get('budget', {}),
                'tokens_saved': optimization_info.get('token_optimization', {}).get('tokens_saved', 0)
            },
            
            'quality_preservation': {
                'diversity_maintained': optimization_info.get('quality_indicators', {}).get('diversity_maintained', False),
                'context_preserved': optimization_info.get('quality_indicators', {}).get('context_preserved', False),
                'min_relevance': optimization_info.get('quality_indicators', {}).get('min_relevance_threshold', 0.0)
            },
            
            'recommendations': self._generate_recommendations(optimization_info)
        }
        
        # Add detailed score breakdowns for top results
        if final_results:
            explanation['top_results_analysis'] = []
            for i, result in enumerate(final_results[:3]):
                if 'relevance_score' in result:
                    score_breakdown = self.scorer.get_score_breakdown(
                        optimization_info['query'], 
                        result, 
                        optimization_info.get('context')
                    )
                    explanation['top_results_analysis'].append({
                        'rank': i + 1,
                        'relevance_score': result['relevance_score'],
                        'score_breakdown': score_breakdown,
                        'title': result.get('metadata', {}).get('title', 'Untitled')
                    })
        
        return explanation
    
    def _analyze_score_distribution(self, scored_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of relevance scores."""
        scores = [r.get('relevance_score', 0.0) for r in scored_results]
        
        if not scores:
            return {'error': 'No scores available'}
        
        return {
            'min_score': min(scores),
            'max_score': max(scores),
            'avg_score': sum(scores) / len(scores),
            'score_range': max(scores) - min(scores),
            'high_quality_count': len([s for s in scores if s >= 0.7]),
            'medium_quality_count': len([s for s in scores if 0.4 <= s < 0.7]),
            'low_quality_count': len([s for s in scores if s < 0.4])
        }
    
    def _generate_recommendations(self, optimization_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        token_reduction = optimization_info.get('token_optimization', {}).get('reduction_percentage', 0.0)
        relevance_improvement = optimization_info.get('relevance_improvements', {}).get('relevance_improvement', 0.0)
        final_count = optimization_info.get('count_changes', {}).get('final_count', 0)
        
        if token_reduction > 50:
            recommendations.append("High token reduction achieved - consider if query scope can be narrowed")
        
        if relevance_improvement < 0.1:
            recommendations.append("Low relevance improvement - query terms might need refinement")
        
        if final_count < 3:
            recommendations.append("Few results returned - consider broadening search criteria")
        
        if not optimization_info.get('quality_indicators', {}).get('diversity_maintained', True):
            recommendations.append("Result diversity could be improved - consider diversifying query terms")
        
        if not recommendations:
            recommendations.append("Optimization performed successfully with good quality preservation")
        
        return recommendations
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization history."""
        return self.optimization_history[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'metrics': self.performance_metrics,
            'recent_optimizations': len(self.optimization_history),
            'avg_optimization_time': self._calculate_avg_optimization_time(),
            'optimization_trends': self._analyze_optimization_trends()
        }
    
    def _calculate_avg_optimization_time(self) -> float:
        """Calculate average optimization time from history."""
        if not self.optimization_history:
            return 0.0
        
        times = [record.get('duration_ms', 0) for record in self.optimization_history]
        return sum(times) / len(times)
    
    def _analyze_optimization_trends(self) -> Dict[str, Any]:
        """Analyze trends in optimization performance."""
        if len(self.optimization_history) < 5:
            return {'insufficient_data': True}
        
        recent = self.optimization_history[-5:]
        older = self.optimization_history[-10:-5] if len(self.optimization_history) >= 10 else []
        
        if not older:
            return {'insufficient_data': True}
        
        recent_avg_tokens_saved = sum(
            r.get('optimization_info', {}).get('token_optimization', {}).get('tokens_saved', 0) 
            for r in recent
        ) / len(recent)
        
        older_avg_tokens_saved = sum(
            r.get('optimization_info', {}).get('token_optimization', {}).get('tokens_saved', 0) 
            for r in older
        ) / len(older)
        
        return {
            'token_saving_trend': 'improving' if recent_avg_tokens_saved > older_avg_tokens_saved else 'stable',
            'recent_avg_tokens_saved': recent_avg_tokens_saved,
            'older_avg_tokens_saved': older_avg_tokens_saved
        }