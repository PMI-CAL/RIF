"""
Integration module connecting context optimization with RIF knowledge interface.

This module extends the knowledge interface to automatically apply context
optimization when retrieving results for agents.
"""

from typing import Dict, List, Any, Optional
import logging
from ..interface import KnowledgeInterface
from .optimizer import ContextOptimizer

logger = logging.getLogger(__name__)


class ContextOptimizedKnowledgeInterface(KnowledgeInterface):
    """
    Knowledge interface wrapper that automatically applies context optimization.
    
    This wrapper extends any KnowledgeInterface implementation to automatically
    optimize query results for agent consumption using the context optimization system.
    """
    
    def __init__(self, base_interface: KnowledgeInterface, enable_optimization: bool = True):
        """
        Initialize context-optimized knowledge interface.
        
        Args:
            base_interface: Underlying knowledge interface implementation
            enable_optimization: Whether to enable automatic context optimization
        """
        self.base_interface = base_interface
        self.optimizer = ContextOptimizer()
        self.enable_optimization = enable_optimization
        self.optimization_stats = {
            'queries_optimized': 0,
            'total_queries': 0,
            'avg_token_reduction': 0.0
        }
    
    def retrieve_knowledge(self, 
                          query: str, 
                          collection: Optional[str] = None, 
                          n_results: int = 5,
                          filters: Optional[Dict[str, Any]] = None,
                          agent_type: str = 'default',
                          optimization_context: Optional[Dict[str, Any]] = None,
                          disable_optimization: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge with automatic context optimization.
        
        Args:
            query: Search query string
            collection: Specific collection to search
            n_results: Maximum number of results
            filters: Optional metadata filters
            agent_type: Target agent type for optimization
            optimization_context: Context for relevance scoring
            disable_optimization: Skip optimization for this query
            
        Returns:
            List of optimized results
        """
        self.optimization_stats['total_queries'] += 1
        
        # Get raw results from base interface
        raw_results = self.base_interface.retrieve_knowledge(
            query=query,
            collection=collection,
            n_results=n_results * 2,  # Get more results for better optimization
            filters=filters
        )
        
        # Apply context optimization if enabled
        if self.enable_optimization and not disable_optimization and raw_results:
            try:
                optimization_result = self.optimizer.optimize_for_agent(
                    results=raw_results,
                    query=query,
                    agent_type=agent_type,
                    context=optimization_context,
                    min_results=min(n_results, len(raw_results))
                )
                
                optimized_results = optimization_result['optimized_results']
                
                # Update stats
                self.optimization_stats['queries_optimized'] += 1
                token_reduction = optimization_result['optimization_info'].get(
                    'token_optimization', {}
                ).get('reduction_percentage', 0.0)
                
                # Running average
                total_optimized = self.optimization_stats['queries_optimized']
                current_avg = self.optimization_stats['avg_token_reduction']
                self.optimization_stats['avg_token_reduction'] = (
                    (current_avg * (total_optimized - 1) + token_reduction) / total_optimized
                )
                
                logger.debug(f"Context optimization applied: {len(raw_results)} → {len(optimized_results)} results")
                
                return optimized_results[:n_results]
                
            except Exception as e:
                logger.warning(f"Context optimization failed, using raw results: {e}")
                return raw_results[:n_results]
        
        return raw_results[:n_results]
    
    def search_patterns(self, 
                       query: str, 
                       complexity: Optional[str] = None, 
                       limit: int = 5,
                       agent_type: str = 'default') -> List[Dict[str, Any]]:
        """
        Search patterns with context optimization.
        
        Extends base pattern search with automatic optimization.
        """
        filters = {"complexity": complexity} if complexity else None
        optimization_context = {
            "search_type": "patterns",
            "complexity": complexity,
            "agent_type": agent_type
        }
        
        return self.retrieve_knowledge(
            query=query,
            collection="patterns",
            n_results=limit,
            filters=filters,
            agent_type=agent_type,
            optimization_context=optimization_context
        )
    
    def search_decisions(self, 
                        query: str, 
                        status: Optional[str] = None, 
                        limit: int = 5,
                        agent_type: str = 'default') -> List[Dict[str, Any]]:
        """
        Search decisions with context optimization.
        """
        filters = {"status": status} if status else None
        optimization_context = {
            "search_type": "decisions",
            "status": status,
            "agent_type": agent_type
        }
        
        return self.retrieve_knowledge(
            query=query,
            collection="decisions",
            n_results=limit,
            filters=filters,
            agent_type=agent_type,
            optimization_context=optimization_context
        )
    
    def find_similar_issues(self, 
                           description: str, 
                           limit: int = 5,
                           agent_type: str = 'default') -> List[Dict[str, Any]]:
        """
        Find similar issues with context optimization.
        """
        optimization_context = {
            "search_type": "issues",
            "agent_type": agent_type
        }
        
        return self.retrieve_knowledge(
            query=description,
            collection="issue_resolutions",
            n_results=limit,
            agent_type=agent_type,
            optimization_context=optimization_context
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get context optimization statistics.
        
        Returns:
            Statistics about optimization performance
        """
        base_stats = self.optimization_stats.copy()
        
        if self.optimization_stats['total_queries'] > 0:
            base_stats['optimization_rate'] = (
                self.optimization_stats['queries_optimized'] / 
                self.optimization_stats['total_queries']
            )
        else:
            base_stats['optimization_rate'] = 0.0
        
        # Add optimizer performance stats
        optimizer_stats = self.optimizer.get_performance_summary()
        base_stats['optimizer_performance'] = optimizer_stats
        
        return base_stats
    
    def configure_optimization(self, 
                             enable: bool = True,
                             agent_windows: Optional[Dict[str, int]] = None) -> None:
        """
        Configure context optimization settings.
        
        Args:
            enable: Enable or disable optimization
            agent_windows: Custom context window sizes for agents
        """
        self.enable_optimization = enable
        
        if agent_windows:
            self.optimizer.pruner.AGENT_WINDOWS.update(agent_windows)
        
        logger.info(f"Context optimization {'enabled' if enable else 'disabled'}")
    
    # Delegate all other methods to base interface
    def store_knowledge(self, *args, **kwargs):
        return self.base_interface.store_knowledge(*args, **kwargs)
    
    def update_knowledge(self, *args, **kwargs):
        return self.base_interface.update_knowledge(*args, **kwargs)
    
    def delete_knowledge(self, *args, **kwargs):
        return self.base_interface.delete_knowledge(*args, **kwargs)
    
    def get_collection_stats(self):
        return self.base_interface.get_collection_stats()
    
    def store_pattern(self, *args, **kwargs):
        return self.base_interface.store_pattern(*args, **kwargs)
    
    def store_decision(self, *args, **kwargs):
        return self.base_interface.store_decision(*args, **kwargs)
    
    def store_learning(self, *args, **kwargs):
        return self.base_interface.store_learning(*args, **kwargs)
    
    def get_recent_patterns(self, *args, **kwargs):
        return self.base_interface.get_recent_patterns(*args, **kwargs)
    
    def get_system_info(self):
        base_info = self.base_interface.get_system_info()
        base_info['context_optimization'] = {
            'enabled': self.enable_optimization,
            'version': '1.0.0',
            'features': [
                'relevance_scoring',
                'context_pruning', 
                'agent_window_optimization',
                'performance_tracking'
            ]
        }
        return base_info


def create_optimized_knowledge_interface(base_interface: KnowledgeInterface) -> ContextOptimizedKnowledgeInterface:
    """
    Factory function to create context-optimized knowledge interface.
    
    Args:
        base_interface: Base knowledge interface implementation
        
    Returns:
        Context-optimized wrapper interface
    """
    return ContextOptimizedKnowledgeInterface(base_interface)


# Convenience function for agents
def optimize_query_results(results: List[Dict[str, Any]], 
                          query: str,
                          agent_type: str = 'default',
                          context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Standalone function to optimize query results.
    
    Useful for applying context optimization to existing query results
    without wrapping the entire knowledge interface.
    
    Args:
        results: Query results to optimize
        query: Original query string
        agent_type: Target agent type
        context: Query context for relevance scoring
        
    Returns:
        Optimized results list
    """
    optimizer = ContextOptimizer()
    
    optimization_result = optimizer.optimize_for_agent(
        results=results,
        query=query,
        agent_type=agent_type,
        context=context
    )
    
    return optimization_result['optimized_results']