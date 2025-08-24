"""
Query Strategy Planner - Issue #33
Determines optimal execution strategy for hybrid searches
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .query_parser import StructuredQuery, QueryIntent, SearchStrategy


class ExecutionMode(Enum):
    """Query execution mode"""
    FAST = "fast"           # Prioritize speed over completeness
    BALANCED = "balanced"   # Balance speed and accuracy
    COMPREHENSIVE = "comprehensive"  # Prioritize completeness over speed


@dataclass 
class ResourceConstraints:
    """Resource limits for query execution"""
    max_latency_ms: int = 500         # Maximum allowed latency
    max_memory_mb: int = 200          # Maximum memory usage
    max_cpu_cores: int = 2            # Maximum CPU cores to use
    max_concurrent_searches: int = 4   # Max parallel searches


@dataclass
class SearchCostEstimate:
    """Estimated cost for a search strategy"""
    latency_ms: int
    memory_mb: int
    cpu_utilization: float  # 0.0 to 1.0
    accuracy_score: float   # 0.0 to 1.0 expected accuracy
    confidence: float       # 0.0 to 1.0 confidence in estimate


@dataclass 
class ExecutionPlan:
    """Detailed execution plan for query"""
    strategy: SearchStrategy
    mode: ExecutionMode
    
    # Vector search plan
    vector_enabled: bool = False
    vector_query: Optional[str] = None
    vector_limit: int = 50
    vector_threshold: float = 0.7
    
    # Graph search plan
    graph_enabled: bool = False
    graph_start_entities: List[Dict[str, Any]] = field(default_factory=list)
    graph_max_depth: int = 3
    graph_relationship_types: List[str] = field(default_factory=list)
    graph_direction: str = "both"  # outgoing, incoming, both
    
    # Direct lookup plan
    direct_enabled: bool = False
    direct_entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Parallel execution settings
    parallel_execution: bool = True
    max_workers: int = 2
    timeout_per_search_ms: int = 1000
    
    # Result fusion settings
    fusion_strategy: str = "weighted_merge"  # weighted_merge, rank_fusion, cascade
    result_limit: int = 20
    
    # Performance estimates
    estimated_cost: SearchCostEstimate = None


class StrategyPlanner:
    """
    Plans optimal execution strategy for hybrid search queries.
    
    Analyzes query complexity, resource constraints, and performance requirements
    to determine the best execution approach.
    """
    
    def __init__(self, constraints: ResourceConstraints = None):
        self.constraints = constraints or ResourceConstraints()
        self.logger = logging.getLogger(__name__)
        
        # Performance history for cost estimation
        self.performance_history = {
            'vector_search_latency': 50,   # ms per search
            'graph_search_latency': 100,   # ms per search  
            'direct_search_latency': 10,   # ms per search
            'parallel_overhead': 20,       # ms overhead for parallel execution
            'result_fusion_latency': 15    # ms for result merging
        }
        
        # Strategy effectiveness scores (learned from usage)
        self.strategy_effectiveness = {
            QueryIntent.ENTITY_SEARCH: {
                SearchStrategy.VECTOR_ONLY: 0.6,
                SearchStrategy.GRAPH_ONLY: 0.8,
                SearchStrategy.HYBRID_PARALLEL: 0.9
            },
            QueryIntent.SIMILARITY_SEARCH: {
                SearchStrategy.VECTOR_ONLY: 0.9,
                SearchStrategy.GRAPH_ONLY: 0.3,
                SearchStrategy.HYBRID_PARALLEL: 0.95
            },
            QueryIntent.DEPENDENCY_ANALYSIS: {
                SearchStrategy.VECTOR_ONLY: 0.2,
                SearchStrategy.GRAPH_ONLY: 0.95,
                SearchStrategy.HYBRID_PARALLEL: 0.9
            },
            QueryIntent.IMPACT_ANALYSIS: {
                SearchStrategy.VECTOR_ONLY: 0.3,
                SearchStrategy.GRAPH_ONLY: 0.9,
                SearchStrategy.HYBRID_PARALLEL: 0.85
            },
            QueryIntent.HYBRID_SEARCH: {
                SearchStrategy.VECTOR_ONLY: 0.6,
                SearchStrategy.GRAPH_ONLY: 0.7,
                SearchStrategy.HYBRID_PARALLEL: 1.0
            }
        }
    
    def plan_execution(self, query: StructuredQuery, 
                      mode: ExecutionMode = ExecutionMode.BALANCED) -> ExecutionPlan:
        """
        Create optimal execution plan for structured query.
        
        Args:
            query: Structured query from parser
            mode: Execution mode preference
            
        Returns:
            Detailed execution plan
        """
        self.logger.debug(f"Planning execution for query: {query.original_query}")
        
        # Estimate costs for different strategies
        strategy_costs = self._estimate_strategy_costs(query, mode)
        
        # Select optimal strategy based on costs and constraints
        optimal_strategy = self._select_optimal_strategy(query, strategy_costs, mode)
        
        # Create detailed execution plan
        execution_plan = self._create_execution_plan(query, optimal_strategy, mode)
        
        # Validate plan against constraints
        self._validate_plan(execution_plan)
        
        self.logger.debug(f"Selected strategy: {execution_plan.strategy.value}")
        return execution_plan
    
    def _estimate_strategy_costs(self, query: StructuredQuery, 
                               mode: ExecutionMode) -> Dict[SearchStrategy, SearchCostEstimate]:
        """Estimate costs for all possible strategies"""
        costs = {}
        
        for strategy in SearchStrategy:
            costs[strategy] = self._estimate_single_strategy_cost(query, strategy, mode)
        
        return costs
    
    def _estimate_single_strategy_cost(self, query: StructuredQuery, 
                                     strategy: SearchStrategy,
                                     mode: ExecutionMode) -> SearchCostEstimate:
        """Estimate cost for a single strategy"""
        latency_ms = 0
        memory_mb = 0
        cpu_utilization = 0.0
        
        # Base costs depending on strategy
        if strategy == SearchStrategy.VECTOR_ONLY:
            latency_ms = self.performance_history['vector_search_latency']
            memory_mb = 50  # Vector search memory
            cpu_utilization = 0.3
            
        elif strategy == SearchStrategy.GRAPH_ONLY:
            latency_ms = self.performance_history['graph_search_latency']
            memory_mb = 30  # Graph traversal memory
            cpu_utilization = 0.4
            
        elif strategy == SearchStrategy.HYBRID_PARALLEL:
            # Parallel execution - max of individual searches
            vector_latency = self.performance_history['vector_search_latency']
            graph_latency = self.performance_history['graph_search_latency']
            parallel_overhead = self.performance_history['parallel_overhead']
            fusion_latency = self.performance_history['result_fusion_latency']
            
            latency_ms = max(vector_latency, graph_latency) + parallel_overhead + fusion_latency
            memory_mb = 80  # Combined memory usage
            cpu_utilization = 0.6
            
        elif strategy == SearchStrategy.SEQUENTIAL:
            # Sequential execution - sum of individual searches
            latency_ms = (self.performance_history['vector_search_latency'] +
                         self.performance_history['graph_search_latency'] +
                         self.performance_history['result_fusion_latency'])
            memory_mb = 60  # Slightly less than parallel due to sequential memory usage
            cpu_utilization = 0.4
        
        # Adjust for query complexity
        complexity_multiplier = self._calculate_complexity_multiplier(query)
        latency_ms = int(latency_ms * complexity_multiplier)
        memory_mb = int(memory_mb * complexity_multiplier)
        
        # Adjust for execution mode
        mode_adjustments = {
            ExecutionMode.FAST: (0.7, 0.8),      # (latency_mult, accuracy_mult)
            ExecutionMode.BALANCED: (1.0, 1.0),
            ExecutionMode.COMPREHENSIVE: (1.5, 1.2)
        }
        
        latency_mult, accuracy_mult = mode_adjustments[mode]
        latency_ms = int(latency_ms * latency_mult)
        
        # Calculate expected accuracy
        base_accuracy = self.strategy_effectiveness.get(
            query.intent.primary_intent, {}
        ).get(strategy, 0.5)
        
        accuracy_score = min(1.0, base_accuracy * accuracy_mult)
        
        # Confidence based on historical data quality
        confidence = 0.8  # TODO: Update based on actual performance data
        
        return SearchCostEstimate(
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            cpu_utilization=cpu_utilization,
            accuracy_score=accuracy_score,
            confidence=confidence
        )
    
    def _calculate_complexity_multiplier(self, query: StructuredQuery) -> float:
        """Calculate query complexity multiplier"""
        multiplier = 1.0
        
        # Entity count factor
        entity_count = len(query.intent.entities)
        if entity_count > 3:
            multiplier += 0.2 * (entity_count - 3)
        
        # Concept count factor  
        concept_count = len(query.intent.concepts)
        if concept_count > 5:
            multiplier += 0.1 * (concept_count - 5)
        
        # Filter complexity
        if query.intent.filters.entity_types:
            multiplier += 0.1
        if query.intent.filters.file_patterns:
            multiplier += 0.15
        
        # Intent complexity
        intent_complexity = {
            QueryIntent.ENTITY_SEARCH: 1.0,
            QueryIntent.SIMILARITY_SEARCH: 1.2,
            QueryIntent.DEPENDENCY_ANALYSIS: 1.4,
            QueryIntent.IMPACT_ANALYSIS: 1.6,
            QueryIntent.HYBRID_SEARCH: 1.8
        }
        
        multiplier *= intent_complexity.get(query.intent.primary_intent, 1.0)
        
        return min(multiplier, 3.0)  # Cap at 3x
    
    def _select_optimal_strategy(self, query: StructuredQuery,
                               strategy_costs: Dict[SearchStrategy, SearchCostEstimate],
                               mode: ExecutionMode) -> SearchStrategy:
        """Select optimal strategy based on costs and constraints"""
        
        # Filter strategies that violate constraints
        viable_strategies = {}
        for strategy, cost in strategy_costs.items():
            if (cost.latency_ms <= self.constraints.max_latency_ms and
                cost.memory_mb <= self.constraints.max_memory_mb):
                viable_strategies[strategy] = cost
        
        if not viable_strategies:
            # If no strategies meet constraints, fallback to fastest
            self.logger.warning("No strategies meet constraints, using fastest available")
            return min(strategy_costs.items(), key=lambda x: x[1].latency_ms)[0]
        
        # Score strategies based on mode preferences
        if mode == ExecutionMode.FAST:
            # Prioritize speed
            best_strategy = min(viable_strategies.items(), 
                              key=lambda x: x[1].latency_ms)[0]
            
        elif mode == ExecutionMode.COMPREHENSIVE:
            # Prioritize accuracy
            best_strategy = max(viable_strategies.items(),
                              key=lambda x: x[1].accuracy_score)[0]
            
        else:  # BALANCED
            # Balance speed and accuracy using weighted score
            best_score = 0
            best_strategy = SearchStrategy.VECTOR_ONLY
            
            for strategy, cost in viable_strategies.items():
                # Normalize metrics (0-1 scale)
                speed_score = 1.0 - (cost.latency_ms / self.constraints.max_latency_ms)
                accuracy_score = cost.accuracy_score
                
                # Weighted combination (equal weight for balanced mode)
                combined_score = 0.5 * speed_score + 0.5 * accuracy_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_strategy = strategy
        
        return best_strategy
    
    def _create_execution_plan(self, query: StructuredQuery, 
                             strategy: SearchStrategy,
                             mode: ExecutionMode) -> ExecutionPlan:
        """Create detailed execution plan"""
        plan = ExecutionPlan(
            strategy=strategy,
            mode=mode,
            result_limit=query.result_limit
        )
        
        # Configure vector search
        if strategy in [SearchStrategy.VECTOR_ONLY, SearchStrategy.HYBRID_PARALLEL]:
            plan.vector_enabled = True
            plan.vector_query = query.vector_query
            plan.vector_limit = self._calculate_vector_limit(query, mode)
            plan.vector_threshold = self._calculate_vector_threshold(query, mode)
        
        # Configure graph search
        if strategy in [SearchStrategy.GRAPH_ONLY, SearchStrategy.HYBRID_PARALLEL]:
            plan.graph_enabled = True
            if query.graph_query:
                plan.graph_start_entities = query.graph_query.get('start_entities', [])
                plan.graph_max_depth = query.graph_query.get('max_depth', 3)
                plan.graph_relationship_types = query.graph_query.get('relationship_types', [])
                plan.graph_direction = query.graph_query.get('direction', 'both')
        
        # Configure direct lookup
        if query.direct_lookup:
            plan.direct_enabled = True
            plan.direct_entities = query.direct_lookup.get('entities', [])
        
        # Configure parallel execution
        if strategy == SearchStrategy.HYBRID_PARALLEL:
            plan.parallel_execution = True
            plan.max_workers = min(self.constraints.max_concurrent_searches, 4)
        else:
            plan.parallel_execution = False
            plan.max_workers = 1
        
        # Set timeouts
        plan.timeout_per_search_ms = min(
            query.timeout_ms // plan.max_workers if plan.max_workers > 0 else query.timeout_ms,
            2000  # Max 2 seconds per individual search
        )
        
        # Configure result fusion
        plan.fusion_strategy = self._select_fusion_strategy(strategy, mode)
        
        return plan
    
    def _calculate_vector_limit(self, query: StructuredQuery, mode: ExecutionMode) -> int:
        """Calculate vector search result limit"""
        base_limit = query.result_limit
        
        if mode == ExecutionMode.FAST:
            return min(base_limit, 30)
        elif mode == ExecutionMode.COMPREHENSIVE:
            return min(base_limit * 2, 100)
        else:  # BALANCED
            return base_limit
    
    def _calculate_vector_threshold(self, query: StructuredQuery, mode: ExecutionMode) -> float:
        """Calculate vector similarity threshold"""
        if mode == ExecutionMode.FAST:
            return 0.8  # Higher threshold, fewer but more relevant results
        elif mode == ExecutionMode.COMPREHENSIVE:
            return 0.5  # Lower threshold, more diverse results
        else:  # BALANCED
            return 0.7
    
    def _select_fusion_strategy(self, strategy: SearchStrategy, mode: ExecutionMode) -> str:
        """Select result fusion strategy"""
        if strategy in [SearchStrategy.VECTOR_ONLY, SearchStrategy.GRAPH_ONLY]:
            return "single_source"  # No fusion needed
        
        if mode == ExecutionMode.FAST:
            return "rank_fusion"     # Faster merging
        elif mode == ExecutionMode.COMPREHENSIVE:
            return "weighted_merge"  # More sophisticated merging  
        else:
            return "weighted_merge"  # Default balanced approach
    
    def _validate_plan(self, plan: ExecutionPlan):
        """Validate execution plan against constraints"""
        if plan.estimated_cost:
            if plan.estimated_cost.latency_ms > self.constraints.max_latency_ms:
                self.logger.warning(
                    f"Plan exceeds latency constraint: {plan.estimated_cost.latency_ms}ms > {self.constraints.max_latency_ms}ms"
                )
            
            if plan.estimated_cost.memory_mb > self.constraints.max_memory_mb:
                self.logger.warning(
                    f"Plan exceeds memory constraint: {plan.estimated_cost.memory_mb}MB > {self.constraints.max_memory_mb}MB"
                )
        
        if plan.max_workers > self.constraints.max_concurrent_searches:
            plan.max_workers = self.constraints.max_concurrent_searches
            self.logger.debug(f"Reduced workers to {plan.max_workers} due to constraints")
    
    def update_performance_history(self, strategy: SearchStrategy, 
                                 actual_latency_ms: int, 
                                 accuracy_score: float):
        """Update performance history with actual measurements"""
        # Simple exponential moving average for now
        alpha = 0.1  # Learning rate
        
        if strategy == SearchStrategy.VECTOR_ONLY:
            old_value = self.performance_history['vector_search_latency']
            self.performance_history['vector_search_latency'] = int(
                alpha * actual_latency_ms + (1 - alpha) * old_value
            )
        elif strategy == SearchStrategy.GRAPH_ONLY:
            old_value = self.performance_history['graph_search_latency']
            self.performance_history['graph_search_latency'] = int(
                alpha * actual_latency_ms + (1 - alpha) * old_value
            )
        
        # Update effectiveness scores
        # TODO: Implement adaptive effectiveness learning
        
        self.logger.debug(f"Updated performance history for {strategy.value}")


# Performance monitoring decorator
def measure_performance(planner: StrategyPlanner):
    """Decorator to measure and update performance metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Extract strategy from result if possible
                if hasattr(result, 'strategy'):
                    accuracy = getattr(result, 'accuracy_score', 0.8)  # Default accuracy
                    planner.update_performance_history(
                        result.strategy, execution_time_ms, accuracy
                    )
                
                return result
            except Exception as e:
                execution_time_ms = int((time.time() - start_time) * 1000)
                # Log failed execution
                logging.getLogger(__name__).error(f"Execution failed in {execution_time_ms}ms: {e}")
                raise
        return wrapper
    return decorator


# Convenience function
def plan_query_execution(query: StructuredQuery, 
                        mode: ExecutionMode = ExecutionMode.BALANCED,
                        constraints: ResourceConstraints = None) -> ExecutionPlan:
    """Convenience function for planning query execution"""
    planner = StrategyPlanner(constraints)
    return planner.plan_execution(query, mode)


# Example usage
if __name__ == "__main__":
    from .query_parser import parse_query
    
    # Test strategy planning
    test_query = parse_query("find functions similar to authentication that also handle errors")
    
    planner = StrategyPlanner()
    execution_plan = planner.plan_execution(test_query, ExecutionMode.BALANCED)
    
    print(f"Strategy: {execution_plan.strategy.value}")
    print(f"Vector enabled: {execution_plan.vector_enabled}")
    print(f"Graph enabled: {execution_plan.graph_enabled}")
    print(f"Parallel execution: {execution_plan.parallel_execution}")
    print(f"Max workers: {execution_plan.max_workers}")