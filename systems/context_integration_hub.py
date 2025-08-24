#!/usr/bin/env python3
"""
RIF Context Integration Hub

Central coordination layer that connects all 4 infrastructure engines and provides
unified context API for agents. Implements the foundational work for Issues #112/#113.

This hub integrates:
- ContextOptimizationEngine: Agent-specific context filtering
- LiveSystemContextEngine: Real-time system overview 
- DynamicDependencyTracker: Live dependency analysis
- DesignBenchmarkingFramework: Spec vs implementation comparison

Provides unified context delivery with relevance orchestration and window management.
"""

import json
import os
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

# Import the four infrastructure engines
try:
    from context_optimization_engine import ContextOptimizer, AgentType as ContextAgentType, ContextType
    from live_system_context_engine import LiveSystemContextEngine, AgentContextOptimizer, UpdateTrigger
    from dynamic_dependency_tracker import DynamicDependencyTracker, ChangeImpact, ImpactLevel
    from design_benchmarking_framework import BenchmarkingEngine, BenchmarkingResult, ComplianceLevel
except ImportError as e:
    print(f"Warning: Could not import all engines: {e}")
    # Create placeholder classes for development
    class ContextOptimizer:
        def __init__(self, *args, **kwargs): pass
        def optimize_for_agent(self, *args, **kwargs): return None
        def format_context_for_agent(self, *args, **kwargs): return ""
    
    class LiveSystemContextEngine:
        def __init__(self, *args, **kwargs): pass
        def get_live_context(self, *args, **kwargs): return None
        def get_optimized_context_for_agent(self, *args, **kwargs): return None
    
    class DynamicDependencyTracker:
        def __init__(self, *args, **kwargs): pass
        def get_component_info(self, *args, **kwargs): return {}
        def assess_change_impact(self, *args, **kwargs): return None
    
    class BenchmarkingEngine:
        def __init__(self, *args, **kwargs): pass
        def benchmark_issue(self, *args, **kwargs): return None

class AgentType(Enum):
    """Unified agent types for the integration hub"""
    ANALYST = "rif-analyst"
    PLANNER = "rif-planner"
    ARCHITECT = "rif-architect"
    IMPLEMENTER = "rif-implementer"
    VALIDATOR = "rif-validator"
    LEARNER = "rif-learner"
    PR_MANAGER = "rif-pr-manager"
    ERROR_ANALYST = "rif-error-analyst"
    SHADOW_AUDITOR = "rif-shadow-auditor"
    PROJECTGEN = "rif-projectgen"

class ContextScope(Enum):
    """Scope of context information requested"""
    MINIMAL = "minimal"          # Only essential context
    STANDARD = "standard"        # Normal comprehensive context
    COMPREHENSIVE = "comprehensive"  # Full detailed context
    DIAGNOSTIC = "diagnostic"    # Context for debugging/analysis

class RelevanceWeight(Enum):
    """Relevance weighting factors for cross-engine scoring"""
    CRITICAL = 1.0      # Always include
    HIGH = 0.8          # High priority for inclusion
    MEDIUM = 0.6        # Moderate priority
    LOW = 0.4           # Low priority
    MINIMAL = 0.2       # Include only if space available

@dataclass
class ContextRequest:
    """Request for unified context from an agent"""
    agent_type: AgentType
    issue_number: Optional[int] = None
    task_description: str = ""
    scope: ContextScope = ContextScope.STANDARD
    specific_components: List[str] = None
    include_dependencies: bool = True
    include_benchmarking: bool = False
    max_context_size: Optional[int] = None
    priority_keywords: List[str] = None
    force_refresh: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        data['scope'] = self.scope.value
        return data

@dataclass
class UnifiedContextResponse:
    """Unified context response combining all engine outputs"""
    agent_type: AgentType
    request_id: str
    context_sections: Dict[str, Any]
    total_relevance_score: float
    context_window_utilization: float
    estimated_tokens: int
    engine_contributions: Dict[str, float]  # How much each engine contributed
    optimization_applied: bool
    generation_timestamp: datetime
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['agent_type'] = self.agent_type.value
        data['generation_timestamp'] = self.generation_timestamp.isoformat()
        return data

class RelevanceOrchestrator:
    """Orchestrates cross-engine relevance scoring and context prioritization"""
    
    def __init__(self):
        # Agent-specific relevance weights for different engine types
        self.engine_weights = {
            AgentType.ANALYST: {
                'optimization': 0.3,    # Context filtering
                'system': 0.4,          # System overview
                'dependency': 0.2,      # Dependency analysis
                'benchmarking': 0.1     # Design comparison
            },
            AgentType.PLANNER: {
                'optimization': 0.25,
                'system': 0.35,
                'dependency': 0.25,
                'benchmarking': 0.15
            },
            AgentType.ARCHITECT: {
                'optimization': 0.2,
                'system': 0.3,
                'dependency': 0.4,
                'benchmarking': 0.1
            },
            AgentType.IMPLEMENTER: {
                'optimization': 0.4,
                'system': 0.2,
                'dependency': 0.3,
                'benchmarking': 0.1
            },
            AgentType.VALIDATOR: {
                'optimization': 0.3,
                'system': 0.2,
                'dependency': 0.2,
                'benchmarking': 0.3
            }
        }
        
        # Default weights for other agents
        self.default_weights = {
            'optimization': 0.3,
            'system': 0.3,
            'dependency': 0.2,
            'benchmarking': 0.2
        }
    
    def score_engine_outputs(self, agent_type: AgentType, 
                           engine_outputs: Dict[str, Any],
                           request: ContextRequest) -> Dict[str, float]:
        """Score outputs from each engine for relevance to agent and task"""
        
        weights = self.engine_weights.get(agent_type, self.default_weights)
        scores = {}
        
        # Score optimization engine output
        if 'optimization' in engine_outputs:
            optimization_score = self._score_optimization_output(
                engine_outputs['optimization'], agent_type, request
            )
            scores['optimization'] = optimization_score * weights['optimization']
        
        # Score system engine output
        if 'system' in engine_outputs:
            system_score = self._score_system_output(
                engine_outputs['system'], agent_type, request
            )
            scores['system'] = system_score * weights['system']
        
        # Score dependency engine output
        if 'dependency' in engine_outputs:
            dependency_score = self._score_dependency_output(
                engine_outputs['dependency'], agent_type, request
            )
            scores['dependency'] = dependency_score * weights['dependency']
        
        # Score benchmarking engine output
        if 'benchmarking' in engine_outputs:
            benchmarking_score = self._score_benchmarking_output(
                engine_outputs['benchmarking'], agent_type, request
            )
            scores['benchmarking'] = benchmarking_score * weights['benchmarking']
        
        return scores
    
    def _score_optimization_output(self, output: Any, agent_type: AgentType, 
                                 request: ContextRequest) -> float:
        """Score context optimization engine output"""
        base_score = 0.8  # High base relevance
        
        # Boost score if task description matches optimization patterns
        if request.task_description:
            if any(keyword in request.task_description.lower() 
                   for keyword in ['context', 'information', 'knowledge', 'pattern']):
                base_score += 0.1
        
        # Agent-specific boosts
        if agent_type in [AgentType.IMPLEMENTER, AgentType.ANALYST]:
            base_score += 0.1
            
        return min(1.0, base_score)
    
    def _score_system_output(self, output: Any, agent_type: AgentType, 
                           request: ContextRequest) -> float:
        """Score live system context engine output"""
        base_score = 0.7
        
        # Boost for system-level tasks
        if request.task_description:
            if any(keyword in request.task_description.lower() 
                   for keyword in ['system', 'architecture', 'overview', 'design']):
                base_score += 0.2
        
        # Agent-specific boosts
        if agent_type in [AgentType.ARCHITECT, AgentType.PLANNER]:
            base_score += 0.1
            
        return min(1.0, base_score)
    
    def _score_dependency_output(self, output: Any, agent_type: AgentType, 
                               request: ContextRequest) -> float:
        """Score dependency tracker engine output"""
        base_score = 0.6
        
        # Boost for dependency-related tasks
        if request.task_description:
            if any(keyword in request.task_description.lower() 
                   for keyword in ['dependency', 'component', 'relationship', 'impact']):
                base_score += 0.3
        
        # Agent-specific boosts
        if agent_type in [AgentType.ARCHITECT, AgentType.IMPLEMENTER]:
            base_score += 0.2
            
        return min(1.0, base_score)
    
    def _score_benchmarking_output(self, output: Any, agent_type: AgentType, 
                                 request: ContextRequest) -> float:
        """Score benchmarking engine output"""
        base_score = 0.4  # Lower base score - more specialized
        
        # Boost for validation and quality tasks
        if request.task_description:
            if any(keyword in request.task_description.lower() 
                   for keyword in ['specification', 'benchmark', 'compliance', 'validate']):
                base_score += 0.4
        
        # Strong boost for validator
        if agent_type == AgentType.VALIDATOR:
            base_score += 0.3
            
        return min(1.0, base_score)

class ContextDeliveryOptimizer:
    """Optimizes context delivery to fit within agent context windows"""
    
    def __init__(self):
        # Context window sizes for different agents (in characters)
        self.context_windows = {
            AgentType.ANALYST: 8000,
            AgentType.PLANNER: 10000,
            AgentType.ARCHITECT: 12000,
            AgentType.IMPLEMENTER: 7000,
            AgentType.VALIDATOR: 9000,
            AgentType.LEARNER: 8000,
            AgentType.PR_MANAGER: 6000,
            AgentType.ERROR_ANALYST: 8000,
            AgentType.SHADOW_AUDITOR: 10000,
            AgentType.PROJECTGEN: 9000,
        }
    
    def optimize_context_delivery(self, agent_type: AgentType, 
                                engine_outputs: Dict[str, Any],
                                relevance_scores: Dict[str, float],
                                request: ContextRequest) -> Dict[str, Any]:
        """Optimize context delivery for agent's context window"""
        
        # Get context window size
        max_size = request.max_context_size or self.context_windows.get(agent_type, 8000)
        
        # Apply scope-based sizing
        if request.scope == ContextScope.MINIMAL:
            max_size = int(max_size * 0.5)
        elif request.scope == ContextScope.COMPREHENSIVE:
            max_size = int(max_size * 1.2)
        elif request.scope == ContextScope.DIAGNOSTIC:
            max_size = int(max_size * 1.5)
        
        # Sort engine outputs by relevance
        sorted_outputs = sorted(
            engine_outputs.items(),
            key=lambda x: relevance_scores.get(x[0], 0),
            reverse=True
        )
        
        # Build optimized context within size limits
        optimized_context = {}
        current_size = 0
        
        for engine_name, output in sorted_outputs:
            if output is None:
                continue
                
            # Estimate size of this output
            output_text = self._extract_text_content(output)
            output_size = len(output_text)
            
            if current_size + output_size <= max_size:
                # Include full output
                optimized_context[engine_name] = output
                current_size += output_size
            else:
                # Try to include partial output if high relevance
                relevance = relevance_scores.get(engine_name, 0)
                if relevance > 0.7 and current_size < max_size * 0.8:
                    # Include truncated version
                    remaining_space = max_size - current_size
                    if remaining_space > 200:  # Minimum useful size
                        truncated_output = self._truncate_output(
                            output, remaining_space
                        )
                        optimized_context[f"{engine_name}_truncated"] = truncated_output
                        current_size += remaining_space
                break
        
        # Add metadata about optimization
        optimized_context['_optimization_metadata'] = {
            'original_engines': len(engine_outputs),
            'included_engines': len([k for k in optimized_context.keys() if not k.startswith('_')]),
            'context_utilization': current_size / max_size,
            'max_context_size': max_size,
            'actual_size': current_size,
            'optimization_applied': current_size < sum(len(self._extract_text_content(output)) 
                                                     for output in engine_outputs.values() if output)
        }
        
        return optimized_context
    
    def _extract_text_content(self, output: Any) -> str:
        """Extract text content from engine output for size estimation"""
        if isinstance(output, str):
            return output
        elif isinstance(output, dict):
            return json.dumps(output, indent=2)
        else:
            return str(output)
    
    def _truncate_output(self, output: Any, max_size: int) -> Any:
        """Truncate output to fit within size limit"""
        if isinstance(output, str):
            if len(output) <= max_size:
                return output
            return output[:max_size-20] + "...[truncated]"
        elif isinstance(output, dict):
            # Truncate dictionary by removing less important keys
            truncated = {}
            current_size = 0
            
            # Prioritize certain keys
            priority_keys = ['summary', 'overview', 'description', 'purpose']
            
            for key in priority_keys:
                if key in output and current_size < max_size * 0.8:
                    value_str = str(output[key])
                    if current_size + len(value_str) < max_size:
                        truncated[key] = output[key]
                        current_size += len(value_str)
            
            # Add remaining keys if space allows
            for key, value in output.items():
                if key not in truncated and current_size < max_size * 0.95:
                    value_str = str(value)
                    if current_size + len(value_str) < max_size:
                        truncated[key] = value
                        current_size += len(value_str)
            
            if len(truncated) < len(output):
                truncated['_truncation_note'] = f"Showing {len(truncated)} of {len(output)} fields"
            
            return truncated
        else:
            text_content = str(output)
            if len(text_content) <= max_size:
                return output
            return text_content[:max_size-20] + "...[truncated]"

class RealTimeSynchronizationCoordinator:
    """Coordinates real-time updates across all engines"""
    
    def __init__(self):
        self.last_sync_times = {}
        self.sync_intervals = {
            'optimization': 300,    # 5 minutes
            'system': 180,          # 3 minutes  
            'dependency': 240,      # 4 minutes
            'benchmarking': 600     # 10 minutes
        }
        self.sync_lock = threading.RLock()
        
    def should_sync_engine(self, engine_name: str) -> bool:
        """Check if engine needs synchronization"""
        with self.sync_lock:
            last_sync = self.last_sync_times.get(engine_name)
            if not last_sync:
                return True
                
            interval = self.sync_intervals.get(engine_name, 300)
            return (datetime.now() - last_sync).total_seconds() > interval
    
    def mark_engine_synced(self, engine_name: str) -> None:
        """Mark engine as synchronized"""
        with self.sync_lock:
            self.last_sync_times[engine_name] = datetime.now()
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status for all engines"""
        with self.sync_lock:
            status = {}
            for engine_name in self.sync_intervals:
                last_sync = self.last_sync_times.get(engine_name)
                needs_sync = self.should_sync_engine(engine_name)
                
                status[engine_name] = {
                    'last_sync': last_sync.isoformat() if last_sync else None,
                    'needs_sync': needs_sync,
                    'sync_interval': self.sync_intervals[engine_name]
                }
            return status

class ContextIntegrationHub:
    """Main Context Integration Hub coordinating all infrastructure engines"""
    
    def __init__(self, repo_path: str = "/Users/cal/DEV/RIF"):
        self.repo_path = repo_path
        
        # Initialize all four infrastructure engines
        print("Initializing Context Integration Hub...")
        
        try:
            self.context_optimizer = ContextOptimizer(
                knowledge_base_path=os.path.join(repo_path, "knowledge")
            )
            print("✓ Context Optimization Engine initialized")
        except Exception as e:
            print(f"⚠ Context Optimization Engine initialization failed: {e}")
            self.context_optimizer = None
        
        try:
            self.system_engine = LiveSystemContextEngine(repo_path)
            print("✓ Live System Context Engine initialized")
        except Exception as e:
            print(f"⚠ Live System Context Engine initialization failed: {e}")
            self.system_engine = None
        
        try:
            self.dependency_tracker = DynamicDependencyTracker(repo_path)
            print("✓ Dynamic Dependency Tracker initialized")
        except Exception as e:
            print(f"⚠ Dynamic Dependency Tracker initialization failed: {e}")
            self.dependency_tracker = None
        
        try:
            self.benchmarking_engine = BenchmarkingEngine(repo_path)
            print("✓ Design Benchmarking Framework initialized")
        except Exception as e:
            print(f"⚠ Design Benchmarking Framework initialization failed: {e}")
            self.benchmarking_engine = None
        
        # Initialize coordination components
        self.relevance_orchestrator = RelevanceOrchestrator()
        self.delivery_optimizer = ContextDeliveryOptimizer()
        self.sync_coordinator = RealTimeSynchronizationCoordinator()
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'average_response_time': 0.0,
            'engine_utilization': {
                'optimization': 0,
                'system': 0, 
                'dependency': 0,
                'benchmarking': 0
            },
            'context_cache_hits': 0,
            'context_cache_misses': 0
        }
        
        # Context cache for performance
        self.context_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Thread pool for parallel engine execution
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        print(f"Context Integration Hub initialized successfully")
        print(f"Available engines: {sum(1 for engine in [self.context_optimizer, self.system_engine, self.dependency_tracker, self.benchmarking_engine] if engine is not None)}/4")
    
    def get_unified_context(self, request: ContextRequest) -> UnifiedContextResponse:
        """Get unified context from all available engines"""
        start_time = time.time()
        request_id = self._generate_request_id(request)
        
        print(f"Processing unified context request: {request_id}")
        print(f"Agent: {request.agent_type.value}, Scope: {request.scope.value}")
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if not request.force_refresh and cache_key in self.context_cache:
                cached_response = self.context_cache[cache_key]
                if self._is_cache_valid(cached_response):
                    print("✓ Returning cached context")
                    self.performance_metrics['context_cache_hits'] += 1
                    return cached_response
            
            self.performance_metrics['context_cache_misses'] += 1
            
            # Gather outputs from all available engines in parallel
            engine_outputs = self._gather_engine_outputs_parallel(request)
            
            # Calculate relevance scores for each engine output
            relevance_scores = self.relevance_orchestrator.score_engine_outputs(
                request.agent_type, engine_outputs, request
            )
            
            # Optimize context delivery for agent's window
            optimized_context = self.delivery_optimizer.optimize_context_delivery(
                request.agent_type, engine_outputs, relevance_scores, request
            )
            
            # Calculate overall metrics
            total_relevance = sum(relevance_scores.values())
            optimization_metadata = optimized_context.get('_optimization_metadata', {})
            
            # Create unified response
            response = UnifiedContextResponse(
                agent_type=request.agent_type,
                request_id=request_id,
                context_sections=optimized_context,
                total_relevance_score=total_relevance,
                context_window_utilization=optimization_metadata.get('context_utilization', 0),
                estimated_tokens=optimization_metadata.get('actual_size', 0) // 4,  # Rough token estimate
                engine_contributions=relevance_scores,
                optimization_applied=optimization_metadata.get('optimization_applied', False),
                generation_timestamp=datetime.now(),
                performance_metrics={
                    'response_time_ms': (time.time() - start_time) * 1000,
                    'engines_used': len([k for k in engine_outputs if engine_outputs[k] is not None]),
                    'cache_used': False
                }
            )
            
            # Cache the response
            self.context_cache[cache_key] = response
            
            # Update performance metrics
            self._update_performance_metrics(response, start_time)
            
            print(f"✓ Unified context generated in {response.performance_metrics['response_time_ms']:.1f}ms")
            print(f"  Relevance score: {total_relevance:.2f}, Window utilization: {response.context_window_utilization:.1%}")
            
            return response
            
        except Exception as e:
            print(f"✗ Error generating unified context: {e}")
            # Return minimal fallback response
            return self._create_fallback_response(request, request_id, start_time)
    
    def _gather_engine_outputs_parallel(self, request: ContextRequest) -> Dict[str, Any]:
        """Gather outputs from all engines in parallel for performance"""
        engine_outputs = {}
        futures = {}
        
        # Submit parallel tasks for each available engine
        if self.context_optimizer:
            futures['optimization'] = self.executor.submit(
                self._get_optimization_output, request
            )
        
        if self.system_engine:
            futures['system'] = self.executor.submit(
                self._get_system_output, request
            )
        
        if self.dependency_tracker:
            futures['dependency'] = self.executor.submit(
                self._get_dependency_output, request
            )
        
        if self.benchmarking_engine and request.include_benchmarking and request.issue_number:
            futures['benchmarking'] = self.executor.submit(
                self._get_benchmarking_output, request
            )
        
        # Collect results with timeout
        for engine_name, future in futures.items():
            try:
                result = future.result(timeout=30)  # 30 second timeout per engine
                engine_outputs[engine_name] = result
                self.performance_metrics['engine_utilization'][engine_name] += 1
                print(f"  ✓ {engine_name} engine completed")
            except Exception as e:
                print(f"  ⚠ {engine_name} engine failed: {e}")
                engine_outputs[engine_name] = None
        
        return engine_outputs
    
    def _get_optimization_output(self, request: ContextRequest) -> Any:
        """Get output from context optimization engine"""
        if not self.context_optimizer:
            return None
        
        try:
            # Convert unified agent type to optimization engine type
            context_agent_type = getattr(ContextAgentType, request.agent_type.name, None)
            if not context_agent_type:
                return None
            
            task_context = {
                'description': request.task_description,
                'issue_number': request.issue_number,
                'scope': request.scope.value
            }
            
            optimized_context = self.context_optimizer.optimize_for_agent(
                context_agent_type, task_context, request.issue_number
            )
            
            return self.context_optimizer.format_context_for_agent(optimized_context)
            
        except Exception as e:
            print(f"Context optimization error: {e}")
            return None
    
    def _get_system_output(self, request: ContextRequest) -> Any:
        """Get output from live system context engine"""
        if not self.system_engine:
            return None
        
        try:
            # Check if engine needs sync
            if self.sync_coordinator.should_sync_engine('system'):
                print("  Syncing system context...")
                self.system_engine.trigger_update(UpdateTrigger.MANUAL_TRIGGER, 
                                                 "Hub-triggered sync")
                self.sync_coordinator.mark_engine_synced('system')
            
            # Get optimized context for agent
            context = self.system_engine.get_optimized_context_for_agent(
                request.agent_type.value, request.force_refresh
            )
            
            if context and 'error' not in context:
                return context
            else:
                return None
                
        except Exception as e:
            print(f"System context error: {e}")
            return None
    
    def _get_dependency_output(self, request: ContextRequest) -> Any:
        """Get output from dependency tracker"""
        if not self.dependency_tracker:
            return None
        
        try:
            # Initialize tracking if needed
            if not self.dependency_tracker.current_graph:
                self.dependency_tracker.initialize_tracking()
            
            # Check if specific components requested
            if request.specific_components:
                dependency_info = {}
                for component_id in request.specific_components:
                    info = self.dependency_tracker.get_component_info(component_id)
                    if 'error' not in info:
                        dependency_info[component_id] = info
                
                return dependency_info if dependency_info else None
            else:
                # Return general dependency overview
                if request.scope == ContextScope.COMPREHENSIVE:
                    return self.dependency_tracker.generate_how_things_work_documentation()
                else:
                    # Return summary analysis
                    analysis = self.dependency_tracker.analyzer.analyze_graph(
                        self.dependency_tracker.current_graph
                    )
                    return analysis['summary']
                    
        except Exception as e:
            print(f"Dependency analysis error: {e}")
            return None
    
    def _get_benchmarking_output(self, request: ContextRequest) -> Any:
        """Get output from benchmarking engine"""
        if not self.benchmarking_engine or not request.issue_number:
            return None
        
        try:
            result = self.benchmarking_engine.benchmark_issue(
                request.issue_number, f"Context request from {request.agent_type.value}"
            )
            
            # Return summary for most requests
            if request.scope == ContextScope.COMPREHENSIVE:
                return result
            else:
                return {
                    'overall_grade': result.quality_grade,
                    'adherence_score': result.overall_adherence_score,
                    'compliance_level': result.overall_compliance_level.value,
                    'specifications_count': len(result.specifications),
                    'key_recommendations': result.recommendations[:3]
                }
                
        except Exception as e:
            print(f"Benchmarking error: {e}")
            return None
    
    def format_unified_context_for_agent(self, response: UnifiedContextResponse) -> str:
        """Format unified context response for agent consumption"""
        formatted = []
        
        # Header with metadata
        formatted.append(f"# Unified Context for {response.agent_type.value.upper()}")
        formatted.append(f"**Request ID**: {response.request_id}")
        formatted.append(f"**Generated**: {response.generation_timestamp.isoformat()}")
        formatted.append(f"**Relevance Score**: {response.total_relevance_score:.2f}")
        formatted.append(f"**Context Utilization**: {response.context_window_utilization:.1%}")
        formatted.append(f"**Response Time**: {response.performance_metrics['response_time_ms']:.1f}ms")
        formatted.append("")
        
        # Engine contributions summary
        formatted.append("## Context Sources")
        for engine, contribution in response.engine_contributions.items():
            status = "✓ Included" if f"{engine}" in response.context_sections else "○ Available"
            formatted.append(f"- **{engine.title()}**: {contribution:.2f} relevance {status}")
        formatted.append("")
        
        # Context sections
        for section_name, section_content in response.context_sections.items():
            if section_name.startswith('_'):  # Skip metadata sections
                continue
                
            formatted.append(f"## {section_name.replace('_', ' ').title()}")
            
            if isinstance(section_content, str):
                formatted.append(section_content)
            elif isinstance(section_content, dict):
                formatted.extend(self._format_dict_content(section_content))
            else:
                formatted.append(str(section_content))
            
            formatted.append("")
        
        # Optimization notes
        if response.optimization_applied:
            formatted.append("## Context Optimization Notes")
            metadata = response.context_sections.get('_optimization_metadata', {})
            formatted.append(f"- Original engines available: {metadata.get('original_engines', 'N/A')}")
            formatted.append(f"- Engines included: {metadata.get('included_engines', 'N/A')}")
            formatted.append(f"- Size optimization applied: {'Yes' if metadata.get('optimization_applied') else 'No'}")
            formatted.append("")
        
        formatted.append("---")
        formatted.append("*Generated by RIF Context Integration Hub*")
        
        return "\n".join(formatted)
    
    def _format_dict_content(self, content: dict) -> List[str]:
        """Format dictionary content for display"""
        lines = []
        
        # Handle special formatting for common structures
        if 'summary' in content or 'overview' in content:
            summary = content.get('summary') or content.get('overview')
            if isinstance(summary, dict):
                for key, value in summary.items():
                    lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
            else:
                lines.append(f"**Summary**: {summary}")
        
        # Handle other key-value pairs
        for key, value in content.items():
            if key in ['summary', 'overview']:
                continue  # Already handled above
            
            if isinstance(value, (list, dict)):
                if len(str(value)) < 200:  # Short structures
                    lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
                else:
                    lines.append(f"**{key.replace('_', ' ').title()}**: {type(value).__name__} with {len(value)} items")
            else:
                lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
        
        return lines
    
    def _generate_request_id(self, request: ContextRequest) -> str:
        """Generate unique request ID"""
        request_data = f"{request.agent_type.value}-{request.task_description}-{request.scope.value}-{int(time.time())}"
        return hashlib.md5(request_data.encode()).hexdigest()[:12]
    
    def _generate_cache_key(self, request: ContextRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.agent_type.value}-{request.issue_number}-{request.scope.value}-{request.task_description}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cached_response: UnifiedContextResponse) -> bool:
        """Check if cached response is still valid"""
        age = datetime.now() - cached_response.generation_timestamp
        return age.total_seconds() < self.cache_ttl
    
    def _update_performance_metrics(self, response: UnifiedContextResponse, start_time: float) -> None:
        """Update performance metrics"""
        self.performance_metrics['total_requests'] += 1
        
        # Update average response time
        response_time = (time.time() - start_time) * 1000
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg
    
    def _create_fallback_response(self, request: ContextRequest, request_id: str, 
                                start_time: float) -> UnifiedContextResponse:
        """Create fallback response when main processing fails"""
        return UnifiedContextResponse(
            agent_type=request.agent_type,
            request_id=request_id,
            context_sections={'error': 'Context generation failed - using fallback'},
            total_relevance_score=0.0,
            context_window_utilization=0.0,
            estimated_tokens=0,
            engine_contributions={},
            optimization_applied=False,
            generation_timestamp=datetime.now(),
            performance_metrics={
                'response_time_ms': (time.time() - start_time) * 1000,
                'engines_used': 0,
                'cache_used': False,
                'fallback_used': True
            }
        )
    
    def get_hub_status(self) -> Dict[str, Any]:
        """Get comprehensive hub status"""
        return {
            'hub_status': 'active',
            'engines': {
                'context_optimizer': self.context_optimizer is not None,
                'system_engine': self.system_engine is not None,
                'dependency_tracker': self.dependency_tracker is not None,
                'benchmarking_engine': self.benchmarking_engine is not None
            },
            'synchronization': self.sync_coordinator.get_sync_status(),
            'performance': self.performance_metrics,
            'cache_status': {
                'cached_responses': len(self.context_cache),
                'cache_ttl_seconds': self.cache_ttl
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        print("Cleaning up Context Integration Hub...")
        
        # Shutdown thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Clean up engines that support it
        if self.system_engine and hasattr(self.system_engine, 'stop_continuous_updates'):
            self.system_engine.stop_continuous_updates()
        
        if self.dependency_tracker and hasattr(self.dependency_tracker, 'stop_real_time_monitoring'):
            self.dependency_tracker.stop_real_time_monitoring()
        
        print("Context Integration Hub cleanup completed")

# Convenience functions for agent integration
def get_context_for_agent(agent_type_str: str, 
                         task_description: str,
                         issue_number: Optional[int] = None,
                         scope: str = "standard",
                         repo_path: str = "/Users/cal/DEV/RIF") -> str:
    """Convenience function to get formatted context for an agent"""
    
    try:
        agent_type = AgentType(agent_type_str)
        context_scope = ContextScope(scope.lower())
    except ValueError as e:
        return f"Error: Invalid agent type or scope: {e}"
    
    hub = ContextIntegrationHub(repo_path)
    
    try:
        request = ContextRequest(
            agent_type=agent_type,
            task_description=task_description,
            issue_number=issue_number,
            scope=context_scope
        )
        
        response = hub.get_unified_context(request)
        return hub.format_unified_context_for_agent(response)
        
    finally:
        hub.cleanup()

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RIF Context Integration Hub")
    parser.add_argument("--agent", type=str, required=True, 
                       choices=[a.value for a in AgentType],
                       help="Agent type requesting context")
    parser.add_argument("--task", type=str, required=True,
                       help="Task description")
    parser.add_argument("--issue", type=int, 
                       help="GitHub issue number")
    parser.add_argument("--scope", type=str, default="standard",
                       choices=['minimal', 'standard', 'comprehensive', 'diagnostic'],
                       help="Context scope")
    parser.add_argument("--repo", type=str, default="/Users/cal/DEV/RIF",
                       help="Repository path")
    parser.add_argument("--status", action="store_true",
                       help="Show hub status")
    parser.add_argument("--benchmark", action="store_true",
                       help="Include benchmarking data")
    
    args = parser.parse_args()
    
    if args.status:
        hub = ContextIntegrationHub(args.repo)
        status = hub.get_hub_status()
        print("=== Context Integration Hub Status ===")
        print(json.dumps(status, indent=2, default=str))
        hub.cleanup()
        
    else:
        context = get_context_for_agent(
            agent_type_str=args.agent,
            task_description=args.task,
            issue_number=args.issue,
            scope=args.scope,
            repo_path=args.repo
        )
        
        print(context)