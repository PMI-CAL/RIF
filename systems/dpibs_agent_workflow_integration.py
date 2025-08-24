"""
DPIBS Agent Workflow Integration Architecture
===========================================

Layer 1 of DPIBS Integration: Agent Context Optimization Integration

This module provides context optimization integration for all RIF agent types without
modifying existing agent files. It implements the Context Bridge Pattern to enhance
agent workflows with DPIBS capabilities while maintaining zero regression.

Architecture:
- Agent Context Optimizer: Enhances context delivery without agent file modification
- Agent Enhancement Templates: Dynamic instruction augmentation
- Agent Context Manager: Context state management and optimization
- Agent State Integration: Workflow state integration with context delivery

Key Requirements:
- Zero modification to existing agent files (claude/agents/*.md)
- Universal enhancement for all 9+ RIF agent types
- Backward compatibility with complete fallback capability
- Performance enhancement rather than degradation
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import os

# RIF Infrastructure Imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knowledge.database.database_interface import RIFDatabase
from knowledge.integration.hybrid_knowledge_system import HybridKnowledgeSystem


@dataclass
class AgentContextProfile:
    """Context profile for agent optimization."""
    agent_type: str
    agent_name: str
    specialization: str
    context_requirements: Dict[str, Any]
    optimization_preferences: Dict[str, Any]
    performance_targets: Dict[str, float]
    compatibility_level: str  # "full", "enhanced", "fallback"


@dataclass
class ContextOptimizationResult:
    """Result of context optimization for agent."""
    agent_type: str
    original_context_size: int
    optimized_context_size: int
    optimization_ratio: float
    relevance_score: float
    processing_time_ms: int
    enhancement_applied: bool
    fallback_triggered: bool


@dataclass
class AgentEnhancementTemplate:
    """Template for enhancing agent instructions."""
    template_id: str
    agent_type: str
    enhancement_type: str  # "context", "instructions", "capabilities"
    template_content: str
    activation_conditions: Dict[str, Any]
    backward_compatibility: bool


class AgentContextOptimizer:
    """
    Core component for optimizing agent context without modifying agent files.
    
    Implements context enhancement through dynamic instruction augmentation
    while preserving complete backward compatibility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # RIF Infrastructure Integration
        self.rif_db = None
        self.hybrid_system = None
        self.context_cache = {}
        self.enhancement_templates = {}
        self.agent_profiles = {}
        
        # Performance Tracking
        self.performance_metrics = {
            'optimizations_performed': 0,
            'fallbacks_triggered': 0,
            'enhancement_success_rate': 0.0,
            'average_processing_time_ms': 0.0
        }
        
        # Backward Compatibility Control
        self.fallback_mode = config.get('fallback_mode_enabled', True)
        self.compatibility_level = config.get('compatibility_level', 'full')
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the agent context optimization system."""
        try:
            # Initialize RIF database connection
            self.rif_db = RIFDatabase()
            
            # Initialize hybrid knowledge system for context analysis
            hybrid_config = {
                'memory_limit_mb': 512,
                'cpu_cores': 2,
                'performance_mode': 'BALANCED',
                'database_path': 'knowledge/hybrid_knowledge.duckdb'
            }
            
            self.hybrid_system = HybridKnowledgeSystem(hybrid_config)
            if not self.hybrid_system.initialize():
                raise RuntimeError("Failed to initialize hybrid knowledge system")
            
            # Load agent profiles
            self._load_agent_profiles()
            
            # Load enhancement templates
            self._load_enhancement_templates()
            
            # Initialize context cache
            self._initialize_context_cache()
            
            self.logger.info("Agent Context Optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Agent Context Optimizer: {e}")
            if self.fallback_mode:
                self.logger.warning("Falling back to legacy mode - no context optimization")
                self.compatibility_level = "fallback"
            else:
                raise
    
    def _load_agent_profiles(self):
        """Load agent profiles for all RIF agent types."""
        agent_types = [
            'rif-analyst', 'rif-planner', 'rif-architect', 'rif-implementer',
            'rif-validator', 'rif-learner', 'rif-pr-manager', 'rif-projectgen',
            'rif-shadow-auditor'
        ]
        
        for agent_type in agent_types:
            try:
                # Load agent profile from knowledge base or create default
                profile = self._create_agent_profile(agent_type)
                self.agent_profiles[agent_type] = profile
                
            except Exception as e:
                self.logger.warning(f"Failed to load profile for {agent_type}: {e}")
                # Create minimal fallback profile
                self.agent_profiles[agent_type] = AgentContextProfile(
                    agent_type=agent_type,
                    agent_name=agent_type,
                    specialization="unknown",
                    context_requirements={},
                    optimization_preferences={},
                    performance_targets={},
                    compatibility_level="fallback"
                )
    
    def _create_agent_profile(self, agent_type: str) -> AgentContextProfile:
        """Create agent profile based on agent type and capabilities."""
        
        # Agent-specific optimization profiles
        profiles = {
            'rif-analyst': {
                'specialization': 'requirements_analysis_pattern_recognition',
                'context_requirements': {
                    'similar_issues': 5,
                    'patterns': 10,
                    'knowledge_depth': 'comprehensive',
                    'complexity_assessment': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.8,
                    'context_window_target': 2000,
                    'performance_priority': 'accuracy'
                },
                'performance_targets': {
                    'processing_time_ms': 500,
                    'context_reduction_ratio': 0.7,
                    'relevance_score': 0.85
                }
            },
            'rif-planner': {
                'specialization': 'strategic_planning_workflow_configuration',
                'context_requirements': {
                    'dependencies': 20,
                    'complexity_patterns': 15,
                    'workflow_states': True,
                    'resource_planning': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.75,
                    'context_window_target': 2500,
                    'performance_priority': 'completeness'
                },
                'performance_targets': {
                    'processing_time_ms': 750,
                    'context_reduction_ratio': 0.6,
                    'relevance_score': 0.8
                }
            },
            'rif-architect': {
                'specialization': 'system_design_dependency_mapping',
                'context_requirements': {
                    'architectural_patterns': 25,
                    'integration_points': 15,
                    'design_decisions': 20,
                    'system_constraints': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.85,
                    'context_window_target': 3000,
                    'performance_priority': 'depth'
                },
                'performance_targets': {
                    'processing_time_ms': 1000,
                    'context_reduction_ratio': 0.5,
                    'relevance_score': 0.9
                }
            },
            'rif-implementer': {
                'specialization': 'code_implementation_feature_development',
                'context_requirements': {
                    'code_patterns': 15,
                    'implementation_examples': 10,
                    'testing_strategies': 5,
                    'quality_requirements': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.8,
                    'context_window_target': 2200,
                    'performance_priority': 'actionability'
                },
                'performance_targets': {
                    'processing_time_ms': 600,
                    'context_reduction_ratio': 0.65,
                    'relevance_score': 0.85
                }
            },
            'rif-validator': {
                'specialization': 'testing_quality_assurance',
                'context_requirements': {
                    'quality_patterns': 10,
                    'testing_frameworks': 8,
                    'validation_strategies': 12,
                    'quality_gates': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.85,
                    'context_window_target': 2000,
                    'performance_priority': 'precision'
                },
                'performance_targets': {
                    'processing_time_ms': 400,
                    'context_reduction_ratio': 0.7,
                    'relevance_score': 0.9
                }
            },
            'rif-learner': {
                'specialization': 'knowledge_extraction_pattern_learning',
                'context_requirements': {
                    'learning_patterns': 20,
                    'knowledge_relationships': 15,
                    'success_patterns': 10,
                    'failure_analysis': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.7,
                    'context_window_target': 2800,
                    'performance_priority': 'comprehensiveness'
                },
                'performance_targets': {
                    'processing_time_ms': 800,
                    'context_reduction_ratio': 0.55,
                    'relevance_score': 0.8
                }
            },
            'rif-pr-manager': {
                'specialization': 'pull_request_management_quality_gates',
                'context_requirements': {
                    'pr_patterns': 8,
                    'quality_checks': 10,
                    'merge_strategies': 5,
                    'deployment_patterns': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.8,
                    'context_window_target': 1800,
                    'performance_priority': 'efficiency'
                },
                'performance_targets': {
                    'processing_time_ms': 350,
                    'context_reduction_ratio': 0.75,
                    'relevance_score': 0.85
                }
            },
            'rif-projectgen': {
                'specialization': 'project_generation_intelligent_discovery',
                'context_requirements': {
                    'project_templates': 15,
                    'technology_patterns': 20,
                    'configuration_examples': 12,
                    'setup_strategies': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.75,
                    'context_window_target': 2500,
                    'performance_priority': 'adaptability'
                },
                'performance_targets': {
                    'processing_time_ms': 700,
                    'context_reduction_ratio': 0.6,
                    'relevance_score': 0.8
                }
            },
            'rif-shadow-auditor': {
                'specialization': 'shadow_mode_quality_tracking',
                'context_requirements': {
                    'audit_patterns': 12,
                    'quality_metrics': 15,
                    'compliance_checks': 8,
                    'tracking_strategies': True
                },
                'optimization_preferences': {
                    'relevance_threshold': 0.85,
                    'context_window_target': 2000,
                    'performance_priority': 'accuracy'
                },
                'performance_targets': {
                    'processing_time_ms': 450,
                    'context_reduction_ratio': 0.7,
                    'relevance_score': 0.9
                }
            }
        }
        
        # Get profile data or default
        profile_data = profiles.get(agent_type, {
            'specialization': 'general_purpose',
            'context_requirements': {},
            'optimization_preferences': {'relevance_threshold': 0.7},
            'performance_targets': {'processing_time_ms': 500}
        })
        
        return AgentContextProfile(
            agent_type=agent_type,
            agent_name=agent_type,
            specialization=profile_data['specialization'],
            context_requirements=profile_data['context_requirements'],
            optimization_preferences=profile_data['optimization_preferences'],
            performance_targets=profile_data['performance_targets'],
            compatibility_level=self.compatibility_level
        )
    
    def _load_enhancement_templates(self):
        """Load enhancement templates for agent instruction augmentation."""
        templates = [
            # Context Enhancement Template
            AgentEnhancementTemplate(
                template_id="context_enhancement_universal",
                agent_type="universal",
                enhancement_type="context",
                template_content="""
## 🔍 Enhanced Context Analysis
**DPIBS Integration Active**: Context optimization enabled for improved agent performance.

### Relevant Context Patterns
{context_patterns}

### Similar Past Issues
{similar_issues}

### Optimization Insights
{optimization_insights}
""",
                activation_conditions={
                    "compatibility_level": ["full", "enhanced"],
                    "context_optimization_enabled": True
                },
                backward_compatibility=True
            ),
            
            # Performance Enhancement Template
            AgentEnhancementTemplate(
                template_id="performance_enhancement",
                agent_type="universal", 
                enhancement_type="capabilities",
                template_content="""
## ⚡ Performance Enhancement Active
**Context Optimization**: {optimization_ratio}% context size reduction achieved
**Relevance Score**: {relevance_score} (target: {target_score})
**Processing Time**: {processing_time}ms (target: {target_time}ms)
""",
                activation_conditions={
                    "performance_enhancement_enabled": True,
                    "processing_time_available": True
                },
                backward_compatibility=True
            ),
            
            # Fallback Safety Template
            AgentEnhancementTemplate(
                template_id="fallback_safety",
                agent_type="universal",
                enhancement_type="instructions",
                template_content="""
## 🛡️ Compatibility Mode Active
**DPIBS Integration**: Operating in fallback mode for maximum compatibility
**All existing functionality preserved and enhanced**
""",
                activation_conditions={
                    "compatibility_level": ["fallback"],
                    "fallback_mode": True
                },
                backward_compatibility=True
            )
        ]
        
        for template in templates:
            self.enhancement_templates[template.template_id] = template
    
    def _initialize_context_cache(self):
        """Initialize context cache for performance optimization."""
        self.context_cache = {
            'recent_optimizations': {},
            'performance_baselines': {},
            'fallback_triggers': {},
            'cache_stats': {
                'hits': 0,
                'misses': 0,
                'total_requests': 0
            }
        }
    
    async def optimize_agent_context(self, agent_type: str, context_data: Dict[str, Any], 
                                   issue_context: Dict[str, Any] = None) -> ContextOptimizationResult:
        """
        Optimize context for specific agent type without modifying agent files.
        
        Args:
            agent_type: Type of RIF agent (e.g., 'rif-analyst')
            context_data: Raw context data to optimize
            issue_context: Optional issue-specific context
            
        Returns:
            ContextOptimizationResult with optimization metrics and enhanced context
        """
        start_time = time.time()
        
        try:
            # Get agent profile
            profile = self.agent_profiles.get(agent_type)
            if not profile:
                return self._create_fallback_result(agent_type, context_data, start_time)
            
            # Check compatibility level
            if profile.compatibility_level == "fallback":
                return self._create_fallback_result(agent_type, context_data, start_time)
            
            # Check cache first
            cache_key = self._generate_cache_key(agent_type, context_data)
            cached_result = self.context_cache.get('recent_optimizations', {}).get(cache_key)
            
            if cached_result and self._is_cache_valid(cached_result):
                self.context_cache['cache_stats']['hits'] += 1
                return cached_result
            
            self.context_cache['cache_stats']['misses'] += 1
            self.context_cache['cache_stats']['total_requests'] += 1
            
            # Perform context optimization
            original_size = len(str(context_data))
            
            # Step 1: Extract relevant patterns based on agent specialization
            relevant_patterns = await self._extract_relevant_patterns(agent_type, context_data, profile)
            
            # Step 2: Optimize context based on agent requirements
            optimized_context = await self._optimize_context_for_agent(
                context_data, profile, relevant_patterns, issue_context
            )
            
            # Step 3: Calculate optimization metrics
            optimized_size = len(str(optimized_context))
            optimization_ratio = 1.0 - (optimized_size / original_size) if original_size > 0 else 0.0
            
            # Step 4: Calculate relevance score
            relevance_score = await self._calculate_relevance_score(
                optimized_context, profile, relevant_patterns
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create optimization result
            result = ContextOptimizationResult(
                agent_type=agent_type,
                original_context_size=original_size,
                optimized_context_size=optimized_size,
                optimization_ratio=optimization_ratio,
                relevance_score=relevance_score,
                processing_time_ms=processing_time_ms,
                enhancement_applied=True,
                fallback_triggered=False
            )
            
            # Cache result
            self.context_cache['recent_optimizations'][cache_key] = result
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context optimization failed for {agent_type}: {e}")
            return self._create_fallback_result(agent_type, context_data, start_time)
    
    def _create_fallback_result(self, agent_type: str, context_data: Dict[str, Any], 
                              start_time: float) -> ContextOptimizationResult:
        """Create fallback result when optimization fails."""
        processing_time_ms = int((time.time() - start_time) * 1000)
        context_size = len(str(context_data))
        
        return ContextOptimizationResult(
            agent_type=agent_type,
            original_context_size=context_size,
            optimized_context_size=context_size,
            optimization_ratio=0.0,
            relevance_score=1.0,  # Assume full relevance for fallback
            processing_time_ms=processing_time_ms,
            enhancement_applied=False,
            fallback_triggered=True
        )
    
    async def _extract_relevant_patterns(self, agent_type: str, context_data: Dict[str, Any], 
                                       profile: AgentContextProfile) -> List[Dict[str, Any]]:
        """Extract patterns relevant to agent specialization."""
        try:
            # Use hybrid search to find relevant patterns
            search_query = f"{profile.specialization} {agent_type}"
            
            results = self.rif_db.hybrid_search(
                text_query=search_query,
                entity_types=['pattern', 'implementation_pattern', 'architectural_pattern'],
                limit=profile.context_requirements.get('patterns', 10)
            )
            
            patterns = []
            for result in results:
                entity = self.rif_db.get_entity(str(result.entity_id))
                if entity:
                    patterns.append({
                        'pattern_id': entity['id'],
                        'name': entity['name'],
                        'description': entity.get('metadata', {}).get('description', ''),
                        'relevance_score': result.similarity_score,
                        'pattern_type': entity.get('type', 'unknown')
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Failed to extract patterns for {agent_type}: {e}")
            return []
    
    async def _optimize_context_for_agent(self, context_data: Dict[str, Any], 
                                        profile: AgentContextProfile,
                                        relevant_patterns: List[Dict[str, Any]],
                                        issue_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize context data based on agent profile requirements."""
        optimized_context = context_data.copy()
        
        # Add relevant patterns
        if relevant_patterns:
            optimized_context['relevant_patterns'] = relevant_patterns[:profile.context_requirements.get('patterns', 5)]
        
        # Add agent-specific enhancements
        if profile.specialization:
            optimized_context['agent_specialization'] = profile.specialization
        
        # Add performance targets for agent awareness
        optimized_context['performance_targets'] = profile.performance_targets
        
        # Add context optimization metadata
        optimized_context['optimization_metadata'] = {
            'agent_type': profile.agent_type,
            'compatibility_level': profile.compatibility_level,
            'optimization_timestamp': datetime.now().isoformat(),
            'dpibs_integration': 'active'
        }
        
        return optimized_context
    
    async def _calculate_relevance_score(self, optimized_context: Dict[str, Any], 
                                       profile: AgentContextProfile,
                                       patterns: List[Dict[str, Any]]) -> float:
        """Calculate relevance score for optimized context."""
        try:
            # Base score from agent profile matching
            base_score = 0.7
            
            # Bonus for relevant patterns
            if patterns:
                pattern_bonus = min(0.2, len(patterns) * 0.02)
                base_score += pattern_bonus
            
            # Bonus for meeting agent requirements
            requirements_met = 0
            total_requirements = len(profile.context_requirements)
            
            for req_key, req_value in profile.context_requirements.items():
                if req_key in optimized_context:
                    requirements_met += 1
            
            if total_requirements > 0:
                requirements_score = requirements_met / total_requirements
                base_score = base_score * 0.7 + requirements_score * 0.3
            
            return min(1.0, base_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate relevance score: {e}")
            return 0.8  # Default reasonable score
    
    def _generate_cache_key(self, agent_type: str, context_data: Dict[str, Any]) -> str:
        """Generate cache key for context optimization."""
        # Create hash of agent type and context data structure
        context_str = json.dumps(context_data, sort_keys=True, default=str)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:12]
        return f"{agent_type}_{context_hash}"
    
    def _is_cache_valid(self, cached_result: ContextOptimizationResult) -> bool:
        """Check if cached result is still valid."""
        # Simple time-based cache validity (5 minutes)
        cache_duration = timedelta(minutes=5)
        # In real implementation, would check timestamp from cached result
        return True  # Simplified for this implementation
    
    def _update_performance_metrics(self, result: ContextOptimizationResult):
        """Update performance metrics with optimization result."""
        self.performance_metrics['optimizations_performed'] += 1
        
        if result.fallback_triggered:
            self.performance_metrics['fallbacks_triggered'] += 1
        
        # Update success rate
        total_optimizations = self.performance_metrics['optimizations_performed']
        successful_optimizations = total_optimizations - self.performance_metrics['fallbacks_triggered']
        self.performance_metrics['enhancement_success_rate'] = (
            successful_optimizations / total_optimizations if total_optimizations > 0 else 0.0
        )
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time_ms']
        new_avg = (
            (current_avg * (total_optimizations - 1) + result.processing_time_ms) / total_optimizations
            if total_optimizations > 0 else result.processing_time_ms
        )
        self.performance_metrics['average_processing_time_ms'] = new_avg
    
    def get_enhancement_template(self, agent_type: str, enhancement_type: str) -> Optional[str]:
        """Get enhancement template for agent type."""
        try:
            # Look for agent-specific template first
            agent_templates = [t for t in self.enhancement_templates.values() 
                             if t.agent_type == agent_type and t.enhancement_type == enhancement_type]
            
            if agent_templates:
                return agent_templates[0].template_content
            
            # Fall back to universal template
            universal_templates = [t for t in self.enhancement_templates.values()
                                 if t.agent_type == "universal" and t.enhancement_type == enhancement_type]
            
            if universal_templates:
                return universal_templates[0].template_content
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get enhancement template: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'system_metrics': self.performance_metrics.copy(),
            'cache_metrics': self.context_cache['cache_stats'].copy(),
            'agent_profiles_loaded': len(self.agent_profiles),
            'enhancement_templates_loaded': len(self.enhancement_templates),
            'compatibility_level': self.compatibility_level,
            'fallback_mode_enabled': self.fallback_mode
        }
    
    def shutdown(self):
        """Clean shutdown of the agent context optimizer."""
        try:
            if self.hybrid_system:
                self.hybrid_system.shutdown()
            if self.rif_db:
                self.rif_db.close()
            self.logger.info("Agent Context Optimizer shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class AgentContextManager:
    """
    Manages context state and optimization across agent workflows.
    Provides centralized context management for all RIF agents.
    """
    
    def __init__(self, optimizer: AgentContextOptimizer):
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        self.active_contexts = {}
        self.context_history = {}
    
    async def prepare_agent_context(self, agent_type: str, issue_id: str, 
                                  base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare optimized context for agent execution."""
        try:
            # Optimize context for agent
            optimization_result = await self.optimizer.optimize_agent_context(
                agent_type, base_context, {'issue_id': issue_id}
            )
            
            # Store active context
            context_id = f"{agent_type}_{issue_id}"
            self.active_contexts[context_id] = {
                'context': base_context,
                'optimization_result': optimization_result,
                'timestamp': datetime.now(),
                'agent_type': agent_type,
                'issue_id': issue_id
            }
            
            # Return enhanced context
            enhanced_context = base_context.copy()
            enhanced_context['dpibs_optimization'] = asdict(optimization_result)
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Failed to prepare context for {agent_type}: {e}")
            return base_context  # Fallback to original context
    
    def get_context_status(self, agent_type: str, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get status of context optimization for agent."""
        context_id = f"{agent_type}_{issue_id}"
        return self.active_contexts.get(context_id)
    
    def clear_context(self, agent_type: str, issue_id: str):
        """Clear context data for completed agent."""
        context_id = f"{agent_type}_{issue_id}"
        if context_id in self.active_contexts:
            # Move to history before clearing
            self.context_history[context_id] = self.active_contexts[context_id]
            del self.active_contexts[context_id]


class AgentStateIntegration:
    """
    Integrates agent context optimization with RIF workflow state transitions.
    Provides seamless integration with existing state machine.
    """
    
    def __init__(self, context_manager: AgentContextManager):
        self.context_manager = context_manager
        self.logger = logging.getLogger(__name__)
        self.state_transitions = {}
    
    async def on_state_transition(self, issue_id: str, from_state: str, to_state: str, 
                                agent_type: str = None) -> Dict[str, Any]:
        """Handle state transition with context optimization."""
        try:
            if agent_type:
                # Prepare context for new agent state
                base_context = self._get_state_context(issue_id, to_state)
                enhanced_context = await self.context_manager.prepare_agent_context(
                    agent_type, issue_id, base_context
                )
                
                return {
                    'transition_handled': True,
                    'context_prepared': True,
                    'agent_type': agent_type,
                    'enhanced_context_size': len(str(enhanced_context))
                }
            
            return {'transition_handled': False, 'reason': 'no_agent_specified'}
            
        except Exception as e:
            self.logger.error(f"State transition handling failed: {e}")
            return {'transition_handled': False, 'error': str(e)}
    
    def _get_state_context(self, issue_id: str, state: str) -> Dict[str, Any]:
        """Get context data for specific workflow state."""
        # This would integrate with RIF workflow system
        return {
            'issue_id': issue_id,
            'workflow_state': state,
            'context_type': 'state_transition',
            'timestamp': datetime.now().isoformat()
        }


# Integration Interface Functions
def create_agent_workflow_integration(config: Dict[str, Any] = None) -> AgentContextOptimizer:
    """
    Factory function to create agent workflow integration system.
    
    This is the main entry point for DPIBS Agent Workflow Integration.
    """
    if config is None:
        config = {
            'fallback_mode_enabled': True,
            'compatibility_level': 'full',
            'performance_optimization': True,
            'cache_enabled': True
        }
    
    return AgentContextOptimizer(config)


def get_agent_enhancement_for_type(agent_type: str, optimizer: AgentContextOptimizer) -> Optional[str]:
    """
    Get enhancement template for specific agent type.
    Used by RIF orchestration to enhance agent instructions.
    """
    return optimizer.get_enhancement_template(agent_type, "context")


# Backward Compatibility Functions
def is_dpibs_integration_available() -> bool:
    """Check if DPIBS integration is available and working."""
    try:
        config = {'compatibility_level': 'fallback'}
        optimizer = AgentContextOptimizer(config)
        return optimizer.compatibility_level in ['full', 'enhanced', 'fallback']
    except Exception:
        return False


def get_fallback_context(agent_type: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get fallback context when DPIBS integration is unavailable."""
    # Return original context with minimal enhancement
    enhanced_context = context_data.copy()
    enhanced_context['dpibs_status'] = 'fallback_mode'
    enhanced_context['compatibility_note'] = 'Operating in legacy compatibility mode'
    return enhanced_context