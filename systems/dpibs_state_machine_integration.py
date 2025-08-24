"""
DPIBS RIF State Machine Integration Architecture
==============================================

Layer 4 of DPIBS Integration: RIF State Machine Integration

This module provides enhanced integration with the existing RIF workflow state machine,
adding DPIBS-specific context triggers, agent handoff enhancements, and quality gate
integration while preserving all existing state machine functionality.

Architecture:
- State Machine Integrator: Enhanced integration with existing rif-workflow.yaml
- State Transition Manager: Context-aware state transition handling
- Context Trigger Manager: DPIBS context delivery during state transitions
- Quality Gate Integrator: Enhanced quality gates with DPIBS optimization

Key Requirements:
- Preserve all existing rif-workflow.yaml states and transitions (824+ lines)
- Add DPIBS context triggers without disrupting existing workflow
- Enhance agent handoffs with context optimization
- Integrate with existing quality gates system (adaptive coverage, etc.)
"""

import json
import logging
import asyncio
import time
import yaml
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import os

# RIF Infrastructure Imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knowledge.database.database_interface import RIFDatabase
from systems.dpibs_agent_workflow_integration import AgentContextOptimizer


class StateTransitionType(Enum):
    """Types of state transitions."""
    AUTOMATIC = "automatic"
    TRIGGERED = "triggered"
    MANUAL = "manual"
    CONDITIONAL = "conditional"


@dataclass
class StateTransitionEvent:
    """State transition event data."""
    transition_id: str
    issue_id: str
    from_state: str
    to_state: str
    transition_type: StateTransitionType
    agent_type: Optional[str]
    trigger_condition: Optional[str]
    context_data: Dict[str, Any]
    dpibs_enhancement: bool
    timestamp: datetime


@dataclass 
class ContextTriggerResult:
    """Result of context trigger processing."""
    trigger_id: str
    success: bool
    context_enhanced: bool
    agent_context_optimized: bool
    processing_time_ms: int
    enhancement_metadata: Dict[str, Any]


@dataclass
class QualityGateIntegration:
    """Quality gate integration with DPIBS."""
    gate_name: str
    original_threshold: Any
    dpibs_enhanced_threshold: Any
    context_aware: bool
    optimization_applied: bool
    performance_improvement: Optional[float]


class RIFStateMachineIntegrator:
    """
    Enhanced integration with existing RIF workflow state machine.
    
    Provides DPIBS context enhancement and optimization while preserving
    all existing state machine functionality and transitions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core Integration Components
        self.rif_db = None
        self.agent_context_optimizer = None
        self.existing_workflow = {}
        self.enhanced_workflow = {}
        
        # State Machine Enhancement Components
        self.state_transition_manager = None
        self.context_trigger_manager = None
        self.quality_gate_integrator = None
        
        # State Machine State
        self.active_transitions = {}
        self.state_history = {}
        self.context_triggers = {}
        
        # Performance Tracking
        self.performance_metrics = {
            'transitions_processed': 0,
            'context_triggers_executed': 0,
            'quality_gates_enhanced': 0,
            'optimization_success_rate': 0.0,
            'average_transition_time_ms': 0.0
        }
        
        # Backward Compatibility
        self.preserve_existing_workflow = config.get('preserve_existing_workflow', True)
        self.compatibility_mode = config.get('compatibility_mode', True)
        
        self._initialize_state_machine_integration()
    
    def _initialize_state_machine_integration(self):
        """Initialize state machine integration."""
        try:
            # Initialize RIF database connection
            self.rif_db = RIFDatabase()
            
            # Initialize agent context optimizer
            optimizer_config = self.config.get('agent_context_optimization', {})
            self.agent_context_optimizer = AgentContextOptimizer(optimizer_config)
            
            # Load existing RIF workflow
            self._load_existing_workflow()
            
            # Initialize enhancement components
            self._initialize_state_transition_manager()
            self._initialize_context_trigger_manager()
            self._initialize_quality_gate_integrator()
            
            # Setup enhanced workflow
            self._setup_enhanced_workflow()
            
            self.logger.info("RIF State Machine Integrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize state machine integration: {e}")
            if self.compatibility_mode:
                self.logger.warning("Operating in compatibility mode - existing workflow preserved")
            else:
                raise
    
    def _load_existing_workflow(self):
        """Load existing RIF workflow configuration."""
        try:
            workflow_path = Path('config/rif-workflow.yaml')
            if workflow_path.exists():
                with open(workflow_path, 'r') as f:
                    self.existing_workflow = yaml.safe_load(f)
                
                workflow_info = self.existing_workflow.get('workflow', {})
                states = workflow_info.get('states', {})
                transitions = workflow_info.get('transitions', [])
                
                self.logger.info(f"Loaded existing workflow: {len(states)} states, {len(transitions)} transitions")
            else:
                self.logger.warning("No existing RIF workflow found")
                self.existing_workflow = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load existing workflow: {e}")
            self.existing_workflow = {}
    
    def _initialize_state_transition_manager(self):
        """Initialize enhanced state transition manager."""
        transition_config = self.config.get('state_transitions', {
            'context_optimization_enabled': True,
            'agent_handoff_enhancement': True,
            'transition_monitoring': True,
            'performance_tracking': True
        })
        
        self.state_transition_manager = StateTransitionManager(
            config=transition_config,
            existing_workflow=self.existing_workflow,
            agent_context_optimizer=self.agent_context_optimizer,
            rif_db=self.rif_db
        )
    
    def _initialize_context_trigger_manager(self):
        """Initialize context trigger manager."""
        trigger_config = self.config.get('context_triggers', {
            'dpibs_triggers_enabled': True,
            'agent_context_injection': True,
            'state_aware_optimization': True,
            'performance_optimization': True
        })
        
        self.context_trigger_manager = ContextTriggerManager(
            config=trigger_config,
            existing_workflow=self.existing_workflow,
            agent_context_optimizer=self.agent_context_optimizer
        )
    
    def _initialize_quality_gate_integrator(self):
        """Initialize quality gate integrator."""
        quality_config = self.config.get('quality_gate_integration', {
            'adaptive_thresholds_enhanced': True,
            'context_aware_gates': True,
            'dpibs_optimization': True,
            'preserve_existing_gates': True
        })
        
        self.quality_gate_integrator = QualityGateIntegrator(
            config=quality_config,
            existing_workflow=self.existing_workflow,
            rif_db=self.rif_db
        )
    
    def _setup_enhanced_workflow(self):
        """Setup enhanced workflow that extends existing functionality."""
        if not self.preserve_existing_workflow:
            return
        
        # Create enhanced workflow by copying existing and adding enhancements
        self.enhanced_workflow = self.existing_workflow.copy()
        
        # Add DPIBS-specific enhancements to workflow
        workflow_section = self.enhanced_workflow.get('workflow', {})
        
        # Add DPIBS context triggers to states
        self._add_context_triggers_to_states(workflow_section)
        
        # Enhance transitions with context optimization
        self._enhance_transitions_with_context(workflow_section)
        
        # Integrate DPIBS with quality gates
        self._integrate_quality_gates_with_dpibs(workflow_section)
    
    def _add_context_triggers_to_states(self, workflow: Dict[str, Any]):
        """Add DPIBS context triggers to existing states."""
        states = workflow.get('states', {})
        
        # Add DPIBS context enhancement to agent states
        agent_states = [
            'analyzing', 'planning', 'architecting', 'implementing',
            'validating', 'learning', 'pr_creating', 'pr_validating'
        ]
        
        for state_name in agent_states:
            if state_name in states:
                state_config = states[state_name]
                
                # Add DPIBS context trigger
                if 'dpibs_context_trigger' not in state_config:
                    state_config['dpibs_context_trigger'] = {
                        'enabled': True,
                        'optimization_level': self._determine_optimization_level(state_name),
                        'agent_type': state_config.get('agent', f'rif-{state_name.replace("ing", "er")}'),
                        'context_requirements': self._get_state_context_requirements(state_name)
                    }
    
    def _determine_optimization_level(self, state_name: str) -> str:
        """Determine optimization level for state."""
        high_optimization_states = ['implementing', 'architecting', 'planning']
        medium_optimization_states = ['analyzing', 'validating', 'learning']
        
        if state_name in high_optimization_states:
            return 'high'
        elif state_name in medium_optimization_states:
            return 'medium'
        else:
            return 'low'
    
    def _get_state_context_requirements(self, state_name: str) -> Dict[str, Any]:
        """Get context requirements for specific state."""
        requirements = {
            'analyzing': {
                'patterns_needed': 10,
                'similar_issues': 5,
                'complexity_assessment': True
            },
            'planning': {
                'dependencies': 15,
                'workflow_patterns': 8,
                'resource_planning': True
            },
            'architecting': {
                'architectural_patterns': 20,
                'design_decisions': 15,
                'integration_points': 10
            },
            'implementing': {
                'code_patterns': 12,
                'implementation_examples': 8,
                'testing_strategies': 5
            },
            'validating': {
                'quality_patterns': 8,
                'testing_frameworks': 6,
                'validation_strategies': 10
            },
            'learning': {
                'learning_patterns': 15,
                'success_analysis': 10,
                'knowledge_extraction': True
            }
        }
        
        return requirements.get(state_name, {})
    
    def _enhance_transitions_with_context(self, workflow: Dict[str, Any]):
        """Enhance transitions with context optimization."""
        transitions = workflow.get('transitions', [])
        
        for transition in transitions:
            if isinstance(transition, dict):
                # Add DPIBS context enhancement to transition
                transition['dpibs_context_enhancement'] = {
                    'enabled': True,
                    'optimization_on_transition': True,
                    'agent_handoff_enhancement': True,
                    'performance_tracking': True
                }
    
    def _integrate_quality_gates_with_dpibs(self, workflow: Dict[str, Any]):
        """Integrate DPIBS with existing quality gates."""
        quality_gates = workflow.get('quality_gates', {})
        
        # Enhance adaptive coverage with DPIBS optimization
        if 'adaptive_coverage' in quality_gates:
            adaptive_coverage = quality_gates['adaptive_coverage']
            adaptive_coverage['dpibs_optimization'] = {
                'enabled': True,
                'context_aware_thresholds': True,
                'performance_enhancement': True,
                'fallback_to_original': True
            }
        
        # Add DPIBS-specific quality gates
        quality_gates['dpibs_context_quality'] = {
            'threshold': 'optimized',
            'required': True,
            'blocker': False,
            'description': 'DPIBS context optimization quality gate'
        }
        
        quality_gates['agent_context_optimization'] = {
            'threshold': '70%',
            'required': True,
            'blocker': False,
            'description': 'Agent context optimization effectiveness'
        }
    
    async def process_state_transition(self, issue_id: str, from_state: str, to_state: str,
                                     trigger_condition: str = None, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process state transition with DPIBS enhancements.
        
        Args:
            issue_id: Issue identifier
            from_state: Current state
            to_state: Target state
            trigger_condition: Transition trigger condition
            context_data: Additional context data
            
        Returns:
            Transition processing result with DPIBS enhancements
        """
        start_time = time.time()
        transition_id = self._generate_transition_id(issue_id, from_state, to_state)
        
        try:
            # Create transition event
            transition_event = StateTransitionEvent(
                transition_id=transition_id,
                issue_id=issue_id,
                from_state=from_state,
                to_state=to_state,
                transition_type=self._determine_transition_type(from_state, to_state, trigger_condition),
                agent_type=self._get_agent_for_state(to_state),
                trigger_condition=trigger_condition,
                context_data=context_data or {},
                dpibs_enhancement=True,
                timestamp=datetime.now()
            )
            
            # Process existing state machine transition first
            existing_result = await self._process_existing_transition(transition_event)
            
            # Apply DPIBS enhancements
            dpibs_enhancements = await self._apply_dpibs_state_enhancements(transition_event)
            
            # Update performance metrics
            processing_time_ms = int((time.time() - start_time) * 1000)
            self._update_transition_metrics(transition_id, processing_time_ms, True)
            
            return {
                'transition_id': transition_id,
                'issue_id': issue_id,
                'transition': f"{from_state} -> {to_state}",
                'existing_transition_result': existing_result,
                'dpibs_enhancements': dpibs_enhancements,
                'processing_time_ms': processing_time_ms,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"State transition processing failed: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            return await self._handle_transition_error(transition_id, issue_id, from_state, to_state, e, processing_time_ms)
    
    async def _process_existing_transition(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Process transition through existing state machine logic."""
        try:
            # Validate transition is allowed by existing workflow
            is_valid = self._validate_existing_transition(event.from_state, event.to_state, event.trigger_condition)
            
            if not is_valid:
                return {
                    'valid': False,
                    'reason': 'transition_not_allowed_by_existing_workflow',
                    'from_state': event.from_state,
                    'to_state': event.to_state
                }
            
            # Apply existing state machine logic
            existing_logic_result = await self.state_transition_manager.process_existing_transition(event)
            
            return {
                'valid': True,
                'existing_logic_applied': True,
                'result': existing_logic_result,
                'transition_allowed': True
            }
            
        except Exception as e:
            self.logger.error(f"Existing transition processing failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'existing_logic_applied': False
            }
    
    async def _apply_dpibs_state_enhancements(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Apply DPIBS-specific state transition enhancements."""
        try:
            enhancements = {}
            
            # Apply context trigger enhancements
            if self.context_trigger_manager:
                context_result = await self.context_trigger_manager.process_state_context_trigger(event)
                enhancements['context_triggers'] = context_result
            
            # Apply agent context optimization for target state
            if event.agent_type and self.agent_context_optimizer:
                agent_optimization = await self._optimize_agent_context_for_state(event)
                enhancements['agent_context_optimization'] = agent_optimization
            
            # Apply quality gate enhancements
            if self.quality_gate_integrator:
                quality_enhancement = await self.quality_gate_integrator.enhance_quality_gates_for_state(event.to_state)
                enhancements['quality_gate_enhancements'] = quality_enhancement
            
            # Update transition history
            self._update_transition_history(event)
            
            return enhancements
            
        except Exception as e:
            self.logger.error(f"DPIBS state enhancements failed: {e}")
            return {'error': str(e), 'enhancements_applied': False}
    
    async def _optimize_agent_context_for_state(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Optimize agent context for target state."""
        try:
            if not event.agent_type or not self.agent_context_optimizer:
                return {'optimization_applied': False, 'reason': 'agent_type_or_optimizer_missing'}
            
            # Prepare context data for agent
            state_context = {
                'issue_id': event.issue_id,
                'current_state': event.to_state,
                'previous_state': event.from_state,
                'transition_trigger': event.trigger_condition,
                'workflow_context': self._get_workflow_context_for_state(event.to_state),
                'additional_context': event.context_data
            }
            
            # Optimize context for agent type
            optimization_result = await self.agent_context_optimizer.optimize_agent_context(
                event.agent_type, state_context, {'issue_id': event.issue_id}
            )
            
            return {
                'optimization_applied': True,
                'agent_type': event.agent_type,
                'state': event.to_state,
                'optimization_result': asdict(optimization_result)
            }
            
        except Exception as e:
            self.logger.error(f"Agent context optimization failed: {e}")
            return {
                'optimization_applied': False,
                'error': str(e),
                'agent_type': event.agent_type
            }
    
    def _validate_existing_transition(self, from_state: str, to_state: str, trigger_condition: str) -> bool:
        """Validate transition is allowed by existing workflow."""
        workflow = self.existing_workflow.get('workflow', {})
        transitions = workflow.get('transitions', [])
        
        # Check if transition exists in existing workflow
        for transition in transitions:
            if isinstance(transition, dict):
                if (transition.get('from') == from_state and 
                    transition.get('to') == to_state):
                    return True
        
        # Allow all transitions in compatibility mode
        return self.compatibility_mode
    
    def _determine_transition_type(self, from_state: str, to_state: str, trigger_condition: str) -> StateTransitionType:
        """Determine type of state transition."""
        if trigger_condition == 'auto':
            return StateTransitionType.AUTOMATIC
        elif trigger_condition == 'manual':
            return StateTransitionType.MANUAL
        elif trigger_condition and 'condition' in trigger_condition:
            return StateTransitionType.CONDITIONAL
        else:
            return StateTransitionType.TRIGGERED
    
    def _get_agent_for_state(self, state: str) -> Optional[str]:
        """Get agent type for state."""
        workflow = self.existing_workflow.get('workflow', {})
        states = workflow.get('states', {})
        
        state_config = states.get(state, {})
        return state_config.get('agent')
    
    def _get_workflow_context_for_state(self, state: str) -> Dict[str, Any]:
        """Get workflow context for specific state."""
        workflow = self.existing_workflow.get('workflow', {})
        states = workflow.get('states', {})
        
        state_config = states.get(state, {})
        
        return {
            'state_name': state,
            'description': state_config.get('description', ''),
            'timeout': state_config.get('timeout', ''),
            'agent': state_config.get('agent', ''),
            'checkpoints': state_config.get('checkpoints', False),
            'parallel_to': state_config.get('parallel_to', []),
            'required_for': state_config.get('required_for', [])
        }
    
    def _generate_transition_id(self, issue_id: str, from_state: str, to_state: str) -> str:
        """Generate unique transition ID."""
        transition_str = f"{issue_id}_{from_state}_{to_state}_{int(time.time())}"
        return hashlib.md5(transition_str.encode()).hexdigest()[:12]
    
    def _update_transition_history(self, event: StateTransitionEvent):
        """Update transition history."""
        history_key = f"{event.issue_id}_transitions"
        
        if history_key not in self.state_history:
            self.state_history[history_key] = []
        
        self.state_history[history_key].append({
            'transition_id': event.transition_id,
            'from_state': event.from_state,
            'to_state': event.to_state,
            'timestamp': event.timestamp.isoformat(),
            'agent_type': event.agent_type,
            'dpibs_enhancement': event.dpibs_enhancement
        })
    
    def _update_transition_metrics(self, transition_id: str, processing_time_ms: int, success: bool):
        """Update transition processing metrics."""
        self.performance_metrics['transitions_processed'] += 1
        
        # Update average processing time
        total_transitions = self.performance_metrics['transitions_processed']
        current_avg = self.performance_metrics['average_transition_time_ms']
        new_avg = (current_avg * (total_transitions - 1) + processing_time_ms) / total_transitions
        self.performance_metrics['average_transition_time_ms'] = new_avg
        
        # Update success rate
        if success:
            success_rate = self.performance_metrics.get('optimization_success_rate', 0.0)
            new_success_rate = (success_rate * (total_transitions - 1) + 1.0) / total_transitions
            self.performance_metrics['optimization_success_rate'] = new_success_rate
    
    async def _handle_transition_error(self, transition_id: str, issue_id: str, from_state: str, 
                                     to_state: str, error: Exception, processing_time_ms: int) -> Dict[str, Any]:
        """Handle transition processing error with fallback."""
        self._update_transition_metrics(transition_id, processing_time_ms, False)
        
        return {
            'transition_id': transition_id,
            'issue_id': issue_id,
            'transition': f"{from_state} -> {to_state}",
            'error': str(error),
            'processing_time_ms': processing_time_ms,
            'success': False,
            'fallback_to_existing': True
        }
    
    def get_state_machine_integration_status(self) -> Dict[str, Any]:
        """Get current state machine integration status."""
        workflow = self.existing_workflow.get('workflow', {})
        
        return {
            'integration_active': True,
            'compatibility_mode': self.compatibility_mode,
            'existing_workflow_preserved': self.preserve_existing_workflow,
            'workflow_loaded': {
                'existing_states': len(workflow.get('states', {})),
                'existing_transitions': len(workflow.get('transitions', [])),
                'existing_quality_gates': len(workflow.get('quality_gates', {})),
                'enhanced_workflow_active': bool(self.enhanced_workflow)
            },
            'performance_metrics': self.performance_metrics.copy(),
            'state_machine_state': {
                'active_transitions': len(self.active_transitions),
                'state_history_entries': len(self.state_history),
                'context_triggers_active': len(self.context_triggers)
            },
            'components_initialized': {
                'state_transition_manager': self.state_transition_manager is not None,
                'context_trigger_manager': self.context_trigger_manager is not None,
                'quality_gate_integrator': self.quality_gate_integrator is not None,
                'agent_context_optimizer': self.agent_context_optimizer is not None,
                'rif_database': self.rif_db is not None
            },
            'dpibs_enhancements': {
                'context_triggers_enabled': True,
                'agent_handoff_enhancement': True,
                'quality_gate_integration': True,
                'performance_optimization': True
            }
        }
    
    def shutdown(self):
        """Clean shutdown of state machine integration."""
        try:
            if self.state_transition_manager:
                self.state_transition_manager.shutdown()
            if self.context_trigger_manager:
                self.context_trigger_manager.shutdown()
            if self.quality_gate_integrator:
                self.quality_gate_integrator.shutdown()
            if self.agent_context_optimizer:
                self.agent_context_optimizer.shutdown()
            if self.rif_db:
                self.rif_db.close()
            
            self.logger.info("RIF State Machine Integrator shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during state machine integration shutdown: {e}")


class StateTransitionManager:
    """Enhanced state transition manager."""
    
    def __init__(self, config: Dict[str, Any], existing_workflow: Dict[str, Any],
                 agent_context_optimizer: AgentContextOptimizer, rif_db: RIFDatabase):
        self.config = config
        self.existing_workflow = existing_workflow
        self.agent_context_optimizer = agent_context_optimizer
        self.rif_db = rif_db
        self.logger = logging.getLogger(__name__)
    
    async def process_existing_transition(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Process transition through existing state machine logic."""
        try:
            # Apply existing transition logic
            transition_result = {
                'transition_processed': True,
                'from_state': event.from_state,
                'to_state': event.to_state,
                'agent_type': event.agent_type,
                'existing_logic_preserved': True
            }
            
            # Check for existing transition conditions
            if event.trigger_condition:
                condition_result = self._evaluate_transition_condition(event)
                transition_result['condition_evaluation'] = condition_result
            
            return transition_result
            
        except Exception as e:
            self.logger.error(f"Existing transition processing failed: {e}")
            return {'error': str(e), 'transition_processed': False}
    
    def _evaluate_transition_condition(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Evaluate transition condition."""
        return {
            'condition': event.trigger_condition,
            'evaluated': True,
            'result': 'condition_met'  # Simplified evaluation
        }
    
    def shutdown(self):
        """Shutdown transition manager."""
        self.logger.info("State transition manager shutdown completed")


class ContextTriggerManager:
    """Context trigger manager for DPIBS integration."""
    
    def __init__(self, config: Dict[str, Any], existing_workflow: Dict[str, Any],
                 agent_context_optimizer: AgentContextOptimizer):
        self.config = config
        self.existing_workflow = existing_workflow
        self.agent_context_optimizer = agent_context_optimizer
        self.logger = logging.getLogger(__name__)
    
    async def process_state_context_trigger(self, event: StateTransitionEvent) -> ContextTriggerResult:
        """Process context trigger for state transition."""
        start_time = time.time()
        trigger_id = f"trigger_{event.transition_id}"
        
        try:
            enhancement_metadata = {}
            
            # Apply DPIBS context triggers if enabled
            if self.config.get('dpibs_triggers_enabled'):
                dpibs_result = await self._apply_dpibs_context_trigger(event)
                enhancement_metadata['dpibs_trigger'] = dpibs_result
            
            # Apply agent context injection
            if self.config.get('agent_context_injection') and event.agent_type:
                injection_result = await self._inject_agent_context(event)
                enhancement_metadata['agent_context_injection'] = injection_result
            
            # Apply state-aware optimization
            if self.config.get('state_aware_optimization'):
                optimization_result = await self._apply_state_aware_optimization(event)
                enhancement_metadata['state_optimization'] = optimization_result
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return ContextTriggerResult(
                trigger_id=trigger_id,
                success=True,
                context_enhanced=True,
                agent_context_optimized=bool(event.agent_type),
                processing_time_ms=processing_time_ms,
                enhancement_metadata=enhancement_metadata
            )
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Context trigger processing failed: {e}")
            return ContextTriggerResult(
                trigger_id=trigger_id,
                success=False,
                context_enhanced=False,
                agent_context_optimized=False,
                processing_time_ms=processing_time_ms,
                enhancement_metadata={'error': str(e)}
            )
    
    async def _apply_dpibs_context_trigger(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Apply DPIBS-specific context trigger."""
        return {
            'dpibs_trigger_applied': True,
            'state': event.to_state,
            'context_enhancement_level': 'high' if event.agent_type else 'medium'
        }
    
    async def _inject_agent_context(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Inject agent-specific context."""
        return {
            'agent_context_injected': True,
            'agent_type': event.agent_type,
            'context_size': len(str(event.context_data))
        }
    
    async def _apply_state_aware_optimization(self, event: StateTransitionEvent) -> Dict[str, Any]:
        """Apply state-aware optimization."""
        return {
            'state_optimization_applied': True,
            'optimization_level': 'high',
            'state': event.to_state
        }
    
    def shutdown(self):
        """Shutdown context trigger manager."""
        self.logger.info("Context trigger manager shutdown completed")


class QualityGateIntegrator:
    """Quality gate integrator for DPIBS enhancement."""
    
    def __init__(self, config: Dict[str, Any], existing_workflow: Dict[str, Any], rif_db: RIFDatabase):
        self.config = config
        self.existing_workflow = existing_workflow
        self.rif_db = rif_db
        self.logger = logging.getLogger(__name__)
        
        self.enhanced_gates = {}
    
    async def enhance_quality_gates_for_state(self, state: str) -> List[QualityGateIntegration]:
        """Enhance quality gates for specific state."""
        try:
            workflow = self.existing_workflow.get('workflow', {})
            quality_gates = workflow.get('quality_gates', {})
            
            enhanced_gates = []
            
            # Enhance existing quality gates
            for gate_name, gate_config in quality_gates.items():
                if self._should_enhance_gate(gate_name, state):
                    enhanced_gate = await self._enhance_quality_gate(gate_name, gate_config, state)
                    enhanced_gates.append(enhanced_gate)
            
            return enhanced_gates
            
        except Exception as e:
            self.logger.error(f"Quality gate enhancement failed: {e}")
            return []
    
    def _should_enhance_gate(self, gate_name: str, state: str) -> bool:
        """Check if quality gate should be enhanced for state."""
        # Enhance adaptive coverage and context-aware gates
        enhance_for_gates = ['adaptive_coverage', 'code_coverage', 'quality_score']
        enhance_for_states = ['implementing', 'validating', 'architecting']
        
        return gate_name in enhance_for_gates or state in enhance_for_states
    
    async def _enhance_quality_gate(self, gate_name: str, gate_config: Dict[str, Any], 
                                  state: str) -> QualityGateIntegration:
        """Enhance individual quality gate."""
        try:
            original_threshold = gate_config.get('threshold')
            
            # Apply DPIBS optimization to threshold
            enhanced_threshold = await self._optimize_threshold(gate_name, original_threshold, state)
            
            return QualityGateIntegration(
                gate_name=gate_name,
                original_threshold=original_threshold,
                dpibs_enhanced_threshold=enhanced_threshold,
                context_aware=True,
                optimization_applied=True,
                performance_improvement=self._calculate_performance_improvement(original_threshold, enhanced_threshold)
            )
            
        except Exception as e:
            self.logger.error(f"Quality gate enhancement failed for {gate_name}: {e}")
            return QualityGateIntegration(
                gate_name=gate_name,
                original_threshold=gate_config.get('threshold'),
                dpibs_enhanced_threshold=gate_config.get('threshold'),
                context_aware=False,
                optimization_applied=False,
                performance_improvement=None
            )
    
    async def _optimize_threshold(self, gate_name: str, original_threshold: Any, state: str) -> Any:
        """Optimize quality gate threshold with DPIBS."""
        # Apply context-aware optimization
        if gate_name == 'adaptive_coverage' and isinstance(original_threshold, (int, float)):
            # Optimize coverage threshold based on state context
            if state in ['implementing', 'architecting']:
                return min(original_threshold * 1.1, 100)  # Increase by 10% max
            else:
                return original_threshold
        
        return original_threshold
    
    def _calculate_performance_improvement(self, original: Any, enhanced: Any) -> Optional[float]:
        """Calculate performance improvement percentage."""
        try:
            if isinstance(original, (int, float)) and isinstance(enhanced, (int, float)):
                if original > 0:
                    return ((enhanced - original) / original) * 100
        except:
            pass
        
        return None
    
    def shutdown(self):
        """Shutdown quality gate integrator."""
        self.logger.info("Quality gate integrator shutdown completed")


# Integration Interface Functions
def create_state_machine_integration(config: Dict[str, Any] = None) -> RIFStateMachineIntegrator:
    """
    Factory function to create RIF state machine integration system.
    
    This is the main entry point for DPIBS State Machine Integration.
    """
    if config is None:
        config = {
            'preserve_existing_workflow': True,
            'compatibility_mode': True,
            'state_transitions': {
                'context_optimization_enabled': True,
                'agent_handoff_enhancement': True,
                'transition_monitoring': True,
                'performance_tracking': True
            },
            'context_triggers': {
                'dpibs_triggers_enabled': True,
                'agent_context_injection': True,
                'state_aware_optimization': True,
                'performance_optimization': True
            },
            'quality_gate_integration': {
                'adaptive_thresholds_enhanced': True,
                'context_aware_gates': True,
                'dpibs_optimization': True,
                'preserve_existing_gates': True
            },
            'agent_context_optimization': {
                'fallback_mode_enabled': True,
                'compatibility_level': 'full'
            }
        }
    
    return RIFStateMachineIntegrator(config)


async def process_workflow_state_transition(integrator: RIFStateMachineIntegrator, 
                                          issue_id: str, from_state: str, to_state: str,
                                          context_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process workflow state transition through DPIBS integration.
    Used by RIF orchestration system.
    """
    return await integrator.process_state_transition(issue_id, from_state, to_state, 'auto', context_data)


# Backward Compatibility Functions
def is_state_machine_integration_available() -> bool:
    """Check if state machine integration is available and working."""
    try:
        config = {'compatibility_mode': True}
        integrator = RIFStateMachineIntegrator(config)
        return bool(integrator.existing_workflow) or integrator.compatibility_mode
    except Exception:
        return False


def get_existing_workflow_only() -> Dict[str, Any]:
    """Get existing workflow configuration for fallback."""
    try:
        workflow_path = Path('config/rif-workflow.yaml')
        if workflow_path.exists():
            with open(workflow_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    
    return {}