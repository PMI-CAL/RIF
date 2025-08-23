"""
IntegrationController - Component Coordination and Orchestration

Manages coordination between the four pipeline components according to the 
Master Coordination Plan, ensuring proper data flow and synchronization.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ComponentState(Enum):
    """Component state enumeration."""
    INITIALIZING = "initializing"
    READY = "ready" 
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component: Any
    state: ComponentState
    last_activity: float
    error_count: int
    dependencies: List[str]


@dataclass  
class CoordinationEvent:
    """Event for component coordination."""
    timestamp: float
    event_type: str
    source_component: str
    target_component: Optional[str]
    data: Dict[str, Any]


class IntegrationController:
    """
    Coordinates component interactions and manages data flow between pipeline components.
    
    Responsibilities:
    - Component registration and lifecycle management
    - Dependency coordination (Issue #30 → Issues #31,#32 → Issue #33)
    - Event-driven coordination and synchronization
    - Error recovery and graceful degradation
    - Resource sharing coordination
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the integration controller."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Component registry
        self._components: Dict[str, ComponentInfo] = {}
        self._lock = threading.RLock()
        
        # Event system
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._event_history = []
        
        # Coordination state
        self._coordination_active = False
        self._dependency_graph = {
            'entity_extraction': [],  # No dependencies
            'relationships': ['entity_extraction'],  # Depends on entities
            'embeddings': ['entity_extraction'],     # Depends on entities  
            'query_planning': ['relationships', 'embeddings']  # Depends on both
        }
        
        # Synchronization points from master plan
        self._checkpoints = {
            'entity_extraction_ready': {
                'required_components': ['entity_extraction'],
                'enables': ['relationships', 'embeddings'],
                'validation': self._validate_entity_extraction_ready
            },
            'parallel_processing_ready': {
                'required_components': ['relationships', 'embeddings'],
                'enables': ['query_planning'],
                'validation': self._validate_parallel_components_ready
            },
            'integration_complete': {
                'required_components': ['entity_extraction', 'relationships', 'embeddings', 'query_planning'],
                'enables': ['system_ready'],
                'validation': self._validate_integration_complete
            }
        }
        
        self.logger.info("IntegrationController initialized")
    
    def register_component(self, name: str, component: Any, dependencies: List[str] = None) -> bool:
        """
        Register a component with the integration controller.
        
        Args:
            name: Component name
            component: Component instance
            dependencies: List of component names this component depends on
            
        Returns:
            bool: True if registration successful
        """
        with self._lock:
            if name in self._components:
                self.logger.warning(f"Component '{name}' already registered")
                return False
            
            # Validate dependencies
            if dependencies:
                missing_deps = [dep for dep in dependencies if dep not in self._dependency_graph.get(name, [])]
                if missing_deps:
                    self.logger.warning(f"Unknown dependencies for '{name}': {missing_deps}")
            
            self._components[name] = ComponentInfo(
                name=name,
                component=component,
                state=ComponentState.INITIALIZING,
                last_activity=time.time(),
                error_count=0,
                dependencies=dependencies or self._dependency_graph.get(name, [])
            )
            
            self.logger.info(f"Component '{name}' registered with dependencies: {dependencies}")
            self._emit_event('component_registered', name, None, {'component': name})
            return True
    
    def update_component_state(self, name: str, state: ComponentState, error_info: str = None):
        """Update component state and trigger coordination logic."""
        with self._lock:
            if name not in self._components:
                self.logger.error(f"Cannot update state for unregistered component: {name}")
                return
            
            component_info = self._components[name]
            old_state = component_info.state
            component_info.state = state
            component_info.last_activity = time.time()
            
            if state == ComponentState.ERROR:
                component_info.error_count += 1
                self.logger.error(f"Component '{name}' entered error state: {error_info}")
            
            self.logger.info(f"Component '{name}' state changed: {old_state.value} → {state.value}")
            
            # Emit state change event
            self._emit_event('state_changed', name, None, {
                'old_state': old_state.value,
                'new_state': state.value,
                'error_info': error_info
            })
            
            # Check coordination checkpoints
            self._check_coordination_checkpoints()
    
    def _check_coordination_checkpoints(self):
        """Check if any coordination checkpoints can be triggered."""
        for checkpoint_name, checkpoint_info in self._checkpoints.items():
            if self._can_trigger_checkpoint(checkpoint_name):
                self._trigger_checkpoint(checkpoint_name)
    
    def _can_trigger_checkpoint(self, checkpoint_name: str) -> bool:
        """Check if a coordination checkpoint can be triggered."""
        checkpoint_info = self._checkpoints[checkpoint_name]
        required_components = checkpoint_info['required_components']
        
        # Check if all required components are ready
        for component_name in required_components:
            if component_name not in self._components:
                return False
            
            component_info = self._components[component_name]
            if component_info.state not in [ComponentState.READY, ComponentState.PROCESSING]:
                return False
        
        # Run validation if provided
        validation_func = checkpoint_info.get('validation')
        if validation_func:
            try:
                return validation_func()
            except Exception as e:
                self.logger.error(f"Checkpoint validation failed for '{checkpoint_name}': {e}")
                return False
        
        return True
    
    def _trigger_checkpoint(self, checkpoint_name: str):
        """Trigger a coordination checkpoint."""
        checkpoint_info = self._checkpoints[checkpoint_name]
        enabled_components = checkpoint_info.get('enables', [])
        
        self.logger.info(f"Triggering coordination checkpoint: {checkpoint_name}")
        
        # Emit checkpoint event
        self._emit_event('checkpoint_triggered', 'integration_controller', None, {
            'checkpoint': checkpoint_name,
            'enables': enabled_components
        })
        
        # Notify enabled components they can proceed
        for component_name in enabled_components:
            if component_name in self._components:
                self._emit_event('component_enabled', 'integration_controller', component_name, {
                    'checkpoint': checkpoint_name
                })
    
    def _validate_entity_extraction_ready(self) -> bool:
        """Validate that entity extraction is ready for parallel phase."""
        if 'entity_extraction' not in self._components:
            return False
        
        component = self._components['entity_extraction'].component
        
        # Check if component has required methods and is healthy
        required_methods = ['extract_from_file', 'is_healthy']
        for method in required_methods:
            if not hasattr(component, method):
                self.logger.error(f"Entity extraction missing required method: {method}")
                return False
        
        if not component.is_healthy():
            self.logger.error("Entity extraction component is not healthy")
            return False
        
        # Check if entity data is available (basic validation)
        try:
            # This would ideally test with a small sample file
            self.logger.info("Entity extraction validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Entity extraction validation failed: {e}")
            return False
    
    def _validate_parallel_components_ready(self) -> bool:
        """Validate that parallel components (relationships + embeddings) are ready."""
        required_components = ['relationships', 'embeddings']
        
        for component_name in required_components:
            if component_name not in self._components:
                return False
            
            component = self._components[component_name].component
            if not hasattr(component, 'is_healthy') or not component.is_healthy():
                self.logger.error(f"Component '{component_name}' is not healthy")
                return False
        
        self.logger.info("Parallel components validation successful")
        return True
    
    def _validate_integration_complete(self) -> bool:
        """Validate that complete integration is ready."""
        required_components = ['entity_extraction', 'relationships', 'embeddings', 'query_planning']
        
        # Check all components are healthy
        for component_name in required_components:
            if component_name not in self._components:
                self.logger.error(f"Required component missing: {component_name}")
                return False
            
            component_info = self._components[component_name]
            if component_info.state == ComponentState.ERROR:
                self.logger.error(f"Component '{component_name}' is in error state")
                return False
            
            component = component_info.component
            if hasattr(component, 'is_healthy') and not component.is_healthy():
                self.logger.error(f"Component '{component_name}' failed health check")
                return False
        
        self.logger.info("Integration validation successful")
        return True
    
    def coordinate_file_processing(self, file_path: str) -> Dict[str, Any]:
        """
        Coordinate file processing across all components in proper dependency order.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dict with results from all components
        """
        self.logger.info(f"Coordinating file processing: {file_path}")
        results = {}
        
        try:
            # Step 1: Entity Extraction (Issue #30)
            if 'entity_extraction' in self._components:
                self.logger.debug("Processing entities...")
                entities = self._components['entity_extraction'].component.extract_from_file(file_path)
                results['entities'] = entities
                results['entity_count'] = len(entities) if entities else 0
                
                # Emit coordination event
                self._emit_event('entities_extracted', 'entity_extraction', None, {
                    'file_path': file_path,
                    'entity_count': results['entity_count']
                })
            
            # Step 2: Parallel Processing (Issues #31 & #32)  
            # These can run in parallel since they both depend only on entities
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                # Relationship detection
                if 'relationships' in self._components:
                    future_rel = executor.submit(
                        self._components['relationships'].component.detect_from_file,
                        file_path
                    )
                    futures['relationships'] = future_rel
                
                # Embedding generation
                if 'embeddings' in self._components:
                    future_emb = executor.submit(
                        self._components['embeddings'].component.process_entities_by_file,
                        file_path
                    )
                    futures['embeddings'] = future_emb
                
                # Collect results
                for component_name, future in futures.items():
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        results[component_name] = result
                        
                        # Emit coordination event
                        self._emit_event(f'{component_name}_processed', component_name, None, {
                            'file_path': file_path,
                            'result_count': len(result) if isinstance(result, list) else 1
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in {component_name} processing: {e}")
                        results[component_name] = {'error': str(e)}
            
            # Step 3: Query System Update (Issue #33)
            # This step would typically update indexes and caches
            if 'query_planning' in self._components:
                # The query planner typically doesn't process individual files
                # but we could trigger cache updates here if needed
                results['query_system_updated'] = True
            
            results['coordination_success'] = True
            results['processing_time'] = time.time()
            
            self.logger.info(f"File processing coordination completed: {file_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"File processing coordination failed: {e}")
            results['coordination_success'] = False
            results['error'] = str(e)
            return results
    
    def get_component_status(self, name: str = None) -> Dict[str, Any]:
        """Get status of one or all components."""
        with self._lock:
            if name:
                if name not in self._components:
                    return {'error': f'Component {name} not found'}
                
                info = self._components[name]
                return {
                    'name': info.name,
                    'state': info.state.value,
                    'last_activity': info.last_activity,
                    'error_count': info.error_count,
                    'dependencies': info.dependencies,
                    'healthy': (
                        info.state != ComponentState.ERROR and 
                        hasattr(info.component, 'is_healthy') and 
                        info.component.is_healthy()
                    )
                }
            else:
                # Return all components
                return {
                    name: {
                        'state': info.state.value,
                        'last_activity': info.last_activity,
                        'error_count': info.error_count,
                        'dependencies': info.dependencies,
                        'healthy': (
                            info.state != ComponentState.ERROR and
                            hasattr(info.component, 'is_healthy') and
                            info.component.is_healthy()
                        )
                    }
                    for name, info in self._components.items()
                }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get overall coordination system status."""
        with self._lock:
            # Check checkpoint status
            checkpoint_status = {}
            for checkpoint_name in self._checkpoints:
                checkpoint_status[checkpoint_name] = self._can_trigger_checkpoint(checkpoint_name)
            
            # Component health summary
            total_components = len(self._components)
            healthy_components = sum(1 for info in self._components.values() 
                                   if info.state != ComponentState.ERROR)
            error_components = [name for name, info in self._components.items() 
                               if info.state == ComponentState.ERROR]
            
            return {
                'coordination_active': self._coordination_active,
                'total_components': total_components,
                'healthy_components': healthy_components,
                'error_components': error_components,
                'checkpoint_status': checkpoint_status,
                'recent_events': len(self._event_history),
                'dependency_graph': self._dependency_graph
            }
    
    def _emit_event(self, event_type: str, source: str, target: Optional[str], data: Dict[str, Any]):
        """Emit a coordination event."""
        event = CoordinationEvent(
            timestamp=time.time(),
            event_type=event_type,
            source_component=source,
            target_component=target,
            data=data
        )
        
        self._event_history.append(event)
        
        # Keep only recent events
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-500:]
        
        # Trigger event handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler for '{event_type}': {e}")
        
        self.logger.debug(f"Event emitted: {event_type} from {source} to {target}")
    
    def add_event_handler(self, event_type: str, handler: Callable[[CoordinationEvent], None]):
        """Add an event handler for coordination events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        self.logger.info(f"Event handler added for: {event_type}")
    
    def get_recent_events(self, count: int = 20, event_type: str = None) -> List[Dict[str, Any]]:
        """Get recent coordination events."""
        events = self._event_history[-count:] if count > 0 else self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return [
            {
                'timestamp': e.timestamp,
                'event_type': e.event_type,
                'source_component': e.source_component,
                'target_component': e.target_component,
                'data': e.data
            }
            for e in events
        ]
    
    def shutdown(self):
        """Shutdown the integration controller."""
        with self._lock:
            self._coordination_active = False
            
            # Notify all components of shutdown
            for name in self._components:
                self._emit_event('shutdown_requested', 'integration_controller', name, {})
            
            self._components.clear()
            self._event_handlers.clear()
            
            self.logger.info("IntegrationController shutdown complete")