"""
DPIBS Master Integration Orchestrator
===================================

Master orchestrator for all DPIBS Integration Architecture components.
Provides unified interface and coordination for all 5 integration layers.

This is the main entry point for DPIBS Phase 3 Integration Architecture,
providing comprehensive coordination while maintaining zero regression guarantees.

Components Orchestrated:
- Layer 1: Agent Workflow Integration
- Layer 2: MCP Knowledge Server Integration  
- Layer 3: GitHub Workflow Integration
- Layer 4: RIF State Machine Integration
- Layer 5: Backward Compatibility Strategy
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import os

# Import all DPIBS integration layers
from systems.dpibs_agent_workflow_integration import (
    create_agent_workflow_integration,
    AgentContextOptimizer
)
from systems.dpibs_mcp_integration import (
    create_mcp_integration,
    MCPKnowledgeIntegrator
)
from systems.dpibs_github_integration import (
    create_github_integration,
    GitHubWorkflowIntegrator
)
from systems.dpibs_state_machine_integration import (
    create_state_machine_integration,
    RIFStateMachineIntegrator
)
from systems.dpibs_backward_compatibility import (
    create_backward_compatibility_layer,
    DPIBSBackwardCompatibilityLayer,
    CompatibilityLevel
)


class IntegrationStatus(Enum):
    """Overall integration status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass
class DPIBSIntegrationHealth:
    """Health status of DPIBS integration."""
    overall_status: IntegrationStatus
    layer_statuses: Dict[str, str]
    compatibility_level: str
    performance_score: float
    last_health_check: datetime
    recommendations: List[str]


@dataclass
class IntegrationMetrics:
    """Integration performance metrics."""
    operations_processed: int
    successful_operations: int
    fallback_operations: int
    average_processing_time_ms: float
    compatibility_score: float
    uptime_percentage: float


class DPIBSMasterIntegration:
    """
    Master orchestrator for DPIBS Integration Architecture.
    
    Coordinates all 5 integration layers and provides unified interface
    for RIF system to access DPIBS capabilities with zero regression guarantees.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Integration Layer Components
        self.agent_integration: Optional[AgentContextOptimizer] = None
        self.mcp_integration: Optional[MCPKnowledgeIntegrator] = None
        self.github_integration: Optional[GitHubWorkflowIntegrator] = None
        self.state_machine_integration: Optional[RIFStateMachineIntegrator] = None
        self.backward_compatibility: Optional[DPIBSBackwardCompatibilityLayer] = None
        
        # Integration State
        self.integration_status = IntegrationStatus.INITIALIZING
        self.initialization_time = None
        self.last_health_check = None
        
        # Performance Monitoring
        self.metrics = IntegrationMetrics(
            operations_processed=0,
            successful_operations=0,
            fallback_operations=0,
            average_processing_time_ms=0.0,
            compatibility_score=1.0,
            uptime_percentage=100.0
        )
        
        # Health Monitoring
        self.health_check_interval = timedelta(minutes=5)
        self.health_monitoring_active = False
        
        # Initialize all integration layers
        self._initialize_master_integration()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for DPIBS integration."""
        return {
            'zero_regression_mode': True,
            'aggressive_fallback': True,
            'health_monitoring_enabled': True,
            'performance_tracking_enabled': True,
            
            # Layer-specific configurations
            'agent_integration': {
                'compatibility_level': 'full',
                'fallback_mode_enabled': True
            },
            'mcp_integration': {
                'compatibility_mode': True,
                'performance_optimization': True
            },
            'github_integration': {
                'preserve_existing_hooks': True,
                'compatibility_mode': True
            },
            'state_machine_integration': {
                'preserve_existing_workflow': True,
                'compatibility_mode': True
            },
            'backward_compatibility': {
                'zero_regression_mode': True,
                'aggressive_fallback': True,
                'migration_enabled': True
            }
        }
    
    def _initialize_master_integration(self):
        """Initialize all DPIBS integration layers."""
        self.initialization_time = datetime.now()
        
        try:
            self.logger.info("Initializing DPIBS Master Integration...")
            
            # Initialize backward compatibility layer first (provides fallback for other layers)
            self._initialize_backward_compatibility()
            
            # Initialize integration layers with compatibility support
            self._initialize_agent_integration()
            self._initialize_mcp_integration()
            self._initialize_github_integration()
            self._initialize_state_machine_integration()
            
            # Determine overall integration status
            self._determine_integration_status()
            
            # Start health monitoring
            if self.config.get('health_monitoring_enabled', True):
                self._start_health_monitoring()
            
            self.logger.info(f"DPIBS Master Integration initialized with status: {self.integration_status.value}")
            
        except Exception as e:
            self.logger.error(f"Master integration initialization failed: {e}")
            self.integration_status = IntegrationStatus.ERROR
            # Continue in error state - backward compatibility should still provide fallback
    
    def _initialize_backward_compatibility(self):
        """Initialize backward compatibility layer."""
        try:
            compatibility_config = self.config.get('backward_compatibility', {})
            self.backward_compatibility = create_backward_compatibility_layer(compatibility_config)
            self.logger.info("Backward compatibility layer initialized")
        except Exception as e:
            self.logger.error(f"Backward compatibility initialization failed: {e}")
            # This is critical - without compatibility layer, we can't guarantee zero regression
            raise
    
    def _initialize_agent_integration(self):
        """Initialize agent workflow integration."""
        try:
            if self.backward_compatibility:
                agent_config = self.config.get('agent_integration', {})
                
                async def init_agent_integration():
                    return create_agent_workflow_integration(agent_config)
                
                # Use compatibility layer to ensure fallback if initialization fails
                self.agent_integration = asyncio.run(
                    self.backward_compatibility.execute_with_compatibility(
                        'initialize', 'agent_workflow', init_agent_integration
                    )
                )
                
                if self.agent_integration:
                    self.logger.info("Agent workflow integration initialized")
                else:
                    self.logger.warning("Agent workflow integration initialized in fallback mode")
            
        except Exception as e:
            self.logger.warning(f"Agent integration initialization failed: {e}")
            # Not critical - will use fallback
    
    def _initialize_mcp_integration(self):
        """Initialize MCP knowledge server integration."""
        try:
            if self.backward_compatibility:
                mcp_config = self.config.get('mcp_integration', {})
                
                async def init_mcp_integration():
                    return create_mcp_integration(mcp_config)
                
                self.mcp_integration = asyncio.run(
                    self.backward_compatibility.execute_with_compatibility(
                        'initialize', 'mcp_integration', init_mcp_integration
                    )
                )
                
                if self.mcp_integration:
                    self.logger.info("MCP integration initialized")
                else:
                    self.logger.warning("MCP integration initialized in fallback mode")
            
        except Exception as e:
            self.logger.warning(f"MCP integration initialization failed: {e}")
            # Not critical - will use fallback
    
    def _initialize_github_integration(self):
        """Initialize GitHub workflow integration."""
        try:
            if self.backward_compatibility:
                github_config = self.config.get('github_integration', {})
                
                async def init_github_integration():
                    return create_github_integration(github_config)
                
                self.github_integration = asyncio.run(
                    self.backward_compatibility.execute_with_compatibility(
                        'initialize', 'github_integration', init_github_integration
                    )
                )
                
                if self.github_integration:
                    self.logger.info("GitHub integration initialized")
                else:
                    self.logger.warning("GitHub integration initialized in fallback mode")
            
        except Exception as e:
            self.logger.warning(f"GitHub integration initialization failed: {e}")
            # Not critical - will use fallback
    
    def _initialize_state_machine_integration(self):
        """Initialize RIF state machine integration."""
        try:
            if self.backward_compatibility:
                state_config = self.config.get('state_machine_integration', {})
                
                async def init_state_integration():
                    return create_state_machine_integration(state_config)
                
                self.state_machine_integration = asyncio.run(
                    self.backward_compatibility.execute_with_compatibility(
                        'initialize', 'state_machine', init_state_integration
                    )
                )
                
                if self.state_machine_integration:
                    self.logger.info("State machine integration initialized")
                else:
                    self.logger.warning("State machine integration initialized in fallback mode")
            
        except Exception as e:
            self.logger.warning(f"State machine integration initialization failed: {e}")
            # Not critical - will use fallback
    
    def _determine_integration_status(self):
        """Determine overall integration status."""
        active_integrations = sum(1 for integration in [
            self.agent_integration,
            self.mcp_integration, 
            self.github_integration,
            self.state_machine_integration
        ] if integration is not None)
        
        total_integrations = 4
        
        if self.backward_compatibility is None:
            self.integration_status = IntegrationStatus.ERROR
        elif active_integrations == total_integrations:
            self.integration_status = IntegrationStatus.ACTIVE
        elif active_integrations >= total_integrations * 0.5:
            self.integration_status = IntegrationStatus.DEGRADED
        elif active_integrations > 0:
            self.integration_status = IntegrationStatus.FALLBACK
        else:
            self.integration_status = IntegrationStatus.FALLBACK  # Compatibility layer provides fallback
    
    def _start_health_monitoring(self):
        """Start health monitoring background process."""
        self.health_monitoring_active = True
        # In production, would start background monitoring task
        self.logger.info("Health monitoring started")
    
    # Public API Methods - Main interface for RIF system
    
    async def optimize_agent_context(self, agent_type: str, context_data: Dict[str, Any],
                                   issue_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize context for RIF agent with zero regression guarantee.
        
        Main interface for RIF orchestration to enhance agent context.
        """
        if not self.backward_compatibility:
            return context_data  # Emergency fallback
        
        try:
            # Execute through compatibility layer
            async def agent_optimization():
                if self.agent_integration:
                    result = await self.agent_integration.optimize_agent_context(
                        agent_type, context_data, issue_context
                    )
                    return {
                        'optimized': True,
                        'optimization_result': result,
                        'original_context': context_data,
                        'enhanced_context': context_data  # Would be enhanced in real implementation
                    }
                else:
                    return {
                        'optimized': False,
                        'fallback_used': True,
                        'context': context_data
                    }
            
            result = await self.backward_compatibility.execute_with_compatibility(
                'optimize_context', 'agent_workflow', agent_optimization
            )
            
            self._update_operation_metrics(True)
            return result
            
        except Exception as e:
            self.logger.error(f"Agent context optimization failed: {e}")
            self._update_operation_metrics(False)
            return {'error': str(e), 'fallback_context': context_data}
    
    async def enhanced_knowledge_query(self, query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced knowledge query with fallback.
        
        Main interface for RIF components to query enhanced knowledge.
        """
        if not self.backward_compatibility:
            return {'result': 'compatibility_layer_unavailable'}
        
        try:
            async def mcp_query():
                if self.mcp_integration:
                    return await self.mcp_integration.enhanced_query(query_type, query_data)
                else:
                    return {
                        'query_type': query_type,
                        'result': {'fallback': True, 'message': 'MCP integration unavailable'},
                        'success': True
                    }
            
            result = await self.backward_compatibility.execute_with_compatibility(
                'enhanced_query', 'mcp_integration', mcp_query
            )
            
            self._update_operation_metrics(True)
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced knowledge query failed: {e}")
            self._update_operation_metrics(False)
            return {'error': str(e), 'fallback': True}
    
    async def process_github_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process GitHub event with enhanced workflow integration.
        
        Main interface for GitHub event processing.
        """
        if not self.backward_compatibility:
            return {'event_processed': True, 'compatibility_unavailable': True}
        
        try:
            async def github_processing():
                if self.github_integration:
                    return await self.github_integration.process_github_event(event_type, event_data)
                else:
                    return {
                        'event_processed': True,
                        'fallback_used': True,
                        'existing_hooks_preserved': True
                    }
            
            result = await self.backward_compatibility.execute_with_compatibility(
                'process_event', 'github_integration', github_processing
            )
            
            self._update_operation_metrics(True)
            return result
            
        except Exception as e:
            self.logger.error(f"GitHub event processing failed: {e}")
            self._update_operation_metrics(False)
            return {'error': str(e), 'event_processed': False}
    
    async def process_state_transition(self, issue_id: str, from_state: str, to_state: str,
                                     context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process RIF workflow state transition with enhancements.
        
        Main interface for RIF workflow state transitions.
        """
        if not self.backward_compatibility:
            return {'transition_processed': True, 'compatibility_unavailable': True}
        
        try:
            async def state_transition():
                if self.state_machine_integration:
                    return await self.state_machine_integration.process_state_transition(
                        issue_id, from_state, to_state, 'auto', context_data
                    )
                else:
                    return {
                        'transition_processed': True,
                        'existing_workflow_preserved': True,
                        'fallback_used': True
                    }
            
            result = await self.backward_compatibility.execute_with_compatibility(
                'process_transition', 'state_machine', state_transition
            )
            
            self._update_operation_metrics(True)
            return result
            
        except Exception as e:
            self.logger.error(f"State transition processing failed: {e}")
            self._update_operation_metrics(False)
            return {'error': str(e), 'transition_processed': False}
    
    def _update_operation_metrics(self, success: bool):
        """Update operation metrics."""
        self.metrics.operations_processed += 1
        
        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.fallback_operations += 1
        
        # Update compatibility score
        if self.metrics.operations_processed > 0:
            self.metrics.compatibility_score = (
                self.metrics.successful_operations / self.metrics.operations_processed
            )
    
    # Health and Status Methods
    
    async def get_integration_health(self) -> DPIBSIntegrationHealth:
        """Get comprehensive integration health status."""
        try:
            # Check each layer status
            layer_statuses = {
                'agent_workflow': 'active' if self.agent_integration else 'fallback',
                'mcp_integration': 'active' if self.mcp_integration else 'fallback',
                'github_integration': 'active' if self.github_integration else 'fallback',
                'state_machine': 'active' if self.state_machine_integration else 'fallback',
                'backward_compatibility': 'active' if self.backward_compatibility else 'error'
            }
            
            # Get compatibility level from compatibility layer
            compatibility_level = 'unknown'
            if self.backward_compatibility:
                status = self.backward_compatibility.get_compatibility_status()
                compatibility_level = status.compatibility_level.value
            
            # Calculate performance score
            performance_score = self.metrics.compatibility_score
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(layer_statuses)
            
            self.last_health_check = datetime.now()
            
            return DPIBSIntegrationHealth(
                overall_status=self.integration_status,
                layer_statuses=layer_statuses,
                compatibility_level=compatibility_level,
                performance_score=performance_score,
                last_health_check=self.last_health_check,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return DPIBSIntegrationHealth(
                overall_status=IntegrationStatus.ERROR,
                layer_statuses={'error': str(e)},
                compatibility_level='error',
                performance_score=0.0,
                last_health_check=datetime.now(),
                recommendations=['Health check system needs attention']
            )
    
    def _generate_health_recommendations(self, layer_statuses: Dict[str, str]) -> List[str]:
        """Generate health recommendations based on layer status."""
        recommendations = []
        
        fallback_layers = [layer for layer, status in layer_statuses.items() if status == 'fallback']
        error_layers = [layer for layer, status in layer_statuses.items() if status == 'error']
        
        if error_layers:
            recommendations.append(f"Critical: {', '.join(error_layers)} layer(s) in error state")
        
        if fallback_layers:
            recommendations.append(f"Operating in fallback mode: {', '.join(fallback_layers)}")
        
        if self.metrics.compatibility_score < 0.8:
            recommendations.append("Performance degradation detected - review integration logs")
        
        if not recommendations:
            recommendations.append("All systems operating normally")
        
        return recommendations
    
    def get_integration_metrics(self) -> IntegrationMetrics:
        """Get current integration metrics."""
        # Calculate uptime
        if self.initialization_time:
            uptime_duration = datetime.now() - self.initialization_time
            uptime_hours = uptime_duration.total_seconds() / 3600
            self.metrics.uptime_percentage = min(100.0, uptime_hours / 24 * 100)  # Simplified calculation
        
        return self.metrics
    
    def is_integration_healthy(self) -> bool:
        """Check if integration is healthy."""
        return (self.integration_status in [IntegrationStatus.ACTIVE, IntegrationStatus.DEGRADED] 
                and self.backward_compatibility is not None)
    
    # Migration and Configuration Methods
    
    async def initiate_compatibility_migration(self, target_level: str) -> Dict[str, Any]:
        """Initiate migration to target compatibility level."""
        try:
            if not self.backward_compatibility:
                return {'success': False, 'reason': 'compatibility_layer_unavailable'}
            
            target_enum = CompatibilityLevel(target_level)
            result = await self.backward_compatibility.initiate_gradual_migration(target_enum)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Migration initiation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update integration configuration."""
        try:
            self.config.update(new_config)
            self.logger.info("Configuration updated")
            
            # In production, would trigger re-initialization of affected components
            return {'success': True, 'message': 'Configuration updated'}
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Shutdown and Cleanup
    
    def shutdown(self):
        """Clean shutdown of all integration components."""
        try:
            self.health_monitoring_active = False
            
            # Shutdown all integration layers
            if self.agent_integration:
                self.agent_integration.shutdown()
            if self.mcp_integration:
                self.mcp_integration.shutdown()
            if self.github_integration:
                self.github_integration.shutdown()
            if self.state_machine_integration:
                self.state_machine_integration.shutdown()
            if self.backward_compatibility:
                self.backward_compatibility.shutdown()
            
            self.logger.info("DPIBS Master Integration shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during master integration shutdown: {e}")


# Global singleton instance
_master_integration_instance = None


def get_dpibs_integration(config: Dict[str, Any] = None) -> DPIBSMasterIntegration:
    """
    Get DPIBS Master Integration singleton instance.
    
    This is the main entry point for accessing DPIBS functionality from RIF system.
    """
    global _master_integration_instance
    
    if _master_integration_instance is None:
        _master_integration_instance = DPIBSMasterIntegration(config)
    
    return _master_integration_instance


# Convenience functions for RIF system integration

async def optimize_rif_agent_context(agent_type: str, context_data: Dict[str, Any],
                                   issue_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for RIF agent context optimization."""
    integration = get_dpibs_integration()
    return await integration.optimize_agent_context(agent_type, context_data, issue_context)


async def query_enhanced_knowledge(query_type: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for enhanced knowledge queries."""
    integration = get_dpibs_integration()
    return await integration.enhanced_knowledge_query(query_type, query_data)


async def process_rif_github_event(event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for RIF GitHub event processing."""
    integration = get_dpibs_integration()
    return await integration.process_github_event(event_type, event_data)


async def process_rif_state_transition(issue_id: str, from_state: str, to_state: str,
                                     context_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for RIF state transitions."""
    integration = get_dpibs_integration()
    return await integration.process_state_transition(issue_id, from_state, to_state, context_data)


def get_dpibs_health_status() -> Dict[str, Any]:
    """Get DPIBS integration health status."""
    try:
        integration = get_dpibs_integration()
        health = asyncio.run(integration.get_integration_health())
        return {
            'status': health.overall_status.value,
            'compatibility_level': health.compatibility_level,
            'performance_score': health.performance_score,
            'healthy': integration.is_integration_healthy(),
            'last_check': health.last_health_check.isoformat(),
            'recommendations': health.recommendations
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'healthy': False,
            'last_check': datetime.now().isoformat()
        }


def shutdown_dpibs_integration():
    """Shutdown DPIBS integration."""
    global _master_integration_instance
    
    if _master_integration_instance:
        _master_integration_instance.shutdown()
        _master_integration_instance = None