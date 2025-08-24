"""
Hybrid Knowledge Pipeline Integration Layer

This module provides the master coordination layer for the RIF hybrid knowledge pipeline,
integrating all four core components (entity extraction, relationship detection, 
vector embeddings, and query planning) into a unified system.

Core Components:
- HybridKnowledgeSystem: Main coordination controller
- KnowledgeAPI: Unified API gateway for all operations  
- SystemMonitor: Health monitoring and resource management
- IntegrationController: Component coordination and orchestration

Issues #30-33 Integration:
- Issue #30: Entity extraction foundation
- Issue #31: Relationship detection system
- Issue #32: Vector embedding generation
- Issue #33: Query planning and hybrid search
"""

from .hybrid_knowledge_system import HybridKnowledgeSystem
from .knowledge_api import KnowledgeAPI
from .system_monitor import SystemMonitor
from .integration_controller import IntegrationController

__all__ = [
    'HybridKnowledgeSystem',
    'KnowledgeAPI', 
    'SystemMonitor',
    'IntegrationController'
]

# Version information
__version__ = '1.0.0'
__build__ = 'issue-40-master-coordination'

# Default configuration
DEFAULT_CONFIG = {
    'memory_limit_mb': 2048,
    'cpu_cores': 4,
    'query_cache_size': 1000,
    'performance_mode': 'BALANCED',
    'enable_monitoring': True,
    'enable_metrics': True
}

def create_hybrid_knowledge_system(**config):
    """
    Factory function to create a fully configured HybridKnowledgeSystem.
    
    Args:
        **config: Configuration overrides for the system
        
    Returns:
        HybridKnowledgeSystem: Configured and ready-to-use knowledge system
    """
    merged_config = {**DEFAULT_CONFIG, **config}
    return HybridKnowledgeSystem(merged_config)