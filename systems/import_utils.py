#!/usr/bin/env python3
"""
Import utilities for RIF modules with hyphens in filenames
Fixes import issues with context-optimization-engine.py and live-system-context-engine.py
"""

import importlib.util
import sys
import os

def import_context_optimization_engine():
    """Import context optimization engine from hyphenated filename"""
    spec = importlib.util.spec_from_file_location(
        "context_optimization_engine", 
        "/Users/cal/DEV/RIF/systems/context-optimization-engine.py"
    )
    context_optimization_engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_optimization_engine)
    
    # Return commonly used classes
    return {
        'ContextOptimizer': context_optimization_engine.ContextOptimizer,
        'AgentType': context_optimization_engine.AgentType,
        'ContextItem': context_optimization_engine.ContextItem,
        'AgentContext': context_optimization_engine.AgentContext,
        'SystemContext': context_optimization_engine.SystemContext,
        'ContextType': context_optimization_engine.ContextType,
        'module': context_optimization_engine
    }

def import_live_system_context_engine():
    """Import live system context engine from hyphenated filename"""
    spec = importlib.util.spec_from_file_location(
        "live_system_context_engine", 
        "/Users/cal/DEV/RIF/systems/live-system-context-engine.py"
    )
    live_system_context_engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live_system_context_engine)
    
    return {
        'LiveSystemContextEngine': live_system_context_engine.LiveSystemContextEngine,
        'module': live_system_context_engine
    }

# Convenience functions for common import patterns
def get_context_optimizer_classes():
    """Get ContextOptimizer and related classes"""
    imports = import_context_optimization_engine()
    return imports['ContextOptimizer'], imports['AgentType'], imports['ContextItem'], imports['AgentContext']

def get_live_context_engine():
    """Get LiveSystemContextEngine class"""
    imports = import_live_system_context_engine()
    return imports['LiveSystemContextEngine']