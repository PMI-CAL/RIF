"""
RIF Arbitration System - Issue #61
Comprehensive arbitration system for resolving agent consensus conflicts and disagreements.

This module provides the core infrastructure for:
- Disagreement detection between agent votes/decisions
- Multi-level escalation mechanisms (weighted → arbitrator → human)
- Decision recording and audit trail
- Integration with RIF workflow state machine

Components:
- ArbitrationSystem: Main orchestration class for conflict resolution
- ConflictDetector: Identifies and analyzes voting conflicts
- EscalationEngine: Manages multi-level resolution escalation
- DecisionRecorder: Maintains complete audit trail
- ArbitratorAgent: Specialized agent for complex conflict resolution
"""

from .arbitration_system import ArbitrationSystem, ArbitrationDecision, ArbitrationResult
from .conflict_detector import ConflictDetector, ConflictAnalysis, ConflictSeverity
from .escalation_engine import EscalationEngine, EscalationLevel, EscalationStrategy
from .decision_recorder import DecisionRecorder, ArbitrationRecord
from .arbitrator_agent import ArbitratorAgent, ArbitratorConfig

__version__ = "1.0.0"
__author__ = "RIF-Implementer"

__all__ = [
    "ArbitrationSystem",
    "ArbitrationDecision", 
    "ArbitrationResult",
    "ConflictDetector",
    "ConflictAnalysis",
    "ConflictSeverity", 
    "EscalationEngine",
    "EscalationLevel",
    "EscalationStrategy",
    "DecisionRecorder",
    "ArbitrationRecord",
    "ArbitratorAgent",
    "ArbitratorConfig"
]