"""
RIF Learning System - Phase 4 Integration
Provides comprehensive feedback loops, analytics, and knowledge refinement.
"""

from .feedback_loop import (
    FeedbackLoop,
    FeedbackEvent,
    PatternEffectiveness,
    get_feedback_loop,
    record_agent_feedback
)

from .analytics import (
    AnalyticsDashboard,
    generate_system_report,
    export_report_markdown
)

from .knowledge_refiner import (
    KnowledgeRefiner,
    get_knowledge_refiner,
    run_knowledge_refinement
)

__all__ = [
    # Feedback Loop
    'FeedbackLoop',
    'FeedbackEvent', 
    'PatternEffectiveness',
    'get_feedback_loop',
    'record_agent_feedback',
    
    # Analytics
    'AnalyticsDashboard',
    'generate_system_report',
    'export_report_markdown',
    
    # Knowledge Refinement
    'KnowledgeRefiner',
    'get_knowledge_refiner',
    'run_knowledge_refinement'
]

# Version info
__version__ = "1.0.0"
__phase__ = "4"
__description__ = "RIF Learning System with feedback loops and real-time optimization"