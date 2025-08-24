"""
RIF Feedback Loop System for LightRAG
Implements real-time learning and pattern effectiveness tracking.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from threading import Thread, Lock
import queue

# Add parent directory to path for LightRAG core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lightrag_core import LightRAGCore, get_lightrag_instance


@dataclass
class FeedbackEvent:
    """Represents a feedback event in the system."""
    timestamp: str
    agent_name: str
    action: str
    success: bool
    pattern_id: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternEffectiveness:
    """Tracks effectiveness of a specific pattern."""
    pattern_id: str
    usage_count: int
    success_count: int
    failure_count: int
    avg_performance: float
    last_used: str
    context_tags: List[str]
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackLoop:
    """
    Main feedback loop system for continuous learning and improvement.
    """
    
    def __init__(self, feedback_window_hours: int = 24):
        """
        Initialize feedback loop system.
        
        Args:
            feedback_window_hours: Hours to keep feedback events for analysis
        """
        self.rag = get_lightrag_instance()
        self.logger = logging.getLogger("rif.feedback")
        self.feedback_window = timedelta(hours=feedback_window_hours)
        
        # Thread-safe collections
        self.event_queue = queue.Queue()
        self.pattern_stats = {}
        self.recent_events = deque(maxlen=1000)
        self.lock = Lock()
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.agent_metrics = defaultdict(dict)
        
        # Learning thresholds
        self.min_usage_for_analysis = 5
        self.low_performance_threshold = 0.6
        self.high_performance_threshold = 0.85
        
        # Start background processing
        self.processing_thread = None
        self.is_running = False
        self._start_background_processing()
    
    def _start_background_processing(self):
        """Start background thread for processing feedback events."""
        self.is_running = True
        self.processing_thread = Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        self.logger.info("Feedback loop background processing started")
    
    def _process_events(self):
        """Background thread for processing feedback events."""
        while self.is_running:
            try:
                # Process events from queue
                while not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    self._process_single_event(event)
                
                # Periodic analysis
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self._analyze_patterns()
                    self._update_knowledge_base()
                
                time.sleep(1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                self.logger.error(f"Error in feedback processing: {str(e)}")
                time.sleep(5)  # Longer delay on error
    
    def record_event(self, 
                    agent_name: str,
                    action: str,
                    success: bool,
                    pattern_id: Optional[str] = None,
                    performance_metrics: Optional[Dict[str, float]] = None,
                    context: Optional[Dict[str, Any]] = None):
        """
        Record a feedback event for later processing.
        
        Args:
            agent_name: Name of the agent generating the event
            action: Action performed
            success: Whether the action was successful
            pattern_id: ID of pattern used (if any)
            performance_metrics: Performance metrics
            context: Additional context
        """
        event = FeedbackEvent(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            action=action,
            success=success,
            pattern_id=pattern_id,
            performance_metrics=performance_metrics or {},
            context=context or {}
        )
        
        self.event_queue.put(event)
        self.logger.debug(f"Recorded event: {agent_name} - {action} - {'success' if success else 'failure'}")
    
    def _process_single_event(self, event: FeedbackEvent):
        """Process a single feedback event."""
        with self.lock:
            # Add to recent events
            self.recent_events.append(event)
            
            # Update pattern statistics
            if event.pattern_id:
                if event.pattern_id not in self.pattern_stats:
                    self.pattern_stats[event.pattern_id] = PatternEffectiveness(
                        pattern_id=event.pattern_id,
                        usage_count=0,
                        success_count=0,
                        failure_count=0,
                        avg_performance=0.0,
                        last_used=event.timestamp,
                        context_tags=[]
                    )
                
                stats = self.pattern_stats[event.pattern_id]
                stats.usage_count += 1
                stats.last_used = event.timestamp
                
                if event.success:
                    stats.success_count += 1
                else:
                    stats.failure_count += 1
                
                # Update performance metrics
                if event.performance_metrics:
                    performance_score = event.performance_metrics.get('score', 0.0)
                    stats.avg_performance = (
                        (stats.avg_performance * (stats.usage_count - 1) + performance_score) 
                        / stats.usage_count
                    )
            
            # Update agent metrics
            self.agent_metrics[event.agent_name][event.action] = {
                'last_success': event.success,
                'timestamp': event.timestamp,
                'performance': event.performance_metrics
            }
    
    def _analyze_patterns(self):
        """Analyze pattern effectiveness and identify improvements."""
        with self.lock:
            current_time = datetime.now()
            
            for pattern_id, stats in self.pattern_stats.items():
                if stats.usage_count < self.min_usage_for_analysis:
                    continue
                
                # Check if pattern needs attention
                if stats.success_rate < self.low_performance_threshold:
                    self._handle_low_performance_pattern(pattern_id, stats)
                elif stats.success_rate > self.high_performance_threshold:
                    self._promote_high_performance_pattern(pattern_id, stats)
                
                # Check for unused patterns
                last_used = datetime.fromisoformat(stats.last_used)
                if current_time - last_used > self.feedback_window:
                    self._handle_unused_pattern(pattern_id, stats)
    
    def _handle_low_performance_pattern(self, pattern_id: str, stats: PatternEffectiveness):
        """Handle patterns with low success rates."""
        self.logger.warning(f"Low performance pattern detected: {pattern_id} "
                          f"(success rate: {stats.success_rate:.2f})")
        
        # Store learning insight
        insight = {
            "type": "pattern_performance",
            "pattern_id": pattern_id,
            "issue": "low_success_rate",
            "success_rate": stats.success_rate,
            "usage_count": stats.usage_count,
            "recommendation": "Review pattern implementation or retire pattern",
            "timestamp": datetime.now().isoformat()
        }
        
        self._store_learning_insight(insight)
    
    def _promote_high_performance_pattern(self, pattern_id: str, stats: PatternEffectiveness):
        """Promote highly successful patterns."""
        self.logger.info(f"High performance pattern identified: {pattern_id} "
                        f"(success rate: {stats.success_rate:.2f})")
        
        # Store learning insight
        insight = {
            "type": "pattern_performance",
            "pattern_id": pattern_id,
            "achievement": "high_success_rate",
            "success_rate": stats.success_rate,
            "usage_count": stats.usage_count,
            "recommendation": "Promote pattern for similar contexts",
            "timestamp": datetime.now().isoformat()
        }
        
        self._store_learning_insight(insight)
    
    def _handle_unused_pattern(self, pattern_id: str, stats: PatternEffectiveness):
        """Handle patterns that haven't been used recently."""
        self.logger.debug(f"Unused pattern detected: {pattern_id}")
        
        # Mark for potential cleanup
        insight = {
            "type": "pattern_usage",
            "pattern_id": pattern_id,
            "issue": "unused_pattern",
            "last_used": stats.last_used,
            "recommendation": "Consider archiving or removing pattern",
            "timestamp": datetime.now().isoformat()
        }
        
        self._store_learning_insight(insight)
    
    def _store_learning_insight(self, insight: Dict[str, Any]):
        """Store a learning insight in the knowledge base."""
        try:
            content = f"Learning Insight: {insight['type']}\n"
            content += f"Generated: {insight['timestamp']}\n"
            content += f"Details: {json.dumps(insight, indent=2)}"
            
            doc_id = self.rag.insert_document(
                content=content,
                metadata={
                    "type": "learning_insight",
                    "insight_type": insight["type"],
                    "timestamp": insight["timestamp"],
                    "agent": "feedback_loop"
                }
            )
            
            self.logger.debug(f"Stored learning insight: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store learning insight: {str(e)}")
    
    def _update_knowledge_base(self):
        """Update knowledge base with latest patterns and insights."""
        try:
            # Create summary of current state
            summary = {
                "total_patterns": len(self.pattern_stats),
                "recent_events": len(self.recent_events),
                "active_agents": list(self.agent_metrics.keys()),
                "timestamp": datetime.now().isoformat(),
                "top_patterns": self._get_top_patterns(),
                "performance_summary": self._get_performance_summary()
            }
            
            content = f"Feedback Loop Status Update\n"
            content += f"Generated: {summary['timestamp']}\n"
            content += f"Summary: {json.dumps(summary, indent=2)}"
            
            self.rag.insert_document(
                content=content,
                metadata={
                    "type": "feedback_status",
                    "timestamp": summary["timestamp"],
                    "agent": "feedback_loop"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge base: {str(e)}")
    
    def _get_top_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing patterns."""
        with self.lock:
            patterns = list(self.pattern_stats.values())
            patterns.sort(key=lambda p: p.success_rate, reverse=True)
            return [p.to_dict() for p in patterns[:limit]]
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all agents."""
        with self.lock:
            summary = {}
            for agent_name, metrics in self.agent_metrics.items():
                summary[agent_name] = {
                    "total_actions": len(metrics),
                    "recent_activity": len([
                        m for m in metrics.values()
                        if datetime.fromisoformat(m['timestamp']) > 
                        datetime.now() - timedelta(hours=1)
                    ])
                }
            return summary
    
    def get_pattern_effectiveness(self, pattern_id: str) -> Optional[PatternEffectiveness]:
        """Get effectiveness data for a specific pattern."""
        with self.lock:
            return self.pattern_stats.get(pattern_id)
    
    def get_agent_performance(self, agent_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance data for a specific agent."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_events = [
                e for e in self.recent_events
                if e.agent_name == agent_name and 
                datetime.fromisoformat(e.timestamp) > cutoff
            ]
            
            if not recent_events:
                return {"events": 0, "success_rate": 0.0, "actions": [], "latest_activity": None}
            
            success_count = sum(1 for e in recent_events if e.success)
            
            return {
                "events": len(recent_events),
                "success_rate": success_count / len(recent_events),
                "actions": list(set(e.action for e in recent_events)),
                "latest_activity": max(e.timestamp for e in recent_events) if recent_events else None
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        with self.lock:
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_events = [
                e for e in self.recent_events
                if datetime.fromisoformat(e.timestamp) > recent_cutoff
            ]
            
            return {
                "total_patterns": len(self.pattern_stats),
                "recent_events": len(recent_events),
                "active_agents": len(self.agent_metrics),
                "queue_size": self.event_queue.qsize(),
                "system_uptime": self.is_running,
                "avg_success_rate": (
                    sum(1 for e in recent_events if e.success) / len(recent_events)
                    if recent_events else 0.0
                )
            }
    
    def shutdown(self):
        """Shutdown the feedback loop system."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("Feedback loop system shutdown")


# Global instance
_feedback_loop_instance = None
_instance_lock = Lock()

def get_feedback_loop() -> FeedbackLoop:
    """Get global feedback loop instance."""
    global _feedback_loop_instance
    
    with _instance_lock:
        if _feedback_loop_instance is None:
            _feedback_loop_instance = FeedbackLoop()
        return _feedback_loop_instance


def record_agent_feedback(agent_name: str, 
                         action: str, 
                         success: bool, 
                         **kwargs):
    """
    Convenience function for agents to record feedback.
    
    Args:
        agent_name: Name of the agent
        action: Action performed
        success: Whether action was successful
        **kwargs: Additional parameters (pattern_id, performance_metrics, context)
    """
    feedback_loop = get_feedback_loop()
    feedback_loop.record_event(agent_name, action, success, **kwargs)