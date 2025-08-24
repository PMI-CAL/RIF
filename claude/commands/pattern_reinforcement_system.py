"""
Pattern Reinforcement System for RIF Framework

This module implements a comprehensive pattern reinforcement system that learns
from pattern application outcomes and continuously improves the pattern library
quality through outcome-based scoring, failure analysis, and automated pruning.

Key Features:
- Outcome-based pattern scoring with success/failure tracking
- Advanced failure analysis and learning extraction
- Automated pattern pruning with quality thresholds
- Time-based pattern decay for relevance maintenance
- Integration with existing RIF knowledge system

Implementation follows the design from Issue #78 planning phase.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Import knowledge interface
try:
    from knowledge.interface import get_knowledge_system, KnowledgeInterface
    KNOWLEDGE_INTERFACE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_INTERFACE_AVAILABLE = False


class OutcomeType(Enum):
    """Types of pattern application outcomes."""
    SUCCESS = "success"
    FAILURE = "failure" 
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    ERROR = "error"


class FailureMode(Enum):
    """Categories of pattern application failures."""
    IMPLEMENTATION_ERROR = "implementation_error"
    CONTEXT_MISMATCH = "context_mismatch"
    DEPENDENCY_FAILURE = "dependency_failure"
    COMPLEXITY_UNDERESTIMATE = "complexity_underestimate"
    INTEGRATION_ISSUE = "integration_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    QUALITY_GATE_FAILURE = "quality_gate_failure"
    UNKNOWN = "unknown"


@dataclass
class PatternOutcome:
    """
    Represents the outcome of applying a pattern.
    
    This data structure captures all relevant information about how
    a pattern performed when applied to a specific task or issue.
    """
    pattern_id: str
    outcome_type: OutcomeType
    success: bool
    issue_id: Optional[str] = None
    agent_type: Optional[str] = None
    complexity_level: Optional[str] = None
    execution_time: Optional[float] = None
    quality_score: Optional[float] = None
    
    # Failure-specific information
    failure_mode: Optional[FailureMode] = None
    error_details: Optional[str] = None
    context_info: Optional[Dict[str, Any]] = None
    
    # Success-specific information
    effectiveness_score: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    outcome_id: str = field(default_factory=lambda: f"outcome_{int(time.time() * 1000)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['outcome_type'] = self.outcome_type.value
        if self.failure_mode:
            data['failure_mode'] = self.failure_mode.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternOutcome':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['outcome_type'] = OutcomeType(data['outcome_type'])
        if data.get('failure_mode'):
            data['failure_mode'] = FailureMode(data['failure_mode'])
        return cls(**data)


@dataclass
class PatternMetrics:
    """
    Comprehensive metrics for a pattern's performance over time.
    
    Tracks success/failure statistics, performance trends, and
    quality indicators for pattern effectiveness assessment.
    """
    pattern_id: str
    
    # Usage statistics
    total_applications: int = 0
    success_count: int = 0
    failure_count: int = 0
    partial_success_count: int = 0
    
    # Performance metrics
    current_score: float = 1.0  # Base score
    success_rate: float = 0.0
    average_execution_time: Optional[float] = None
    average_quality_score: Optional[float] = None
    
    # Time-based metrics
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    creation_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Decay and aging
    decay_factor: float = 1.0  # Multiplier for time-based decay
    relevance_score: float = 1.0  # Current relevance based on usage patterns
    
    # Failure analysis
    failure_modes: Dict[str, int] = field(default_factory=dict)  # failure_mode -> count
    common_contexts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality indicators
    effectiveness_trend: List[float] = field(default_factory=list)  # Last 10 effectiveness scores
    pruning_candidate: bool = False
    
    def calculate_success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_applications == 0:
            return 0.0
        return self.success_count / self.total_applications
    
    def calculate_weighted_score(self) -> float:
        """Calculate weighted score considering multiple factors."""
        base_score = self.current_score
        
        # Apply success rate weighting
        success_rate = self.calculate_success_rate()
        success_weight = 0.4
        
        # Apply time decay weighting
        time_weight = 0.2
        
        # Apply effectiveness trend weighting
        trend_weight = 0.3
        trend_factor = 1.0
        if self.effectiveness_trend:
            recent_trend = self.effectiveness_trend[-5:]  # Last 5 applications
            if len(recent_trend) >= 2:
                trend_factor = sum(recent_trend) / len(recent_trend)
        
        # Apply usage frequency weighting
        frequency_weight = 0.1
        frequency_factor = min(1.0, self.total_applications / 10)  # Normalize to max 10 applications
        
        weighted_score = (
            base_score * 
            (1 + success_rate * success_weight) * 
            (1 + trend_factor * trend_weight) *
            (1 + frequency_factor * frequency_weight) *
            max(0.8, self.decay_factor)  # Don't let decay reduce score too much in weighting
        )
        
        return max(0.0, min(2.0, weighted_score))  # Clamp between 0 and 2
    
    def is_pruning_candidate(self) -> bool:
        """Determine if pattern should be considered for pruning."""
        # Low success rate with sufficient data
        if self.total_applications >= 10 and self.calculate_success_rate() < 0.3:
            return True
        
        # No recent usage (over 90 days)
        if self.last_used and (datetime.now() - self.last_used).days > 90:
            return True
        
        # Consistently declining effectiveness
        if len(self.effectiveness_trend) >= 5:
            recent_trend = self.effectiveness_trend[-5:]
            if all(recent_trend[i] <= recent_trend[i-1] for i in range(1, len(recent_trend))):
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['creation_time'] = self.creation_time.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        if self.last_success:
            data['last_success'] = self.last_success.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternMetrics':
        """Create from dictionary."""
        data = data.copy()
        data['creation_time'] = datetime.fromisoformat(data['creation_time'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        if data.get('last_used'):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        if data.get('last_success'):
            data['last_success'] = datetime.fromisoformat(data['last_success'])
        return cls(**data)


class PatternReinforcementSystem:
    """
    Core pattern reinforcement system for learning from pattern application outcomes.
    
    This system implements a comprehensive pattern learning and quality management
    approach that:
    1. Updates pattern scores based on application outcomes
    2. Analyzes failures to extract learnings and improvements
    3. Prunes ineffective patterns to maintain library quality
    4. Applies time-based decay to keep patterns relevant
    
    The system integrates with the existing RIF knowledge system for storage
    and retrieval of pattern data and metrics.
    """
    
    def __init__(self, 
                 knowledge_system: Optional[KnowledgeInterface] = None,
                 performance_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern reinforcement system.
        
        Args:
            knowledge_system: Knowledge system interface for pattern storage
            performance_config: Configuration for performance optimization
        """
        self.logger = logging.getLogger(f"{__name__}.PatternReinforcementSystem")
        
        # Initialize knowledge system
        if knowledge_system:
            self.knowledge = knowledge_system
        elif KNOWLEDGE_INTERFACE_AVAILABLE:
            try:
                self.knowledge = get_knowledge_system()
            except Exception as e:
                self.logger.error(f"Failed to initialize knowledge system: {e}")
                self.knowledge = None
        else:
            self.knowledge = None
            self.logger.warning("Knowledge interface not available - using in-memory storage")
        
        # Configuration
        self.config = {
            'success_boost_factor': 1.1,      # 10% boost for successful patterns
            'failure_penalty_factor': 0.9,    # 10% penalty for failed patterns
            'time_decay_rate': 0.05,          # 5% decay per month of non-use
            'pruning_threshold': 0.3,         # Success rate threshold for pruning
            'min_applications_for_pruning': 10, # Minimum applications before pruning
            'max_execution_time': 30.0,       # Maximum time for reinforcement processing (seconds)
            'batch_size': 50,                 # Batch size for bulk operations
            'enable_async_processing': True,   # Enable asynchronous processing
        }
        
        if performance_config:
            self.config.update(performance_config)
        
        # In-memory storage for when knowledge system is unavailable
        self.pattern_metrics_cache: Dict[str, PatternMetrics] = {}
        self.outcomes_cache: List[PatternOutcome] = []
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_lock = threading.RLock()
        
        # Performance monitoring
        self.performance_metrics = {
            'outcomes_processed': 0,
            'patterns_updated': 0,
            'patterns_pruned': 0,
            'average_processing_time': 0.0,
            'last_maintenance_run': None,
            'errors_encountered': 0
        }
        
        self.logger.info("Pattern reinforcement system initialized successfully")
    
    def process_pattern_outcome(self, outcome: PatternOutcome) -> bool:
        """
        Process a pattern application outcome and update scores.
        
        This is the main entry point for the reinforcement system.
        When a pattern is applied (successfully or unsuccessfully),
        this method should be called with the outcome details.
        
        Args:
            outcome: PatternOutcome containing application results
            
        Returns:
            True if processing succeeded, False otherwise
        """
        start_time = time.time()
        
        try:
            with self.processing_lock:
                self.logger.debug(f"Processing outcome for pattern {outcome.pattern_id}: {outcome.outcome_type.value}")
                
                # Get or create pattern metrics
                metrics = self._get_pattern_metrics(outcome.pattern_id)
                if not metrics:
                    metrics = PatternMetrics(pattern_id=outcome.pattern_id)
                    self.logger.info(f"Created new metrics for pattern {outcome.pattern_id}")
                
                # Update metrics based on outcome
                self._update_pattern_metrics(metrics, outcome)
                
                # Perform failure analysis if needed
                if not outcome.success:
                    self._analyze_failure(outcome, metrics)
                
                # Update pattern score
                old_score = metrics.current_score
                self._update_pattern_score(metrics, outcome)
                
                # Store updated metrics
                if not self._store_pattern_metrics(metrics):
                    self.logger.error(f"Failed to store updated metrics for pattern {outcome.pattern_id}")
                    return False
                
                # Store outcome for learning
                if not self._store_outcome(outcome):
                    self.logger.error(f"Failed to store outcome {outcome.outcome_id}")
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self._update_performance_metrics(processing_time)
                
                self.logger.info(f"Pattern {outcome.pattern_id} score updated: {old_score:.3f} -> {metrics.current_score:.3f} "
                               f"(success_rate: {metrics.success_rate:.2%}, processing_time: {processing_time:.3f}s)")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error processing outcome for pattern {outcome.pattern_id}: {e}")
            self.performance_metrics['errors_encountered'] += 1
            return False
    
    def _get_pattern_metrics(self, pattern_id: str) -> Optional[PatternMetrics]:
        """Get pattern metrics from storage or cache."""
        # Check cache first
        if pattern_id in self.pattern_metrics_cache:
            return self.pattern_metrics_cache[pattern_id]
        
        # Try knowledge system
        if self.knowledge:
            try:
                results = self.knowledge.retrieve_knowledge(
                    query=f"pattern_id:{pattern_id}",
                    collection="pattern_metrics",
                    n_results=1,
                    filters={"pattern_id": pattern_id}
                )
                
                if results:
                    metrics_data = json.loads(results[0]['content'])
                    metrics = PatternMetrics.from_dict(metrics_data)
                    self.pattern_metrics_cache[pattern_id] = metrics
                    return metrics
                    
            except Exception as e:
                self.logger.warning(f"Error retrieving metrics for pattern {pattern_id}: {e}")
        
        return None
    
    def _update_pattern_metrics(self, metrics: PatternMetrics, outcome: PatternOutcome):
        """Update pattern metrics with new outcome data."""
        # Update usage statistics
        metrics.total_applications += 1
        metrics.last_used = outcome.timestamp
        metrics.last_updated = datetime.now()
        
        if outcome.success:
            metrics.success_count += 1
            metrics.last_success = outcome.timestamp
            
            # Update effectiveness trend
            if outcome.effectiveness_score:
                metrics.effectiveness_trend.append(outcome.effectiveness_score)
                # Keep only last 10 scores
                if len(metrics.effectiveness_trend) > 10:
                    metrics.effectiveness_trend = metrics.effectiveness_trend[-10:]
        else:
            metrics.failure_count += 1
            
            # Track failure modes
            if outcome.failure_mode:
                failure_key = outcome.failure_mode.value
                metrics.failure_modes[failure_key] = metrics.failure_modes.get(failure_key, 0) + 1
        
        # Update partial successes
        if outcome.outcome_type == OutcomeType.PARTIAL_SUCCESS:
            metrics.partial_success_count += 1
        
        # Update success rate
        metrics.success_rate = metrics.calculate_success_rate()
        
        # Update average execution time
        if outcome.execution_time:
            if metrics.average_execution_time:
                # Rolling average
                metrics.average_execution_time = (
                    (metrics.average_execution_time * (metrics.total_applications - 1) + outcome.execution_time) /
                    metrics.total_applications
                )
            else:
                metrics.average_execution_time = outcome.execution_time
        
        # Update average quality score
        if outcome.quality_score:
            if metrics.average_quality_score:
                # Rolling average
                metrics.average_quality_score = (
                    (metrics.average_quality_score * (metrics.total_applications - 1) + outcome.quality_score) /
                    metrics.total_applications
                )
            else:
                metrics.average_quality_score = outcome.quality_score
        
        # Check if pattern is pruning candidate
        metrics.pruning_candidate = metrics.is_pruning_candidate()
    
    def _update_pattern_score(self, metrics: PatternMetrics, outcome: PatternOutcome):
        """Update the pattern score based on outcome."""
        if outcome.success:
            # Boost score for successful applications
            boost_factor = self.config['success_boost_factor']
            
            # Additional boost for high effectiveness
            if outcome.effectiveness_score and outcome.effectiveness_score > 0.8:
                boost_factor *= 1.05  # Extra 5% boost for highly effective applications
            
            metrics.current_score *= boost_factor
            
        else:
            # Penalize score for failed applications
            penalty_factor = self.config['failure_penalty_factor']
            
            # Larger penalty for certain failure types
            if outcome.failure_mode in [FailureMode.IMPLEMENTATION_ERROR, FailureMode.QUALITY_GATE_FAILURE]:
                penalty_factor *= 0.95  # Extra 5% penalty for serious failures
            
            metrics.current_score *= penalty_factor
        
        # Apply trend-based adjustments
        if len(metrics.effectiveness_trend) >= 3:
            recent_trend = metrics.effectiveness_trend[-3:]
            if all(recent_trend[i] > recent_trend[i-1] for i in range(1, len(recent_trend))):
                # Improving trend - small boost
                metrics.current_score *= 1.02
            elif all(recent_trend[i] < recent_trend[i-1] for i in range(1, len(recent_trend))):
                # Declining trend - small penalty
                metrics.current_score *= 0.98
        
        # Clamp score to reasonable bounds
        metrics.current_score = max(0.1, min(2.0, metrics.current_score))
    
    def _analyze_failure(self, outcome: PatternOutcome, metrics: PatternMetrics):
        """Perform advanced failure analysis to extract learnings."""
        try:
            failure_analysis = {
                'pattern_id': outcome.pattern_id,
                'failure_mode': outcome.failure_mode.value if outcome.failure_mode else 'unknown',
                'error_details': outcome.error_details,
                'context_info': outcome.context_info,
                'timestamp': outcome.timestamp.isoformat(),
                'analysis_results': {}
            }
            
            # Context-based analysis
            if outcome.context_info:
                context = outcome.context_info
                
                # Analyze complexity mismatch
                if outcome.complexity_level:
                    if outcome.failure_mode == FailureMode.COMPLEXITY_UNDERESTIMATE:
                        failure_analysis['analysis_results']['complexity_insight'] = {
                            'expected': outcome.complexity_level,
                            'recommendation': 'Consider upgrading pattern complexity classification',
                            'suggested_adjustment': 'increase_complexity_score'
                        }
                
                # Analyze context patterns
                context_patterns = self._extract_context_patterns(context)
                failure_analysis['analysis_results']['context_patterns'] = context_patterns
            
            # Failure frequency analysis
            failure_mode_key = outcome.failure_mode.value if outcome.failure_mode else 'unknown'
            failure_count = metrics.failure_modes.get(failure_mode_key, 0)
            if failure_count >= 3:
                failure_analysis['analysis_results']['recurring_failure'] = {
                    'mode': failure_mode_key,
                    'frequency': failure_count,
                    'recommendation': 'Pattern may need significant revision or retirement',
                    'severity': 'high' if failure_count >= 5 else 'medium'
                }
            
            # Store failure analysis
            self._store_failure_analysis(failure_analysis)
            
            self.logger.debug(f"Failure analysis completed for pattern {outcome.pattern_id}")
            
        except Exception as e:
            self.logger.error(f"Error during failure analysis: {e}")
    
    def _extract_context_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patterns from failure context."""
        patterns = {}
        
        # Technology context
        if 'technology' in context:
            patterns['technology'] = context['technology']
        
        # Project size/complexity indicators
        if 'files_modified' in context:
            file_count = context['files_modified']
            if isinstance(file_count, int):
                if file_count > 20:
                    patterns['scale'] = 'large'
                elif file_count > 5:
                    patterns['scale'] = 'medium'
                else:
                    patterns['scale'] = 'small'
        
        # Time-based patterns
        if 'execution_time' in context:
            exec_time = context['execution_time']
            if isinstance(exec_time, (int, float)) and exec_time > 300:  # 5 minutes
                patterns['duration'] = 'long_running'
        
        return patterns
    
    def _store_pattern_metrics(self, metrics: PatternMetrics) -> bool:
        """Store pattern metrics to knowledge system or cache."""
        try:
            # Update cache
            self.pattern_metrics_cache[metrics.pattern_id] = metrics
            
            # Store to knowledge system if available
            if self.knowledge:
                metrics_data = metrics.to_dict()
                doc_id = f"pattern_metrics_{metrics.pattern_id}"
                
                result = self.knowledge.store_knowledge(
                    collection="pattern_metrics",
                    content=json.dumps(metrics_data, indent=2),
                    metadata={
                        "pattern_id": metrics.pattern_id,
                        "type": "pattern_metrics",
                        "last_updated": metrics.last_updated.isoformat(),
                        "success_rate": metrics.success_rate,
                        "pruning_candidate": metrics.pruning_candidate
                    },
                    doc_id=doc_id
                )
                
                return result is not None
            
            return True  # Cache storage always succeeds
            
        except Exception as e:
            self.logger.error(f"Error storing pattern metrics: {e}")
            return False
    
    def _store_outcome(self, outcome: PatternOutcome) -> bool:
        """Store pattern outcome for learning."""
        try:
            # Add to cache
            self.outcomes_cache.append(outcome)
            
            # Keep cache size manageable
            if len(self.outcomes_cache) > 1000:
                self.outcomes_cache = self.outcomes_cache[-500:]  # Keep last 500
            
            # Store to knowledge system if available
            if self.knowledge:
                outcome_data = outcome.to_dict()
                
                result = self.knowledge.store_knowledge(
                    collection="pattern_outcomes",
                    content=json.dumps(outcome_data, indent=2),
                    metadata={
                        "pattern_id": outcome.pattern_id,
                        "outcome_type": outcome.outcome_type.value,
                        "success": outcome.success,
                        "timestamp": outcome.timestamp.isoformat(),
                        "type": "pattern_outcome"
                    },
                    doc_id=outcome.outcome_id
                )
                
                return result is not None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing outcome: {e}")
            return False
    
    def _store_failure_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Store failure analysis results."""
        try:
            if self.knowledge:
                doc_id = f"failure_analysis_{analysis['pattern_id']}_{int(time.time())}"
                
                result = self.knowledge.store_knowledge(
                    collection="failure_analyses",
                    content=json.dumps(analysis, indent=2),
                    metadata={
                        "pattern_id": analysis['pattern_id'],
                        "failure_mode": analysis['failure_mode'],
                        "type": "failure_analysis",
                        "timestamp": analysis['timestamp']
                    },
                    doc_id=doc_id
                )
                
                return result is not None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing failure analysis: {e}")
            return False
    
    def _update_performance_metrics(self, processing_time: float):
        """Update system performance metrics."""
        self.performance_metrics['outcomes_processed'] += 1
        
        # Update rolling average processing time
        current_avg = self.performance_metrics['average_processing_time']
        processed = self.performance_metrics['outcomes_processed']
        
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (processed - 1) + processing_time) / processed
        )
    
    def get_pattern_score(self, pattern_id: str) -> Optional[float]:
        """Get current score for a pattern."""
        metrics = self._get_pattern_metrics(pattern_id)
        return metrics.calculate_weighted_score() if metrics else None
    
    def get_pattern_metrics(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive metrics for a pattern."""
        metrics = self._get_pattern_metrics(pattern_id)
        if metrics:
            return metrics.to_dict()
        return None
    
    def get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top-performing patterns by weighted score."""
        try:
            all_patterns = []
            
            # Get from knowledge system if available
            if self.knowledge:
                results = self.knowledge.retrieve_knowledge(
                    query="*",
                    collection="pattern_metrics",
                    n_results=limit * 2
                )
                
                for result in results:
                    try:
                        metrics_data = json.loads(result['content'])
                        metrics = PatternMetrics.from_dict(metrics_data)
                        all_patterns.append({
                            'pattern_id': metrics.pattern_id,
                            'score': metrics.calculate_weighted_score(),
                            'success_rate': metrics.success_rate,
                            'applications': metrics.total_applications,
                            'last_used': metrics.last_used.isoformat() if metrics.last_used else None
                        })
                    except Exception as e:
                        self.logger.warning(f"Error parsing pattern metrics: {e}")
            else:
                # Use cache
                for pattern_id, metrics in self.pattern_metrics_cache.items():
                    all_patterns.append({
                        'pattern_id': pattern_id,
                        'score': metrics.calculate_weighted_score(),
                        'success_rate': metrics.success_rate,
                        'applications': metrics.total_applications,
                        'last_used': metrics.last_used.isoformat() if metrics.last_used else None
                    })
            
            # Sort by score and return top patterns
            all_patterns.sort(key=lambda x: x['score'], reverse=True)
            return all_patterns[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting top patterns: {e}")
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            **self.performance_metrics,
            'cache_size': len(self.pattern_metrics_cache),
            'outcomes_cache_size': len(self.outcomes_cache),
            'knowledge_system_available': self.knowledge is not None
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.logger.info("Pattern reinforcement system cleanup completed")