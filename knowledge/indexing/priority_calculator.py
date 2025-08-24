"""
Priority Calculator for Auto-Reindexing Scheduler
Issue #69: Build auto-reindexing scheduler

This module provides dynamic priority calculation for reindexing jobs:
- Multi-factor priority analysis (urgency, impact, staleness, dependencies)
- Trigger-based priority adjustments
- Adaptive learning from historical performance
- Context-aware priority boosting

Author: RIF-Implementer
Date: 2025-08-23
"""

import logging
import statistics
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json


class ReindexPriority(IntEnum):
    """Priority levels for reindexing operations (lower = higher priority)"""
    CRITICAL = 0    # Data integrity issues, must run immediately
    HIGH = 1        # Content updates, performance critical
    MEDIUM = 2      # Relationship updates, optimization 
    LOW = 3         # Background optimization, maintenance


@dataclass
class PriorityFactors:
    """Individual factors that contribute to priority calculation"""
    urgency_factor: float = 0.5         # How urgent (0.0-1.0)
    impact_factor: float = 0.5          # How impactful (0.0-1.0)
    staleness_factor: float = 0.0       # How stale the data is (0.0-1.0)
    dependency_factor: float = 0.0      # Dependency on other entities (0.0-1.0)
    system_load_factor: float = 0.0     # Current system load consideration (0.0-1.0)
    historical_factor: float = 0.0      # Based on historical performance (0.0-1.0)
    
    def calculate_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted priority score"""
        factors = {
            'urgency': self.urgency_factor,
            'impact': self.impact_factor,
            'staleness': self.staleness_factor,
            'dependency': self.dependency_factor,
            'system_load': self.system_load_factor,
            'historical': self.historical_factor
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor_name, factor_value in factors.items():
            weight = weights.get(factor_name, 1.0)
            weighted_sum += factor_value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class TriggerAnalyzer:
    """
    Analyzes different types of reindexing triggers to determine base priority.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"trigger_analyzer_{id(self)}")
        
        # Base priority mappings for different triggers
        self.trigger_priorities = {
            "validation_issue": {
                "critical": ReindexPriority.CRITICAL,
                "error": ReindexPriority.HIGH,
                "warning": ReindexPriority.MEDIUM,
                "info": ReindexPriority.LOW
            },
            "file_change": {
                "source_code": ReindexPriority.HIGH,
                "configuration": ReindexPriority.HIGH,
                "documentation": ReindexPriority.MEDIUM,
                "test_file": ReindexPriority.MEDIUM,
                "other": ReindexPriority.LOW
            },
            "manual_request": ReindexPriority.MEDIUM,
            "scheduled_maintenance": ReindexPriority.LOW,
            "dependency_update": ReindexPriority.MEDIUM,
            "system_startup": ReindexPriority.HIGH,
            "performance_optimization": ReindexPriority.LOW
        }
        
        # Entity type importance mapping
        self.entity_importance = {
            "function": 0.8,
            "class": 0.9,
            "module": 1.0,
            "interface": 0.7,
            "constant": 0.4,
            "variable": 0.3,
            "configuration": 0.9,
            "relationship": 0.6
        }
    
    def analyze_trigger(self, trigger_type: str, trigger_context: Dict[str, Any]) -> Tuple[ReindexPriority, float]:
        """
        Analyze a trigger to determine base priority and confidence.
        
        Args:
            trigger_type: Type of trigger (file_change, validation_issue, etc.)
            trigger_context: Additional context about the trigger
            
        Returns:
            Tuple of (base_priority, confidence_score)
        """
        try:
            if trigger_type == "validation_issue":
                return self._analyze_validation_trigger(trigger_context)
            elif trigger_type == "file_change":
                return self._analyze_file_change_trigger(trigger_context)
            elif trigger_type in self.trigger_priorities:
                priority = self.trigger_priorities[trigger_type]
                if isinstance(priority, ReindexPriority):
                    return priority, 0.8  # High confidence for direct mappings
                else:
                    return ReindexPriority.MEDIUM, 0.5  # Medium confidence for complex mappings
            else:
                self.logger.warning(f"Unknown trigger type: {trigger_type}")
                return ReindexPriority.MEDIUM, 0.3  # Low confidence for unknown triggers
                
        except Exception as e:
            self.logger.error(f"Error analyzing trigger {trigger_type}: {e}")
            return ReindexPriority.MEDIUM, 0.1
    
    def _analyze_validation_trigger(self, context: Dict[str, Any]) -> Tuple[ReindexPriority, float]:
        """Analyze validation issue trigger"""
        severity = context.get("validation_severity", "warning").lower()
        issue_category = context.get("validation_category", "").lower()
        
        # Get base priority from severity
        base_priorities = self.trigger_priorities["validation_issue"]
        base_priority = base_priorities.get(severity, ReindexPriority.MEDIUM)
        
        # Adjust based on issue category
        if "referential_integrity" in issue_category or "critical" in issue_category:
            # Boost priority for integrity issues
            if base_priority.value > ReindexPriority.CRITICAL.value:
                base_priority = ReindexPriority.CRITICAL
        elif "performance" in issue_category:
            # Lower priority for performance issues
            if base_priority.value < ReindexPriority.LOW.value:
                base_priority = ReindexPriority.LOW
        
        confidence = 0.9  # High confidence for validation-triggered reindexing
        
        self.logger.debug(f"Validation trigger analysis: {severity} -> {base_priority.name}")
        return base_priority, confidence
    
    def _analyze_file_change_trigger(self, context: Dict[str, Any]) -> Tuple[ReindexPriority, float]:
        """Analyze file change trigger"""
        file_path = context.get("file_path", "")
        change_type = context.get("file_change_type", "modified")
        
        # Determine file category
        file_category = self._categorize_file(file_path)
        
        # Get base priority from file category
        file_priorities = self.trigger_priorities["file_change"]
        base_priority = file_priorities.get(file_category, ReindexPriority.MEDIUM)
        
        # Adjust based on change type
        if change_type == "deleted":
            # Deletions are more critical
            if base_priority.value > ReindexPriority.HIGH.value:
                base_priority = ReindexPriority.HIGH
        elif change_type == "created":
            # New files are important
            if base_priority.value > ReindexPriority.MEDIUM.value:
                base_priority = ReindexPriority.MEDIUM
        
        confidence = 0.7  # Good confidence for file-based analysis
        
        self.logger.debug(f"File change trigger analysis: {file_category} {change_type} -> {base_priority.name}")
        return base_priority, confidence
    
    def _categorize_file(self, file_path: str) -> str:
        """Categorize a file based on its path and extension"""
        file_path = file_path.lower()
        
        # Source code files
        if any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h']):
            return "source_code"
        
        # Configuration files
        elif any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg']):
            return "configuration"
        
        # Documentation files
        elif any(file_path.endswith(ext) for ext in ['.md', '.txt', '.rst', '.adoc']) or 'doc' in file_path:
            return "documentation"
        
        # Test files
        elif 'test' in file_path or file_path.endswith('_test.py') or file_path.endswith('.test.js'):
            return "test_file"
        
        else:
            return "other"


class PriorityCalculator:
    """
    Calculates dynamic priority for reindexing jobs using multiple factors
    and adaptive learning from historical performance.
    """
    
    def __init__(self, knowledge_path: str = "/Users/cal/DEV/RIF/knowledge"):
        self.knowledge_path = knowledge_path
        self.logger = logging.getLogger(f"priority_calculator_{id(self)}")
        
        # Initialize trigger analyzer
        self.trigger_analyzer = TriggerAnalyzer()
        
        # Priority calculation weights (can be tuned based on learning)
        self.priority_weights = {
            'urgency': 2.0,        # Most important factor
            'impact': 1.5,         # Second most important
            'staleness': 1.0,      # Moderate importance
            'dependency': 1.2,     # Important for correctness
            'system_load': 0.8,    # Performance consideration
            'historical': 0.5      # Learning from past performance
        }
        
        # Historical performance data for learning
        self.performance_history = deque(maxlen=1000)
        self.entity_staleness_cache = {}  # entity_id -> last_indexed_time
        self.dependency_graph = defaultdict(set)  # entity_id -> set of dependent entities
        
        # Adaptive learning parameters
        self.learning_enabled = True
        self.weight_adjustment_factor = 0.1  # How quickly to adjust weights
        
        # Entity type priority modifiers
        self.entity_type_modifiers = {
            "module": 1.2,        # Modules are important
            "class": 1.1,         # Classes are important
            "function": 1.0,      # Functions are baseline
            "interface": 1.1,     # Interfaces affect many things
            "constant": 0.8,      # Constants are less critical
            "variable": 0.7,      # Variables are less critical
            "relationship": 0.9   # Relationships are moderately important
        }
        
        self.logger.info("PriorityCalculator initialized")
    
    def calculate_priority(self, job: Any) -> ReindexPriority:
        """
        Calculate the priority for a reindexing job using multiple factors.
        
        Args:
            job: ReindexJob instance
            
        Returns:
            Calculated priority level
        """
        try:
            # For manual requests, be more conservative and respect user intent
            if job.trigger.value == "manual_request":
                # Use the originally requested priority as base
                base_priority = job.priority
                self.logger.debug(f"Manual request detected, using original priority: {base_priority.name}")
            else:
                # Start with base priority from trigger analysis
                base_priority, trigger_confidence = self.trigger_analyzer.analyze_trigger(
                    job.trigger.value, 
                    job.metadata
                )
            
            # Calculate individual priority factors
            factors = self._calculate_priority_factors(job)
            
            # Calculate weighted score
            weighted_score = factors.calculate_weighted_score(self.priority_weights)
            
            # Apply entity type modifiers
            entity_modifier = self.entity_type_modifiers.get(job.entity_type, 1.0)
            adjusted_score = weighted_score * entity_modifier
            
            # For manual requests, only adjust priority if score is very compelling
            if job.trigger.value == "manual_request":
                # Only change priority for very high or very low scores
                if adjusted_score > 0.9:
                    # Exceptional case - boost priority
                    calculated_priority = ReindexPriority(max(0, base_priority.value - 1))
                elif adjusted_score < 0.1:
                    # Exceptional case - reduce priority
                    calculated_priority = ReindexPriority(min(3, base_priority.value + 1))
                else:
                    # Respect user's intended priority
                    calculated_priority = base_priority
            else:
                # Convert score to priority level for non-manual triggers
                calculated_priority = self._score_to_priority(adjusted_score, base_priority)
            
            # Apply learning-based adjustments
            if self.learning_enabled:
                calculated_priority = self._apply_historical_learning(job, calculated_priority)
            
            self.logger.debug(
                f"Priority calculation for {job.id}: "
                f"base={base_priority.name}, weighted_score={weighted_score:.3f}, "
                f"entity_mod={entity_modifier}, final={calculated_priority.name}"
            )
            
            return calculated_priority
            
        except Exception as e:
            self.logger.error(f"Error calculating priority for job {job.id}: {e}")
            return ReindexPriority.MEDIUM  # Safe default
    
    def _calculate_priority_factors(self, job: Any) -> PriorityFactors:
        """Calculate individual priority factors for a job"""
        
        # Urgency factor (based on trigger type and context)
        urgency_factor = self._calculate_urgency_factor(job)
        
        # Impact factor (based on entity type and dependencies)
        impact_factor = self._calculate_impact_factor(job)
        
        # Staleness factor (how old is the indexed data)
        staleness_factor = self._calculate_staleness_factor(job)
        
        # Dependency factor (how many things depend on this entity)
        dependency_factor = self._calculate_dependency_factor(job)
        
        # System load factor (current system performance)
        system_load_factor = self._calculate_system_load_factor()
        
        # Historical factor (based on past performance)
        historical_factor = self._calculate_historical_factor(job)
        
        return PriorityFactors(
            urgency_factor=urgency_factor,
            impact_factor=impact_factor,
            staleness_factor=staleness_factor,
            dependency_factor=dependency_factor,
            system_load_factor=system_load_factor,
            historical_factor=historical_factor
        )
    
    def _calculate_urgency_factor(self, job: Any) -> float:
        """Calculate urgency based on trigger and context"""
        # Start with trigger-based urgency
        if job.trigger.value == "validation_issue":
            severity = job.metadata.get("validation_severity", "warning")
            severity_urgency = {
                "critical": 1.0,
                "error": 0.8,
                "warning": 0.5,
                "info": 0.2
            }
            return severity_urgency.get(severity.lower(), 0.5)
        
        elif job.trigger.value == "file_change":
            change_type = job.metadata.get("file_change_type", "modified")
            change_urgency = {
                "deleted": 0.9,
                "created": 0.7,
                "modified": 0.6,
                "moved": 0.5
            }
            return change_urgency.get(change_type, 0.5)
        
        elif job.trigger.value == "system_startup":
            return 0.8  # High urgency for system startup
        
        elif job.trigger.value == "manual_request":
            return 0.7  # User requested, moderately urgent
        
        else:
            return 0.3  # Low urgency for background tasks
    
    def _calculate_impact_factor(self, job: Any) -> float:
        """Calculate impact based on entity type and affected systems"""
        base_impact = {
            "module": 0.9,
            "class": 0.8,
            "interface": 0.8,
            "function": 0.6,
            "constant": 0.4,
            "variable": 0.3,
            "relationship": 0.5
        }.get(job.entity_type, 0.5)
        
        # Boost impact if this is a configuration or core system file
        if job.file_path:
            if any(keyword in job.file_path.lower() for keyword in ['config', 'settings', 'core', 'main']):
                base_impact = min(base_impact * 1.3, 1.0)
        
        return base_impact
    
    def _calculate_staleness_factor(self, job: Any) -> float:
        """Calculate staleness based on when entity was last indexed"""
        entity_key = f"{job.entity_type}:{job.entity_id}" if job.entity_id else f"file:{job.file_path}"
        
        last_indexed = self.entity_staleness_cache.get(entity_key)
        if not last_indexed:
            return 0.5  # Unknown staleness, moderate priority boost
        
        # Calculate age in hours
        age_hours = (datetime.now() - last_indexed).total_seconds() / 3600
        
        # Convert age to staleness factor (0-1)
        if age_hours < 1:
            return 0.1  # Very fresh
        elif age_hours < 6:
            return 0.3  # Moderately fresh
        elif age_hours < 24:
            return 0.6  # Getting stale
        else:
            return 1.0  # Very stale
    
    def _calculate_dependency_factor(self, job: Any) -> float:
        """Calculate dependency factor based on how many entities depend on this one"""
        entity_key = f"{job.entity_type}:{job.entity_id}" if job.entity_id else f"file:{job.file_path}"
        
        dependents = self.dependency_graph.get(entity_key, set())
        
        # Convert dependent count to factor (0-1)
        if len(dependents) == 0:
            return 0.1
        elif len(dependents) < 3:
            return 0.3
        elif len(dependents) < 10:
            return 0.6
        else:
            return 1.0  # Many dependents, high impact
    
    def _calculate_system_load_factor(self) -> float:
        """Calculate system load factor (higher load = lower priority boost)"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Calculate overall load
            overall_load = (cpu_percent + memory_percent) / 200.0  # Normalize to 0-1
            
            # Return inverse of load (high load = low factor)
            return max(0.0, 1.0 - overall_load)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate system load: {e}")
            return 0.5  # Default to moderate factor
    
    def _calculate_historical_factor(self, job: Any) -> float:
        """Calculate factor based on historical performance of similar jobs"""
        if not self.performance_history:
            return 0.0  # No history available
        
        # Find similar jobs in history
        similar_jobs = [
            perf for perf in self.performance_history
            if (perf.get("entity_type") == job.entity_type and 
                perf.get("trigger") == job.trigger.value)
        ]
        
        if not similar_jobs:
            return 0.0  # No similar jobs
        
        # Calculate average success rate
        success_rate = sum(1 for job in similar_jobs if job.get("success", False)) / len(similar_jobs)
        
        # Higher success rate suggests good priority calculation
        return success_rate
    
    def _score_to_priority(self, score: float, base_priority: ReindexPriority) -> ReindexPriority:
        """Convert weighted score to priority level"""
        # Use base priority as starting point, then adjust based on score
        base_value = base_priority.value
        
        # Score influences priority adjustment
        if score > 0.8:
            # Very high score, boost priority
            adjusted_value = max(0, base_value - 1)
        elif score > 0.6:
            # High score, slight boost
            adjusted_value = max(0, base_value - 0.5)
        elif score < 0.3:
            # Low score, reduce priority
            adjusted_value = min(3, base_value + 1)
        elif score < 0.5:
            # Moderate-low score, slight reduction
            adjusted_value = min(3, base_value + 0.5)
        else:
            # Moderate score, keep base priority
            adjusted_value = base_value
        
        # Convert back to priority enum
        priority_values = [ReindexPriority.CRITICAL, ReindexPriority.HIGH, ReindexPriority.MEDIUM, ReindexPriority.LOW]
        return priority_values[max(0, min(3, int(round(adjusted_value))))]
    
    def _apply_historical_learning(self, job: Any, calculated_priority: ReindexPriority) -> ReindexPriority:
        """Apply learning-based adjustments to priority"""
        # This is a placeholder for adaptive learning
        # In a full implementation, this would analyze past performance
        # and adjust priorities based on what worked well
        
        return calculated_priority
    
    def update_entity_staleness(self, entity_type: str, entity_id: Optional[str], file_path: Optional[str]):
        """Update staleness tracking when an entity is indexed"""
        entity_key = f"{entity_type}:{entity_id}" if entity_id else f"file:{file_path}"
        self.entity_staleness_cache[entity_key] = datetime.now()
    
    def add_dependency(self, entity: str, dependent: str):
        """Add a dependency relationship for factor calculation"""
        self.dependency_graph[entity].add(dependent)
    
    def record_job_performance(self, job: Any, success: bool, duration: float, metrics: Dict[str, Any]):
        """Record job performance for learning"""
        performance_record = {
            "entity_type": job.entity_type,
            "trigger": job.trigger.value,
            "priority": job.priority.name,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.performance_history.append(performance_record)
        
        # Trigger learning update if enabled
        if self.learning_enabled and len(self.performance_history) % 50 == 0:
            self._update_weights_from_performance()
    
    def _update_weights_from_performance(self):
        """Update priority calculation weights based on performance history"""
        # This is a placeholder for machine learning-based weight adjustment
        # In a full implementation, this would analyze performance patterns
        # and adjust weights to optimize scheduling effectiveness
        
        self.logger.debug("Weight adjustment triggered (placeholder implementation)")
    
    def get_priority_statistics(self) -> Dict[str, Any]:
        """Get statistics about priority calculations"""
        if not self.performance_history:
            return {"no_data": True}
        
        # Analyze priority distribution
        priority_counts = defaultdict(int)
        trigger_counts = defaultdict(int)
        success_by_priority = defaultdict(list)
        
        for record in self.performance_history:
            priority = record.get("priority", "UNKNOWN")
            trigger = record.get("trigger", "UNKNOWN")
            success = record.get("success", False)
            
            priority_counts[priority] += 1
            trigger_counts[trigger] += 1
            success_by_priority[priority].append(success)
        
        # Calculate success rates by priority
        success_rates = {}
        for priority, successes in success_by_priority.items():
            success_rates[priority] = sum(successes) / len(successes) if successes else 0.0
        
        return {
            "total_jobs_analyzed": len(self.performance_history),
            "priority_distribution": dict(priority_counts),
            "trigger_distribution": dict(trigger_counts),
            "success_rates_by_priority": success_rates,
            "current_weights": self.priority_weights.copy(),
            "entity_staleness_tracked": len(self.entity_staleness_cache),
            "dependency_relationships": sum(len(deps) for deps in self.dependency_graph.values())
        }


# Factory function for easy instantiation
def create_priority_calculator(knowledge_path: str = "/Users/cal/DEV/RIF/knowledge") -> PriorityCalculator:
    """Create a PriorityCalculator with default configuration"""
    return PriorityCalculator(knowledge_path=knowledge_path)


# Example usage and testing
if __name__ == "__main__":
    # Test priority calculation system
    print("Testing Priority Calculation System")
    print("=" * 40)
    
    # Create priority calculator
    calculator = create_priority_calculator()
    
    # Create mock job for testing
    from dataclasses import dataclass
    from enum import Enum
    
    class MockTrigger(Enum):
        FILE_CHANGE = "file_change"
        VALIDATION_ISSUE = "validation_issue"
        MANUAL_REQUEST = "manual_request"
    
    @dataclass
    class MockJob:
        id: str
        entity_type: str
        entity_id: str
        file_path: str
        trigger: MockTrigger
        metadata: dict
        priority: ReindexPriority = ReindexPriority.MEDIUM
    
    # Test different job scenarios
    test_jobs = [
        MockJob(
            id="test-1",
            entity_type="module",
            entity_id="main_module",
            file_path="/src/main.py",
            trigger=MockTrigger.FILE_CHANGE,
            metadata={"file_change_type": "modified"}
        ),
        MockJob(
            id="test-2",
            entity_type="class",
            entity_id="critical_class",
            file_path="/src/core/critical.py",
            trigger=MockTrigger.VALIDATION_ISSUE,
            metadata={"validation_severity": "critical", "validation_category": "referential_integrity"}
        ),
        MockJob(
            id="test-3",
            entity_type="variable",
            entity_id="temp_var",
            file_path="/src/utils/helpers.py",
            trigger=MockTrigger.MANUAL_REQUEST,
            metadata={}
        )
    ]
    
    # Calculate priorities
    for job in test_jobs:
        calculated_priority = calculator.calculate_priority(job)
        print(f"Job {job.id} ({job.entity_type}, {job.trigger.value}): {calculated_priority.name}")
        
        # Simulate job completion for performance tracking
        calculator.record_job_performance(
            job, 
            success=True, 
            duration=1.5, 
            metrics={"cpu": 20, "memory": 100}
        )
    
    # Get priority statistics
    stats = calculator.get_priority_statistics()
    print(f"\nPriority Statistics:")
    for key, value in stats.items():
        if key != "current_weights":  # Skip detailed weights for brevity
            print(f"  {key}: {value}")
    
    print("\nPriority calculation system test completed")