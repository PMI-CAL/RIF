"""
Pattern Reinforcement Integration Module

This module provides integration components for the pattern reinforcement system,
including mock interfaces for Issue #77 dependency and complete system integration.

Key Features:
- Mock pattern application engine interface for parallel development
- Complete system integration with knowledge interface
- Performance monitoring and analytics
- Agent integration helpers
- Testing utilities and validation

Designed to handle Issue #77 dependency gracefully while providing full functionality.
"""

import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import core components
from .pattern_reinforcement_system import (
    PatternReinforcementSystem, PatternOutcome, PatternMetrics,
    OutcomeType, FailureMode
)
from .pattern_maintenance_system import (
    PatternMaintenanceSystem, MaintenanceReport, MaintenanceMode
)

# Import knowledge interface
try:
    from knowledge.interface import get_knowledge_system, KnowledgeInterface
    KNOWLEDGE_INTERFACE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_INTERFACE_AVAILABLE = False


class MockPatternApplicationEngine:
    """
    Mock implementation of Pattern Application Engine (Issue #77).
    
    This mock allows parallel development while Issue #77 is being implemented.
    It simulates pattern application outcomes with realistic data patterns.
    """
    
    def __init__(self, simulation_config: Optional[Dict[str, Any]] = None):
        """Initialize mock engine with simulation parameters."""
        self.logger = logging.getLogger(f"{__name__}.MockPatternApplicationEngine")
        
        # Simulation configuration
        self.config = {
            'success_probability': 0.7,      # 70% success rate for simulations
            'partial_success_probability': 0.15, # 15% partial success rate
            'timeout_probability': 0.05,     # 5% timeout rate
            'error_probability': 0.1,        # 10% error rate
            'min_execution_time': 1.0,       # Minimum execution time (seconds)
            'max_execution_time': 30.0,      # Maximum execution time (seconds)
            'quality_score_range': (0.3, 1.0), # Quality score range
            'effectiveness_range': (0.2, 0.95), # Effectiveness range
        }
        
        if simulation_config:
            self.config.update(simulation_config)
        
        # Predefined patterns for simulation
        self.mock_patterns = [
            'rest_api_implementation',
            'database_migration_pattern',
            'error_handling_standard',
            'logging_configuration',
            'authentication_flow',
            'caching_strategy',
            'validation_middleware',
            'file_upload_handler',
            'email_notification_system',
            'background_job_processing'
        ]
        
        # Simulation state
        self.application_count = 0
        self.pattern_usage = {pattern: 0 for pattern in self.mock_patterns}
        
        self.logger.info("Mock Pattern Application Engine initialized")
    
    def apply_pattern(self, 
                     pattern_id: str,
                     issue_context: Dict[str, Any],
                     agent_type: Optional[str] = None) -> PatternOutcome:
        """
        Mock pattern application that generates realistic outcomes.
        
        Args:
            pattern_id: ID of pattern to apply
            issue_context: Context information about the issue
            agent_type: Type of agent applying the pattern
            
        Returns:
            PatternOutcome with simulated results
        """
        import random
        
        self.application_count += 1
        self.pattern_usage[pattern_id] = self.pattern_usage.get(pattern_id, 0) + 1
        
        # Determine outcome based on probabilities
        rand = random.random()
        
        if rand < self.config['success_probability']:
            outcome_type = OutcomeType.SUCCESS
            success = True
            failure_mode = None
            error_details = None
            effectiveness_score = random.uniform(*self.config['effectiveness_range'])
        elif rand < (self.config['success_probability'] + self.config['partial_success_probability']):
            outcome_type = OutcomeType.PARTIAL_SUCCESS
            success = False  # Partial success counts as failure for scoring
            failure_mode = FailureMode.COMPLEXITY_UNDERESTIMATE
            error_details = "Pattern partially applicable - complexity underestimated"
            effectiveness_score = random.uniform(0.4, 0.7)
        elif rand < (self.config['success_probability'] + 
                     self.config['partial_success_probability'] + 
                     self.config['timeout_probability']):
            outcome_type = OutcomeType.TIMEOUT
            success = False
            failure_mode = FailureMode.PERFORMANCE_ISSUE
            error_details = "Pattern application timed out"
            effectiveness_score = None
        else:
            outcome_type = OutcomeType.FAILURE
            success = False
            failure_mode = random.choice(list(FailureMode))
            error_details = f"Simulated failure: {failure_mode.value}"
            effectiveness_score = None
        
        # Generate realistic execution time
        execution_time = random.uniform(
            self.config['min_execution_time'],
            self.config['max_execution_time']
        )
        
        # Generate quality score
        quality_score = random.uniform(*self.config['quality_score_range'])
        
        # Create outcome
        outcome = PatternOutcome(
            pattern_id=pattern_id,
            outcome_type=outcome_type,
            success=success,
            issue_id=issue_context.get('issue_id'),
            agent_type=agent_type,
            complexity_level=issue_context.get('complexity', 'medium'),
            execution_time=execution_time,
            quality_score=quality_score,
            failure_mode=failure_mode,
            error_details=error_details,
            context_info=issue_context,
            effectiveness_score=effectiveness_score,
            performance_metrics={
                'mock_engine_version': '1.0',
                'simulation_iteration': self.application_count
            }
        )
        
        self.logger.debug(f"Mock pattern application: {pattern_id} -> {outcome_type.value} "
                         f"(execution_time: {execution_time:.2f}s)")
        
        return outcome
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get statistics about mock simulations."""
        return {
            'total_applications': self.application_count,
            'pattern_usage': self.pattern_usage.copy(),
            'config': self.config.copy(),
            'available_patterns': self.mock_patterns.copy()
        }


@dataclass
class IntegrationConfig:
    """Configuration for pattern reinforcement system integration."""
    enable_mock_engine: bool = True
    enable_maintenance: bool = True
    enable_performance_monitoring: bool = True
    maintenance_interval_hours: int = 24
    performance_report_interval_hours: int = 6
    max_concurrent_outcomes: int = 10
    outcome_processing_timeout: int = 30


class PatternReinforcementIntegration:
    """
    Complete integration system for pattern reinforcement.
    
    This class provides a unified interface for the complete pattern reinforcement
    system, integrating all components and providing high-level APIs for agents.
    """
    
    def __init__(self, 
                 knowledge_system: Optional[KnowledgeInterface] = None,
                 config: Optional[IntegrationConfig] = None):
        """
        Initialize the complete pattern reinforcement integration.
        
        Args:
            knowledge_system: Knowledge system interface
            config: Integration configuration
        """
        self.logger = logging.getLogger(f"{__name__}.PatternReinforcementIntegration")
        
        self.config = config or IntegrationConfig()
        
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
            self.logger.warning("Knowledge interface not available")
        
        # Initialize core components
        self.reinforcement_system = PatternReinforcementSystem(
            knowledge_system=self.knowledge
        )
        
        if self.config.enable_maintenance:
            self.maintenance_system = PatternMaintenanceSystem(
                reinforcement_system=self.reinforcement_system
            )
        else:
            self.maintenance_system = None
        
        # Initialize mock engine if enabled
        if self.config.enable_mock_engine:
            self.mock_engine = MockPatternApplicationEngine()
        else:
            self.mock_engine = None
        
        # Performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None
        
        # Integration metrics
        self.integration_metrics = {
            'system_start_time': datetime.now(),
            'total_patterns_processed': 0,
            'total_outcomes_processed': 0,
            'total_maintenance_runs': 0,
            'system_errors': 0,
            'last_health_check': None
        }
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_outcomes)
        self.shutdown_event = threading.Event()
        
        self.logger.info("Pattern reinforcement integration system initialized successfully")
    
    def process_pattern_outcome_async(self, outcome: PatternOutcome) -> bool:
        """
        Process pattern outcome asynchronously.
        
        Args:
            outcome: Pattern outcome to process
            
        Returns:
            True if processing was initiated successfully
        """
        try:
            future = self.executor.submit(self._process_outcome_with_monitoring, outcome)
            
            # Don't wait for completion - let it run asynchronously
            self.logger.debug(f"Async processing initiated for outcome {outcome.outcome_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initiate async processing: {e}")
            self.integration_metrics['system_errors'] += 1
            return False
    
    def process_pattern_outcome_sync(self, outcome: PatternOutcome) -> bool:
        """
        Process pattern outcome synchronously.
        
        Args:
            outcome: Pattern outcome to process
            
        Returns:
            True if processing succeeded
        """
        return self._process_outcome_with_monitoring(outcome)
    
    def _process_outcome_with_monitoring(self, outcome: PatternOutcome) -> bool:
        """Process outcome with performance monitoring."""
        start_time = time.time()
        
        try:
            # Process with reinforcement system
            success = self.reinforcement_system.process_pattern_outcome(outcome)
            
            if success:
                self.integration_metrics['total_outcomes_processed'] += 1
                
                # Update performance monitoring
                if self.performance_monitor:
                    processing_time = time.time() - start_time
                    self.performance_monitor.record_outcome_processing(
                        outcome.pattern_id,
                        processing_time,
                        success
                    )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing outcome with monitoring: {e}")
            self.integration_metrics['system_errors'] += 1
            return False
    
    def simulate_pattern_application(self, 
                                   pattern_id: str,
                                   issue_context: Dict[str, Any],
                                   agent_type: Optional[str] = None,
                                   process_outcome: bool = True) -> Optional[PatternOutcome]:
        """
        Simulate pattern application using mock engine.
        
        Args:
            pattern_id: Pattern to simulate
            issue_context: Issue context
            agent_type: Agent type
            process_outcome: Whether to process the outcome through reinforcement system
            
        Returns:
            PatternOutcome if simulation succeeded, None otherwise
        """
        if not self.mock_engine:
            self.logger.error("Mock engine not available")
            return None
        
        try:
            # Generate simulated outcome
            outcome = self.mock_engine.apply_pattern(pattern_id, issue_context, agent_type)
            
            # Process outcome if requested
            if process_outcome:
                success = self.process_pattern_outcome_sync(outcome)
                if not success:
                    self.logger.warning(f"Failed to process simulated outcome for pattern {pattern_id}")
            
            return outcome
            
        except Exception as e:
            self.logger.error(f"Error simulating pattern application: {e}")
            return None
    
    def run_maintenance(self, dry_run: bool = False) -> Optional[MaintenanceReport]:
        """
        Run pattern maintenance operations.
        
        Args:
            dry_run: If True, don't actually modify patterns
            
        Returns:
            MaintenanceReport if maintenance ran, None if not available
        """
        if not self.maintenance_system:
            self.logger.warning("Maintenance system not available")
            return None
        
        try:
            report = self.maintenance_system.run_maintenance(dry_run=dry_run)
            self.integration_metrics['total_maintenance_runs'] += 1
            
            self.logger.info(f"Maintenance completed: {report.patterns_evaluated} evaluated, "
                           f"{report.patterns_pruned} pruned, {report.patterns_archived} archived")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error running maintenance: {e}")
            self.integration_metrics['system_errors'] += 1
            return None
    
    def get_pattern_recommendations(self, 
                                  issue_context: Dict[str, Any], 
                                  limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get pattern recommendations based on issue context.
        
        Args:
            issue_context: Context about the current issue
            limit: Maximum number of recommendations
            
        Returns:
            List of pattern recommendations with scores
        """
        try:
            # Get top patterns from reinforcement system
            top_patterns = self.reinforcement_system.get_top_patterns(limit * 2)
            
            # Filter and score based on context
            recommendations = []
            
            for pattern_info in top_patterns[:limit]:
                recommendation = {
                    'pattern_id': pattern_info['pattern_id'],
                    'reinforcement_score': pattern_info['score'],
                    'success_rate': pattern_info['success_rate'],
                    'total_applications': pattern_info['applications'],
                    'last_used': pattern_info['last_used'],
                    'recommendation_reason': self._generate_recommendation_reason(pattern_info, issue_context)
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting pattern recommendations: {e}")
            return []
    
    def _generate_recommendation_reason(self, 
                                      pattern_info: Dict[str, Any], 
                                      issue_context: Dict[str, Any]) -> str:
        """Generate explanation for why pattern is recommended."""
        reasons = []
        
        if pattern_info['success_rate'] > 0.8:
            reasons.append(f"High success rate ({pattern_info['success_rate']:.1%})")
        
        if pattern_info['score'] > 1.2:
            reasons.append(f"Strong reinforcement score ({pattern_info['score']:.2f})")
        
        if pattern_info['applications'] >= 10:
            reasons.append(f"Well-tested ({pattern_info['applications']} applications)")
        
        if not reasons:
            reasons.append("Available pattern")
        
        return "; ".join(reasons)
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration system metrics."""
        return self.integration_metrics.copy()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        health = {
            'status': 'healthy',
            'components': {},
            'metrics': self.integration_metrics.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check component health
        try:
            # Reinforcement system
            reinforcement_metrics = self.reinforcement_system.get_performance_metrics()
            health['components']['reinforcement_system'] = {
                'status': 'healthy',
                'metrics': reinforcement_metrics
            }
        except Exception as e:
            health['components']['reinforcement_system'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Maintenance system
        if self.maintenance_system:
            try:
                maintenance_metrics = self.maintenance_system.get_maintenance_metrics()
                health['components']['maintenance_system'] = {
                    'status': 'healthy',
                    'metrics': maintenance_metrics
                }
            except Exception as e:
                health['components']['maintenance_system'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        # Knowledge system
        health['components']['knowledge_system'] = {
            'status': 'healthy' if self.knowledge else 'unavailable',
            'available': self.knowledge is not None
        }
        
        # Performance monitor
        if self.performance_monitor:
            try:
                perf_metrics = self.performance_monitor.get_metrics()
                health['components']['performance_monitor'] = {
                    'status': 'healthy',
                    'metrics': perf_metrics
                }
            except Exception as e:
                health['components']['performance_monitor'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.integration_metrics['last_health_check'] = datetime.now()
        
        return health
    
    def cleanup(self):
        """Cleanup all system resources."""
        self.shutdown_event.set()
        
        # Cleanup components
        if self.reinforcement_system:
            self.reinforcement_system.cleanup()
        
        if self.maintenance_system:
            self.maintenance_system.cleanup()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Pattern reinforcement integration cleanup completed")


class PerformanceMonitor:
    """Performance monitoring for pattern reinforcement system."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self.metrics = {
            'outcome_processing': {
                'total_processed': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'max_time': 0.0,
                'min_time': float('inf'),
                'success_count': 0,
                'failure_count': 0
            },
            'pattern_performance': {},  # pattern_id -> performance data
            'recent_samples': []  # Last 100 processing samples
        }
    
    def record_outcome_processing(self, 
                                 pattern_id: str, 
                                 processing_time: float, 
                                 success: bool):
        """Record outcome processing performance."""
        # Update overall metrics
        outcome_metrics = self.metrics['outcome_processing']
        outcome_metrics['total_processed'] += 1
        outcome_metrics['total_time'] += processing_time
        outcome_metrics['average_time'] = outcome_metrics['total_time'] / outcome_metrics['total_processed']
        outcome_metrics['max_time'] = max(outcome_metrics['max_time'], processing_time)
        outcome_metrics['min_time'] = min(outcome_metrics['min_time'], processing_time)
        
        if success:
            outcome_metrics['success_count'] += 1
        else:
            outcome_metrics['failure_count'] += 1
        
        # Update pattern-specific metrics
        if pattern_id not in self.metrics['pattern_performance']:
            self.metrics['pattern_performance'][pattern_id] = {
                'total_processed': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'success_count': 0,
                'failure_count': 0
            }
        
        pattern_metrics = self.metrics['pattern_performance'][pattern_id]
        pattern_metrics['total_processed'] += 1
        pattern_metrics['total_time'] += processing_time
        pattern_metrics['average_time'] = pattern_metrics['total_time'] / pattern_metrics['total_processed']
        
        if success:
            pattern_metrics['success_count'] += 1
        else:
            pattern_metrics['failure_count'] += 1
        
        # Add to recent samples
        self.metrics['recent_samples'].append({
            'pattern_id': pattern_id,
            'processing_time': processing_time,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only last 100 samples
        if len(self.metrics['recent_samples']) > 100:
            self.metrics['recent_samples'] = self.metrics['recent_samples'][-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return self.metrics.copy()


# Convenience functions for agent integration
def create_pattern_reinforcement_system(knowledge_system: Optional[KnowledgeInterface] = None) -> PatternReinforcementIntegration:
    """
    Create a complete pattern reinforcement system for agent use.
    
    Args:
        knowledge_system: Optional knowledge system interface
        
    Returns:
        Configured PatternReinforcementIntegration instance
    """
    return PatternReinforcementIntegration(knowledge_system=knowledge_system)


def create_mock_pattern_outcome(pattern_id: str, 
                               success: bool = True,
                               issue_id: Optional[str] = None,
                               agent_type: Optional[str] = None) -> PatternOutcome:
    """
    Create a mock pattern outcome for testing.
    
    Args:
        pattern_id: Pattern ID
        success: Whether the outcome was successful
        issue_id: Optional issue ID
        agent_type: Optional agent type
        
    Returns:
        Mock PatternOutcome instance
    """
    return PatternOutcome(
        pattern_id=pattern_id,
        outcome_type=OutcomeType.SUCCESS if success else OutcomeType.FAILURE,
        success=success,
        issue_id=issue_id,
        agent_type=agent_type,
        execution_time=1.5,
        quality_score=0.8 if success else 0.3,
        effectiveness_score=0.9 if success else None,
        failure_mode=None if success else FailureMode.IMPLEMENTATION_ERROR,
        error_details=None if success else "Mock failure for testing"
    )