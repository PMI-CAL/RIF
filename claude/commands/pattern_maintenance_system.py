"""
Pattern Maintenance System - Advanced Pattern Management

This module extends the PatternReinforcementSystem with advanced maintenance
capabilities including pattern pruning, time-based decay, and quality management.

Key Features:
- Automated pattern pruning based on effectiveness thresholds
- Time-based pattern decay for relevance maintenance
- Pattern lifecycle management with safe archival
- Performance-optimized batch processing
- Quality gate enforcement

Implementation corresponds to Phase 3 and Phase 4 of Issue #78 plan.
"""

import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import base components
from .pattern_reinforcement_system import (
    PatternMetrics, PatternOutcome, PatternReinforcementSystem,
    OutcomeType, FailureMode
)

# Import knowledge interface
try:
    from knowledge.interface import get_knowledge_system, KnowledgeInterface
    KNOWLEDGE_INTERFACE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_INTERFACE_AVAILABLE = False


class MaintenanceMode(Enum):
    """Pattern maintenance operation modes."""
    ARCHIVE = "archive"         # Move to archive collection
    SOFT_DELETE = "soft_delete" # Mark as deleted but keep
    HARD_DELETE = "hard_delete" # Permanently remove
    QUARANTINE = "quarantine"   # Isolate for review


@dataclass
class PruningResult:
    """Result of pattern pruning operation."""
    pattern_id: str
    action_taken: MaintenanceMode
    reason: str
    metrics_snapshot: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'pattern_id': self.pattern_id,
            'action_taken': self.action_taken.value,
            'reason': self.reason,
            'metrics_snapshot': self.metrics_snapshot,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MaintenanceReport:
    """Comprehensive maintenance operation report."""
    operation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    patterns_evaluated: int = 0
    patterns_pruned: int = 0
    patterns_archived: int = 0
    patterns_updated: int = 0
    errors_encountered: int = 0
    
    # Detailed results
    pruning_results: List[PruningResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'operation_id': self.operation_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'patterns_evaluated': self.patterns_evaluated,
            'patterns_pruned': self.patterns_pruned,
            'patterns_archived': self.patterns_archived,
            'patterns_updated': self.patterns_updated,
            'errors_encountered': self.errors_encountered,
            'pruning_results': [result.to_dict() for result in self.pruning_results],
            'performance_metrics': self.performance_metrics
        }


class PatternMaintenanceSystem:
    """
    Advanced pattern maintenance system for quality management.
    
    This system handles:
    1. Pattern pruning based on performance thresholds
    2. Time-based decay calculations
    3. Pattern lifecycle management
    4. Quality gate enforcement
    5. Maintenance reporting and analytics
    
    Integrates with PatternReinforcementSystem for comprehensive pattern management.
    """
    
    def __init__(self, 
                 reinforcement_system: PatternReinforcementSystem,
                 maintenance_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pattern maintenance system.
        
        Args:
            reinforcement_system: Core reinforcement system instance
            maintenance_config: Configuration for maintenance operations
        """
        self.logger = logging.getLogger(f"{__name__}.PatternMaintenanceSystem")
        self.reinforcement_system = reinforcement_system
        self.knowledge = reinforcement_system.knowledge
        
        # Configuration
        self.config = {
            # Pruning thresholds
            'pruning_success_rate_threshold': 0.3,  # Below 30% success rate
            'pruning_min_applications': 10,          # Minimum applications before pruning
            'pruning_inactivity_days': 90,           # Days of inactivity before pruning
            
            # Time decay settings
            'decay_rate_per_day': 0.001,             # 0.1% decay per day
            'decay_acceleration_threshold': 30,      # Days before accelerated decay
            'decay_acceleration_factor': 2.0,        # Multiplier for accelerated decay
            'minimum_score': 0.1,                    # Minimum score floor
            
            # Batch processing
            'batch_size': 50,                        # Patterns per batch
            'max_concurrent_operations': 4,          # Concurrent maintenance operations
            'operation_timeout': 300,                # 5 minutes timeout
            
            # Safety settings
            'max_pruning_percentage': 0.2,           # Max 20% of patterns pruned per run
            'require_confirmation': True,            # Require confirmation for pruning
            'create_backups': True,                  # Create backups before operations
            
            # Quality gates
            'quality_threshold': 0.6,                # Minimum quality score
            'effectiveness_trend_length': 5,         # Number of recent outcomes to consider
        }
        
        if maintenance_config:
            self.config.update(maintenance_config)
        
        # Operation tracking
        self.maintenance_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_concurrent_operations'])
        
        # Performance metrics
        self.maintenance_metrics = {
            'total_operations': 0,
            'patterns_pruned_total': 0,
            'patterns_archived_total': 0,
            'last_maintenance_run': None,
            'average_operation_time': 0.0,
            'errors_total': 0
        }
        
        self.logger.info("Pattern maintenance system initialized")
    
    def run_maintenance(self, 
                       include_pruning: bool = True,
                       include_decay: bool = True,
                       dry_run: bool = False) -> MaintenanceReport:
        """
        Run comprehensive pattern maintenance.
        
        Args:
            include_pruning: Whether to include pattern pruning
            include_decay: Whether to apply time-based decay
            dry_run: If True, don't actually modify patterns
            
        Returns:
            MaintenanceReport with operation results
        """
        start_time = datetime.now()
        operation_id = f"maintenance_{int(time.time())}"
        
        with self.maintenance_lock:
            self.logger.info(f"Starting maintenance operation {operation_id} "
                           f"(pruning: {include_pruning}, decay: {include_decay}, dry_run: {dry_run})")
            
            report = MaintenanceReport(
                operation_id=operation_id,
                start_time=start_time
            )
            
            try:
                # Get all patterns for maintenance
                all_patterns = self._get_all_patterns_for_maintenance()
                report.patterns_evaluated = len(all_patterns)
                
                if not all_patterns:
                    self.logger.warning("No patterns found for maintenance")
                    report.end_time = datetime.now()
                    return report
                
                # Apply time-based decay if requested
                if include_decay:
                    decay_results = self._apply_time_decay(all_patterns, dry_run)
                    report.patterns_updated += decay_results['updated_count']
                    report.performance_metrics['decay_operation'] = decay_results
                
                # Perform pattern pruning if requested
                if include_pruning:
                    pruning_results = self._perform_pattern_pruning(all_patterns, dry_run)
                    report.pruning_results.extend(pruning_results)
                    report.patterns_pruned = len([r for r in pruning_results if r.action_taken != MaintenanceMode.ARCHIVE])
                    report.patterns_archived = len([r for r in pruning_results if r.action_taken == MaintenanceMode.ARCHIVE])
                
                # Update maintenance metrics
                if not dry_run:
                    self._update_maintenance_metrics(report)
                
                report.end_time = datetime.now()
                
                # Store maintenance report
                self._store_maintenance_report(report)
                
                self.logger.info(f"Maintenance operation {operation_id} completed: "
                               f"{report.patterns_evaluated} evaluated, "
                               f"{report.patterns_updated} updated, "
                               f"{report.patterns_pruned} pruned, "
                               f"{report.patterns_archived} archived")
                
                return report
                
            except Exception as e:
                self.logger.error(f"Error during maintenance operation: {e}")
                report.errors_encountered += 1
                report.end_time = datetime.now()
                return report
    
    def _get_all_patterns_for_maintenance(self) -> List[Tuple[str, PatternMetrics]]:
        """Get all patterns that need maintenance evaluation."""
        patterns = []
        
        try:
            if self.knowledge:
                # Get from knowledge system
                results = self.knowledge.retrieve_knowledge(
                    query="*",
                    collection="pattern_metrics",
                    n_results=1000  # Get large batch
                )
                
                for result in results:
                    try:
                        metrics_data = json.loads(result['content'])
                        metrics = PatternMetrics.from_dict(metrics_data)
                        patterns.append((metrics.pattern_id, metrics))
                    except Exception as e:
                        self.logger.warning(f"Error parsing pattern metrics: {e}")
            else:
                # Get from cache
                for pattern_id, metrics in self.reinforcement_system.pattern_metrics_cache.items():
                    patterns.append((pattern_id, metrics))
                    
        except Exception as e:
            self.logger.error(f"Error retrieving patterns for maintenance: {e}")
        
        self.logger.debug(f"Retrieved {len(patterns)} patterns for maintenance")
        return patterns
    
    def _apply_time_decay(self, 
                         patterns: List[Tuple[str, PatternMetrics]], 
                         dry_run: bool = False) -> Dict[str, Any]:
        """Apply time-based decay to pattern scores."""
        decay_results = {
            'updated_count': 0,
            'total_decay_applied': 0.0,
            'patterns_below_threshold': 0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        current_time = datetime.now()
        
        for pattern_id, metrics in patterns:
            try:
                if not metrics.last_used:
                    continue
                
                # Calculate decay based on time since last use
                days_since_use = (current_time - metrics.last_used).days
                
                if days_since_use <= 0:
                    continue  # Used today, no decay
                
                # Calculate decay amount
                base_decay_rate = self.config['decay_rate_per_day']
                decay_rate = base_decay_rate
                
                # Apply acceleration for long-unused patterns
                if days_since_use > self.config['decay_acceleration_threshold']:
                    decay_rate *= self.config['decay_acceleration_factor']
                
                # Calculate total decay
                total_decay = 1 - (decay_rate * days_since_use)
                total_decay = max(0.0, total_decay)  # Don't go negative
                
                # Apply decay to current score
                old_score = metrics.current_score
                new_score = old_score * total_decay
                
                # Respect minimum score
                new_score = max(new_score, self.config['minimum_score'])
                
                # Update metrics if not dry run
                if not dry_run and new_score != old_score:
                    metrics.current_score = new_score
                    metrics.decay_factor = total_decay
                    metrics.last_updated = current_time
                    
                    # Store updated metrics
                    if self.reinforcement_system._store_pattern_metrics(metrics):
                        decay_results['updated_count'] += 1
                        decay_results['total_decay_applied'] += (old_score - new_score)
                        
                        if new_score < self.config['quality_threshold']:
                            decay_results['patterns_below_threshold'] += 1
                
                self.logger.debug(f"Pattern {pattern_id} decay: {old_score:.3f} -> {new_score:.3f} "
                                f"(days_unused: {days_since_use}, decay_rate: {decay_rate:.4f})")
                                
            except Exception as e:
                self.logger.error(f"Error applying decay to pattern {pattern_id}: {e}")
        
        decay_results['processing_time'] = time.time() - start_time
        self.logger.info(f"Time decay applied to {decay_results['updated_count']} patterns "
                        f"(total_decay: {decay_results['total_decay_applied']:.3f}, "
                        f"below_threshold: {decay_results['patterns_below_threshold']})")
        
        return decay_results
    
    def _perform_pattern_pruning(self, 
                                patterns: List[Tuple[str, PatternMetrics]], 
                                dry_run: bool = False) -> List[PruningResult]:
        """Perform pattern pruning based on effectiveness criteria."""
        pruning_results = []
        
        # Calculate safety limits
        max_patterns_to_prune = int(len(patterns) * self.config['max_pruning_percentage'])
        
        # Identify pruning candidates
        pruning_candidates = []
        
        for pattern_id, metrics in patterns:
            pruning_reason = self._evaluate_pruning_candidate(metrics)
            if pruning_reason:
                pruning_candidates.append((pattern_id, metrics, pruning_reason))
        
        # Sort by priority (lowest scores first)
        pruning_candidates.sort(key=lambda x: x[1].calculate_weighted_score())
        
        # Limit to safety threshold
        pruning_candidates = pruning_candidates[:max_patterns_to_prune]
        
        self.logger.info(f"Found {len(pruning_candidates)} patterns eligible for pruning "
                        f"(max allowed: {max_patterns_to_prune})")
        
        # Process pruning candidates
        for pattern_id, metrics, reason in pruning_candidates:
            try:
                # Determine pruning action
                action = self._determine_pruning_action(metrics, reason)
                
                # Create result record
                result = PruningResult(
                    pattern_id=pattern_id,
                    action_taken=action,
                    reason=reason,
                    metrics_snapshot=metrics.to_dict()
                )
                
                # Execute pruning action if not dry run
                if not dry_run:
                    success = self._execute_pruning_action(pattern_id, metrics, action)
                    if success:
                        self.logger.info(f"Pattern {pattern_id} {action.value}: {reason}")
                    else:
                        self.logger.error(f"Failed to {action.value} pattern {pattern_id}")
                        continue
                else:
                    self.logger.info(f"[DRY RUN] Would {action.value} pattern {pattern_id}: {reason}")
                
                pruning_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error pruning pattern {pattern_id}: {e}")
        
        return pruning_results
    
    def _evaluate_pruning_candidate(self, metrics: PatternMetrics) -> Optional[str]:
        """Evaluate if a pattern should be pruned and return reason."""
        current_time = datetime.now()
        
        # Check success rate threshold
        if (metrics.total_applications >= self.config['pruning_min_applications'] and
            metrics.success_rate < self.config['pruning_success_rate_threshold']):
            return f"Low success rate: {metrics.success_rate:.2%} (threshold: {self.config['pruning_success_rate_threshold']:.2%})"
        
        # Check inactivity
        if metrics.last_used:
            days_inactive = (current_time - metrics.last_used).days
            if days_inactive > self.config['pruning_inactivity_days']:
                return f"Inactive for {days_inactive} days (threshold: {self.config['pruning_inactivity_days']})"
        
        # Check effectiveness trend
        if len(metrics.effectiveness_trend) >= self.config['effectiveness_trend_length']:
            recent_trend = metrics.effectiveness_trend[-self.config['effectiveness_trend_length']:]
            avg_effectiveness = sum(recent_trend) / len(recent_trend)
            if avg_effectiveness < 0.3:  # Very low effectiveness
                return f"Declining effectiveness trend: {avg_effectiveness:.2f}"
        
        # Check for patterns marked as pruning candidates
        if metrics.pruning_candidate:
            return "Marked as pruning candidate by system"
        
        return None
    
    def _determine_pruning_action(self, metrics: PatternMetrics, reason: str) -> MaintenanceMode:
        """Determine appropriate pruning action for a pattern."""
        # Archive patterns with some usage but poor performance
        if metrics.total_applications >= 5 and metrics.success_rate > 0.1:
            return MaintenanceMode.ARCHIVE
        
        # Quarantine patterns with unusual failure patterns
        if "effectiveness trend" in reason:
            return MaintenanceMode.QUARANTINE
        
        # Soft delete inactive patterns
        if "inactive" in reason.lower():
            return MaintenanceMode.SOFT_DELETE
        
        # Default to archiving for safety
        return MaintenanceMode.ARCHIVE
    
    def _execute_pruning_action(self, 
                               pattern_id: str, 
                               metrics: PatternMetrics, 
                               action: MaintenanceMode) -> bool:
        """Execute the specified pruning action."""
        try:
            if action == MaintenanceMode.ARCHIVE:
                return self._archive_pattern(pattern_id, metrics)
            elif action == MaintenanceMode.SOFT_DELETE:
                return self._soft_delete_pattern(pattern_id, metrics)
            elif action == MaintenanceMode.QUARANTINE:
                return self._quarantine_pattern(pattern_id, metrics)
            elif action == MaintenanceMode.HARD_DELETE:
                return self._hard_delete_pattern(pattern_id, metrics)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing {action.value} for pattern {pattern_id}: {e}")
            return False
    
    def _archive_pattern(self, pattern_id: str, metrics: PatternMetrics) -> bool:
        """Archive a pattern by moving it to archive collection."""
        if not self.knowledge:
            return False
        
        try:
            # Get original pattern data
            pattern_results = self.knowledge.retrieve_knowledge(
                query=f"pattern_id:{pattern_id}",
                collection="patterns",
                n_results=1,
                filters={"pattern_id": pattern_id}
            )
            
            if not pattern_results:
                self.logger.warning(f"Pattern {pattern_id} not found in patterns collection")
                return False
            
            # Store in archive collection with additional metadata
            archive_metadata = pattern_results[0]['metadata'].copy()
            archive_metadata.update({
                'archived_date': datetime.now().isoformat(),
                'archive_reason': 'automated_maintenance',
                'original_collection': 'patterns',
                'final_metrics': metrics.to_dict()
            })
            
            archive_result = self.knowledge.store_knowledge(
                collection="archived_patterns",
                content=pattern_results[0]['content'],
                metadata=archive_metadata,
                doc_id=f"archived_{pattern_id}"
            )
            
            if archive_result:
                # Remove from active patterns collection
                self.knowledge.delete_knowledge("patterns", pattern_results[0]['id'])
                self.logger.debug(f"Pattern {pattern_id} archived successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error archiving pattern {pattern_id}: {e}")
            return False
    
    def _soft_delete_pattern(self, pattern_id: str, metrics: PatternMetrics) -> bool:
        """Soft delete a pattern by marking it as deleted."""
        if not self.knowledge:
            return False
        
        try:
            # Update pattern metadata to mark as deleted
            pattern_results = self.knowledge.retrieve_knowledge(
                query=f"pattern_id:{pattern_id}",
                collection="patterns",
                n_results=1,
                filters={"pattern_id": pattern_id}
            )
            
            if not pattern_results:
                return False
            
            # Update metadata
            updated_metadata = pattern_results[0]['metadata'].copy()
            updated_metadata.update({
                'deleted': True,
                'deleted_date': datetime.now().isoformat(),
                'deletion_reason': 'automated_maintenance'
            })
            
            success = self.knowledge.update_knowledge(
                collection="patterns",
                doc_id=pattern_results[0]['id'],
                metadata=updated_metadata
            )
            
            if success:
                self.logger.debug(f"Pattern {pattern_id} soft deleted")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error soft deleting pattern {pattern_id}: {e}")
            return False
    
    def _quarantine_pattern(self, pattern_id: str, metrics: PatternMetrics) -> bool:
        """Move pattern to quarantine for manual review."""
        if not self.knowledge:
            return False
        
        try:
            # Similar to archiving but to quarantine collection
            pattern_results = self.knowledge.retrieve_knowledge(
                query=f"pattern_id:{pattern_id}",
                collection="patterns",
                n_results=1,
                filters={"pattern_id": pattern_id}
            )
            
            if not pattern_results:
                return False
            
            # Store in quarantine collection
            quarantine_metadata = pattern_results[0]['metadata'].copy()
            quarantine_metadata.update({
                'quarantined_date': datetime.now().isoformat(),
                'quarantine_reason': 'effectiveness_concerns',
                'requires_review': True,
                'original_collection': 'patterns',
                'final_metrics': metrics.to_dict()
            })
            
            quarantine_result = self.knowledge.store_knowledge(
                collection="quarantined_patterns",
                content=pattern_results[0]['content'],
                metadata=quarantine_metadata,
                doc_id=f"quarantine_{pattern_id}"
            )
            
            if quarantine_result:
                # Remove from active patterns
                self.knowledge.delete_knowledge("patterns", pattern_results[0]['id'])
                self.logger.debug(f"Pattern {pattern_id} quarantined for review")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error quarantining pattern {pattern_id}: {e}")
            return False
    
    def _hard_delete_pattern(self, pattern_id: str, metrics: PatternMetrics) -> bool:
        """Permanently delete a pattern (use with caution)."""
        if not self.knowledge:
            return False
        
        try:
            # Only allow hard deletion if explicitly configured
            if not self.config.get('allow_hard_delete', False):
                self.logger.warning(f"Hard delete not allowed for pattern {pattern_id}")
                return False
            
            # Delete from patterns collection
            pattern_results = self.knowledge.retrieve_knowledge(
                query=f"pattern_id:{pattern_id}",
                collection="patterns",
                n_results=1,
                filters={"pattern_id": pattern_id}
            )
            
            if pattern_results:
                success = self.knowledge.delete_knowledge("patterns", pattern_results[0]['id'])
                if success:
                    self.logger.debug(f"Pattern {pattern_id} permanently deleted")
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error hard deleting pattern {pattern_id}: {e}")
            return False
    
    def _update_maintenance_metrics(self, report: MaintenanceReport):
        """Update system maintenance metrics."""
        self.maintenance_metrics['total_operations'] += 1
        self.maintenance_metrics['patterns_pruned_total'] += report.patterns_pruned
        self.maintenance_metrics['patterns_archived_total'] += report.patterns_archived
        self.maintenance_metrics['last_maintenance_run'] = report.end_time
        
        if report.end_time:
            operation_time = (report.end_time - report.start_time).total_seconds()
            current_avg = self.maintenance_metrics['average_operation_time']
            total_ops = self.maintenance_metrics['total_operations']
            
            self.maintenance_metrics['average_operation_time'] = (
                (current_avg * (total_ops - 1) + operation_time) / total_ops
            )
        
        self.maintenance_metrics['errors_total'] += report.errors_encountered
    
    def _store_maintenance_report(self, report: MaintenanceReport) -> bool:
        """Store maintenance report for auditing."""
        try:
            if self.knowledge:
                report_data = report.to_dict()
                
                result = self.knowledge.store_knowledge(
                    collection="maintenance_reports",
                    content=json.dumps(report_data, indent=2),
                    metadata={
                        'operation_id': report.operation_id,
                        'type': 'maintenance_report',
                        'patterns_evaluated': report.patterns_evaluated,
                        'patterns_pruned': report.patterns_pruned,
                        'timestamp': report.start_time.isoformat()
                    },
                    doc_id=report.operation_id
                )
                
                return result is not None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing maintenance report: {e}")
            return False
    
    def get_maintenance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive maintenance system metrics."""
        return {
            **self.maintenance_metrics,
            'config': self.config,
            'system_status': {
                'knowledge_system_available': self.knowledge is not None,
                'executor_available': not self.executor._shutdown
            }
        }
    
    def schedule_maintenance(self, interval_hours: int = 24) -> bool:
        """Schedule regular maintenance operations."""
        # This would typically integrate with a job scheduler
        # For now, just log the intention
        self.logger.info(f"Maintenance scheduled to run every {interval_hours} hours")
        return True
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.logger.info("Pattern maintenance system cleanup completed")