#!/usr/bin/env python3
"""
GitHub Batch Resilience - Batch Operation Resilience
Issue #153: High Priority Error Investigation - err_20250824_2f0392aa

Batch operation resilience system for handling bulk GitHub operations with timeout
recovery, operation fragmentation, and coordinated state management.
"""

import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading
import queue
import hashlib
from collections import defaultdict

from .github_timeout_manager import GitHubEndpoint, get_timeout_manager
from .github_request_context import RequestContext, RequestState, ContextScope, get_context_manager

logger = logging.getLogger(__name__)

class BatchOperationType(Enum):
    """Types of batch operations"""
    ISSUE_BULK_UPDATE = "issue_bulk_update"
    ISSUE_BULK_LABEL = "issue_bulk_label"
    ISSUE_BULK_CLOSE = "issue_bulk_close"
    PR_BULK_REVIEW = "pr_bulk_review"
    BULK_COMMENT = "bulk_comment"
    BULK_SEARCH = "bulk_search"
    BULK_CLONE = "bulk_clone"
    CUSTOM_BATCH = "custom_batch"

class BatchStrategy(Enum):
    """Batch execution strategies"""
    SEQUENTIAL = "sequential"          # One operation at a time
    PARALLEL_LIMITED = "parallel_limited"  # Limited parallel execution
    CHUNKED = "chunked"               # Break into chunks
    ADAPTIVE = "adaptive"             # Adapt based on performance

class BatchItemState(Enum):
    """State of individual batch items"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

@dataclass
class BatchItem:
    """Individual item in a batch operation"""
    item_id: str
    operation_data: Dict[str, Any]
    state: BatchItemState
    attempt_count: int
    max_attempts: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_info: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    timeout_used: Optional[float]
    context_id: Optional[str]  # Link to request context

@dataclass
class BatchConfiguration:
    """Configuration for batch operation execution"""
    chunk_size: int = 10
    parallel_limit: int = 3
    item_timeout: float = 60.0
    batch_timeout: float = 3600.0  # 1 hour
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    retry_failed_items: bool = True
    continue_on_errors: bool = True
    failure_threshold: float = 0.2  # Stop if >20% fail
    recovery_delay: float = 5.0
    progress_callback: Optional[Callable] = None

@dataclass
class BatchOperation:
    """A batch operation with items and execution state"""
    batch_id: str
    operation_type: BatchOperationType
    endpoint: GitHubEndpoint
    items: List[BatchItem]
    config: BatchConfiguration
    state: RequestState
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    # Progress tracking
    total_items: int
    completed_items: int
    failed_items: int
    skipped_items: int
    
    # Performance metrics
    total_duration: Optional[float]
    avg_item_duration: Optional[float]
    throughput: Optional[float]  # items per second
    
    # Error tracking
    error_summary: Dict[str, int]
    critical_errors: List[Dict[str, Any]]
    
    def get_progress_percentage(self) -> float:
        """Get completion percentage"""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items + self.failed_items + self.skipped_items) / self.total_items * 100.0
    
    def get_success_rate(self) -> float:
        """Get success rate for completed items"""
        processed = self.completed_items + self.failed_items
        if processed == 0:
            return 100.0
        return self.completed_items / processed * 100.0

class GitHubBatchResilienceManager:
    """
    Manages resilient execution of batch GitHub operations with:
    - Operation fragmentation for timeout resilience
    - Individual item state tracking and recovery
    - Adaptive execution strategies
    - Coordinated timeout and rate limit management
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "knowledge/batch_operations")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Dependencies
        self.timeout_manager = get_timeout_manager()
        self.context_manager = get_context_manager()
        
        # Batch tracking
        self.active_batches: Dict[str, BatchOperation] = {}
        self.completed_batches: Dict[str, BatchOperation] = {}
        
        # Execution management
        self.executor_threads: Dict[str, threading.Thread] = {}
        self.batch_queues: Dict[str, queue.Queue] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Rate limiting coordination
        self.rate_limit_state = {
            "remaining": 5000,
            "reset_time": None,
            "last_update": None
        }
        
        # Load persisted batches
        self._load_persisted_batches()
        
        logger.info(f"Initialized GitHub Batch Resilience Manager with {len(self.active_batches)} active batches")
    
    def create_batch_operation(
        self,
        operation_type: BatchOperationType,
        endpoint: GitHubEndpoint,
        items_data: List[Dict[str, Any]],
        config: Optional[BatchConfiguration] = None
    ) -> BatchOperation:
        """
        Create a new batch operation.
        
        Args:
            operation_type: Type of batch operation
            endpoint: Primary GitHub endpoint for the operation
            items_data: List of data for individual items
            config: Batch configuration (uses defaults if None)
            
        Returns:
            Created BatchOperation
        """
        batch_id = self._generate_batch_id(operation_type, len(items_data))
        config = config or BatchConfiguration()
        
        # Create batch items
        items = []
        for i, item_data in enumerate(items_data):
            item = BatchItem(
                item_id=f"{batch_id}_item_{i:04d}",
                operation_data=item_data,
                state=BatchItemState.PENDING,
                attempt_count=0,
                max_attempts=3,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                error_info=None,
                result=None,
                timeout_used=None,
                context_id=None
            )
            items.append(item)
        
        # Create batch operation
        batch = BatchOperation(
            batch_id=batch_id,
            operation_type=operation_type,
            endpoint=endpoint,
            items=items,
            config=config,
            state=RequestState.INITIALIZED,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            total_items=len(items),
            completed_items=0,
            failed_items=0,
            skipped_items=0,
            total_duration=None,
            avg_item_duration=None,
            throughput=None,
            error_summary=defaultdict(int),
            critical_errors=[]
        )
        
        with self.lock:
            self.active_batches[batch_id] = batch
            self._persist_batch(batch)
        
        logger.info(f"Created batch operation {batch_id} with {len(items)} items")
        return batch
    
    def execute_batch(
        self,
        batch_id: str,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ) -> bool:
        """
        Execute a batch operation with resilience handling.
        
        Args:
            batch_id: Batch operation identifier
            item_executor: Function to execute individual items
                         Returns (success, result_data, error_message)
        
        Returns:
            True if batch execution started successfully
        """
        with self.lock:
            batch = self.active_batches.get(batch_id)
            if not batch:
                logger.error(f"Batch {batch_id} not found")
                return False
            
            if batch.state != RequestState.INITIALIZED:
                logger.error(f"Batch {batch_id} is not in initialized state")
                return False
            
            batch.state = RequestState.EXECUTING
            batch.started_at = datetime.now()
            self._persist_batch(batch)
        
        # Start execution in background thread
        def execute_batch_thread():
            try:
                self._execute_batch_internal(batch, item_executor)
            except Exception as e:
                logger.error(f"Batch execution error for {batch_id}: {e}")
                with self.lock:
                    batch.state = RequestState.FAILED
                    batch.completed_at = datetime.now()
                    self._persist_batch(batch)
        
        thread = threading.Thread(target=execute_batch_thread, daemon=False)
        thread.start()
        
        with self.lock:
            self.executor_threads[batch_id] = thread
        
        logger.info(f"Started batch execution for {batch_id}")
        return True
    
    def _execute_batch_internal(
        self,
        batch: BatchOperation,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ):
        """Internal batch execution logic"""
        start_time = time.time()
        
        try:
            if batch.config.strategy == BatchStrategy.SEQUENTIAL:
                self._execute_sequential(batch, item_executor)
            elif batch.config.strategy == BatchStrategy.PARALLEL_LIMITED:
                self._execute_parallel_limited(batch, item_executor)
            elif batch.config.strategy == BatchStrategy.CHUNKED:
                self._execute_chunked(batch, item_executor)
            else:  # ADAPTIVE
                self._execute_adaptive(batch, item_executor)
            
            # Calculate final metrics
            end_time = time.time()
            batch.total_duration = end_time - start_time
            
            processed_items = batch.completed_items + batch.failed_items
            if processed_items > 0:
                batch.avg_item_duration = batch.total_duration / processed_items
                batch.throughput = processed_items / batch.total_duration
            
            # Determine final state
            if batch.failed_items == 0:
                batch.state = RequestState.COMPLETED
            elif batch.failed_items / batch.total_items > batch.config.failure_threshold:
                batch.state = RequestState.FAILED
            else:
                batch.state = RequestState.COMPLETED  # Partial success
            
            batch.completed_at = datetime.now()
            
        except Exception as e:
            batch.state = RequestState.FAILED
            batch.completed_at = datetime.now()
            batch.critical_errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "type": "batch_execution_error"
            })
            logger.error(f"Batch execution failed for {batch.batch_id}: {e}")
        
        finally:
            with self.lock:
                # Move to completed batches
                self.completed_batches[batch.batch_id] = batch
                if batch.batch_id in self.active_batches:
                    del self.active_batches[batch.batch_id]
                
                # Clean up thread tracking
                if batch.batch_id in self.executor_threads:
                    del self.executor_threads[batch.batch_id]
                
                self._persist_batch(batch, completed=True)
            
            logger.info(f"Batch {batch.batch_id} completed with {batch.completed_items}/{batch.total_items} successful items")
    
    def _execute_sequential(
        self,
        batch: BatchOperation,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ):
        """Execute batch items sequentially"""
        for item in batch.items:
            if batch.state == RequestState.FAILED:
                break  # Stop if batch failed
            
            self._execute_single_item(batch, item, item_executor)
            
            # Check failure threshold
            if self._should_stop_batch(batch):
                break
    
    def _execute_parallel_limited(
        self,
        batch: BatchOperation,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ):
        """Execute batch items with limited parallelism"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch.config.parallel_limit) as executor:
            # Submit all items
            future_to_item = {}
            for item in batch.items:
                future = executor.submit(self._execute_single_item, batch, item, item_executor)
                future_to_item[future] = item
            
            # Process completions
            for future in concurrent.futures.as_completed(future_to_item, timeout=batch.config.batch_timeout):
                if self._should_stop_batch(batch):
                    # Cancel remaining futures
                    for f in future_to_item:
                        if not f.done():
                            f.cancel()
                    break
    
    def _execute_chunked(
        self,
        batch: BatchOperation,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ):
        """Execute batch items in chunks"""
        chunk_size = batch.config.chunk_size
        items = batch.items
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            
            # Execute chunk sequentially
            for item in chunk:
                if batch.state == RequestState.FAILED:
                    break
                
                self._execute_single_item(batch, item, item_executor)
                
                if self._should_stop_batch(batch):
                    break
            
            # Delay between chunks for rate limiting
            if i + chunk_size < len(items):
                time.sleep(batch.config.recovery_delay)
            
            if self._should_stop_batch(batch):
                break
    
    def _execute_adaptive(
        self,
        batch: BatchOperation,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ):
        """Execute with adaptive strategy based on performance"""
        # Start with chunked approach
        initial_chunk_size = min(5, len(batch.items))
        success_rate = 0.0
        avg_duration = 0.0
        
        items = batch.items[:]
        processed = 0
        
        while items and not self._should_stop_batch(batch):
            # Determine current chunk size based on performance
            if success_rate > 0.9 and avg_duration < 10.0:
                # Good performance, increase chunk size
                chunk_size = min(initial_chunk_size * 2, 20)
            elif success_rate < 0.7 or avg_duration > 30.0:
                # Poor performance, reduce to sequential
                chunk_size = 1
            else:
                chunk_size = initial_chunk_size
            
            # Process next chunk
            chunk = items[:chunk_size]
            items = items[chunk_size:]
            
            chunk_start = time.time()
            chunk_successes = 0
            
            for item in chunk:
                if batch.state == RequestState.FAILED:
                    break
                
                success = self._execute_single_item(batch, item, item_executor)
                if success:
                    chunk_successes += 1
                
                processed += 1
            
            # Update performance metrics
            chunk_duration = time.time() - chunk_start
            success_rate = chunk_successes / len(chunk) if chunk else 0.0
            avg_duration = chunk_duration / len(chunk) if chunk else 0.0
            
            # Adaptive delay based on performance
            if success_rate < 0.8:
                time.sleep(batch.config.recovery_delay * 2)
            elif success_rate < 0.9:
                time.sleep(batch.config.recovery_delay)
    
    def _execute_single_item(
        self,
        batch: BatchOperation,
        item: BatchItem,
        item_executor: Callable[[Dict[str, Any]], Tuple[bool, Optional[Dict[str, Any]], Optional[str]]]
    ) -> bool:
        """Execute a single batch item with timeout and context management"""
        item.state = BatchItemState.EXECUTING
        item.started_at = datetime.now()
        item.attempt_count += 1
        
        # Get timeout for this operation
        timeout = self.timeout_manager.get_timeout(batch.endpoint, item.attempt_count - 1)
        item.timeout_used = timeout
        
        # Create request context for this item
        context = self.context_manager.create_context(
            endpoint=batch.endpoint,
            operation_type=f"{batch.operation_type.value}_item",
            command_args=[],  # Will be filled by executor
            scope=ContextScope.MEMORY_ONLY,
            max_attempts=item.max_attempts,
            tags=[f"batch:{batch.batch_id}", f"item:{item.item_id}"]
        )
        item.context_id = context.context_id
        
        try:
            # Execute the item
            start_time = time.time()
            success, result, error_message = item_executor(item.operation_data)
            duration = time.time() - start_time
            
            # Record metrics
            self.timeout_manager.record_request_metrics(
                endpoint=batch.endpoint,
                duration=duration,
                success=success,
                timeout_used=timeout,
                retry_count=item.attempt_count - 1,
                error_type=error_message if not success else None
            )
            
            if success:
                item.state = BatchItemState.COMPLETED
                item.result = result
                item.completed_at = datetime.now()
                
                with self.lock:
                    batch.completed_items += 1
                
                # Complete context
                self.context_manager.complete_context(context.context_id, result, True)
                
                # Progress callback
                if batch.config.progress_callback:
                    batch.config.progress_callback(batch.get_progress_percentage())
                
                return True
            
            else:
                # Handle failure
                item.error_info = {
                    "attempt": item.attempt_count,
                    "error": error_message,
                    "duration": duration,
                    "timeout_used": timeout
                }
                
                with self.lock:
                    batch.error_summary[error_message or "unknown"] += 1
                
                # Determine if we should retry
                if item.attempt_count < item.max_attempts and batch.config.retry_failed_items:
                    item.state = BatchItemState.RETRYING
                    
                    # Update context for retry
                    self.context_manager.update_context_state(
                        context.context_id,
                        RequestState.RETRYING,
                        error_info=item.error_info
                    )
                    
                    # Retry delay
                    time.sleep(batch.config.recovery_delay)
                    
                    # Recursive retry
                    return self._execute_single_item(batch, item, item_executor)
                    
                else:
                    # Max attempts reached or no retry
                    item.state = BatchItemState.FAILED
                    item.completed_at = datetime.now()
                    
                    with self.lock:
                        batch.failed_items += 1
                    
                    # Complete context with failure
                    self.context_manager.complete_context(context.context_id, None, False)
                    
                    # Add to critical errors if needed
                    if "timeout" in (error_message or "").lower():
                        batch.critical_errors.append({
                            "timestamp": datetime.now().isoformat(),
                            "item_id": item.item_id,
                            "error": error_message,
                            "attempts": item.attempt_count,
                            "type": "timeout_failure"
                        })
                
                return False
                
        except Exception as e:
            # Unexpected error during execution
            item.state = BatchItemState.FAILED
            item.completed_at = datetime.now()
            item.error_info = {
                "attempt": item.attempt_count,
                "error": str(e),
                "type": "execution_exception"
            }
            
            with self.lock:
                batch.failed_items += 1
                batch.error_summary["execution_exception"] += 1
                batch.critical_errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "item_id": item.item_id,
                    "error": str(e),
                    "type": "execution_exception"
                })
            
            # Complete context with failure
            if item.context_id:
                self.context_manager.complete_context(item.context_id, None, False)
            
            logger.error(f"Unexpected error executing batch item {item.item_id}: {e}")
            return False
    
    def _should_stop_batch(self, batch: BatchOperation) -> bool:
        """Determine if batch execution should be stopped"""
        processed_items = batch.completed_items + batch.failed_items
        
        if processed_items == 0:
            return False
        
        failure_rate = batch.failed_items / processed_items
        return failure_rate > batch.config.failure_threshold
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a batch operation"""
        with self.lock:
            batch = self.active_batches.get(batch_id) or self.completed_batches.get(batch_id)
            if not batch:
                return None
            
            return {
                "batch_id": batch_id,
                "operation_type": batch.operation_type.value,
                "endpoint": batch.endpoint.value,
                "state": batch.state.value,
                "progress_percentage": batch.get_progress_percentage(),
                "success_rate": batch.get_success_rate(),
                "total_items": batch.total_items,
                "completed_items": batch.completed_items,
                "failed_items": batch.failed_items,
                "skipped_items": batch.skipped_items,
                "created_at": batch.created_at.isoformat(),
                "started_at": batch.started_at.isoformat() if batch.started_at else None,
                "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
                "total_duration": batch.total_duration,
                "avg_item_duration": batch.avg_item_duration,
                "throughput": batch.throughput,
                "error_summary": dict(batch.error_summary),
                "critical_errors_count": len(batch.critical_errors)
            }
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get comprehensive batch operation statistics"""
        with self.lock:
            stats = {
                "active_batches": len(self.active_batches),
                "completed_batches": len(self.completed_batches),
                "total_batches": len(self.active_batches) + len(self.completed_batches),
                "batches_by_operation": defaultdict(int),
                "batches_by_state": defaultdict(int),
                "total_items_processed": 0,
                "total_successful_items": 0,
                "overall_success_rate": 0.0
            }
            
            all_batches = list(self.active_batches.values()) + list(self.completed_batches.values())
            
            for batch in all_batches:
                stats["batches_by_operation"][batch.operation_type.value] += 1
                stats["batches_by_state"][batch.state.value] += 1
                stats["total_items_processed"] += batch.completed_items + batch.failed_items
                stats["total_successful_items"] += batch.completed_items
            
            if stats["total_items_processed"] > 0:
                stats["overall_success_rate"] = stats["total_successful_items"] / stats["total_items_processed"]
            
            return dict(stats)
    
    def _generate_batch_id(self, operation_type: BatchOperationType, item_count: int) -> str:
        """Generate unique batch ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{operation_type.value}_{item_count}_{timestamp}_{time.time()}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"batch_{timestamp}_{hash_suffix}"
    
    def _persist_batch(self, batch: BatchOperation, completed: bool = False):
        """Persist batch operation to disk"""
        try:
            # Convert batch to serializable format
            batch_data = {
                "batch_id": batch.batch_id,
                "operation_type": batch.operation_type.value,
                "endpoint": batch.endpoint.value,
                "state": batch.state.value,
                "created_at": batch.created_at.isoformat(),
                "started_at": batch.started_at.isoformat() if batch.started_at else None,
                "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
                "total_items": batch.total_items,
                "completed_items": batch.completed_items,
                "failed_items": batch.failed_items,
                "skipped_items": batch.skipped_items,
                "total_duration": batch.total_duration,
                "avg_item_duration": batch.avg_item_duration,
                "throughput": batch.throughput,
                "error_summary": dict(batch.error_summary),
                "critical_errors": batch.critical_errors,
                "config": {**asdict(batch.config), "strategy": batch.config.strategy.value},  # Serialize enum
                "items": []
            }
            
            # Add item data (limited for large batches)
            for item in batch.items[:100]:  # Limit to first 100 items for storage
                item_data = {
                    "item_id": item.item_id,
                    "state": item.state.value,
                    "attempt_count": item.attempt_count,
                    "created_at": item.created_at.isoformat(),
                    "started_at": item.started_at.isoformat() if item.started_at else None,
                    "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                    "error_info": item.error_info,
                    "timeout_used": item.timeout_used,
                    "context_id": item.context_id
                }
                batch_data["items"].append(item_data)
            
            # Write to file
            filename = f"batch_{batch.batch_id}.json"
            subdir = "completed" if completed else "active"
            filepath = self.storage_path / subdir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(batch_data, f, indent=2)
            
            # Remove from other directory if moving
            if completed:
                old_filepath = self.storage_path / "active" / filename
                if old_filepath.exists():
                    old_filepath.unlink()
                    
        except Exception as e:
            logger.error(f"Failed to persist batch {batch.batch_id}: {e}")
    
    def _load_persisted_batches(self):
        """Load persisted batch operations from disk"""
        # Note: This is a simplified version - in production, we'd need full item restoration
        try:
            active_dir = self.storage_path / "active"
            if active_dir.exists():
                for batch_file in active_dir.glob("batch_*.json"):
                    try:
                        with open(batch_file, 'r') as f:
                            data = json.load(f)
                        
                        # Create minimal batch object for status tracking
                        # Full restoration would require recreating all items
                        logger.info(f"Found persisted active batch: {data['batch_id']}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load batch from {batch_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load persisted batches: {e}")

# Global batch manager instance (singleton pattern)
_batch_manager = None

def get_batch_manager() -> GitHubBatchResilienceManager:
    """Get the global batch manager instance"""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = GitHubBatchResilienceManager()
    return _batch_manager