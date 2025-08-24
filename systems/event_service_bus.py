#!/usr/bin/env python3
"""
Context Intelligence Platform - Event Service Bus
Issue #119: DPIBS Architecture Phase 1

Event Service Bus with priority-based processing and circuit breaker patterns.
Implements real-time event processing for context synchronization.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import queue
import hashlib
from collections import defaultdict
import sqlite3
import os

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events processed by the service bus"""
    CONTEXT_UPDATE = "context_update"
    SYSTEM_CHANGE = "system_change"
    AGENT_REQUEST = "agent_request"
    KNOWLEDGE_UPDATE = "knowledge_update"
    PERFORMANCE_ALERT = "performance_alert"
    CACHE_INVALIDATION = "cache_invalidation"
    SERVICE_STATUS = "service_status"

class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 1  # System failures, security issues
    HIGH = 2      # Performance alerts, urgent updates
    NORMAL = 3    # Regular context updates
    LOW = 4       # Background maintenance, analytics

class EventStatus(Enum):
    """Event processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class Event:
    """Event with metadata and processing information"""
    event_id: str
    event_type: EventType
    priority: EventPriority
    source: str
    target: Optional[str]
    payload: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime]
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    status: EventStatus = EventStatus.QUEUED
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    error_message: Optional[str] = None

class EventHandler:
    """Base class for event handlers"""
    
    def __init__(self, handler_id: str, event_types: Set[EventType]):
        self.handler_id = handler_id
        self.event_types = event_types
        self.processing_count = 0
        self.success_count = 0
        self.failure_count = 0
        
    async def handle_event(self, event: Event) -> bool:
        """Handle an event. Return True for success, False for failure."""
        raise NotImplementedError
    
    def can_handle(self, event_type: EventType) -> bool:
        """Check if handler can process this event type"""
        return event_type in self.event_types

class CircuitBreaker:
    """Circuit breaker pattern for event handler resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        with self.lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            else:  # half-open
                return True
    
    def record_success(self):
        """Record successful execution"""
        with self.lock:
            self.failure_count = 0
            if self.state == "half-open":
                self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"

class EventQueue:
    """Priority-based event queue with persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.queues = {priority: queue.PriorityQueue() for priority in EventPriority}
        self.event_history = []
        self.lock = threading.RLock()
        self._initialize_database()
        self._load_persisted_events()
    
    def _initialize_database(self):
        """Initialize SQLite database for event persistence"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    priority INTEGER,
                    source TEXT,
                    target TEXT,
                    payload TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    correlation_id TEXT,
                    retry_count INTEGER,
                    max_retries INTEGER,
                    status TEXT,
                    processing_started TIMESTAMP,
                    processing_completed TIMESTAMP,
                    error_message TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON events(priority, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON events(status)")
    
    def _load_persisted_events(self):
        """Load persisted events from database on startup"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM events 
                    WHERE status IN ('queued', 'retrying') 
                    AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY priority, created_at
                """, (datetime.now(),))
                
                for row in cursor.fetchall():
                    event = self._row_to_event(row)
                    self.queues[event.priority].put((event.created_at, event))
                    
        except Exception as e:
            logger.error(f"Failed to load persisted events: {e}")
    
    def _row_to_event(self, row) -> Event:
        """Convert database row to Event object"""
        return Event(
            event_id=row[0],
            event_type=EventType(row[1]),
            priority=EventPriority(row[2]),
            source=row[3],
            target=row[4],
            payload=json.loads(row[5]) if row[5] else {},
            created_at=datetime.fromisoformat(row[6]),
            expires_at=datetime.fromisoformat(row[7]) if row[7] else None,
            correlation_id=row[8],
            retry_count=row[9],
            max_retries=row[10],
            status=EventStatus(row[11]),
            processing_started=datetime.fromisoformat(row[12]) if row[12] else None,
            processing_completed=datetime.fromisoformat(row[13]) if row[13] else None,
            error_message=row[14]
        )
    
    def enqueue(self, event: Event):
        """Add event to priority queue"""
        with self.lock:
            # Check if event has expired
            if event.expires_at and event.expires_at <= datetime.now():
                logger.warning(f"Skipping expired event: {event.event_id}")
                return
            
            # Add to in-memory queue
            self.queues[event.priority].put((event.created_at, event))
            
            # Persist to database
            self._persist_event(event)
            
            logger.debug(f"Enqueued event {event.event_id} with priority {event.priority.value}")
    
    def dequeue(self, timeout: float = 1.0) -> Optional[Event]:
        """Dequeue highest priority event"""
        with self.lock:
            # Try each priority level from highest to lowest
            for priority in sorted(EventPriority, key=lambda x: x.value):
                try:
                    if not self.queues[priority].empty():
                        _, event = self.queues[priority].get(timeout=timeout)
                        event.status = EventStatus.PROCESSING
                        event.processing_started = datetime.now()
                        self._update_event_status(event)
                        return event
                except queue.Empty:
                    continue
            
            return None
    
    def complete_event(self, event: Event, success: bool, error_message: Optional[str] = None):
        """Mark event as completed"""
        with self.lock:
            event.processing_completed = datetime.now()
            event.status = EventStatus.COMPLETED if success else EventStatus.FAILED
            event.error_message = error_message
            
            self._update_event_status(event)
            
            # Add to history for analytics
            self.event_history.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "priority": event.priority.value,
                "success": success,
                "processing_time_ms": (event.processing_completed - event.processing_started).total_seconds() * 1000
                if event.processing_started else 0,
                "retry_count": event.retry_count,
                "completed_at": event.processing_completed.isoformat()
            })
            
            # Keep only last 1000 history entries
            if len(self.event_history) > 1000:
                self.event_history = self.event_history[-1000:]
    
    def retry_event(self, event: Event):
        """Retry a failed event if retries remain"""
        with self.lock:
            if event.retry_count < event.max_retries:
                event.retry_count += 1
                event.status = EventStatus.RETRYING
                event.processing_started = None
                event.processing_completed = None
                
                # Add back to queue with delay
                retry_delay = min(2 ** event.retry_count, 60)  # Exponential backoff, max 60s
                event.created_at = datetime.now() + timedelta(seconds=retry_delay)
                
                self.queues[event.priority].put((event.created_at, event))
                self._update_event_status(event)
                
                logger.info(f"Retrying event {event.event_id} (attempt {event.retry_count}/{event.max_retries})")
                return True
            else:
                logger.error(f"Event {event.event_id} exceeded max retries")
                return False
    
    def _persist_event(self, event: Event):
        """Persist event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO events VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.priority.value,
                    event.source,
                    event.target,
                    json.dumps(event.payload),
                    event.created_at.isoformat(),
                    event.expires_at.isoformat() if event.expires_at else None,
                    event.correlation_id,
                    event.retry_count,
                    event.max_retries,
                    event.status.value,
                    event.processing_started.isoformat() if event.processing_started else None,
                    event.processing_completed.isoformat() if event.processing_completed else None,
                    event.error_message
                ))
        except Exception as e:
            logger.error(f"Failed to persist event {event.event_id}: {e}")
    
    def _update_event_status(self, event: Event):
        """Update event status in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE events SET 
                        status = ?, processing_started = ?, 
                        processing_completed = ?, error_message = ?,
                        retry_count = ?
                    WHERE event_id = ?
                """, (
                    event.status.value,
                    event.processing_started.isoformat() if event.processing_started else None,
                    event.processing_completed.isoformat() if event.processing_completed else None,
                    event.error_message,
                    event.retry_count,
                    event.event_id
                ))
        except Exception as e:
            logger.error(f"Failed to update event status {event.event_id}: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            stats = {}
            for priority in EventPriority:
                stats[priority.value] = self.queues[priority].qsize()
            
            recent_history = [h for h in self.event_history 
                            if datetime.fromisoformat(h["completed_at"]) > datetime.now() - timedelta(hours=1)]
            
            return {
                "queue_sizes": stats,
                "total_queued": sum(stats.values()),
                "recent_processed": len(recent_history),
                "recent_success_rate": (len([h for h in recent_history if h["success"]]) / len(recent_history)
                                       if recent_history else 0),
                "avg_processing_time_ms": (sum(h["processing_time_ms"] for h in recent_history) / len(recent_history)
                                         if recent_history else 0)
            }

class ContextUpdateHandler(EventHandler):
    """Handler for context update events"""
    
    def __init__(self):
        super().__init__("context-update-handler", {EventType.CONTEXT_UPDATE, EventType.CACHE_INVALIDATION})
        
    async def handle_event(self, event: Event) -> bool:
        """Handle context update events"""
        try:
            if event.event_type == EventType.CONTEXT_UPDATE:
                # Update context in relevant caches
                logger.info(f"Processing context update: {event.payload.get('context_type')}")
                
                # Simulate context update processing
                await asyncio.sleep(0.01)  # 10ms processing time
                
                # Generate cache invalidation event if needed
                if event.payload.get('invalidate_cache'):
                    cache_event = Event(
                        event_id=f"cache_inv_{event.event_id}",
                        event_type=EventType.CACHE_INVALIDATION,
                        priority=EventPriority.HIGH,
                        source=self.handler_id,
                        target="cache-manager",
                        payload={
                            "cache_keys": event.payload.get('affected_keys', []),
                            "reason": f"Context update {event.event_id}"
                        },
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(minutes=5),
                        correlation_id=event.correlation_id
                    )
                    # This would be published back to the bus
                    logger.debug(f"Generated cache invalidation event: {cache_event.event_id}")
                
                self.success_count += 1
                return True
                
            elif event.event_type == EventType.CACHE_INVALIDATION:
                # Process cache invalidation
                cache_keys = event.payload.get('cache_keys', [])
                logger.info(f"Processing cache invalidation for {len(cache_keys)} keys")
                
                # Simulate cache invalidation
                await asyncio.sleep(0.005)  # 5ms processing time
                
                self.success_count += 1
                return True
                
        except Exception as e:
            logger.error(f"Context update handler error: {e}")
            self.failure_count += 1
            return False

class SystemChangeHandler(EventHandler):
    """Handler for system change events"""
    
    def __init__(self):
        super().__init__("system-change-handler", {EventType.SYSTEM_CHANGE, EventType.KNOWLEDGE_UPDATE})
        
    async def handle_event(self, event: Event) -> bool:
        """Handle system change events"""
        try:
            if event.event_type == EventType.SYSTEM_CHANGE:
                change_type = event.payload.get('change_type')
                logger.info(f"Processing system change: {change_type}")
                
                # Simulate different processing times based on change type
                if change_type == "file_change":
                    await asyncio.sleep(0.02)  # 20ms for file changes
                elif change_type == "dependency_update":
                    await asyncio.sleep(0.05)  # 50ms for dependency updates
                else:
                    await asyncio.sleep(0.01)  # 10ms for other changes
                
                self.success_count += 1
                return True
                
            elif event.event_type == EventType.KNOWLEDGE_UPDATE:
                update_type = event.payload.get('update_type')
                logger.info(f"Processing knowledge update: {update_type}")
                
                # Simulate knowledge update processing
                await asyncio.sleep(0.03)  # 30ms for knowledge updates
                
                self.success_count += 1
                return True
                
        except Exception as e:
            logger.error(f"System change handler error: {e}")
            self.failure_count += 1
            return False

class EventServiceBus:
    """Main Event Service Bus with priority-based processing"""
    
    def __init__(self, base_path: str = "/Users/cal/DEV/RIF/systems"):
        self.base_path = base_path
        self.event_queue = EventQueue(os.path.join(base_path, "events", "event_queue.db"))
        self.handlers = {}
        self.circuit_breakers = {}
        self.worker_tasks = []
        self.running = False
        self.stats = {
            "events_processed": 0,
            "events_failed": 0,
            "handler_stats": defaultdict(dict)
        }
        
        # Register default handlers
        self._register_default_handlers()
        
        # Start worker tasks
        self.worker_count = 4  # Configurable number of workers
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        context_handler = ContextUpdateHandler()
        system_handler = SystemChangeHandler()
        
        self.register_handler(context_handler)
        self.register_handler(system_handler)
    
    def register_handler(self, handler: EventHandler):
        """Register an event handler"""
        self.handlers[handler.handler_id] = handler
        self.circuit_breakers[handler.handler_id] = CircuitBreaker()
        logger.info(f"Registered handler: {handler.handler_id} for types: {[t.value for t in handler.event_types]}")
    
    def publish_event(self, event: Event):
        """Publish an event to the bus"""
        # Add to queue for processing
        self.event_queue.enqueue(event)
        logger.debug(f"Published event {event.event_id} of type {event.event_type.value}")
    
    def publish_context_update(self, context_type: str, affected_keys: List[str], 
                             priority: EventPriority = EventPriority.NORMAL,
                             correlation_id: Optional[str] = None):
        """Convenience method to publish context update events"""
        event = Event(
            event_id=f"ctx_update_{int(time.time() * 1000)}_{hashlib.md5(context_type.encode()).hexdigest()[:8]}",
            event_type=EventType.CONTEXT_UPDATE,
            priority=priority,
            source="context-service",
            target=None,
            payload={
                "context_type": context_type,
                "affected_keys": affected_keys,
                "invalidate_cache": True,
                "timestamp": datetime.now().isoformat()
            },
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=10),
            correlation_id=correlation_id
        )
        
        self.publish_event(event)
        return event.event_id
    
    def publish_system_change(self, change_type: str, details: Dict[str, Any],
                            priority: EventPriority = EventPriority.NORMAL,
                            correlation_id: Optional[str] = None):
        """Convenience method to publish system change events"""
        event = Event(
            event_id=f"sys_change_{int(time.time() * 1000)}_{hashlib.md5(change_type.encode()).hexdigest()[:8]}",
            event_type=EventType.SYSTEM_CHANGE,
            priority=priority,
            source="system-monitor",
            target=None,
            payload={
                "change_type": change_type,
                "details": details,
                "timestamp": datetime.now().isoformat()
            },
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
            correlation_id=correlation_id
        )
        
        self.publish_event(event)
        return event.event_id
    
    async def start(self):
        """Start the event service bus"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(self.worker_count):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Event Service Bus started with {self.worker_count} workers")
    
    async def stop(self):
        """Stop the event service bus"""
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Event Service Bus stopped")
    
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing events"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Dequeue next event
                event = self.event_queue.dequeue(timeout=1.0)
                if not event:
                    continue
                
                # Find appropriate handler
                handler = self._find_handler(event.event_type)
                if not handler:
                    logger.error(f"No handler found for event type {event.event_type.value}")
                    self.event_queue.complete_event(event, False, "No handler available")
                    self.stats["events_failed"] += 1
                    continue
                
                # Check circuit breaker
                circuit_breaker = self.circuit_breakers[handler.handler_id]
                if not circuit_breaker.can_execute():
                    logger.warning(f"Circuit breaker open for handler {handler.handler_id}")
                    self.event_queue.retry_event(event)
                    continue
                
                # Process event
                start_time = time.time()
                try:
                    success = await handler.handle_event(event)
                    processing_time = (time.time() - start_time) * 1000
                    
                    if success:
                        circuit_breaker.record_success()
                        self.event_queue.complete_event(event, True)
                        self.stats["events_processed"] += 1
                        logger.debug(f"Worker {worker_id} processed event {event.event_id} in {processing_time:.1f}ms")
                    else:
                        circuit_breaker.record_failure()
                        if not self.event_queue.retry_event(event):
                            self.stats["events_failed"] += 1
                        
                except Exception as e:
                    logger.error(f"Handler {handler.handler_id} error: {e}")
                    circuit_breaker.record_failure()
                    
                    if not self.event_queue.retry_event(event):
                        self.event_queue.complete_event(event, False, str(e))
                        self.stats["events_failed"] += 1
                
                # Update handler stats
                handler.processing_count += 1
                self.stats["handler_stats"][handler.handler_id] = {
                    "processing_count": handler.processing_count,
                    "success_count": handler.success_count,
                    "failure_count": handler.failure_count,
                    "circuit_breaker_state": circuit_breaker.state
                }
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _find_handler(self, event_type: EventType) -> Optional[EventHandler]:
        """Find appropriate handler for event type"""
        for handler in self.handlers.values():
            if handler.can_handle(event_type):
                return handler
        return None
    
    def get_bus_stats(self) -> Dict[str, Any]:
        """Get event service bus statistics"""
        queue_stats = self.event_queue.get_queue_stats()
        
        return {
            "running": self.running,
            "worker_count": self.worker_count,
            "events_processed": self.stats["events_processed"],
            "events_failed": self.stats["events_failed"],
            "success_rate": (self.stats["events_processed"] / 
                           max(self.stats["events_processed"] + self.stats["events_failed"], 1)),
            "queue_stats": queue_stats,
            "handler_stats": dict(self.stats["handler_stats"]),
            "circuit_breakers": {
                handler_id: breaker.state 
                for handler_id, breaker in self.circuit_breakers.items()
            }
        }

# Testing and CLI Interface
async def main():
    """Main function for testing the Event Service Bus"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Event Service Bus")
    parser.add_argument("--test", action="store_true", help="Run event processing tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Initialize event service bus
    bus = EventServiceBus()
    await bus.start()
    
    try:
        if args.test:
            print("=== Event Service Bus Test ===\n")
            
            # Publish various types of events
            context_event_id = bus.publish_context_update(
                "agent_optimization", 
                ["agent_contexts", "optimization_cache"],
                priority=EventPriority.HIGH
            )
            print(f"Published context update event: {context_event_id}")
            
            system_event_id = bus.publish_system_change(
                "file_change",
                {"file_path": "/systems/test.py", "change_type": "modified"},
                priority=EventPriority.NORMAL
            )
            print(f"Published system change event: {system_event_id}")
            
            # Publish performance alert
            perf_event = Event(
                event_id=f"perf_alert_{int(time.time())}",
                event_type=EventType.PERFORMANCE_ALERT,
                priority=EventPriority.CRITICAL,
                source="performance-monitor",
                target="alert-handler",
                payload={
                    "metric": "response_time",
                    "value": 250,
                    "threshold": 200,
                    "service": "context-optimization"
                },
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=5)
            )
            bus.publish_event(perf_event)
            print(f"Published performance alert: {perf_event.event_id}")
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Show stats
            stats = bus.get_bus_stats()
            print(f"\n=== Bus Statistics ===")
            print(f"Events processed: {stats['events_processed']}")
            print(f"Events failed: {stats['events_failed']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
            print(f"Queue stats: {stats['queue_stats']}")
            print(f"Handler stats: {json.dumps(stats['handler_stats'], indent=2)}")
            
        elif args.benchmark:
            print("=== Event Service Bus Benchmark ===\n")
            
            start_time = time.time()
            event_count = 0
            
            # Publish events continuously for specified duration
            end_time = start_time + args.duration
            
            while time.time() < end_time:
                # Mix of different event types and priorities
                event_types = [
                    ("context_update", EventPriority.NORMAL),
                    ("system_change", EventPriority.NORMAL),
                    ("context_update", EventPriority.HIGH),
                    ("performance_alert", EventPriority.CRITICAL)
                ]
                
                for event_type, priority in event_types:
                    if event_type == "context_update":
                        bus.publish_context_update(
                            f"benchmark_context_{event_count}",
                            [f"key_{event_count}"],
                            priority=priority
                        )
                    elif event_type == "system_change":
                        bus.publish_system_change(
                            "benchmark_change",
                            {"test_id": event_count},
                            priority=priority
                        )
                    elif event_type == "performance_alert":
                        alert_event = Event(
                            event_id=f"benchmark_alert_{event_count}",
                            event_type=EventType.PERFORMANCE_ALERT,
                            priority=priority,
                            source="benchmark",
                            target="test",
                            payload={"test_id": event_count},
                            created_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(minutes=1)
                        )
                        bus.publish_event(alert_event)
                    
                    event_count += 1
                
                # Brief pause to avoid overwhelming the system
                await asyncio.sleep(0.01)
            
            # Wait for all events to be processed
            print(f"Published {event_count} events, waiting for processing...")
            await asyncio.sleep(5)
            
            # Final stats
            stats = bus.get_bus_stats()
            total_time = time.time() - start_time
            
            print(f"Benchmark Results:")
            print(f"- Duration: {total_time:.2f} seconds")
            print(f"- Events published: {event_count}")
            print(f"- Events processed: {stats['events_processed']}")
            print(f"- Events failed: {stats['events_failed']}")
            print(f"- Events per second: {event_count / total_time:.1f}")
            print(f"- Processing success rate: {stats['success_rate']:.1%}")
            print(f"- Average queue processing time: {stats['queue_stats']['avg_processing_time_ms']:.1f}ms")
    
        else:
            parser.print_help()
    
    finally:
        await bus.stop()

if __name__ == "__main__":
    asyncio.run(main())