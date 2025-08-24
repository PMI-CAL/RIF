#!/usr/bin/env python3
"""
Context Request Router - Load Balancing and Prioritization
Issue #123: DPIBS Development Phase 1

Implements intelligent request routing with:
- Load balancing across service instances
- Priority-based request queuing  
- Circuit breaker patterns
- <50ms routing overhead target
- Graceful degradation strategies

Based on RIF-Architect specifications for Context Intelligence Platform.
"""

import asyncio
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from collections import deque, defaultdict
import random

# Import context platform components
from context_intelligence_platform import (
    ContextRequest, ContextResponse, ServiceStatus, 
    PerformanceMetrics, AgentType
)

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Request routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_RANDOM = "weighted_random"
    AGENT_AFFINITY = "agent_affinity"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class ServiceInstance:
    """Service instance with health tracking"""
    instance_id: str
    service_name: str
    endpoint: str
    status: ServiceStatus
    current_load: int
    max_capacity: int
    avg_response_time_ms: float
    success_rate: float
    last_health_check: datetime
    circuit_state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure: Optional[datetime] = None

@dataclass
class RequestMetrics:
    """Request routing metrics"""
    request_id: str
    agent_type: AgentType
    priority: int
    routing_start_time: float
    routing_end_time: float
    selected_instance: str
    routing_strategy: RoutingStrategy
    queue_wait_time_ms: float
    
    @property
    def routing_overhead_ms(self) -> float:
        return (self.routing_end_time - self.routing_start_time) * 1000

class PriorityQueue:
    """Priority-based request queue with fairness"""
    
    def __init__(self, max_size: int = 1000):
        self.queues = {
            1: deque(),  # High priority
            2: deque(),  # Medium priority  
            3: deque(),  # Low priority
        }
        self.max_size = max_size
        self.total_size = 0
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.metrics = {
            "enqueued": 0,
            "dequeued": 0,
            "dropped": 0,
            "wait_times": []
        }
    
    def enqueue(self, request: ContextRequest, enqueue_time: float) -> bool:
        """Enqueue request with priority"""
        with self.lock:
            if self.total_size >= self.max_size:
                # Drop lowest priority requests first
                if self.queues[3]:
                    dropped = self.queues[3].popleft()
                    self.metrics["dropped"] += 1
                    logger.warning(f"Dropped low priority request: {dropped[0].request_id}")
                elif self.queues[2]:
                    dropped = self.queues[2].popleft()
                    self.metrics["dropped"] += 1
                    logger.warning(f"Dropped medium priority request: {dropped[0].request_id}")
                else:
                    # Cannot drop high priority
                    logger.error(f"Queue full, cannot enqueue high priority request: {request.request_id}")
                    return False
            
            priority_queue = self.queues.get(request.priority, self.queues[3])
            priority_queue.append((request, enqueue_time))
            self.total_size += 1
            self.metrics["enqueued"] += 1
            
            self.not_empty.notify()
            return True
    
    def dequeue(self, timeout: float = 1.0) -> Optional[Tuple[ContextRequest, float]]:
        """Dequeue highest priority request"""
        with self.not_empty:
            if not self.not_empty.wait_for(self._has_requests, timeout):
                return None
            
            # Service high priority first, then fair round-robin for others
            for priority in [1, 2, 3]:
                if self.queues[priority]:
                    request, enqueue_time = self.queues[priority].popleft()
                    self.total_size -= 1
                    self.metrics["dequeued"] += 1
                    
                    wait_time = time.time() - enqueue_time
                    self.metrics["wait_times"].append(wait_time * 1000)
                    
                    return request, enqueue_time
            
            return None
    
    def _has_requests(self) -> bool:
        """Check if any requests are available"""
        return any(queue for queue in self.queues.values())
    
    def size(self) -> int:
        """Get total queue size"""
        with self.lock:
            return self.total_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics"""
        with self.lock:
            wait_times = self.metrics["wait_times"][-1000:]  # Last 1000 requests
            return {
                "total_size": self.total_size,
                "priority_breakdown": {p: len(q) for p, q in self.queues.items()},
                "enqueued": self.metrics["enqueued"],
                "dequeued": self.metrics["dequeued"], 
                "dropped": self.metrics["dropped"],
                "avg_wait_time_ms": sum(wait_times) / len(wait_times) if wait_times else 0,
                "p95_wait_time_ms": sorted(wait_times)[int(len(wait_times) * 0.95)] if wait_times else 0
            }

class LoadBalancer:
    """Intelligent load balancer with multiple strategies"""
    
    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED):
        self.routing_strategy = routing_strategy
        self.service_instances: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.agent_affinity: Dict[AgentType, List[str]] = {}
        
    def register_instance(self, instance: ServiceInstance):
        """Register service instance"""
        self.service_instances[instance.service_name][instance.instance_id] = instance
        logger.info(f"Registered instance {instance.instance_id} for {instance.service_name}")
    
    def select_instance(self, service_name: str, request: ContextRequest) -> Optional[ServiceInstance]:
        """Select best instance using routing strategy"""
        instances = self.service_instances.get(service_name, {})
        if not instances:
            return None
        
        # Filter healthy instances
        healthy_instances = [
            inst for inst in instances.values()
            if inst.status == ServiceStatus.RUNNING and 
               inst.circuit_state in [CircuitState.CLOSED, CircuitState.HALF_OPEN] and
               inst.current_load < inst.max_capacity
        ]
        
        if not healthy_instances:
            # Fallback to degraded instances if no healthy ones
            degraded_instances = [
                inst for inst in instances.values()
                if inst.status == ServiceStatus.DEGRADED and 
                   inst.circuit_state != CircuitState.OPEN
            ]
            healthy_instances = degraded_instances
        
        if not healthy_instances:
            return None
        
        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(service_name, healthy_instances)
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(healthy_instances)
        elif self.routing_strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_selection(healthy_instances)
        elif self.routing_strategy == RoutingStrategy.AGENT_AFFINITY:
            return self._agent_affinity_selection(request.agent_type, healthy_instances)
        else:
            return healthy_instances[0]  # Fallback
    
    def _round_robin_selection(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin instance selection"""
        counter = self.round_robin_counters[service_name]
        selected = instances[counter % len(instances)]
        self.round_robin_counters[service_name] = (counter + 1) % len(instances)
        return selected
    
    def _least_loaded_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select least loaded instance"""
        return min(instances, key=lambda x: x.current_load / x.max_capacity)
    
    def _weighted_random_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection based on capacity and performance"""
        weights = []
        for inst in instances:
            # Higher weight for lower load and better performance
            load_factor = 1.0 - (inst.current_load / inst.max_capacity)
            performance_factor = min(1.0, 200 / max(inst.avg_response_time_ms, 50))  # Better perf = higher weight
            success_factor = inst.success_rate
            
            weight = load_factor * performance_factor * success_factor
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(instances)
        
        r = random.random() * total_weight
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return instances[i]
        
        return instances[-1]  # Fallback
    
    def _agent_affinity_selection(self, agent_type: AgentType, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with agent affinity preference"""
        # Check if we have affinity mappings for this agent type
        affinity_instances = self.agent_affinity.get(agent_type, [])
        
        if affinity_instances:
            # Prefer instances with affinity
            preferred = [inst for inst in instances if inst.instance_id in affinity_instances]
            if preferred:
                return self._least_loaded_selection(preferred)
        
        # Fallback to least loaded
        return self._least_loaded_selection(instances)
    
    def update_instance_metrics(self, instance_id: str, service_name: str, 
                               load_delta: int, response_time_ms: float, success: bool):
        """Update instance metrics after request"""
        if service_name in self.service_instances and instance_id in self.service_instances[service_name]:
            instance = self.service_instances[service_name][instance_id]
            
            # Update load
            instance.current_load = max(0, instance.current_load + load_delta)
            
            # Update response time (exponential moving average)
            alpha = 0.3  # Smoothing factor
            instance.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * instance.avg_response_time_ms
            )
            
            # Update success rate (exponential moving average)
            success_value = 1.0 if success else 0.0
            instance.success_rate = (
                alpha * success_value + 
                (1 - alpha) * instance.success_rate
            )
            
            # Circuit breaker logic
            if success:
                instance.failure_count = max(0, instance.failure_count - 1)
                if instance.circuit_state == CircuitState.HALF_OPEN and instance.failure_count == 0:
                    instance.circuit_state = CircuitState.CLOSED
            else:
                instance.failure_count += 1
                instance.last_failure = datetime.now()
                
                if instance.failure_count >= 5:  # Threshold
                    instance.circuit_state = CircuitState.OPEN
                elif instance.circuit_state == CircuitState.HALF_OPEN:
                    instance.circuit_state = CircuitState.OPEN

class ContextRequestRouter:
    """Main context request router with load balancing and prioritization"""
    
    def __init__(self, max_queue_size: int = 1000, max_concurrent: int = 50):
        self.request_queue = PriorityQueue(max_queue_size)
        self.load_balancer = LoadBalancer(RoutingStrategy.LEAST_LOADED)
        self.max_concurrent = max_concurrent
        self.current_concurrent = 0
        self.routing_metrics = []
        self.performance_targets = {
            "routing_overhead_ms": 50,
            "queue_wait_time_ms": 100,
            "total_routing_time_ms": 150
        }
        
        # Start request processing workers
        self.workers_running = True
        self.workers = []
        for i in range(4):  # 4 worker threads
            worker = threading.Thread(target=self._request_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def route_request(self, request: ContextRequest, 
                     service_name: str = "agent-context-delivery") -> asyncio.Future:
        """Route request through load balancer with prioritization"""
        routing_start_time = time.time()
        
        # Create future for async response
        future = asyncio.Future()
        
        # Add routing metadata to request
        request_with_metadata = (request, service_name, future, routing_start_time)
        
        # Enqueue request
        enqueue_time = time.time()
        if not self.request_queue.enqueue(request, enqueue_time):
            future.set_exception(Exception("Request queue full"))
            return future
        
        logger.debug(f"Enqueued request {request.request_id} with priority {request.priority}")
        return future
    
    def _request_worker(self):
        """Background worker to process requests"""
        while self.workers_running:
            try:
                # Dequeue request
                result = self.request_queue.dequeue(timeout=1.0)
                if not result:
                    continue
                
                request, enqueue_time = result
                
                # Check concurrent limit
                if self.current_concurrent >= self.max_concurrent:
                    # Re-enqueue and wait
                    self.request_queue.enqueue(request, enqueue_time)
                    time.sleep(0.1)
                    continue
                
                # Process request
                self.current_concurrent += 1
                threading.Thread(
                    target=self._process_single_request, 
                    args=(request, enqueue_time),
                    daemon=True
                ).start()
                
            except Exception as e:
                logger.error(f"Request worker error: {e}")
    
    def _process_single_request(self, request: ContextRequest, enqueue_time: float):
        """Process individual request"""
        routing_start_time = time.time()
        queue_wait_time = routing_start_time - enqueue_time
        
        try:
            # Select service instance
            instance = self.load_balancer.select_instance("agent-context-delivery", request)
            
            if not instance:
                raise Exception("No healthy service instances available")
            
            routing_end_time = time.time()
            
            # Create routing metrics
            routing_metrics = RequestMetrics(
                request_id=request.request_id,
                agent_type=request.agent_type,
                priority=request.priority,
                routing_start_time=routing_start_time,
                routing_end_time=routing_end_time,
                selected_instance=instance.instance_id,
                routing_strategy=self.load_balancer.routing_strategy,
                queue_wait_time_ms=queue_wait_time * 1000
            )
            
            # Update instance load
            self.load_balancer.update_instance_metrics(
                instance.instance_id, 
                instance.service_name,
                1,  # +1 load
                0,  # Will update after response
                True
            )
            
            # Simulate request processing (in production, this would call the actual service)
            response_start_time = time.time()
            
            # Mock successful response
            time.sleep(0.05)  # Simulate 50ms processing time
            
            response_time = (time.time() - response_start_time) * 1000
            
            # Update instance metrics
            self.load_balancer.update_instance_metrics(
                instance.instance_id,
                instance.service_name,
                -1,  # -1 load (request complete)
                response_time,
                True  # Success
            )
            
            # Store routing metrics
            self.routing_metrics.append(routing_metrics)
            
            # Log performance
            total_time = (time.time() - enqueue_time) * 1000
            if routing_metrics.routing_overhead_ms > self.performance_targets["routing_overhead_ms"]:
                logger.warning(f"Routing overhead exceeded target: {routing_metrics.routing_overhead_ms:.1f}ms")
            
            logger.info(f"Routed {request.request_id} to {instance.instance_id} "
                       f"(routing: {routing_metrics.routing_overhead_ms:.1f}ms, "
                       f"queue: {routing_metrics.queue_wait_time_ms:.1f}ms, "
                       f"total: {total_time:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            
            # Update failure metrics if instance was selected
            if 'instance' in locals():
                self.load_balancer.update_instance_metrics(
                    instance.instance_id,
                    instance.service_name,
                    -1,  # -1 load
                    1000,  # High response time for failure
                    False  # Failure
                )
        
        finally:
            self.current_concurrent -= 1
    
    def register_service_instance(self, service_name: str, instance_id: str, 
                                 endpoint: str, max_capacity: int = 10):
        """Register a service instance"""
        instance = ServiceInstance(
            instance_id=instance_id,
            service_name=service_name,
            endpoint=endpoint,
            status=ServiceStatus.RUNNING,
            current_load=0,
            max_capacity=max_capacity,
            avg_response_time_ms=100.0,
            success_rate=1.0,
            last_health_check=datetime.now(),
            circuit_state=CircuitState.CLOSED,
            failure_count=0
        )
        
        self.load_balancer.register_instance(instance)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        if not self.routing_metrics:
            return {"status": "no_data"}
        
        recent_metrics = [
            m for m in self.routing_metrics[-1000:]  # Last 1000 requests
            if datetime.fromtimestamp(m.routing_start_time) > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return {"status": "no_recent_data"}
        
        routing_times = [m.routing_overhead_ms for m in recent_metrics]
        queue_times = [m.queue_wait_time_ms for m in recent_metrics]
        
        stats = {
            "total_requests_routed": len(recent_metrics),
            "avg_routing_overhead_ms": sum(routing_times) / len(routing_times),
            "p95_routing_overhead_ms": sorted(routing_times)[int(len(routing_times) * 0.95)],
            "avg_queue_wait_time_ms": sum(queue_times) / len(queue_times),
            "p95_queue_wait_time_ms": sorted(queue_times)[int(len(queue_times) * 0.95)],
            "routing_target_compliance": len([t for t in routing_times if t < self.performance_targets["routing_overhead_ms"]]) / len(routing_times),
            "current_concurrent": self.current_concurrent,
            "max_concurrent": self.max_concurrent,
            "queue_metrics": self.request_queue.get_metrics(),
            "service_instances": {
                service: {
                    instance_id: {
                        "status": inst.status.value,
                        "current_load": inst.current_load,
                        "max_capacity": inst.max_capacity,
                        "avg_response_time_ms": inst.avg_response_time_ms,
                        "success_rate": inst.success_rate,
                        "circuit_state": inst.circuit_state.value
                    }
                    for instance_id, inst in instances.items()
                }
                for service, instances in self.load_balancer.service_instances.items()
            }
        }
        
        return stats
    
    def shutdown(self):
        """Shutdown the router gracefully"""
        self.workers_running = False
        for worker in self.workers:
            worker.join(timeout=5)

# CLI and Testing Interface
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Context Request Router")
    parser.add_argument("--test", action="store_true", help="Run router test")
    parser.add_argument("--benchmark", action="store_true", help="Run routing benchmark")
    parser.add_argument("--requests", type=int, default=100, help="Number of test requests")
    
    args = parser.parse_args()
    
    # Initialize router
    router = ContextRequestRouter()
    
    # Register test service instances
    for i in range(3):
        router.register_service_instance(
            "agent-context-delivery", 
            f"instance-{i}", 
            f"http://localhost:800{i}",
            max_capacity=5
        )
    
    if args.test:
        print("=== Context Request Router Test ===\n")
        
        # Test requests with different priorities
        test_requests = []
        for i in range(10):
            request = ContextRequest(
                request_id=f"test-{i}",
                agent_type=AgentType.IMPLEMENTER,
                task_context={"description": f"Test request {i}"},
                issue_number=123,
                priority=1 if i < 3 else 2 if i < 7 else 3  # Mix priorities
            )
            test_requests.append(request)
        
        # Route requests
        start_time = time.time()
        futures = [router.route_request(req) for req in test_requests]
        
        # Wait for completion
        time.sleep(2)  # Allow processing
        
        total_time = time.time() - start_time
        print(f"Processed {len(test_requests)} requests in {total_time:.2f}s")
        
        # Print statistics
        stats = router.get_routing_stats()
        print(json.dumps(stats, indent=2))
        
    elif args.benchmark:
        print(f"=== Routing Benchmark ({args.requests} requests) ===\n")
        
        import concurrent.futures
        
        def create_test_request(i):
            return ContextRequest(
                request_id=f"bench-{i}",
                agent_type=random.choice(list(AgentType)),
                task_context={"description": f"Benchmark request {i}"},
                issue_number=123,
                priority=random.randint(1, 3)
            )
        
        # Generate test requests
        requests = [create_test_request(i) for i in range(args.requests)]
        
        # Route all requests
        start_time = time.time()
        futures = [router.route_request(req) for req in requests]
        
        # Wait for processing
        time.sleep(args.requests * 0.01 + 2)  # Estimated time + buffer
        
        total_time = time.time() - start_time
        
        print(f"Routed {len(requests)} requests in {total_time:.2f}s")
        print(f"Throughput: {len(requests) / total_time:.1f} requests/second")
        
        # Print detailed statistics
        stats = router.get_routing_stats()
        print(json.dumps(stats, indent=2))
    
    else:
        print("Context Request Router initialized")
        print("Use --test or --benchmark to run tests")
        print("Registered service instances:")
        stats = router.get_routing_stats()
        if "service_instances" in stats:
            for service, instances in stats["service_instances"].items():
                print(f"  {service}: {len(instances)} instances")
    
    # Cleanup
    router.shutdown()