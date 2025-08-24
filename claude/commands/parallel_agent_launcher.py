#!/usr/bin/env python3
"""
RIF Parallel Agent Launcher - Issue #59
System to spawn multiple agents simultaneously with resource allocation and result aggregation.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Coroutine, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Status of individual agents in parallel execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class LaunchStrategy(Enum):
    """Different strategies for launching agents"""
    PARALLEL = "parallel"          # Launch all agents simultaneously
    SEQUENTIAL = "sequential"      # Launch agents one after another
    BATCHED = "batched"           # Launch in configurable batches
    PRIORITY = "priority"         # Launch based on priority scores
    RESOURCE_AWARE = "resource_aware"  # Launch based on available resources

@dataclass
class AgentConfig:
    """Configuration for a single agent launch"""
    agent_id: str
    agent_type: str
    task_description: str
    priority: int = 50  # 0-100, higher = more priority
    max_runtime_minutes: int = 30
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 25.0
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 2
    context: Dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    expected_outputs: List[str] = field(default_factory=list)

@dataclass
class AgentResult:
    """Result from a single agent execution"""
    agent_id: str
    status: AgentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    runtime_seconds: float = 0.0
    result_data: Any = None
    error_message: str = ""
    resource_usage: Dict[str, float] = field(default_factory=dict)
    output_artifacts: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LaunchSession:
    """A complete parallel agent launch session"""
    session_id: str
    launch_strategy: LaunchStrategy
    total_agents: int
    start_time: datetime
    end_time: Optional[datetime] = None
    agents: List[AgentConfig] = field(default_factory=list)
    results: Dict[str, AgentResult] = field(default_factory=dict)
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    total_runtime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResourceMonitor:
    """Monitor and manage system resources for agent execution"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # Maximum CPU usage percentage
        self.memory_threshold = 80.0  # Maximum memory usage percentage
        self.max_concurrent_agents = 4  # Maximum number of concurrent agents
        self.resource_history: List[Dict[str, float]] = []
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval_seconds: int = 5):
        """Start resource monitoring in background thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("Resource monitoring stopped")

    def _monitor_resources(self, interval_seconds: int):
        """Background resource monitoring loop"""
        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                resource_snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024 ** 3),
                    "disk_usage_percent": psutil.disk_usage("/").percent
                }
                
                self.resource_history.append(resource_snapshot)
                
                # Keep only last 100 measurements
                if len(self.resource_history) > 100:
                    self.resource_history.pop(0)
                    
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval_seconds)

    def can_launch_agent(self, agent_config: AgentConfig) -> Tuple[bool, str]:
        """Check if resources are available to launch an agent"""
        try:
            current_cpu = psutil.cpu_percent(interval=0.1)
            current_memory = psutil.virtual_memory().percent
            
            # Check CPU availability
            if current_cpu + agent_config.cpu_limit_percent > self.cpu_threshold:
                return False, f"CPU usage too high: {current_cpu:.1f}% + {agent_config.cpu_limit_percent:.1f}% > {self.cpu_threshold}%"
            
            # Check memory availability
            memory_needed_percent = (agent_config.memory_limit_mb / 1024) / (psutil.virtual_memory().total / (1024 ** 3)) * 100
            if current_memory + memory_needed_percent > self.memory_threshold:
                return False, f"Memory usage too high: {current_memory:.1f}% + {memory_needed_percent:.1f}% > {self.memory_threshold}%"
            
            return True, "Resources available"
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False, f"Resource check error: {e}"

    def get_current_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3),
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {"error": str(e)}

    def suggest_optimal_batch_size(self) -> int:
        """Suggest optimal batch size based on current resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Conservative approach: reduce batch size if resources are high
            if cpu_percent > 70 or memory_percent > 70:
                return 1
            elif cpu_percent > 50 or memory_percent > 50:
                return 2
            elif cpu_percent > 30 or memory_percent > 30:
                return 3
            else:
                return self.max_concurrent_agents
                
        except Exception:
            return 2  # Safe default

class ParallelAgentLauncher:
    """
    Core parallel agent launcher with resource management and result aggregation
    """
    
    def __init__(self, max_concurrent_agents: int = 4):
        self.max_concurrent_agents = max_concurrent_agents
        self.resource_monitor = ResourceMonitor()
        self.active_sessions: Dict[str, LaunchSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_agents)
        self.launch_history: List[LaunchSession] = []
        
        # Performance metrics
        self.metrics = {
            "total_launches": 0,
            "successful_launches": 0,
            "failed_launches": 0,
            "average_session_time": 0.0,
            "resource_efficiency": 0.0,
            "agent_success_rates": {}
        }

    def __enter__(self):
        """Context manager entry"""
        self.resource_monitor.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.shutdown()

    def shutdown(self):
        """Graceful shutdown of launcher and resources"""
        logger.info("Shutting down parallel agent launcher")
        
        # Wait for active sessions to complete
        for session_id in list(self.active_sessions.keys()):
            self.wait_for_session_completion(session_id, timeout_seconds=30)
        
        self.executor.shutdown(wait=True)
        self.resource_monitor.stop_monitoring()

    async def launch_agents_parallel(self, agents: List[AgentConfig], 
                                   strategy: LaunchStrategy = LaunchStrategy.PARALLEL,
                                   session_metadata: Dict[str, Any] = None) -> LaunchSession:
        """
        Launch multiple agents in parallel with specified strategy
        
        Args:
            agents: List of agent configurations to launch
            strategy: Launch strategy to use
            session_metadata: Additional metadata for the session
            
        Returns:
            LaunchSession: Complete session results
        """
        session_id = str(uuid.uuid4())
        session = LaunchSession(
            session_id=session_id,
            launch_strategy=strategy,
            total_agents=len(agents),
            start_time=datetime.now(),
            agents=agents,
            metadata=session_metadata or {}
        )
        
        self.active_sessions[session_id] = session
        
        try:
            logger.info(f"Starting parallel agent launch session {session_id} with {len(agents)} agents using {strategy.value} strategy")
            
            if strategy == LaunchStrategy.PARALLEL:
                await self._launch_parallel(session)
            elif strategy == LaunchStrategy.SEQUENTIAL:
                await self._launch_sequential(session)
            elif strategy == LaunchStrategy.BATCHED:
                await self._launch_batched(session)
            elif strategy == LaunchStrategy.PRIORITY:
                await self._launch_priority_based(session)
            elif strategy == LaunchStrategy.RESOURCE_AWARE:
                await self._launch_resource_aware(session)
            else:
                raise ValueError(f"Unsupported launch strategy: {strategy}")
            
            session.end_time = datetime.now()
            session.total_runtime_seconds = (session.end_time - session.start_time).total_seconds()
            session.success_rate = self._calculate_success_rate(session)
            
            # Update metrics
            self.metrics["total_launches"] += 1
            if session.success_rate >= 0.8:  # Consider 80%+ success rate as successful
                self.metrics["successful_launches"] += 1
            else:
                self.metrics["failed_launches"] += 1
            
            self._update_performance_metrics(session)
            
            # Archive session
            self.launch_history.append(session)
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            logger.info(f"Session {session_id} completed with {session.success_rate:.1%} success rate in {session.total_runtime_seconds:.2f}s")
            
            return session
            
        except Exception as e:
            logger.error(f"Session {session_id} failed: {e}")
            session.end_time = datetime.now()
            session.total_runtime_seconds = (session.end_time - session.start_time).total_seconds()
            session.metadata["error"] = str(e)
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            raise

    async def _launch_parallel(self, session: LaunchSession):
        """Launch all agents simultaneously"""
        tasks = []
        for agent_config in session.agents:
            task = asyncio.create_task(self._launch_single_agent(agent_config, session))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            agent_id = session.agents[i].agent_id
            if isinstance(result, Exception):
                session.results[agent_id] = AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(result)
                )
            else:
                session.results[agent_id] = result

    async def _launch_sequential(self, session: LaunchSession):
        """Launch agents one after another"""
        for agent_config in session.agents:
            try:
                result = await self._launch_single_agent(agent_config, session)
                session.results[agent_config.agent_id] = result
            except Exception as e:
                session.results[agent_config.agent_id] = AgentResult(
                    agent_id=agent_config.agent_id,
                    status=AgentStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(e)
                )

    async def _launch_batched(self, session: LaunchSession):
        """Launch agents in batches based on resource availability"""
        batch_size = self.resource_monitor.suggest_optimal_batch_size()
        logger.info(f"Using batch size: {batch_size}")
        
        for i in range(0, len(session.agents), batch_size):
            batch = session.agents[i:i + batch_size]
            
            # Launch batch in parallel
            tasks = [self._launch_single_agent(agent_config, session) for agent_config in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                agent_id = batch[j].agent_id
                if isinstance(result, Exception):
                    session.results[agent_id] = AgentResult(
                        agent_id=agent_id,
                        status=AgentStatus.FAILED,
                        start_time=datetime.now(),
                        error_message=str(result)
                    )
                else:
                    session.results[agent_id] = result
            
            # Brief pause between batches to allow resource recovery
            if i + batch_size < len(session.agents):
                await asyncio.sleep(2)

    async def _launch_priority_based(self, session: LaunchSession):
        """Launch agents based on priority scores"""
        # Sort agents by priority (highest first)
        sorted_agents = sorted(session.agents, key=lambda a: a.priority, reverse=True)
        
        # Launch high priority agents first, then batch the rest
        high_priority_agents = [a for a in sorted_agents if a.priority >= 80]
        medium_priority_agents = [a for a in sorted_agents if 50 <= a.priority < 80]
        low_priority_agents = [a for a in sorted_agents if a.priority < 50]
        
        # Launch high priority sequentially
        for agent_config in high_priority_agents:
            result = await self._launch_single_agent(agent_config, session)
            session.results[agent_config.agent_id] = result
        
        # Launch medium priority in small batches
        batch_size = min(2, self.resource_monitor.suggest_optimal_batch_size())
        for agents_batch in [medium_priority_agents[i:i + batch_size] 
                           for i in range(0, len(medium_priority_agents), batch_size)]:
            tasks = [self._launch_single_agent(agent_config, session) for agent_config in agents_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                agent_id = agents_batch[j].agent_id
                session.results[agent_id] = result if not isinstance(result, Exception) else AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(result)
                )
        
        # Launch low priority in larger batches if resources allow
        batch_size = self.resource_monitor.suggest_optimal_batch_size()
        for agents_batch in [low_priority_agents[i:i + batch_size] 
                           for i in range(0, len(low_priority_agents), batch_size)]:
            tasks = [self._launch_single_agent(agent_config, session) for agent_config in agents_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                agent_id = agents_batch[j].agent_id
                session.results[agent_id] = result if not isinstance(result, Exception) else AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(result)
                )

    async def _launch_resource_aware(self, session: LaunchSession):
        """Launch agents based on available system resources"""
        remaining_agents = session.agents.copy()
        
        while remaining_agents:
            # Find agents that can be launched with current resources
            launchable_agents = []
            for agent_config in remaining_agents:
                can_launch, reason = self.resource_monitor.can_launch_agent(agent_config)
                if can_launch:
                    launchable_agents.append(agent_config)
            
            if not launchable_agents:
                logger.warning("No agents can be launched with current resources. Waiting...")
                await asyncio.sleep(5)  # Wait for resources to free up
                continue
            
            # Launch the agents that can run
            logger.info(f"Launching {len(launchable_agents)} agents based on resource availability")
            tasks = [self._launch_single_agent(agent_config, session) for agent_config in launchable_agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                agent_id = launchable_agents[i].agent_id
                session.results[agent_id] = result if not isinstance(result, Exception) else AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(result)
                )
            
            # Remove launched agents from remaining list
            for agent_config in launchable_agents:
                remaining_agents.remove(agent_config)
            
            # Brief pause before checking resources again
            if remaining_agents:
                await asyncio.sleep(3)

    async def _launch_single_agent(self, agent_config: AgentConfig, session: LaunchSession) -> AgentResult:
        """
        Launch a single agent with resource monitoring and error handling
        
        Args:
            agent_config: Configuration for the agent to launch
            session: Parent launch session
            
        Returns:
            AgentResult: Result of agent execution
        """
        start_time = datetime.now()
        agent_result = AgentResult(
            agent_id=agent_config.agent_id,
            status=AgentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            logger.info(f"Launching agent {agent_config.agent_id} of type {agent_config.agent_type}")
            
            # Check resource availability
            can_launch, reason = self.resource_monitor.can_launch_agent(agent_config)
            if not can_launch:
                raise RuntimeError(f"Cannot launch agent: {reason}")
            
            # Record initial resource usage
            initial_resources = self.resource_monitor.get_current_resources()
            
            # Simulate agent execution (in real implementation, this would call actual agents)
            result_data = await self._simulate_agent_execution(agent_config)
            
            # Record final resource usage
            final_resources = self.resource_monitor.get_current_resources()
            
            agent_result.status = AgentStatus.COMPLETED
            agent_result.result_data = result_data
            agent_result.confidence_score = result_data.get("confidence", 0.8)
            agent_result.output_artifacts = result_data.get("artifacts", [])
            agent_result.resource_usage = {
                "cpu_delta": final_resources.get("cpu_percent", 0) - initial_resources.get("cpu_percent", 0),
                "memory_delta": final_resources.get("memory_percent", 0) - initial_resources.get("memory_percent", 0)
            }
            
        except asyncio.TimeoutError:
            agent_result.status = AgentStatus.TIMEOUT
            agent_result.error_message = f"Agent execution timed out after {agent_config.max_runtime_minutes} minutes"
            logger.warning(f"Agent {agent_config.agent_id} timed out")
            
        except Exception as e:
            agent_result.status = AgentStatus.FAILED
            agent_result.error_message = str(e)
            logger.error(f"Agent {agent_config.agent_id} failed: {e}")
        
        finally:
            agent_result.end_time = datetime.now()
            agent_result.runtime_seconds = (agent_result.end_time - agent_result.start_time).total_seconds()
        
        return agent_result

    async def _simulate_agent_execution(self, agent_config: AgentConfig) -> Dict[str, Any]:
        """
        Simulate agent execution (placeholder for actual agent integration)
        
        In real implementation, this would:
        1. Create the appropriate agent instance
        2. Set up agent context and resources
        3. Execute the agent's task
        4. Collect results and artifacts
        5. Handle agent-specific error conditions
        """
        # Simulate work time based on agent complexity
        work_time = min(agent_config.max_runtime_minutes * 60, 30)  # Cap at 30 seconds for demo
        await asyncio.sleep(work_time / 10)  # Scaled down for demo
        
        # Simulate different outcomes based on agent type
        if agent_config.agent_type == "rif-validator":
            return {
                "validation_results": {"tests_passed": 95, "coverage": 0.85, "quality_score": 0.9},
                "confidence": 0.9,
                "artifacts": [f"validation_report_{agent_config.agent_id}.json"]
            }
        elif agent_config.agent_type == "rif-implementer":
            return {
                "implementation_status": "completed",
                "files_modified": 5,
                "lines_changed": 150,
                "confidence": 0.85,
                "artifacts": [f"implementation_summary_{agent_config.agent_id}.md"]
            }
        elif agent_config.agent_type == "rif-security":
            return {
                "security_scan": {"vulnerabilities": 0, "warnings": 2, "score": 0.95},
                "compliance_check": "passed",
                "confidence": 0.95,
                "artifacts": [f"security_report_{agent_config.agent_id}.json"]
            }
        else:
            return {
                "task_status": "completed",
                "confidence": 0.8,
                "artifacts": [f"agent_output_{agent_config.agent_id}.json"]
            }

    def aggregate_results(self, session: LaunchSession) -> Dict[str, Any]:
        """
        Aggregate results from all agents in a session
        
        Args:
            session: Completed launch session
            
        Returns:
            Dict: Aggregated results and metrics
        """
        if not session.results:
            return {"error": "No results to aggregate"}
        
        results = list(session.results.values())
        
        # Status aggregation
        status_counts = {}
        for status in AgentStatus:
            status_counts[status.value] = sum(1 for r in results if r.status == status)
        
        # Performance aggregation
        successful_results = [r for r in results if r.status == AgentStatus.COMPLETED]
        
        avg_runtime = sum(r.runtime_seconds for r in results) / len(results) if results else 0
        avg_confidence = sum(r.confidence_score for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Resource usage aggregation
        total_cpu_delta = sum(r.resource_usage.get("cpu_delta", 0) for r in results)
        total_memory_delta = sum(r.resource_usage.get("memory_delta", 0) for r in results)
        
        # Quality metrics
        quality_scores = []
        for result in successful_results:
            if result.result_data and isinstance(result.result_data, dict):
                if "quality_score" in result.result_data:
                    quality_scores.append(result.result_data["quality_score"])
                elif "validation_results" in result.result_data:
                    quality_scores.append(result.result_data["validation_results"].get("quality_score", 0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "session_summary": {
                "session_id": session.session_id,
                "total_agents": session.total_agents,
                "success_rate": session.success_rate,
                "total_runtime_seconds": session.total_runtime_seconds,
                "launch_strategy": session.launch_strategy.value
            },
            "status_distribution": status_counts,
            "performance_metrics": {
                "average_runtime_seconds": avg_runtime,
                "average_confidence_score": avg_confidence,
                "average_quality_score": avg_quality,
                "resource_efficiency": (len(successful_results) / len(results)) if results else 0
            },
            "resource_usage": {
                "total_cpu_delta": total_cpu_delta,
                "total_memory_delta": total_memory_delta,
                "resource_efficiency_score": self._calculate_resource_efficiency(session)
            },
            "artifacts": {
                "total_artifacts": sum(len(r.output_artifacts) for r in successful_results),
                "artifact_list": [artifact for r in successful_results for artifact in r.output_artifacts]
            },
            "recommendations": self._generate_session_recommendations(session)
        }

    def _calculate_success_rate(self, session: LaunchSession) -> float:
        """Calculate success rate for a session"""
        if not session.results:
            return 0.0
        
        successful_count = sum(1 for result in session.results.values() 
                             if result.status == AgentStatus.COMPLETED)
        return successful_count / len(session.results)

    def _calculate_resource_efficiency(self, session: LaunchSession) -> float:
        """Calculate resource efficiency score"""
        if not session.results:
            return 0.0
        
        successful_results = [r for r in session.results.values() if r.status == AgentStatus.COMPLETED]
        if not successful_results:
            return 0.0
        
        # Simple efficiency: successful agents / total runtime
        efficiency = len(successful_results) / session.total_runtime_seconds if session.total_runtime_seconds > 0 else 0
        return min(1.0, efficiency * 100)  # Normalize to 0-1 scale

    def _update_performance_metrics(self, session: LaunchSession):
        """Update overall launcher performance metrics"""
        # Update average session time
        total_sessions = len(self.launch_history) + 1  # +1 for current session
        current_avg = self.metrics["average_session_time"]
        self.metrics["average_session_time"] = (
            (current_avg * (total_sessions - 1) + session.total_runtime_seconds) / total_sessions
        )
        
        # Update agent-specific success rates
        for agent_config in session.agents:
            agent_type = agent_config.agent_type
            if agent_type not in self.metrics["agent_success_rates"]:
                self.metrics["agent_success_rates"][agent_type] = {"successes": 0, "attempts": 0}
            
            self.metrics["agent_success_rates"][agent_type]["attempts"] += 1
            if agent_config.agent_id in session.results:
                result = session.results[agent_config.agent_id]
                if result.status == AgentStatus.COMPLETED:
                    self.metrics["agent_success_rates"][agent_type]["successes"] += 1

    def _generate_session_recommendations(self, session: LaunchSession) -> List[str]:
        """Generate recommendations for improving future sessions"""
        recommendations = []
        
        if session.success_rate < 0.7:
            recommendations.append("Consider using sequential launch strategy for better error handling")
        
        if session.total_runtime_seconds > 300:  # 5 minutes
            recommendations.append("Session took longer than expected. Consider breaking into smaller batches")
        
        failed_agents = [r for r in session.results.values() if r.status == AgentStatus.FAILED]
        if len(failed_agents) > len(session.results) * 0.2:  # More than 20% failed
            recommendations.append("High failure rate detected. Review agent configurations and resource limits")
        
        timeout_agents = [r for r in session.results.values() if r.status == AgentStatus.TIMEOUT]
        if timeout_agents:
            recommendations.append("Some agents timed out. Consider increasing timeout limits or optimizing agent tasks")
        
        return recommendations

    def wait_for_session_completion(self, session_id: str, timeout_seconds: int = 300) -> Optional[LaunchSession]:
        """
        Wait for a session to complete with optional timeout
        
        Args:
            session_id: ID of session to wait for
            timeout_seconds: Maximum time to wait
            
        Returns:
            LaunchSession: Completed session or None if timeout
        """
        start_time = time.time()
        
        while session_id in self.active_sessions:
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Timeout waiting for session {session_id}")
                return None
            
            time.sleep(1)
        
        # Look for completed session in history
        for session in self.launch_history:
            if session.session_id == session_id:
                return session
        
        return None

    def get_launcher_metrics(self) -> Dict[str, Any]:
        """Get current launcher performance metrics"""
        return {
            **self.metrics,
            "active_sessions": len(self.active_sessions),
            "session_history_count": len(self.launch_history),
            "current_resources": self.resource_monitor.get_current_resources()
        }


# Convenience functions for common use cases

def create_agent_config(agent_id: str, agent_type: str, task_description: str, 
                       priority: int = 50, **kwargs) -> AgentConfig:
    """Create agent configuration with sensible defaults"""
    return AgentConfig(
        agent_id=agent_id,
        agent_type=agent_type,
        task_description=task_description,
        priority=priority,
        **kwargs
    )

def create_rif_agents_for_issue(issue_number: int, issue_title: str, issue_complexity: str = "medium") -> List[AgentConfig]:
    """Create standard RIF agent configurations for an issue"""
    agents = []
    
    # Adjust based on complexity
    if issue_complexity == "high":
        agents.extend([
            create_agent_config(
                f"rif-analyst-{issue_number}",
                "rif-analyst",
                f"Analyze requirements for issue #{issue_number}: {issue_title}",
                priority=90,
                max_runtime_minutes=20
            ),
            create_agent_config(
                f"rif-architect-{issue_number}",
                "rif-architect", 
                f"Design architecture for issue #{issue_number}: {issue_title}",
                priority=85,
                max_runtime_minutes=30
            )
        ])
    
    # Standard agents for all issues
    agents.extend([
        create_agent_config(
            f"rif-implementer-{issue_number}",
            "rif-implementer",
            f"Implement solution for issue #{issue_number}: {issue_title}",
            priority=70,
            max_runtime_minutes=45
        ),
        create_agent_config(
            f"rif-validator-{issue_number}",
            "rif-validator",
            f"Validate implementation for issue #{issue_number}: {issue_title}",
            priority=80,
            max_runtime_minutes=20
        )
    ])
    
    # Add security agent for security-related issues
    if "security" in issue_title.lower() or "auth" in issue_title.lower():
        agents.append(
            create_agent_config(
                f"rif-security-{issue_number}",
                "rif-security",
                f"Security review for issue #{issue_number}: {issue_title}",
                priority=95,
                max_runtime_minutes=25
            )
        )
    
    return agents

async def main():
    """Demonstration of parallel agent launcher"""
    with ParallelAgentLauncher(max_concurrent_agents=4) as launcher:
        
        # Example 1: Launch agents for a medium complexity issue
        print("\n=== Example 1: Medium Complexity Issue ===")
        agents = create_rif_agents_for_issue(58, "Design consensus architecture", "medium")
        
        session = await launcher.launch_agents_parallel(
            agents,
            LaunchStrategy.PRIORITY,
            {"issue_number": 58, "complexity": "medium"}
        )
        
        aggregated_results = launcher.aggregate_results(session)
        print(f"Session completed with {aggregated_results['session_summary']['success_rate']:.1%} success rate")
        print(f"Total runtime: {aggregated_results['session_summary']['total_runtime_seconds']:.2f}s")
        
        # Example 2: Resource-aware launching
        print("\n=== Example 2: Resource-Aware Strategy ===")
        agents = create_rif_agents_for_issue(59, "Parallel agent launcher", "high")
        
        session = await launcher.launch_agents_parallel(
            agents,
            LaunchStrategy.RESOURCE_AWARE,
            {"issue_number": 59, "complexity": "high"}
        )
        
        aggregated_results = launcher.aggregate_results(session)
        print(f"Resource-aware session: {aggregated_results['session_summary']['success_rate']:.1%} success")
        print(f"Resource efficiency: {aggregated_results['resource_usage']['resource_efficiency_score']:.2f}")
        
        # Print launcher metrics
        print(f"\n=== Launcher Metrics ===")
        metrics = launcher.get_launcher_metrics()
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"{key}: {json.dumps(value, indent=2)}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())