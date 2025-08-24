"""
Dynamic MCP Loader

Automatically detects project requirements and loads appropriate MCP servers
with security validation and resource optimization.

Issue: #82 - Implement dynamic MCP loader
Agent: RIF-Implementer
"""

import asyncio
import logging
import time
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .requirement_detector import RequirementDetector
from .server_mapper import ServerMapper
from ..registry.server_registry import MCPServerRegistry
from ..security.security_gateway import SecurityGateway
from ..monitor.health_monitor import MCPHealthMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Context information about the project requiring MCP server integration"""
    project_path: str
    technology_stack: Set[str]
    integrations: Set[str]
    needs_reasoning: bool
    needs_memory: bool
    needs_database: bool
    needs_cloud: bool
    complexity: str  # low, medium, high, very-high
    agent_type: Optional[str] = None
    performance_requirements: Optional[Dict[str, Any]] = None


@dataclass
class ServerLoadResult:
    """Result of loading an MCP server"""
    server_id: str
    server_name: str
    status: str  # success, failed, skipped
    load_time_ms: int
    error_message: Optional[str] = None
    health_status: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None


class DynamicMCPLoader:
    """
    Dynamic MCP server loader that detects project requirements
    and loads appropriate servers with security and resource validation.
    
    Features:
    - Automatic requirement detection from project context
    - Intelligent server mapping based on capabilities
    - Security validation through MCP Security Gateway
    - Resource optimization with load balancing
    - Health monitoring and graceful error handling
    """
    
    def __init__(self, 
                 server_registry: Optional[MCPServerRegistry] = None,
                 security_gateway: Optional[SecurityGateway] = None,
                 health_monitor: Optional[MCPHealthMonitor] = None,
                 max_concurrent_loads: int = 4,
                 resource_budget_mb: int = 512):
        """
        Initialize the Dynamic MCP Loader
        
        Args:
            server_registry: Server registry for capability lookup
            security_gateway: Security validation gateway
            health_monitor: Health monitoring system
            max_concurrent_loads: Maximum concurrent server loads
            resource_budget_mb: Total memory budget for loaded servers
        """
        self.server_registry = server_registry or MCPServerRegistry()
        self.security_gateway = security_gateway or SecurityGateway()
        self.health_monitor = health_monitor or MCPHealthMonitor()
        
        self.requirement_detector = RequirementDetector()
        self.server_mapper = ServerMapper(self.server_registry)
        
        # Resource management
        self.max_concurrent_loads = max_concurrent_loads
        self.resource_budget_mb = resource_budget_mb
        self.load_semaphore = asyncio.Semaphore(max_concurrent_loads)
        
        # State tracking
        self.active_servers: Dict[str, Any] = {}
        self.server_configs: Dict[str, Dict[str, Any]] = {}  # Store configs for active servers
        self.loading_history: List[ServerLoadResult] = []
        self.total_resource_usage_mb = 0
        
        logger.info(f"DynamicMCPLoader initialized with {max_concurrent_loads} concurrent loads, "
                   f"{resource_budget_mb}MB resource budget")
    
    async def load_servers_for_project(self, project_context: ProjectContext) -> List[ServerLoadResult]:
        """
        Load MCP servers based on project requirements
        
        Args:
            project_context: Project analysis context
            
        Returns:
            List of server load results
        """
        logger.info(f"Loading servers for project: {project_context.project_path}")
        start_time = time.time()
        
        try:
            # Step 1: Detect requirements from project context
            requirements = await self.detect_requirements(project_context)
            logger.info(f"Detected requirements: {requirements}")
            
            # Step 2: Map requirements to server configurations
            required_servers = await self.map_requirements_to_servers(requirements, project_context)
            logger.info(f"Mapped to {len(required_servers)} servers")
            
            # Step 3: Load servers with validation and optimization
            load_results = await self._load_servers_parallel(required_servers)
            
            # Step 4: Update loading history and metrics
            total_time_ms = int((time.time() - start_time) * 1000)
            self.loading_history.extend(load_results)
            await self._record_loading_session(project_context, requirements, load_results, total_time_ms)
            
            successful_loads = [r for r in load_results if r.status == "success"]
            logger.info(f"Successfully loaded {len(successful_loads)}/{len(required_servers)} servers "
                       f"in {total_time_ms}ms")
            
            return load_results
            
        except Exception as e:
            logger.error(f"Failed to load servers for project: {e}")
            return [ServerLoadResult(
                server_id="unknown",
                server_name="failed_load",
                status="failed",
                load_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e)
            )]
    
    async def detect_requirements(self, project_context: ProjectContext) -> Set[str]:
        """
        Detect MCP server requirements from project context
        
        Args:
            project_context: Project analysis context
            
        Returns:
            Set of requirement identifiers
        """
        requirements = set()
        
        # Technology stack analysis
        if 'github' in project_context.integrations or 'git' in project_context.technology_stack:
            requirements.add('github_mcp')
            requirements.add('git_mcp')
        
        if 'nodejs' in project_context.technology_stack or 'javascript' in project_context.technology_stack:
            requirements.add('filesystem_mcp')
            requirements.add('nodejs_tools')
            
        if 'python' in project_context.technology_stack:
            requirements.add('filesystem_mcp') 
            requirements.add('python_tools')
            
        if 'docker' in project_context.technology_stack:
            requirements.add('docker_mcp')
            
        # Capability-based requirements
        if project_context.needs_reasoning:
            requirements.add('sequential_thinking')
            
        if project_context.needs_memory:
            requirements.add('memory_mcp')
            
        if project_context.needs_database:
            requirements.add('database_mcp')
            
        if project_context.needs_cloud:
            requirements.add('aws_mcp')
            requirements.add('azure_mcp')
            
        # Agent-specific requirements
        if project_context.agent_type == 'rif-analyst':
            requirements.add('analysis_tools')
            requirements.add('pattern_recognition')
            
        elif project_context.agent_type == 'rif-implementer':
            requirements.add('code_generation')
            requirements.add('testing_tools')
            
        elif project_context.agent_type == 'rif-validator':
            requirements.add('quality_gates')
            requirements.add('security_scanner')
            
        # Complexity-based requirements
        if project_context.complexity in ['high', 'very-high']:
            requirements.add('performance_monitoring')
            requirements.add('resource_optimization')
            
        return requirements
    
    async def map_requirements_to_servers(self, requirements: Set[str], 
                                        project_context: ProjectContext) -> List[Dict[str, Any]]:
        """
        Map requirements to specific MCP server configurations
        
        Args:
            requirements: Set of requirement identifiers
            project_context: Project context for optimization
            
        Returns:
            List of server configurations to load
        """
        return await self.server_mapper.map_requirements_to_servers(requirements, project_context)
    
    async def validate_resources(self, server_config: Dict[str, Any]) -> bool:
        """
        Validate that server can be loaded within resource constraints
        
        Args:
            server_config: Server configuration to validate
            
        Returns:
            True if server can be loaded, False otherwise
        """
        try:
            # Check memory requirements
            required_memory = server_config.get('resource_requirements', {}).get('memory_mb', 64)
            if self.total_resource_usage_mb + required_memory > self.resource_budget_mb:
                logger.warning(f"Server {server_config['server_id']} requires {required_memory}MB "
                              f"but only {self.resource_budget_mb - self.total_resource_usage_mb}MB available")
                return False
            
            # Check if server is already loaded
            if server_config['server_id'] in self.active_servers:
                logger.info(f"Server {server_config['server_id']} already loaded, skipping")
                return False
            
            # Validate dependencies
            dependencies = server_config.get('dependencies', [])
            for dep in dependencies:
                if not await self._validate_dependency(dep):
                    logger.warning(f"Server {server_config['server_id']} dependency {dep} not satisfied")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Resource validation failed for {server_config.get('server_id')}: {e}")
            return False
    
    async def load_server(self, server_config: Dict[str, Any]) -> ServerLoadResult:
        """
        Load a single MCP server with security validation and health checks
        
        Args:
            server_config: Server configuration
            
        Returns:
            Server load result
        """
        server_id = server_config['server_id']
        server_name = server_config['name']
        start_time = time.time()
        
        try:
            logger.info(f"Loading server: {server_name} ({server_id})")
            
            # Step 1: Security validation
            if not await self.security_gateway.validate_server_security(server_config):
                return ServerLoadResult(
                    server_id=server_id,
                    server_name=server_name,
                    status="failed",
                    load_time_ms=int((time.time() - start_time) * 1000),
                    error_message="Security validation failed"
                )
            
            # Step 2: Resource validation
            if not await self.validate_resources(server_config):
                return ServerLoadResult(
                    server_id=server_id,
                    server_name=server_name,
                    status="skipped",
                    load_time_ms=int((time.time() - start_time) * 1000),
                    error_message="Resource validation failed"
                )
            
            # Step 3: Initialize server
            server = await self._initialize_server(server_config)
            
            # Step 4: Health check
            health_status = await server.health_check()
            if health_status != "healthy":
                await server.cleanup()
                return ServerLoadResult(
                    server_id=server_id,
                    server_name=server_name,
                    status="failed",
                    load_time_ms=int((time.time() - start_time) * 1000),
                    error_message=f"Health check failed: {health_status}"
                )
            
            # Step 5: Register with systems
            await self._register_server_with_systems(server, server_config)
            
            # Step 6: Update resource tracking
            memory_usage = server_config.get('resource_requirements', {}).get('memory_mb', 64)
            self.total_resource_usage_mb += memory_usage
            self.active_servers[server_id] = server
            self.server_configs[server_id] = server_config
            
            load_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Successfully loaded {server_name} in {load_time_ms}ms")
            
            return ServerLoadResult(
                server_id=server_id,
                server_name=server_name,
                status="success",
                load_time_ms=load_time_ms,
                health_status=health_status,
                resource_usage={
                    "memory_mb": memory_usage,
                    "cpu_percent": server_config.get('resource_requirements', {}).get('cpu_percent', 5)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load server {server_name}: {e}")
            return ServerLoadResult(
                server_id=server_id,
                server_name=server_name,
                status="failed",
                load_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e)
            )
    
    async def _load_servers_parallel(self, server_configs: List[Dict[str, Any]]) -> List[ServerLoadResult]:
        """
        Load multiple servers in parallel with resource management
        
        Args:
            server_configs: List of server configurations
            
        Returns:
            List of server load results
        """
        async def load_with_semaphore(config):
            async with self.load_semaphore:
                return await self.load_server(config)
        
        # Create tasks for parallel execution
        tasks = [load_with_semaphore(config) for config in server_configs]
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for all loads
            )
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    config = server_configs[i]
                    processed_results.append(ServerLoadResult(
                        server_id=config['server_id'],
                        server_name=config['name'],
                        status="failed",
                        load_time_ms=30000,  # Timeout
                        error_message=f"Load timeout or exception: {str(result)}"
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            logger.error("Server loading timed out after 30 seconds")
            return [ServerLoadResult(
                server_id=config['server_id'],
                server_name=config['name'],
                status="failed",
                load_time_ms=30000,
                error_message="Load timeout"
            ) for config in server_configs]
    
    async def _initialize_server(self, server_config: Dict[str, Any]) -> Any:
        """
        Initialize MCP server instance
        
        Args:
            server_config: Server configuration
            
        Returns:
            Initialized server instance
        """
        # This is a placeholder for actual MCP server initialization
        # In a real implementation, this would:
        # 1. Import the appropriate MCP server module
        # 2. Configure it with the provided settings
        # 3. Start the server process
        # 4. Return a client interface
        
        from ..mock.mock_server import MockMCPServer
        server = MockMCPServer(server_config)
        await server.initialize()
        return server
    
    async def _register_server_with_systems(self, server: Any, server_config: Dict[str, Any]):
        """
        Register loaded server with monitoring and context systems
        
        Args:
            server: Loaded server instance
            server_config: Server configuration
        """
        # Register with health monitor
        await self.health_monitor.register_server(server, server_config)
        
        # TODO: Register with context aggregator when implemented
        logger.debug(f"Registered {server_config['server_id']} with monitoring systems")
    
    async def _validate_dependency(self, dependency: str) -> bool:
        """
        Validate that a dependency is available
        
        Args:
            dependency: Dependency identifier
            
        Returns:
            True if dependency is satisfied
        """
        # Check common system dependencies
        if dependency == 'git':
            import shutil
            return shutil.which('git') is not None
        elif dependency == 'gh_cli':
            import shutil
            return shutil.which('gh') is not None
        elif dependency == 'docker':
            import shutil
            return shutil.which('docker') is not None
        
        # Check if dependency is an already loaded server
        return dependency in self.active_servers
    
    async def _record_loading_session(self, project_context: ProjectContext, 
                                    requirements: Set[str], 
                                    load_results: List[ServerLoadResult],
                                    total_time_ms: int):
        """
        Record loading session for metrics and learning
        
        Args:
            project_context: Project context
            requirements: Detected requirements
            load_results: Server load results
            total_time_ms: Total loading time
        """
        session_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "project_path": project_context.project_path,
            "requirements": list(requirements),
            "servers_attempted": len(load_results),
            "servers_successful": len([r for r in load_results if r.status == "success"]),
            "total_time_ms": total_time_ms,
            "resource_usage_mb": self.total_resource_usage_mb,
            "load_results": [
                {
                    "server_id": r.server_id,
                    "status": r.status,
                    "load_time_ms": r.load_time_ms,
                    "error": r.error_message
                }
                for r in load_results
            ]
        }
        
        # Store in knowledge base for learning
        import json
        import os
        
        metrics_dir = "/Users/cal/DEV/RIF/knowledge/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = os.path.join(metrics_dir, "mcp-loader-sessions.jsonl")
        with open(metrics_file, "a") as f:
            f.write(json.dumps(session_record) + "\n")
        
        logger.debug(f"Recorded loading session: {len(load_results)} servers in {total_time_ms}ms")
    
    # Management methods
    
    async def unload_server(self, server_id: str) -> bool:
        """
        Unload a specific MCP server
        
        Args:
            server_id: Server identifier to unload
            
        Returns:
            True if successfully unloaded
        """
        if server_id not in self.active_servers:
            logger.warning(f"Server {server_id} not found in active servers")
            return False
        
        try:
            server = self.active_servers[server_id]
            await server.cleanup()
            
            # Update resource tracking
            server_config = self.server_configs.get(server_id, {})
            memory_usage = server_config.get('resource_requirements', {}).get('memory_mb', 64)
            self.total_resource_usage_mb = max(0, self.total_resource_usage_mb - memory_usage)
            
            del self.active_servers[server_id]
            if server_id in self.server_configs:
                del self.server_configs[server_id]
            await self.health_monitor.unregister_server(server_id)
            
            logger.info(f"Successfully unloaded server {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload server {server_id}: {e}")
            return False
    
    async def get_active_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about currently active servers
        
        Returns:
            Dictionary of active server information
        """
        active_info = {}
        for server_id, server in self.active_servers.items():
            try:
                health = await server.health_check()
                server_config = self.server_configs.get(server_id, {})
                
                active_info[server_id] = {
                    "name": server_config.get('name', 'Unknown'),
                    "health": health,
                    "resource_usage": server_config.get('resource_requirements', {})
                }
            except Exception as e:
                active_info[server_id] = {
                    "name": "Unknown",
                    "health": "error",
                    "error": str(e)
                }
        
        return active_info
    
    async def get_loading_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for server loading
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.loading_history:
            return {"message": "No loading history available"}
        
        successful_loads = [r for r in self.loading_history if r.status == "success"]
        
        return {
            "total_load_attempts": len(self.loading_history),
            "successful_loads": len(successful_loads),
            "success_rate": len(successful_loads) / len(self.loading_history) * 100,
            "average_load_time_ms": sum(r.load_time_ms for r in successful_loads) / len(successful_loads) if successful_loads else 0,
            "active_servers": len(self.active_servers),
            "total_resource_usage_mb": self.total_resource_usage_mb,
            "resource_budget_mb": self.resource_budget_mb,
            "resource_utilization_percent": (self.total_resource_usage_mb / self.resource_budget_mb) * 100
        }