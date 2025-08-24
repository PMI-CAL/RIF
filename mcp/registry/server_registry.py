"""
MCP Server Registry

Full implementation for server catalog management, capability tracking,
and version management.

Features:
- Dynamic server registration and cataloging
- Capability-based indexing for fast lookups  
- Version management with compatibility tracking
- Health status integration
- Thread-safe concurrent operations
- Persistent storage for registry state

Issue: #81 - Create MCP server registry
Component: Enterprise server registry with capability tracking
"""

import logging
import asyncio
import json
import threading
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ServerRecord:
    """Registry record for an MCP server"""
    server_id: str
    name: str
    version: str
    capabilities: List[str]
    resource_requirements: Dict[str, Any]
    health_status: str = "unknown"
    last_updated: Optional[datetime] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    registered_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    total_health_checks: int = 0
    failed_health_checks: int = 0


class MCPServerRegistry:
    """
    Enterprise-grade MCP server registry with comprehensive capability tracking,
    version management, and health integration.
    
    Features:
    - Thread-safe server registration and management
    - Capability-based indexing for efficient server discovery
    - Version compatibility tracking and management
    - Health status integration with monitoring systems
    - Persistent storage for registry state and recovery
    - Advanced querying and filtering capabilities
    - Performance metrics and usage analytics
    
    Thread Safety:
    All public methods are thread-safe and can be called concurrently
    from multiple threads or async tasks.
    """
    
    def __init__(self, registry_file: Optional[str] = None, auto_save: bool = True):
        """
        Initialize the server registry
        
        Args:
            registry_file: Path to persistent registry storage file
            auto_save: Whether to automatically save changes to disk
        """
        # Core data structures
        self.servers: Dict[str, ServerRecord] = {}
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set) 
        self.version_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Persistence settings
        self.registry_file = Path(registry_file) if registry_file else None
        self.auto_save = auto_save
        
        # Health monitoring integration
        self.health_callback: Optional[Callable[[str, str], None]] = None
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'capability_queries': 0,
            'version_queries': 0,
            'tag_queries': 0,
            'avg_query_time_ms': 0.0
        }
        
        # Initialize registry
        self._initialize_registry()
        logger.info(f"MCPServerRegistry initialized with {len(self.servers)} servers")
    
    def _initialize_registry(self):
        """Initialize registry by loading from persistence or creating defaults"""
        # Try to load from persistent storage
        if self.registry_file and self.registry_file.exists():
            try:
                self._load_from_file()
                logger.info(f"Loaded {len(self.servers)} servers from {self.registry_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load registry from {self.registry_file}: {e}")
        
        # Initialize with default server configurations
        default_servers = [
            ServerRecord(
                server_id="github-mcp-v1.2.0",
                name="GitHub MCP Server",
                version="1.2.0", 
                capabilities=["github_api", "issue_management", "pr_operations"],
                resource_requirements={"memory_mb": 128, "cpu_percent": 8},
                description="GitHub integration MCP server for issue and PR management",
                tags=["github", "issues", "development"],
                registered_at=datetime.utcnow()
            ),
            ServerRecord(
                server_id="git-mcp-v1.0.0",
                name="Git MCP Server",
                version="1.0.0",
                capabilities=["git_operations", "repository_analysis", "branch_management"],
                resource_requirements={"memory_mb": 64, "cpu_percent": 5},
                description="Git operations MCP server for repository management",
                tags=["git", "repository", "development"],
                registered_at=datetime.utcnow()
            ),
            ServerRecord(
                server_id="filesystem-mcp-v1.1.0",
                name="Filesystem MCP Server",
                version="1.1.0",
                capabilities=["file_operations", "directory_scanning", "file_watching"],
                resource_requirements={"memory_mb": 96, "cpu_percent": 6},
                description="File system operations MCP server",
                tags=["filesystem", "files", "monitoring"],
                registered_at=datetime.utcnow()
            ),
            ServerRecord(
                server_id="sequential-thinking-mcp-v1.0.0",
                name="Sequential Thinking MCP",
                version="1.0.0",
                capabilities=["structured_reasoning", "problem_decomposition", "logic_chains"],
                resource_requirements={"memory_mb": 256, "cpu_percent": 12},
                description="Structured reasoning and problem decomposition MCP server",
                tags=["reasoning", "thinking", "analysis"],
                registered_at=datetime.utcnow()
            ),
            ServerRecord(
                server_id="memory-mcp-v1.0.0",
                name="Memory MCP Server",
                version="1.0.0",
                capabilities=["context_storage", "knowledge_retrieval", "pattern_memory"],
                resource_requirements={"memory_mb": 192, "cpu_percent": 10},
                description="Context storage and knowledge retrieval MCP server",
                tags=["memory", "context", "knowledge"],
                registered_at=datetime.utcnow()
            )
        ]
        
        for server_record in default_servers:
            self._register_server_record(server_record)
        
        # Save to persistence if enabled
        if self.auto_save:
            self._save_to_file()
        
        logger.debug(f"Initialized {len(default_servers)} default servers")
    
    def _register_server_record(self, server_record: ServerRecord):
        """
        Register a server record and update all indices (internal method)
        
        Args:
            server_record: Server record to register
        """
        with self._lock:
            self.servers[server_record.server_id] = server_record
            
            # Update capability index
            for capability in server_record.capabilities:
                self.capability_index[capability].add(server_record.server_id)
            
            # Update tag index
            for tag in server_record.tags:
                self.tag_index[tag].add(server_record.server_id)
                
            # Update version index  
            self.version_index[server_record.version].add(server_record.server_id)
    
    def _unregister_server_record(self, server_id: str):
        """
        Unregister a server record and clean up all indices (internal method)
        
        Args:
            server_id: Server ID to unregister
        """
        with self._lock:
            if server_id not in self.servers:
                return
            
            server_record = self.servers[server_id]
            
            # Clean capability index
            for capability in server_record.capabilities:
                self.capability_index[capability].discard(server_id)
                if not self.capability_index[capability]:
                    del self.capability_index[capability]
            
            # Clean tag index
            for tag in server_record.tags:
                self.tag_index[tag].discard(server_id)
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
                    
            # Clean version index
            self.version_index[server_record.version].discard(server_id)
            if not self.version_index[server_record.version]:
                del self.version_index[server_record.version]
            
            # Remove server record
            del self.servers[server_id]
    
    def _load_from_file(self):
        """Load registry from persistent storage file"""
        if not self.registry_file or not self.registry_file.exists():
            return
        
        with open(self.registry_file, 'r') as f:
            data = json.load(f)
        
        # Load servers
        for server_data in data.get('servers', []):
            # Convert datetime strings back to datetime objects
            for date_field in ['last_updated', 'registered_at', 'last_health_check']:
                if server_data.get(date_field):
                    server_data[date_field] = datetime.fromisoformat(server_data[date_field])
            
            server_record = ServerRecord(**server_data)
            self._register_server_record(server_record)
        
        # Load query stats
        if 'query_stats' in data:
            self.query_stats.update(data['query_stats'])
    
    def _save_to_file(self):
        """Save registry to persistent storage file"""
        if not self.registry_file:
            return
        
        # Ensure directory exists
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        servers_data = []
        for server_record in self.servers.values():
            server_dict = asdict(server_record)
            
            # Convert datetime objects to ISO strings
            for date_field in ['last_updated', 'registered_at', 'last_health_check']:
                if server_dict.get(date_field):
                    server_dict[date_field] = server_dict[date_field].isoformat()
            
            servers_data.append(server_dict)
        
        data = {
            'servers': servers_data,
            'query_stats': self.query_stats,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _server_record_to_dict(self, server_record: ServerRecord, 
                             include_metrics: bool = False) -> Dict[str, Any]:
        """
        Convert server record to dictionary representation
        
        Args:
            server_record: Server record to convert
            include_metrics: Whether to include metrics data
            
        Returns:
            Dictionary representation of server record
        """
        result = {
            "server_id": server_record.server_id,
            "name": server_record.name,
            "version": server_record.version,
            "capabilities": server_record.capabilities,
            "resource_requirements": server_record.resource_requirements,
            "health_status": server_record.health_status,
            "description": server_record.description,
            "tags": server_record.tags,
            "dependencies": server_record.dependencies,
            "configuration": server_record.configuration,
            "registered_at": server_record.registered_at.isoformat() if server_record.registered_at else None,
            "last_updated": server_record.last_updated.isoformat() if server_record.last_updated else None,
            "last_health_check": server_record.last_health_check.isoformat() if server_record.last_health_check else None,
            "total_health_checks": server_record.total_health_checks,
            "failed_health_checks": server_record.failed_health_checks
        }
        
        if include_metrics:
            result["metrics"] = server_record.metrics
            
        return result
    
    async def get_server(self, server_id: str, include_metrics: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get server configuration by ID
        
        Args:
            server_id: Server identifier
            include_metrics: Whether to include performance metrics
            
        Returns:
            Server configuration dictionary or None
        """
        with self._lock:
            if server_id in self.servers:
                self.query_stats['total_queries'] += 1
                return self._server_record_to_dict(self.servers[server_id], include_metrics)
            return None
    
    async def list_servers(self, include_metrics: bool = False, 
                          health_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered servers with optional filtering
        
        Args:
            include_metrics: Whether to include performance metrics
            health_status: Filter by health status (optional)
            
        Returns:
            List of server configurations
        """
        with self._lock:
            self.query_stats['total_queries'] += 1
            servers = []
            
            for record in self.servers.values():
                # Apply health status filter if specified
                if health_status and record.health_status != health_status:
                    continue
                    
                servers.append(self._server_record_to_dict(record, include_metrics))
                
            return servers
    
    async def find_servers_by_capability(self, capability: str,
                                        include_metrics: bool = False) -> List[Dict[str, Any]]:
        """
        Find servers with specific capability using indexed lookup
        
        Args:
            capability: Capability to search for
            include_metrics: Whether to include performance metrics
            
        Returns:
            List of matching server configurations
        """
        with self._lock:
            self.query_stats['total_queries'] += 1
            self.query_stats['capability_queries'] += 1
            
            # Use capability index for fast lookup
            server_ids = self.capability_index.get(capability, set())
            matching_servers = []
            
            for server_id in server_ids:
                if server_id in self.servers:
                    matching_servers.append(
                        self._server_record_to_dict(self.servers[server_id], include_metrics)
                    )
                    
            return matching_servers
    
    async def register_server(self, server_config: Dict[str, Any]) -> bool:
        """
        Register a new server with comprehensive validation and indexing
        
        Args:
            server_config: Server configuration to register
            
        Required fields:
            - server_id: Unique identifier for the server
            - name: Human-readable name
            - capabilities: List of capabilities provided by the server
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            # Validate required fields
            required_fields = ["server_id", "name", "capabilities"]
            for field in required_fields:
                if field not in server_config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Check if server already exists
            if server_config["server_id"] in self.servers:
                logger.warning(f"Server already registered: {server_config['server_id']}")
                return False
            
            # Create server record with all fields
            now = datetime.utcnow()
            record = ServerRecord(
                server_id=server_config["server_id"],
                name=server_config["name"],
                version=server_config.get("version", "1.0.0"),
                capabilities=server_config["capabilities"],
                resource_requirements=server_config.get("resource_requirements", {}),
                description=server_config.get("description"),
                tags=server_config.get("tags", []),
                dependencies=server_config.get("dependencies", []),
                configuration=server_config.get("configuration", {}),
                health_status="unknown",
                registered_at=now,
                last_updated=now
            )
            
            # Register with indexing
            self._register_server_record(record)
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_to_file()
            
            logger.info(f"Successfully registered server: {record.name} ({record.server_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register server {server_config.get('server_id', 'unknown')}: {e}")
            return False
    
    async def update_server_health(self, server_id: str, health_status: str,
                                 health_check_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update server health status with enhanced tracking
        
        Args:
            server_id: Server identifier
            health_status: New health status (healthy, degraded, unhealthy, unknown)
            health_check_result: Optional health check result data
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            if server_id not in self.servers:
                logger.warning(f"Cannot update health for unknown server: {server_id}")
                return False
            
            record = self.servers[server_id]
            now = datetime.utcnow()
            
            # Update basic health info
            old_status = record.health_status
            record.health_status = health_status
            record.last_health_check = now
            record.last_updated = now
            record.total_health_checks += 1
            
            # Track failures
            if health_status in ["unhealthy", "degraded"]:
                record.failed_health_checks += 1
            
            # Update metrics if provided
            if health_check_result:
                if "response_time_ms" in health_check_result:
                    record.metrics["last_response_time_ms"] = health_check_result["response_time_ms"]
                
                if "error" in health_check_result:
                    record.metrics["last_error"] = health_check_result["error"]
                    
                if "uptime_percent" in health_check_result:
                    record.metrics["uptime_percent"] = health_check_result["uptime_percent"]
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_to_file()
            
            # Call health callback if registered
            if self.health_callback and old_status != health_status:
                try:
                    self.health_callback(server_id, health_status)
                except Exception as e:
                    logger.error(f"Health callback failed for {server_id}: {e}")
            
            logger.debug(f"Updated health for {server_id}: {old_status} -> {health_status}")
            return True
    
    async def unregister_server(self, server_id: str) -> bool:
        """
        Unregister a server from the registry
        
        Args:
            server_id: Server identifier to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if server_id not in self.servers:
                logger.warning(f"Cannot unregister unknown server: {server_id}")
                return False
            
            server_name = self.servers[server_id].name
            self._unregister_server_record(server_id)
            
            # Auto-save if enabled
            if self.auto_save:
                self._save_to_file()
            
            logger.info(f"Unregistered server: {server_name} ({server_id})")
            return True
    
    async def find_servers_by_tag(self, tag: str, 
                                 include_metrics: bool = False) -> List[Dict[str, Any]]:
        """
        Find servers by tag using indexed lookup
        
        Args:
            tag: Tag to search for
            include_metrics: Whether to include performance metrics
            
        Returns:
            List of matching server configurations
        """
        with self._lock:
            self.query_stats['total_queries'] += 1
            self.query_stats['tag_queries'] += 1
            
            server_ids = self.tag_index.get(tag, set())
            matching_servers = []
            
            for server_id in server_ids:
                if server_id in self.servers:
                    matching_servers.append(
                        self._server_record_to_dict(self.servers[server_id], include_metrics)
                    )
                    
            return matching_servers
    
    async def find_servers_by_version(self, version: str,
                                    include_metrics: bool = False) -> List[Dict[str, Any]]:
        """
        Find servers by version using indexed lookup
        
        Args:
            version: Version to search for
            include_metrics: Whether to include performance metrics
            
        Returns:
            List of matching server configurations
        """
        with self._lock:
            self.query_stats['total_queries'] += 1
            self.query_stats['version_queries'] += 1
            
            server_ids = self.version_index.get(version, set())
            matching_servers = []
            
            for server_id in server_ids:
                if server_id in self.servers:
                    matching_servers.append(
                        self._server_record_to_dict(self.servers[server_id], include_metrics)
                    )
                    
            return matching_servers
    
    async def find_servers_by_resource_requirements(self, max_memory_mb: Optional[int] = None,
                                                   max_cpu_percent: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find servers that fit within specified resource constraints
        
        Args:
            max_memory_mb: Maximum memory requirement in MB
            max_cpu_percent: Maximum CPU requirement as percentage
            
        Returns:
            List of matching server configurations
        """
        with self._lock:
            self.query_stats['total_queries'] += 1
            matching_servers = []
            
            for record in self.servers.values():
                requirements = record.resource_requirements
                
                # Check memory constraint
                if max_memory_mb is not None:
                    server_memory = requirements.get('memory_mb', 0)
                    if server_memory > max_memory_mb:
                        continue
                
                # Check CPU constraint
                if max_cpu_percent is not None:
                    server_cpu = requirements.get('cpu_percent', 0)
                    if server_cpu > max_cpu_percent:
                        continue
                
                matching_servers.append(self._server_record_to_dict(record))
            
            return matching_servers
    
    async def get_server_dependencies(self, server_id: str) -> List[str]:
        """
        Get dependency list for a specific server
        
        Args:
            server_id: Server identifier
            
        Returns:
            List of dependency server IDs
        """
        with self._lock:
            if server_id in self.servers:
                return self.servers[server_id].dependencies.copy()
            return []
    
    async def get_capability_catalog(self) -> Dict[str, List[str]]:
        """
        Get complete catalog of capabilities and servers that provide them
        
        Returns:
            Dictionary mapping capabilities to lists of server IDs
        """
        with self._lock:
            catalog = {}
            for capability, server_ids in self.capability_index.items():
                catalog[capability] = list(server_ids)
            return catalog
    
    async def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics and metrics
        
        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            total_servers = len(self.servers)
            health_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
            
            for record in self.servers.values():
                health_counts[record.health_status] = health_counts.get(record.health_status, 0) + 1
            
            return {
                "total_servers": total_servers,
                "health_distribution": health_counts,
                "total_capabilities": len(self.capability_index),
                "total_tags": len(self.tag_index),
                "total_versions": len(self.version_index),
                "query_statistics": self.query_stats.copy(),
                "uptime_percent": round((health_counts["healthy"] / total_servers * 100), 2) if total_servers > 0 else 0
            }
    
    def set_health_callback(self, callback: Callable[[str, str], None]):
        """
        Set callback function for health status changes
        
        Args:
            callback: Function to call when server health changes
                     Signature: (server_id: str, new_status: str) -> None
        """
        self.health_callback = callback
        logger.info("Health status callback registered")
    
    async def validate_server_dependencies(self, server_id: str) -> Dict[str, Any]:
        """
        Validate that all dependencies for a server are registered and healthy
        
        Args:
            server_id: Server identifier to validate
            
        Returns:
            Validation result with status and details
        """
        with self._lock:
            if server_id not in self.servers:
                return {"valid": False, "error": "Server not found"}
            
            dependencies = self.servers[server_id].dependencies
            if not dependencies:
                return {"valid": True, "message": "No dependencies"}
            
            missing_deps = []
            unhealthy_deps = []
            
            for dep_id in dependencies:
                if dep_id not in self.servers:
                    missing_deps.append(dep_id)
                elif self.servers[dep_id].health_status in ["unhealthy", "unknown"]:
                    unhealthy_deps.append(dep_id)
            
            if missing_deps or unhealthy_deps:
                return {
                    "valid": False,
                    "missing_dependencies": missing_deps,
                    "unhealthy_dependencies": unhealthy_deps
                }
            
            return {"valid": True, "message": "All dependencies satisfied"}
    
    async def export_registry(self, export_format: str = "json") -> str:
        """
        Export registry data in specified format
        
        Args:
            export_format: Export format ("json" or "yaml")
            
        Returns:
            Serialized registry data
        """
        with self._lock:
            servers_data = []
            for record in self.servers.values():
                servers_data.append(self._server_record_to_dict(record, include_metrics=True))
            
            export_data = {
                "registry_export": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_servers": len(servers_data),
                    "servers": servers_data,
                    "statistics": await self.get_registry_statistics()
                }
            }
            
            if export_format.lower() == "json":
                import json
                return json.dumps(export_data, indent=2)
            elif export_format.lower() == "yaml":
                try:
                    import yaml
                    return yaml.dump(export_data, default_flow_style=False)
                except ImportError:
                    logger.warning("PyYAML not available, falling back to JSON")
                    import json
                    return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
    
    async def cleanup_stale_servers(self, max_age_days: int = 30) -> int:
        """
        Clean up servers that haven't been updated in specified days
        
        Args:
            max_age_days: Maximum age in days for inactive servers
            
        Returns:
            Number of servers cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        stale_servers = []
        
        with self._lock:
            for server_id, record in self.servers.items():
                last_activity = record.last_updated or record.registered_at
                if last_activity and last_activity < cutoff_date:
                    stale_servers.append(server_id)
            
            # Remove stale servers
            for server_id in stale_servers:
                self._unregister_server_record(server_id)
            
            # Auto-save if enabled
            if self.auto_save and stale_servers:
                self._save_to_file()
        
        if stale_servers:
            logger.info(f"Cleaned up {len(stale_servers)} stale servers")
        
        return len(stale_servers)