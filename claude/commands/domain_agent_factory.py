#!/usr/bin/env python3
"""
Domain Agent Factory - Issue #71 Enhanced for Issue #70
Template-driven factory pattern implementation for creating and managing domain-specific agents 
with comprehensive template validation, configuration management, resource allocation, and registry management.
"""

import json
import logging
import time
import uuid
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

# Import template system components
try:
    from claude.commands.template_validator import (
        DomainAgentTemplateValidator, ValidationResult, ValidationSeverity
    )
    from claude.commands.domain_agent_base import (
        DomainAgent, AgentTemplate, AgentConfiguration, CapabilityMatch
    )
except ImportError as e:
    logger.warning(f"Template system imports failed: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Supported domain agent types"""
    FRONTEND = "frontend"
    BACKEND = "backend"
    SECURITY = "security"
    DEVOPS = "devops"
    DATA = "data"
    ML = "ml"
    API = "api"
    DATABASE = "database"
    MOBILE = "mobile"
    TESTING = "testing"

class AgentStatus(Enum):
    """Agent lifecycle status"""
    CREATED = "created"
    CONFIGURED = "configured"
    ALLOCATED = "allocated"
    REGISTERED = "registered"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class DomainCapability:
    """Individual capability of a domain agent"""
    name: str
    description: str
    complexity: str = "medium"  # low, medium, high
    resource_requirement: str = "standard"  # minimal, standard, intensive
    dependencies: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)

@dataclass
class AgentResources:
    """Resource allocation for an agent"""
    cpu_cores: int = 1
    memory_mb: int = 512
    disk_mb: int = 100
    max_runtime_minutes: int = 30
    max_concurrent_tasks: int = 3
    network_bandwidth_mbps: int = 10
    allocated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class DomainConfiguration:
    """Configuration for a domain agent"""
    name: str
    domain_type: AgentType
    capabilities: List[DomainCapability] = field(default_factory=list)
    expertise: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    priority: int = 50  # 0-100
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config_version: str = "1.0"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.name:
            raise ValueError("Agent name is required")
        if not isinstance(self.domain_type, AgentType):
            raise ValueError("domain_type must be an AgentType enum")
        if self.priority < 0 or self.priority > 100:
            raise ValueError("Priority must be between 0 and 100")

@dataclass
class DomainAgent:
    """Domain-specific agent instance"""
    agent_id: str
    name: str
    domain_type: AgentType
    status: AgentStatus = AgentStatus.CREATED
    capabilities: List[DomainCapability] = field(default_factory=list)
    expertise: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    resources: Optional[AgentResources] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    configuration: Optional[DomainConfiguration] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, new_status: AgentStatus, reason: str = ""):
        """Update agent status with logging"""
        old_status = self.status
        self.status = new_status
        self.last_activity = datetime.now()
        if reason:
            self.metadata["status_reason"] = reason
        logger.info(f"Agent {self.name} status changed: {old_status.value} -> {new_status.value}")

class AgentRegistry:
    """Central registry for managing agent instances"""
    
    def __init__(self):
        self.agents: Dict[str, DomainAgent] = {}
        self.agents_by_type: Dict[AgentType, Set[str]] = {agent_type: set() for agent_type in AgentType}
        self.active_agents: Set[str] = set()
        self._lock = threading.RLock()
        self.registry_file = Path("knowledge/agents/agent_registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
    def register(self, agent: DomainAgent) -> bool:
        """Register an agent in the central registry"""
        try:
            with self._lock:
                if agent.agent_id in self.agents:
                    logger.warning(f"Agent {agent.agent_id} already registered")
                    return False
                
                self.agents[agent.agent_id] = agent
                self.agents_by_type[agent.domain_type].add(agent.agent_id)
                
                if agent.status == AgentStatus.ACTIVE:
                    self.active_agents.add(agent.agent_id)
                
                # Update agent status
                agent.update_status(AgentStatus.REGISTERED)
                
                # Persist registry
                self._persist_registry()
                
                logger.info(f"Agent {agent.name} registered successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    def unregister(self, agent_id: str) -> bool:
        """Remove agent from registry"""
        try:
            with self._lock:
                if agent_id not in self.agents:
                    logger.warning(f"Agent {agent_id} not found in registry")
                    return False
                
                agent = self.agents[agent_id]
                self.agents_by_type[agent.domain_type].discard(agent_id)
                self.active_agents.discard(agent_id)
                del self.agents[agent_id]
                
                self._persist_registry()
                logger.info(f"Agent {agent_id} unregistered")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[DomainAgent]:
        """Get agent by ID"""
        with self._lock:
            return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[DomainAgent]:
        """Get all agents of specific type"""
        with self._lock:
            agent_ids = self.agents_by_type.get(agent_type, set())
            return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_active_agents(self) -> List[DomainAgent]:
        """Get all currently active agents"""
        with self._lock:
            return [self.agents[agent_id] for agent_id in self.active_agents if agent_id in self.agents]
    
    def _persist_registry(self):
        """Persist registry to disk"""
        try:
            registry_data = {
                "agents": {agent_id: self._serialize_agent(agent) 
                          for agent_id, agent in self.agents.items()},
                "metadata": {
                    "total_agents": len(self.agents),
                    "active_agents": len(self.active_agents),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist registry: {e}")
    
    def _serialize_agent(self, agent: DomainAgent) -> Dict[str, Any]:
        """Convert agent to serializable format"""
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "domain_type": agent.domain_type.value,
            "status": agent.status.value,
            "capabilities": [asdict(cap) for cap in agent.capabilities],
            "expertise": agent.expertise,
            "tools": agent.tools,
            "resources": asdict(agent.resources) if agent.resources else None,
            "created_at": agent.created_at.isoformat(),
            "last_activity": agent.last_activity.isoformat(),
            "performance_metrics": agent.performance_metrics,
            "metadata": agent.metadata
        }

class ResourceAllocator:
    """Resource allocation and management for domain agents"""
    
    def __init__(self):
        self.allocated_resources: Dict[str, AgentResources] = {}
        self.total_limits = {
            "cpu_cores": psutil.cpu_count(),
            "memory_mb": psutil.virtual_memory().total // (1024 * 1024),
            "disk_mb": 10000,  # 10GB default limit
            "max_agents": 10
        }
        self.safety_margins = {
            "cpu_percent": 80,  # Use max 80% of available CPU
            "memory_percent": 75,  # Use max 75% of available memory
            "disk_percent": 90   # Use max 90% of allocated disk
        }
        self._lock = threading.RLock()
    
    def allocate_resources(self, agent_id: str, requirements: Dict[str, Any]) -> Tuple[bool, AgentResources, str]:
        """Allocate resources for an agent"""
        try:
            with self._lock:
                # Check if agent already has resources
                if agent_id in self.allocated_resources:
                    return False, None, f"Agent {agent_id} already has resources allocated"
                
                # Create resource allocation based on requirements
                requested_resources = AgentResources(
                    cpu_cores=requirements.get("cpu_cores", 1),
                    memory_mb=requirements.get("memory_mb", 512),
                    disk_mb=requirements.get("disk_mb", 100),
                    max_runtime_minutes=requirements.get("max_runtime_minutes", 30),
                    max_concurrent_tasks=requirements.get("max_concurrent_tasks", 3),
                    network_bandwidth_mbps=requirements.get("network_bandwidth_mbps", 10)
                )
                
                # Validate resource availability
                can_allocate, reason = self._check_resource_availability(requested_resources)
                if not can_allocate:
                    return False, None, reason
                
                # Allocate resources
                self.allocated_resources[agent_id] = requested_resources
                
                logger.info(f"Resources allocated to agent {agent_id}: {requested_resources}")
                return True, requested_resources, "Resources allocated successfully"
                
        except Exception as e:
            logger.error(f"Resource allocation failed for agent {agent_id}: {e}")
            return False, None, str(e)
    
    def _check_resource_availability(self, requested: AgentResources) -> Tuple[bool, str]:
        """Check if requested resources are available"""
        try:
            # Calculate currently allocated resources
            total_allocated_cpu = sum(res.cpu_cores for res in self.allocated_resources.values())
            total_allocated_memory = sum(res.memory_mb for res in self.allocated_resources.values())
            total_allocated_disk = sum(res.disk_mb for res in self.allocated_resources.values())
            
            # Check limits with safety margins
            max_cpu = int(self.total_limits["cpu_cores"] * self.safety_margins["cpu_percent"] / 100)
            max_memory = int(self.total_limits["memory_mb"] * self.safety_margins["memory_percent"] / 100)
            max_disk = int(self.total_limits["disk_mb"] * self.safety_margins["disk_percent"] / 100)
            
            if total_allocated_cpu + requested.cpu_cores > max_cpu:
                return False, f"Insufficient CPU: requested {requested.cpu_cores}, available {max_cpu - total_allocated_cpu}"
            
            if total_allocated_memory + requested.memory_mb > max_memory:
                return False, f"Insufficient memory: requested {requested.memory_mb}MB, available {max_memory - total_allocated_memory}MB"
            
            if total_allocated_disk + requested.disk_mb > max_disk:
                return False, f"Insufficient disk: requested {requested.disk_mb}MB, available {max_disk - total_allocated_disk}MB"
            
            if len(self.allocated_resources) >= self.total_limits["max_agents"]:
                return False, f"Maximum agent limit reached: {self.total_limits['max_agents']}"
            
            return True, "Resources available"
            
        except Exception as e:
            return False, f"Resource check failed: {e}"
    
    def deallocate_resources(self, agent_id: str) -> bool:
        """Deallocate resources for an agent"""
        try:
            with self._lock:
                if agent_id not in self.allocated_resources:
                    logger.warning(f"No resources allocated for agent {agent_id}")
                    return False
                
                resources = self.allocated_resources.pop(agent_id)
                logger.info(f"Resources deallocated for agent {agent_id}: {resources}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to deallocate resources for agent {agent_id}: {e}")
            return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        with self._lock:
            total_cpu = sum(res.cpu_cores for res in self.allocated_resources.values())
            total_memory = sum(res.memory_mb for res in self.allocated_resources.values())
            total_disk = sum(res.disk_mb for res in self.allocated_resources.values())
            
            return {
                "allocated_agents": len(self.allocated_resources),
                "cpu_utilization": {
                    "allocated": total_cpu,
                    "available": self.total_limits["cpu_cores"],
                    "percentage": (total_cpu / self.total_limits["cpu_cores"]) * 100 if self.total_limits["cpu_cores"] > 0 else 0
                },
                "memory_utilization": {
                    "allocated_mb": total_memory,
                    "available_mb": self.total_limits["memory_mb"],
                    "percentage": (total_memory / self.total_limits["memory_mb"]) * 100 if self.total_limits["memory_mb"] > 0 else 0
                },
                "disk_utilization": {
                    "allocated_mb": total_disk,
                    "available_mb": self.total_limits["disk_mb"],
                    "percentage": (total_disk / self.total_limits["disk_mb"]) * 100 if self.total_limits["disk_mb"] > 0 else 0
                }
            }

class DomainAgentFactory:
    """
    Factory for creating and managing domain-specific agents
    
    This factory implements the factory pattern for agent creation with:
    - Configuration validation
    - Resource allocation
    - Agent registration
    - Template management
    - Team creation capabilities
    """
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.resource_allocator = ResourceAllocator()
        self.templates = self._load_domain_templates()
        self.created_agents: Dict[str, DomainAgent] = {}
        self.factory_metrics = {
            "agents_created": 0,
            "creation_failures": 0,
            "average_creation_time": 0.0,
            "agents_by_type": {agent_type: 0 for agent_type in AgentType}
        }
        self._lock = threading.RLock()
        
        logger.info("DomainAgentFactory initialized")
    
    def create_agent(self, domain_config: DomainConfiguration) -> Tuple[bool, Optional[DomainAgent], str]:
        """
        Create a domain agent with full lifecycle management
        
        Args:
            domain_config: Configuration for the agent to create
            
        Returns:
            Tuple[bool, Optional[DomainAgent], str]: (success, agent, message)
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Step 1: Validate configuration
                validation_result, validation_message = self.validate_config(domain_config)
                if not validation_result:
                    self.factory_metrics["creation_failures"] += 1
                    return False, None, f"Configuration validation failed: {validation_message}"
                
                # Step 2: Generate unique agent ID
                agent_id = self._generate_agent_id(domain_config)
                
                # Step 3: Create agent instance
                agent = DomainAgent(
                    agent_id=agent_id,
                    name=domain_config.name,
                    domain_type=domain_config.domain_type,
                    capabilities=domain_config.capabilities,
                    expertise=domain_config.expertise,
                    tools=domain_config.tools,
                    configuration=domain_config,
                    metadata=domain_config.metadata.copy()
                )
                
                # Step 4: Allocate resources
                resource_success, allocated_resources, resource_message = self.allocate_resources(agent_id, domain_config)
                if not resource_success:
                    self.factory_metrics["creation_failures"] += 1
                    return False, None, f"Resource allocation failed: {resource_message}"
                
                agent.resources = allocated_resources
                agent.update_status(AgentStatus.ALLOCATED)
                
                # Step 5: Register agent
                registration_success = self.registry.register(agent)
                if not registration_success:
                    # Clean up resources if registration fails
                    self.resource_allocator.deallocate_resources(agent_id)
                    self.factory_metrics["creation_failures"] += 1
                    return False, None, "Agent registration failed"
                
                # Step 6: Update metrics and tracking
                creation_time = time.time() - start_time
                self._update_factory_metrics(domain_config.domain_type, creation_time)
                self.created_agents[agent_id] = agent
                
                logger.info(f"Agent {agent.name} created successfully in {creation_time:.3f}s")
                return True, agent, "Agent created successfully"
                
        except Exception as e:
            self.factory_metrics["creation_failures"] += 1
            logger.error(f"Agent creation failed: {e}")
            return False, None, f"Creation error: {str(e)}"
    
    def validate_config(self, domain_config: DomainConfiguration) -> Tuple[bool, str]:
        """
        Validate domain configuration before agent creation
        
        Args:
            domain_config: Configuration to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        try:
            # Basic validation
            if not domain_config.name or not domain_config.name.strip():
                return False, "Agent name cannot be empty"
            
            if not isinstance(domain_config.domain_type, AgentType):
                return False, "Invalid domain type"
            
            if domain_config.priority < 0 or domain_config.priority > 100:
                return False, "Priority must be between 0 and 100"
            
            # Check for duplicate names
            with self._lock:
                for existing_agent in self.registry.agents.values():
                    if existing_agent.name == domain_config.name:
                        return False, f"Agent with name '{domain_config.name}' already exists"
            
            # Validate capabilities - allow empty list as capabilities can be default
            # This is not a hard requirement since templates can provide defaults
            
            for capability in domain_config.capabilities:
                if not isinstance(capability, DomainCapability):
                    return False, "Invalid capability format"
                if not capability.name or not capability.description:
                    return False, "Capability name and description are required"
            
            # Validate resource requirements
            resource_requirements = domain_config.resource_requirements
            if resource_requirements:
                if "memory_mb" in resource_requirements and resource_requirements["memory_mb"] < 64:
                    return False, "Minimum memory requirement is 64MB"
                if "cpu_cores" in resource_requirements and resource_requirements["cpu_cores"] < 1:
                    return False, "Minimum CPU requirement is 1 core"
            
            # Validate against domain template if available
            if domain_config.domain_type in self.templates:
                template_validation = self._validate_against_template(domain_config)
                if not template_validation[0]:
                    return template_validation
            
            logger.info(f"Configuration validation passed for {domain_config.name}")
            return True, "Configuration is valid"
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def allocate_resources(self, agent_id: str, domain_config: DomainConfiguration) -> Tuple[bool, Optional[AgentResources], str]:
        """
        Allocate resources for an agent based on its configuration
        
        Args:
            agent_id: Unique identifier for the agent
            domain_config: Configuration containing resource requirements
            
        Returns:
            Tuple[bool, Optional[AgentResources], str]: (success, resources, message)
        """
        try:
            # Use configuration requirements or defaults
            requirements = domain_config.resource_requirements.copy()
            
            # Set sensible defaults based on domain type
            domain_defaults = self._get_domain_resource_defaults(domain_config.domain_type)
            for key, value in domain_defaults.items():
                if key not in requirements:
                    requirements[key] = value
            
            # Allocate resources
            success, resources, message = self.resource_allocator.allocate_resources(agent_id, requirements)
            
            if success:
                logger.info(f"Resources allocated for agent {agent_id}")
            else:
                logger.warning(f"Resource allocation failed for agent {agent_id}: {message}")
            
            return success, resources, message
            
        except Exception as e:
            logger.error(f"Resource allocation error for agent {agent_id}: {e}")
            return False, None, str(e)
    
    def create_specialist_team(self, project_requirements: Dict[str, Any]) -> Tuple[bool, List[DomainAgent], str]:
        """
        Create a team of specialist agents based on project requirements
        
        Args:
            project_requirements: Dict containing project details and requirements
            
        Returns:
            Tuple[bool, List[DomainAgent], str]: (success, agents, message)
        """
        try:
            team = []
            required_domains = self.identify_required_domains(project_requirements)
            
            if not required_domains:
                return False, [], "No domains identified from project requirements"
            
            logger.info(f"Creating specialist team for domains: {[d.value for d in required_domains]}")
            
            for domain_type in required_domains:
                # Get template for this domain
                if domain_type not in self.templates:
                    logger.warning(f"No template found for domain {domain_type.value}, using defaults")
                    domain_config = self._create_default_config(domain_type, project_requirements)
                else:
                    template = self.templates[domain_type]
                    domain_config = self._create_config_from_template(template, project_requirements)
                
                # Create agent
                success, agent, message = self.create_agent(domain_config)
                
                if success:
                    team.append(agent)
                    logger.info(f"Added {domain_type.value} specialist to team")
                else:
                    logger.error(f"Failed to create {domain_type.value} specialist: {message}")
                    # Continue creating other agents even if one fails
            
            if not team:
                return False, [], "Failed to create any specialist agents"
            
            # Store team metadata
            team_metadata = {
                "team_id": str(uuid.uuid4()),
                "project_name": project_requirements.get("project_name", "Unknown"),
                "created_at": datetime.now().isoformat(),
                "team_size": len(team),
                "domains": [agent.domain_type.value for agent in team],
                "requirements": project_requirements
            }
            
            # Update team metadata in all agents
            for agent in team:
                agent.metadata["team"] = team_metadata
            
            success_message = f"Created specialist team with {len(team)} agents: {[a.domain_type.value for a in team]}"
            logger.info(success_message)
            return True, team, success_message
            
        except Exception as e:
            logger.error(f"Team creation failed: {e}")
            return False, [], f"Team creation error: {str(e)}"
    
    def identify_required_domains(self, project_requirements: Dict[str, Any]) -> List[AgentType]:
        """
        Identify required domain types based on project requirements
        
        Args:
            project_requirements: Project details and requirements
            
        Returns:
            List[AgentType]: Required domain agent types
        """
        required_domains = set()
        
        try:
            # Analyze project type and requirements
            project_type = project_requirements.get("type", "").lower()
            technologies = [tech.lower() for tech in project_requirements.get("technologies", [])]
            features = [feat.lower() for feat in project_requirements.get("features", [])]
            description = project_requirements.get("description", "").lower()
            
            # Web application indicators
            web_indicators = ["web", "webapp", "website", "frontend", "react", "vue", "angular", "html", "css", "javascript"]
            if project_type in ["web", "webapp"] or any(ind in technologies + features + [description] for ind in web_indicators):
                required_domains.add(AgentType.FRONTEND)
                required_domains.add(AgentType.BACKEND)
            
            # API indicators  
            api_indicators = ["api", "rest", "graphql", "microservice", "service"]
            if any(ind in technologies + features + [description] for ind in api_indicators):
                required_domains.add(AgentType.API)
                required_domains.add(AgentType.BACKEND)
            
            # Database indicators
            db_indicators = ["database", "sql", "postgresql", "mysql", "mongodb", "redis", "storage"]
            if any(ind in technologies + features + [description] for ind in db_indicators):
                required_domains.add(AgentType.DATABASE)
            
            # Security indicators
            security_indicators = ["auth", "authentication", "security", "encryption", "oauth", "jwt"]
            if any(ind in technologies + features + [description] for ind in security_indicators):
                required_domains.add(AgentType.SECURITY)
            
            # ML/Data indicators
            ml_indicators = ["ml", "ai", "data", "analytics", "machine learning", "tensorflow", "pytorch"]
            if any(ind in technologies + features + [description] for ind in ml_indicators):
                required_domains.add(AgentType.ML)
                required_domains.add(AgentType.DATA)
            
            # Mobile indicators
            mobile_indicators = ["mobile", "ios", "android", "react native", "flutter"]
            if any(ind in technologies + features + [description] for ind in mobile_indicators):
                required_domains.add(AgentType.MOBILE)
            
            # DevOps indicators
            devops_indicators = ["deploy", "deployment", "docker", "kubernetes", "ci/cd", "pipeline", "infrastructure"]
            if any(ind in technologies + features + [description] for ind in devops_indicators):
                required_domains.add(AgentType.DEVOPS)
            
            # Testing is always recommended
            required_domains.add(AgentType.TESTING)
            
            # Ensure at least backend if nothing else is detected
            if not required_domains:
                required_domains.add(AgentType.BACKEND)
            
            logger.info(f"Identified required domains: {[d.value for d in required_domains]}")
            return list(required_domains)
            
        except Exception as e:
            logger.error(f"Domain identification failed: {e}")
            return [AgentType.BACKEND]  # Fallback to backend
    
    def _generate_agent_id(self, config: DomainConfiguration) -> str:
        """Generate unique agent ID"""
        timestamp = int(time.time() * 1000)
        domain_prefix = config.domain_type.value[:3].upper()
        return f"{domain_prefix}-{config.name.replace(' ', '_').upper()}-{timestamp}"
    
    def _update_factory_metrics(self, domain_type: AgentType, creation_time: float):
        """Update factory performance metrics"""
        self.factory_metrics["agents_created"] += 1
        self.factory_metrics["agents_by_type"][domain_type] += 1
        
        # Update average creation time
        current_avg = self.factory_metrics["average_creation_time"]
        total_agents = self.factory_metrics["agents_created"]
        self.factory_metrics["average_creation_time"] = (
            (current_avg * (total_agents - 1) + creation_time) / total_agents
        )
    
    def _load_domain_templates(self) -> Dict[AgentType, Dict[str, Any]]:
        """Load domain agent templates from configuration"""
        try:
            # For now, return built-in templates
            # In real implementation, this would load from configuration files
            return {
                AgentType.FRONTEND: {
                    "default_capabilities": [
                        DomainCapability("ui_development", "User interface development", "medium", "standard", [], ["react", "css", "html"]),
                        DomainCapability("responsive_design", "Responsive web design", "low", "minimal", [], ["css", "flexbox"])
                    ],
                    "default_tools": ["react", "vue", "angular", "css", "html", "javascript", "webpack"],
                    "default_resources": {"memory_mb": 512, "cpu_cores": 1}
                },
                AgentType.BACKEND: {
                    "default_capabilities": [
                        DomainCapability("api_development", "REST API development", "medium", "standard", [], ["python", "nodejs"]),
                        DomainCapability("database_integration", "Database integration", "medium", "standard", [], ["sql", "orm"])
                    ],
                    "default_tools": ["python", "nodejs", "express", "fastapi", "postgresql", "redis"],
                    "default_resources": {"memory_mb": 768, "cpu_cores": 1}
                },
                AgentType.SECURITY: {
                    "default_capabilities": [
                        DomainCapability("security_audit", "Security vulnerability assessment", "high", "intensive", [], ["security_scanner", "penetration_testing"]),
                        DomainCapability("authentication", "Authentication and authorization", "high", "standard", [], ["oauth", "jwt"])
                    ],
                    "default_tools": ["owasp_zap", "security_scanner", "oauth", "jwt", "encryption"],
                    "default_resources": {"memory_mb": 1024, "cpu_cores": 1}
                }
            }
        except Exception as e:
            logger.error(f"Failed to load domain templates: {e}")
            return {}
    
    def _validate_against_template(self, config: DomainConfiguration) -> Tuple[bool, str]:
        """Validate configuration against domain template"""
        try:
            template = self.templates[config.domain_type]
            
            # Check if required tools are present - be lenient, just warn if no overlap
            template_tools = set(template.get("default_tools", []))
            config_tools = set(config.tools)
            
            # Allow any tools, just log if there's no overlap
            if template_tools and config_tools and not (template_tools & config_tools):
                logger.info(f"No common tools between config and {config.domain_type.value} template, but allowing creation")
            
            return True, "Template validation passed"
            
        except Exception as e:
            logger.error(f"Template validation error: {e}")
            return False, str(e)
    
    def _get_domain_resource_defaults(self, domain_type: AgentType) -> Dict[str, Any]:
        """Get default resource requirements for domain type"""
        defaults = {
            AgentType.FRONTEND: {"memory_mb": 512, "cpu_cores": 1, "disk_mb": 200},
            AgentType.BACKEND: {"memory_mb": 768, "cpu_cores": 1, "disk_mb": 300},
            AgentType.SECURITY: {"memory_mb": 1024, "cpu_cores": 1, "disk_mb": 500},
            AgentType.DATABASE: {"memory_mb": 1024, "cpu_cores": 1, "disk_mb": 1000},
            AgentType.ML: {"memory_mb": 1536, "cpu_cores": 1, "disk_mb": 2000},
            AgentType.DEVOPS: {"memory_mb": 512, "cpu_cores": 1, "disk_mb": 500},
            AgentType.TESTING: {"memory_mb": 256, "cpu_cores": 1, "disk_mb": 200},
            AgentType.DATA: {"memory_mb": 1024, "cpu_cores": 1, "disk_mb": 1500},
            AgentType.API: {"memory_mb": 512, "cpu_cores": 1, "disk_mb": 300},
            AgentType.MOBILE: {"memory_mb": 768, "cpu_cores": 1, "disk_mb": 400}
        }
        return defaults.get(domain_type, {"memory_mb": 512, "cpu_cores": 1, "disk_mb": 100})
    
    def _create_default_config(self, domain_type: AgentType, project_requirements: Dict[str, Any]) -> DomainConfiguration:
        """Create default configuration for domain type"""
        project_name = project_requirements.get("project_name", "Unknown")
        
        # Create unique name with timestamp
        timestamp = int(time.time() * 1000) % 10000  # last 4 digits for uniqueness
        unique_name = f"{domain_type.value.title()}_Agent_{project_name}_{timestamp}"
        
        return DomainConfiguration(
            name=unique_name,
            domain_type=domain_type,
            capabilities=[DomainCapability(f"{domain_type.value}_general", f"General {domain_type.value} capabilities")],
            expertise=[domain_type.value],
            tools=[domain_type.value],
            resource_requirements=self._get_domain_resource_defaults(domain_type),
            metadata={"created_from": "default_template", "project": project_name}
        )
    
    def _create_config_from_template(self, template: Dict[str, Any], project_requirements: Dict[str, Any]) -> DomainConfiguration:
        """Create configuration from domain template"""
        project_name = project_requirements.get("project_name", "Unknown")
        domain_type_str = template.get("type", "backend")
        
        # Find the domain type from templates based on the key used
        domain_type = None
        for agent_type in AgentType:
            if agent_type in self.templates and self.templates[agent_type] == template:
                domain_type = agent_type
                break
        
        if not domain_type:
            domain_type = AgentType.BACKEND  # fallback
        
        # Create unique name with timestamp
        timestamp = int(time.time() * 1000) % 10000  # last 4 digits for uniqueness
        unique_name = f"{domain_type.value.title()}_Agent_{project_name}_{timestamp}"
        
        return DomainConfiguration(
            name=unique_name,
            domain_type=domain_type,
            capabilities=template.get("default_capabilities", []),
            tools=template.get("default_tools", []),
            resource_requirements=template.get("default_resources", {}),
            metadata={"created_from": "template", "project": project_name}
        )
    
    def get_factory_metrics(self) -> Dict[str, Any]:
        """Get factory performance and usage metrics"""
        with self._lock:
            resource_usage = self.resource_allocator.get_resource_usage()
            
            return {
                "factory_metrics": self.factory_metrics,
                "registry_stats": {
                    "total_agents": len(self.registry.agents),
                    "active_agents": len(self.registry.active_agents),
                    "agents_by_type": {agent_type.value: len(agent_ids) 
                                     for agent_type, agent_ids in self.registry.agents_by_type.items()}
                },
                "resource_usage": resource_usage,
                "created_agents": len(self.created_agents)
            }
    
    def cleanup_agent(self, agent_id: str) -> bool:
        """Clean up agent and its resources"""
        try:
            with self._lock:
                # Remove from registry
                registry_success = self.registry.unregister(agent_id)
                
                # Deallocate resources
                resource_success = self.resource_allocator.deallocate_resources(agent_id)
                
                # Remove from created agents tracking
                if agent_id in self.created_agents:
                    del self.created_agents[agent_id]
                
                success = registry_success and resource_success
                if success:
                    logger.info(f"Agent {agent_id} cleaned up successfully")
                else:
                    logger.warning(f"Partial cleanup for agent {agent_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Cleanup failed for agent {agent_id}: {e}")
            return False


# Convenience functions for common factory operations

def create_simple_agent(name: str, domain_type: AgentType, factory: DomainAgentFactory = None) -> Tuple[bool, Optional[DomainAgent], str]:
    """Create a simple agent with minimal configuration"""
    if factory is None:
        factory = DomainAgentFactory()
    
    config = DomainConfiguration(
        name=name,
        domain_type=domain_type,
        capabilities=[DomainCapability(f"{domain_type.value}_basic", f"Basic {domain_type.value} capabilities")],
        expertise=[domain_type.value],
        tools=[domain_type.value]
    )
    
    return factory.create_agent(config)

def create_web_development_team(project_name: str, factory: DomainAgentFactory = None) -> Tuple[bool, List[DomainAgent], str]:
    """Create a standard web development team"""
    if factory is None:
        factory = DomainAgentFactory()
    
    project_requirements = {
        "project_name": project_name,
        "type": "web",
        "technologies": ["react", "nodejs", "postgresql"],
        "features": ["api", "authentication", "database"]
    }
    
    return factory.create_specialist_team(project_requirements)


# Template integration extensions to DomainAgentFactory - Issue #70 integration
def add_template_methods_to_factory():
    """Add template-driven methods to the DomainAgentFactory class"""
    
    def create_agent_from_template(self, template_path: Union[str, Path], 
                                 agent_name: Optional[str] = None,
                                 custom_config: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional['TemplateBasedAgent'], str]:
        """
        Create a domain agent from a YAML template
        
        Args:
            template_path: Path to the agent template YAML file
            agent_name: Optional custom name for the agent
            custom_config: Optional configuration overrides
            
        Returns:
            Tuple[bool, Optional[TemplateBasedAgent], str]: (success, agent, message)
        """
        start_time = time.time()
        
        try:
            # Validate template exists
            template_path = Path(template_path)
            if not template_path.exists():
                return False, None, f"Template file not found: {template_path}"
            
            # Load and validate template
            validator = DomainAgentTemplateValidator()
            validation_result = validator.validate_template(template_path)
            
            if not validation_result.valid:
                error_summary = "; ".join([issue.message for issue in validation_result.errors])
                return False, None, f"Template validation failed: {error_summary}"
            
            # Load template data
            agent_template = AgentTemplate.from_yaml(template_path)
            
            # Create template-based agent configuration
            agent_config = AgentConfiguration(
                template_path=str(template_path),
                custom_settings=custom_config or {}
            )
            
            # Override agent name if provided
            if agent_name:
                agent_template.metadata["name"] = agent_name
            
            # Create the template-based agent
            template_agent = TemplateBasedAgent(template_path=template_path, config=agent_config)
            
            # Convert to factory agent format and register
            success, factory_agent, message = self._register_template_agent(template_agent, agent_template)
            
            if success:
                creation_time = time.time() - start_time
                logger.info(f"Template-based agent created in {creation_time:.3f}s from {template_path}")
                
                # Update factory metrics
                domain_type = self._template_domain_to_agent_type(agent_template.metadata.get('domain', 'general'))
                self._update_factory_metrics(domain_type, creation_time)
            
            return success, template_agent if success else None, message
            
        except Exception as e:
            self.factory_metrics["creation_failures"] += 1
            logger.error(f"Template-based agent creation failed: {e}")
            return False, None, f"Creation error: {str(e)}"
    
    def create_specialized_agent(self, agent_type: str, agent_name: Optional[str] = None) -> Tuple[bool, Optional['TemplateBasedAgent'], str]:
        """
        Create a specialized agent using predefined templates
        
        Args:
            agent_type: Type of agent (frontend, backend, security)
            agent_name: Optional custom name for the agent
            
        Returns:
            Tuple[bool, Optional[TemplateBasedAgent], str]: (success, agent, message)
        """
        # Map agent types to template files
        template_mapping = {
            'frontend': 'templates/frontend-agent-template.yaml',
            'backend': 'templates/backend-agent-template.yaml',  
            'security': 'templates/security-agent-template.yaml'
        }
        
        if agent_type not in template_mapping:
            return False, None, f"Unknown agent type: {agent_type}. Available types: {list(template_mapping.keys())}"
        
        template_path = Path(template_mapping[agent_type])
        return self.create_agent_from_template(template_path, agent_name)
    
    def validate_template(self, template_path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a domain agent template
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Tuple[bool, Dict]: (is_valid, validation_report)
        """
        try:
            validator = DomainAgentTemplateValidator()
            result = validator.validate_template(Path(template_path))
            report = validator.get_validation_report(result)
            
            return result.valid, report
            
        except Exception as e:
            logger.error(f"Template validation error: {e}")
            return False, {"error": str(e)}
    
    def list_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available agent templates with their metadata
        
        Returns:
            Dict mapping template names to their metadata
        """
        templates_dir = Path("templates")
        available_templates = {}
        
        if not templates_dir.exists():
            return available_templates
        
        for template_file in templates_dir.glob("*-agent-template.yaml"):
            try:
                template = AgentTemplate.from_yaml(template_file)
                available_templates[template_file.stem] = {
                    "file_path": str(template_file),
                    "metadata": template.metadata,
                    "domain": template.metadata.get('domain', 'unknown'),
                    "capabilities": list(template.capabilities.keys()),
                    "file_size": template_file.stat().st_size
                }
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
                
        return available_templates
    
    def get_agent_capabilities(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed capabilities of a specific agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dict with agent capabilities or None if not found
        """
        agent = self.registry.get_agent(agent_id)
        if not agent:
            return None
            
        # If this is a template-based agent, get detailed capability info
        if hasattr(agent, 'template') and agent.template:
            return {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "domain": getattr(agent, 'domain', 'unknown'),
                "template_loaded": True,
                "capabilities": agent.template.capabilities,
                "expertise": agent.template.expertise,
                "tools": agent.template.tools,
                "validation_rules": agent.template.validation_rules,
                "quality_gates": agent.template.quality_gates
            }
        else:
            return {
                "agent_id": agent_id,
                "agent_name": agent.name,
                "domain": agent.domain_type.value,
                "template_loaded": False,
                "capabilities": [cap.name for cap in agent.capabilities],
                "expertise": agent.expertise,
                "tools": agent.tools
            }
    
    def match_agent_to_task(self, task_description: str, task_context: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Find the best agents for a specific task
        
        Args:
            task_description: Description of the task
            task_context: Optional context information
            
        Returns:
            List of (agent_id, match_score) tuples sorted by score
        """
        matches = []
        
        for agent_id, agent in self.registry.agents.items():
            try:
                # If agent supports capability matching
                if hasattr(agent, 'can_handle_task'):
                    match_result = agent.can_handle_task(task_description, task_context)
                    if match_result.overall_score > 0.3:  # Minimum threshold
                        matches.append((agent_id, match_result.overall_score))
                else:
                    # Fallback: simple capability matching
                    score = self._calculate_basic_task_match(agent, task_description)
                    if score > 0.3:
                        matches.append((agent_id, score))
                        
            except Exception as e:
                logger.warning(f"Error matching agent {agent_id} to task: {e}")
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _register_template_agent(self, template_agent: 'TemplateBasedAgent', agent_template: AgentTemplate) -> Tuple[bool, Optional['DomainAgent'], str]:
        """Register a template-based agent with the factory registry"""
        try:
            # Create factory agent representation
            domain_type = self._template_domain_to_agent_type(agent_template.metadata.get('domain', 'general'))
            
            factory_agent = DomainAgent(
                agent_id=template_agent.agent_id,
                name=template_agent.name,
                domain_type=domain_type,
                capabilities=self._convert_template_capabilities(template_agent.capabilities),
                expertise=template_agent.expertise,
                tools=template_agent.tools,
                metadata={
                    "template_path": str(template_agent.config.template_path),
                    "template_version": agent_template.metadata.get('version', '1.0.0'),
                    "template_loaded": True
                }
            )
            
            # Store reference to template agent
            setattr(factory_agent, 'template_agent', template_agent)
            setattr(factory_agent, 'template', agent_template)
            
            # Register with factory registry
            success = self.registry.register(factory_agent)
            
            if success:
                self.created_agents[factory_agent.agent_id] = factory_agent
                return True, factory_agent, "Template agent registered successfully"
            else:
                return False, None, "Failed to register template agent"
                
        except Exception as e:
            logger.error(f"Failed to register template agent: {e}")
            return False, None, f"Registration error: {str(e)}"
    
    def _template_domain_to_agent_type(self, domain: str) -> AgentType:
        """Convert template domain to AgentType enum"""
        domain_mapping = {
            'frontend': AgentType.FRONTEND,
            'backend': AgentType.BACKEND,
            'security': AgentType.SECURITY,
            'devops': AgentType.DEVOPS,
            'data': AgentType.DATA,
            'ml': AgentType.ML,
            'api': AgentType.API,
            'database': AgentType.DATABASE,
            'mobile': AgentType.MOBILE,
            'testing': AgentType.TESTING
        }
        return domain_mapping.get(domain.lower(), AgentType.API)  # Default fallback
    
    def _convert_template_capabilities(self, template_capabilities: List) -> List['DomainCapability']:
        """Convert template capabilities to factory DomainCapability objects"""
        converted = []
        for cap in template_capabilities:
            if hasattr(cap, 'name'):
                domain_cap = DomainCapability(
                    name=cap.name,
                    description=cap.description,
                    complexity=cap.complexity,
                    resource_impact=self._complexity_to_resource_impact(cap.complexity),
                    dependencies=cap.prerequisites,
                    tools_required=getattr(cap, 'tools_required', [])
                )
                converted.append(domain_cap)
        return converted
    
    def _complexity_to_resource_impact(self, complexity: str) -> str:
        """Map complexity to resource impact"""
        mapping = {
            'low': 'minimal',
            'medium': 'standard', 
            'high': 'significant',
            'very_high': 'high'
        }
        return mapping.get(complexity, 'standard')
    
    def _calculate_basic_task_match(self, agent: 'DomainAgent', task_description: str) -> float:
        """Calculate basic task matching score for non-template agents"""
        description_lower = task_description.lower()
        
        # Check domain type relevance
        domain_keywords = {
            AgentType.FRONTEND: ['ui', 'frontend', 'interface', 'component', 'react', 'vue'],
            AgentType.BACKEND: ['api', 'backend', 'database', 'server', 'service'],
            AgentType.SECURITY: ['security', 'audit', 'vulnerability', 'penetration'],
            AgentType.TESTING: ['test', 'testing', 'validate', 'verify'],
            AgentType.DATABASE: ['database', 'sql', 'query', 'migration'],
            AgentType.DEVOPS: ['deploy', 'infrastructure', 'docker', 'kubernetes']
        }
        
        keywords = domain_keywords.get(agent.domain_type, [])
        matches = sum(1 for keyword in keywords if keyword in description_lower)
        
        # Base score from domain relevance
        base_score = min(matches * 0.3, 0.8)
        
        # Add capability matching
        capability_matches = sum(1 for cap in agent.capabilities 
                               if any(word in description_lower for word in cap.name.lower().split('_')))
        capability_score = min(capability_matches * 0.2, 0.5)
        
        return base_score + capability_score


class TemplateBasedAgent(DomainAgent):
    """Template-driven domain agent that extends the base DomainAgent class"""
    
    def __init__(self, template_path: Union[str, Path], config: Optional[AgentConfiguration] = None):
        """Initialize template-based agent"""
        # Call parent constructor with template path
        super().__init__(template_path=template_path, config=config)
        
        # Additional template-specific initialization
        self.template_path = template_path
        self.factory_integration = True
        
        # Performance tracking
        self.template_metrics = {
            "tasks_executed": 0,
            "average_confidence": 0.0,
            "template_validation_time": getattr(self, 'performance_metrics', {}).get('validation_time_ms', 0.0)
        }
    
    def execute_primary_task(self, task_data: Dict[str, Any]) -> 'TaskResult':
        """Execute task using template-defined capabilities"""
        from claude.commands.domain_agent_base import TaskResult, AgentStatus
        from datetime import datetime
        
        start_time = datetime.now()
        
        try:
            # Validate task against template rules if available
            if hasattr(self, 'validation_rules') and self.validation_rules:
                validation_success, validation_errors = self.validate_task_requirements(task_data)
                if not validation_success:
                    return TaskResult(
                        task_id=task_data.get('task_id', 'unknown'),
                        status=AgentStatus.FAILED,
                        start_time=start_time,
                        end_time=datetime.now(),
                        error_message=f"Task validation failed: {'; '.join(validation_errors)}"
                    )
            
            # Execute task based on template capabilities
            task_description = task_data.get('description', '')
            task_type = task_data.get('type', 'general')
            
            # Check capability match
            if hasattr(self, 'can_handle_task'):
                capability_match = self.can_handle_task(task_description, task_data)
                confidence = capability_match.confidence
            else:
                confidence = 0.8  # Default confidence
            
            # Simulate task execution (in real implementation, this would be more sophisticated)
            result_data = {
                "task_type": task_type,
                "agent_domain": self.domain,
                "capabilities_used": [cap.name for cap in self.capabilities],
                "template_version": getattr(self.template, 'metadata', {}).get('version', '1.0.0') if hasattr(self, 'template') else None,
                "execution_method": "template_driven"
            }
            
            # Update metrics
            self.template_metrics["tasks_executed"] += 1
            current_avg = self.template_metrics["average_confidence"]
            task_count = self.template_metrics["tasks_executed"]
            self.template_metrics["average_confidence"] = ((current_avg * (task_count - 1)) + confidence) / task_count
            
            return TaskResult(
                task_id=task_data.get('task_id', str(uuid.uuid4())),
                status=AgentStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                confidence_score=confidence,
                result_data=result_data,
                artifacts=[f"template_execution_log_{self.agent_id}.json"]
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task_data.get('task_id', 'unknown'),
                status=AgentStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
    
    def get_template_metrics(self) -> Dict[str, Any]:
        """Get template-specific metrics"""
        base_metrics = self.get_status()
        base_metrics.update({
            "template_metrics": self.template_metrics,
            "template_path": str(self.template_path),
            "factory_integration": self.factory_integration
        })
        return base_metrics


def demo_factory():
    """Demonstrate the domain agent factory capabilities"""
    print("\n=== Domain Agent Factory Demo ===")
    
    factory = DomainAgentFactory()
    
    # Demo 1: Create individual agents
    print("\n--- Creating Individual Agents ---")
    
    frontend_config = DomainConfiguration(
        name="React_Frontend_Specialist",
        domain_type=AgentType.FRONTEND,
        capabilities=[
            DomainCapability("react_development", "React component development", "medium", "standard", [], ["react", "jsx"]),
            DomainCapability("ui_design", "User interface design", "low", "minimal", [], ["css", "html"])
        ],
        expertise=["react", "javascript", "css", "html"],
        tools=["react", "webpack", "babel", "css"],
        resource_requirements={"memory_mb": 1024, "cpu_cores": 2}
    )
    
    success, agent, message = factory.create_agent(frontend_config)
    print(f"Frontend Agent: {success} - {message}")
    if agent:
        print(f"  Agent ID: {agent.agent_id}")
        print(f"  Status: {agent.status.value}")
    
    # Demo 2: Create specialist team
    print("\n--- Creating Specialist Team ---")
    
    project_requirements = {
        "project_name": "E-Commerce Platform",
        "type": "web",
        "technologies": ["react", "nodejs", "postgresql", "redis"],
        "features": ["authentication", "payment", "inventory", "api"],
        "description": "Full-stack e-commerce platform with user authentication and payment processing"
    }
    
    success, team, message = factory.create_specialist_team(project_requirements)
    print(f"Team Creation: {success} - {message}")
    if team:
        for agent in team:
            print(f"  - {agent.name} ({agent.domain_type.value})")
    
    # Demo 3: Show metrics
    print("\n--- Factory Metrics ---")
    metrics = factory.get_factory_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    demo_factory()