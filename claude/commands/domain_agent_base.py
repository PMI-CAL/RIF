#!/usr/bin/env python3
"""
RIF Domain Agent Base Class - Issue #70 Foundation
Template-driven base class for specialized domain agents with comprehensive functionality
"""

import json
import logging
import time
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

# Import template validation system
try:
    from claude.commands.template_validator import (
        DomainAgentTemplateValidator, ValidationResult, ValidationSeverity
    )
except ImportError:
    # Fallback if validator not available
    DomainAgentTemplateValidator = None
    ValidationResult = None
    ValidationSeverity = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Status of agent execution"""
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class TaskPriority(Enum):
    """Priority levels for agent tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentCapability:
    """Represents a specific capability of a domain agent"""
    name: str
    description: str
    confidence_level: float = 0.8
    prerequisites: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 30
    complexity: str = "medium"
    category: str = "general"

@dataclass
class TaskResult:
    """Result of executing a task"""
    task_id: str
    status: AgentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    confidence_score: float = 0.0
    result_data: Any = None
    error_message: str = ""
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentTemplate:
    """Domain agent template loaded from YAML"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    expertise: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    integration: Dict[str, Any] = field(default_factory=dict)
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, template_path: Path) -> 'AgentTemplate':
        """Load template from YAML file"""
        with open(template_path, 'r') as f:
            data = yaml.safe_load(f)
        
        template_data = data.get('domain_agent_template', {})
        return cls(
            metadata=template_data.get('metadata', {}),
            capabilities=template_data.get('capabilities', {}),
            expertise=template_data.get('expertise', {}),
            tools=template_data.get('tools', {}),
            validation_rules=template_data.get('validation_rules', {}),
            configuration=template_data.get('configuration', {}),
            integration=template_data.get('integration', {}),
            quality_gates=template_data.get('quality_gates', {})
        )

@dataclass
class AgentConfiguration:
    """Configuration for domain agent"""
    timeout_seconds: int = 300
    max_retries: int = 3
    memory_limit_mb: int = 512
    enable_logging: bool = True
    output_directory: str = "./output"
    template_path: Optional[str] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapabilityMatch:
    """Result of capability matching for a task"""
    overall_score: float
    capability_breakdown: Dict[str, float]
    confidence: float
    limitations: List[str]
    recommendations: List[str] = field(default_factory=list)

class DomainAgent(ABC):
    """
    Template-driven base class for specialized domain agents
    
    Provides comprehensive functionality for:
    - Template-based agent configuration
    - Capability matching and execution
    - Task execution and tracking
    - Result aggregation and reporting  
    - Resource management
    - Logging and monitoring
    - Error handling and recovery
    - Integration with RIF workflow system
    """
    
    def __init__(self, 
                 template_path: Optional[Union[str, Path]] = None,
                 domain: Optional[str] = None,
                 capabilities: Optional[List[str]] = None,
                 name: Optional[str] = None,
                 expertise: Optional[List[str]] = None,
                 tools: Optional[List[str]] = None,
                 config: Optional[AgentConfiguration] = None):
        """
        Initialize domain agent with template-based configuration
        
        Args:
            template_path: Path to YAML template file (preferred method)
            domain: Primary domain (fallback if no template)
            capabilities: Capability names (fallback if no template)
            name: Optional display name for the agent
            expertise: Areas of expertise (fallback if no template)
            tools: Tools required (fallback if no template)
            config: Agent configuration settings
        """
        self.agent_id = str(uuid.uuid4())
        self.template = None
        self.validation_result = None
        
        # Initialize performance metrics early
        self.performance_metrics = {
            "template_load_time_ms": 0.0,
            "validation_time_ms": 0.0,
            "initialization_time_ms": 0.0
        }
        
        # Initialize from template if provided
        if template_path:
            self.template = self._load_and_validate_template(template_path)
            self._initialize_from_template()
        else:
            # Fallback to direct initialization
            self.domain = domain or "general"
            self.name = name or f"{self.domain.title()}Agent"
            self.capabilities = [AgentCapability(name=cap, description=f"{cap} capability") for cap in capabilities or []]
            self.expertise = expertise or []
            self.tools = tools or []
            
        self.config = config or AgentConfiguration(template_path=str(template_path) if template_path else None)
        
        # Execution state
        self.status = AgentStatus.IDLE
        self.active_tasks: Dict[str, TaskResult] = {}
        self.completed_tasks: List[TaskResult] = []
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_confidence": 0.0,
            "total_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        # Setup output directory
        self.output_dir = Path(self.config.output_directory) / f"agent_{self.agent_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.name} agent ({self.agent_id}) with {len(self.capabilities)} capabilities")
        if self.template:
            logger.info(f"Agent loaded from template: {self.config.template_path}")

    def get_capability_names(self) -> List[str]:
        """Get list of capability names"""
        return [cap.name for cap in self.capabilities]
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has specific capability"""
        return capability_name in self.get_capability_names()
    
    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get capability by name"""
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    @abstractmethod
    def execute_primary_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """
        Execute the agent's primary task
        Must be implemented by concrete agent classes
        
        Args:
            task_data: Input data for the task
            
        Returns:
            TaskResult: Result of task execution
        """
        pass

    def execute_task(self, task_id: str, task_data: Dict[str, Any], 
                    priority: TaskPriority = TaskPriority.MEDIUM) -> TaskResult:
        """
        Execute a task with full lifecycle management
        
        Args:
            task_id: Unique identifier for the task
            task_data: Input data for the task
            priority: Priority level of the task
            
        Returns:
            TaskResult: Complete task result with metadata
        """
        start_time = datetime.now()
        
        # Create task result
        result = TaskResult(
            task_id=task_id,
            status=AgentStatus.ACTIVE,
            start_time=start_time,
            metadata={"priority": priority.name, "agent_id": self.agent_id}
        )
        
        self.active_tasks[task_id] = result
        self.status = AgentStatus.ACTIVE
        
        try:
            logger.info(f"Agent {self.name} starting task {task_id} with priority {priority.name}")
            
            # Execute the primary task (implemented by subclasses)
            result = self.execute_primary_task(task_data)
            result.task_id = task_id
            result.start_time = start_time
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.metadata.update({"priority": priority.name, "agent_id": self.agent_id})
            
            if result.status != AgentStatus.FAILED:
                result.status = AgentStatus.COMPLETED
                self.metrics["tasks_completed"] += 1
                logger.info(f"Task {task_id} completed successfully in {result.duration_seconds:.2f}s")
            else:
                self.metrics["tasks_failed"] += 1
                logger.error(f"Task {task_id} failed: {result.error_message}")
            
        except Exception as e:
            result.status = AgentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task_id} failed with exception: {e}")
            
        finally:
            # Update metrics and cleanup
            self.metrics["total_execution_time"] += result.duration_seconds
            total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
            if total_tasks > 0:
                self.metrics["success_rate"] = self.metrics["tasks_completed"] / total_tasks
            
            # Move from active to completed
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            self.completed_tasks.append(result)
            
            # Update status
            self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.ACTIVE
            
            # Save result if configured
            if self.config.enable_logging:
                self._save_task_result(result)
        
        return result

    def _save_task_result(self, result: TaskResult):
        """Save task result to disk for later analysis"""
        try:
            result_file = self.output_dir / f"task_{result.task_id}_result.json"
            result_dict = {
                "task_id": result.task_id,
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": result.duration_seconds,
                "confidence_score": result.confidence_score,
                "result_data": result.result_data,
                "error_message": result.error_message,
                "artifacts": result.artifacts,
                "metadata": result.metadata
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save task result: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "domain": self.domain,
            "status": self.status.value,
            "capabilities": self.get_capability_names(),
            "expertise": self.expertise,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "metrics": self.metrics.copy()
        }
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history"""
        recent_tasks = sorted(self.completed_tasks, 
                            key=lambda t: t.start_time, reverse=True)[:limit]
        
        return [{
            "task_id": task.task_id,
            "status": task.status.value,
            "duration_seconds": task.duration_seconds,
            "confidence_score": task.confidence_score,
            "start_time": task.start_time.isoformat(),
            "artifacts": task.artifacts
        } for task in recent_tasks]

    def reset_metrics(self):
        """Reset agent metrics"""
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_confidence": 0.0,
            "total_execution_time": 0.0,
            "success_rate": 0.0
        }
        self.completed_tasks.clear()
        logger.info(f"Metrics reset for agent {self.name}")

    def validate_task_data(self, task_data: Dict[str, Any]) -> bool:
        """
        Validate task data before execution
        Can be overridden by subclasses for custom validation
        
        Args:
            task_data: Task data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return isinstance(task_data, dict) and len(task_data) > 0

    def get_required_tools(self) -> List[str]:
        """Get list of tools required by this agent"""
        return self.tools.copy()

    def check_prerequisites(self) -> Dict[str, bool]:
        """
        Check if prerequisites are met for agent operation
        Can be overridden by subclasses
        
        Returns:
            Dict mapping prerequisite name to status
        """
        return {"basic_setup": True}

    def _load_and_validate_template(self, template_path: Union[str, Path]) -> AgentTemplate:
        """Load and validate domain agent template"""
        start_time = time.time()
        template_path = Path(template_path)
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
            
        # Validate template if validator is available
        if DomainAgentTemplateValidator:
            validator = DomainAgentTemplateValidator()
            self.validation_result = validator.validate_template(template_path)
            
            if not self.validation_result.valid:
                error_summary = "\n".join([f"- {issue.message}" for issue in self.validation_result.errors])
                raise ValueError(f"Template validation failed:\n{error_summary}")
                
            self.performance_metrics["validation_time_ms"] = self.validation_result.validation_time_ms
        
        # Load template data
        template = AgentTemplate.from_yaml(template_path)
        
        load_time = (time.time() - start_time) * 1000
        self.performance_metrics["template_load_time_ms"] = load_time
        
        return template
    
    def _initialize_from_template(self):
        """Initialize agent properties from loaded template"""
        if not self.template:
            return
            
        # Extract metadata
        metadata = self.template.metadata
        self.domain = metadata.get('domain', 'general')
        self.name = metadata.get('name', f"{self.domain.title()}Agent")
        self.version = metadata.get('version', '1.0.0')
        self.description = metadata.get('description', '')
        
        # Build capabilities from template
        self.capabilities = self._build_capabilities_from_template()
        
        # Extract expertise areas
        expertise = self.template.expertise
        self.expertise = self._flatten_expertise(expertise)
        
        # Extract tools
        tools = self.template.tools
        self.tools = self._flatten_tools(tools)
        
        # Store validation rules for runtime use
        self.validation_rules = self.template.validation_rules
        
        # Store quality gates
        self.quality_gates = self.template.quality_gates
        
        logger.info(f"Initialized agent from template - Domain: {self.domain}, Capabilities: {len(self.capabilities)}")
    
    def _build_capabilities_from_template(self) -> List[AgentCapability]:
        """Build capability objects from template definition"""
        capabilities = []
        
        # Load capability definitions from config
        capability_config_path = Path("config/domain-capabilities.yaml")
        capability_definitions = {}
        
        if capability_config_path.exists():
            try:
                with open(capability_config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    capability_definitions = config_data.get('capabilities', {})
            except Exception as e:
                logger.warning(f"Failed to load capability definitions: {e}")
        
        # Process all capability types from template
        template_capabilities = self.template.capabilities
        for cap_type, cap_list in template_capabilities.items():
            if isinstance(cap_list, list):
                for cap_name in cap_list:
                    # Look up capability definition
                    cap_def = self._find_capability_definition(cap_name, capability_definitions)
                    
                    capabilities.append(AgentCapability(
                        name=cap_name,
                        description=cap_def.get('description', f"{cap_name} capability"),
                        confidence_level=0.9 if cap_type == 'primary' else 0.7,
                        prerequisites=cap_def.get('prerequisites', []),
                        outputs=cap_def.get('outputs', []),
                        estimated_time_minutes=cap_def.get('estimated_time_minutes', 30),
                        complexity=cap_def.get('complexity', 'medium'),
                        category=cap_def.get('category', cap_type)
                    ))
                    
        return capabilities
    
    def _find_capability_definition(self, cap_name: str, capability_definitions: Dict) -> Dict[str, Any]:
        """Find capability definition in nested config structure"""
        for category, capabilities in capability_definitions.items():
            if isinstance(capabilities, dict) and cap_name in capabilities:
                return capabilities[cap_name]
        return {}
    
    def _flatten_expertise(self, expertise: Dict[str, Any]) -> List[str]:
        """Flatten expertise areas from template structure"""
        flattened = []
        
        for area_type, areas in expertise.items():
            if isinstance(areas, dict):
                for sub_type, items in areas.items():
                    if isinstance(items, list):
                        flattened.extend(items)
            elif isinstance(areas, list):
                flattened.extend(areas)
                
        return flattened
    
    def _flatten_tools(self, tools: Dict[str, Any]) -> List[str]:
        """Flatten tools from template structure"""
        flattened = []
        
        for tool_type, tool_categories in tools.items():
            if isinstance(tool_categories, dict):
                for category, items in tool_categories.items():
                    if isinstance(items, list):
                        flattened.extend(items)
            elif isinstance(tool_categories, list):
                flattened.extend(tool_categories)
                
        return flattened
    
    def can_handle_task(self, task_description: str, context: Optional[Dict] = None) -> CapabilityMatch:
        """Determine if this agent can handle a specific task"""
        context = context or {}
        
        # Extract required capabilities from task description
        required_capabilities = self._extract_required_capabilities(task_description)
        
        # Calculate capability matching scores
        capability_scores = {}
        total_score = 0.0
        matched_capabilities = 0
        
        for required_cap in required_capabilities:
            best_match_score = 0.0
            for agent_cap in self.capabilities:
                # Exact match
                if agent_cap.name == required_cap:
                    score = agent_cap.confidence_level
                # Partial match based on similarity
                else:
                    score = self._calculate_capability_similarity(required_cap, agent_cap.name) * agent_cap.confidence_level
                
                best_match_score = max(best_match_score, score)
            
            capability_scores[required_cap] = best_match_score
            total_score += best_match_score
            if best_match_score > 0.5:
                matched_capabilities += 1
        
        # Calculate overall match score
        if required_capabilities:
            overall_score = total_score / len(required_capabilities)
        else:
            overall_score = 0.8  # Default score if no specific capabilities required
        
        # Calculate confidence based on expertise match
        expertise_match = self._calculate_expertise_match(task_description, context)
        confidence = (overall_score + expertise_match) / 2
        
        # Identify limitations
        limitations = []
        for cap, score in capability_scores.items():
            if score < 0.5:
                limitations.append(f"Low confidence in {cap} (score: {score:.2f})")
        
        # Generate recommendations
        recommendations = []
        if overall_score < 0.7:
            recommendations.append("Consider using multiple agents for this task")
        if expertise_match < 0.6:
            recommendations.append("Task may require domain knowledge outside agent expertise")
        
        return CapabilityMatch(
            overall_score=overall_score,
            capability_breakdown=capability_scores,
            confidence=confidence,
            limitations=limitations,
            recommendations=recommendations
        )
    
    def _extract_required_capabilities(self, task_description: str) -> List[str]:
        """Extract required capabilities from task description"""
        # Simple keyword matching - can be enhanced with NLP
        capability_keywords = {
            'analysis': ['analyze', 'review', 'examine', 'investigate'],
            'implementation': ['implement', 'develop', 'build', 'create', 'code'],
            'testing': ['test', 'verify', 'validate', 'check'],
            'optimization': ['optimize', 'improve', 'enhance', 'performance'],
            'documentation': ['document', 'write docs', 'readme', 'guide'],
            'ui_development': ['ui', 'interface', 'frontend', 'component'],
            'api_development': ['api', 'endpoint', 'rest', 'graphql'],
            'security_audit': ['security', 'audit', 'vulnerability', 'secure'],
            'database_design': ['database', 'schema', 'sql', 'migration']
        }
        
        description_lower = task_description.lower()
        required_caps = []
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                required_caps.append(capability)
        
        return required_caps or ['analysis']  # Default to analysis if nothing specific found
    
    def _calculate_capability_similarity(self, cap1: str, cap2: str) -> float:
        """Calculate similarity between two capability names"""
        # Simple similarity based on common words
        words1 = set(cap1.lower().split('_'))
        words2 = set(cap2.lower().split('_'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_expertise_match(self, task_description: str, context: Dict) -> float:
        """Calculate how well agent expertise matches task requirements"""
        description_lower = task_description.lower()
        
        matched_expertise = 0
        total_expertise = len(self.expertise) if self.expertise else 1
        
        for expertise_area in self.expertise:
            if expertise_area.lower() in description_lower:
                matched_expertise += 1
        
        # Check context for technology mentions
        context_str = str(context).lower()
        for expertise_area in self.expertise:
            if expertise_area.lower() in context_str:
                matched_expertise += 0.5
        
        return min(matched_expertise / total_expertise, 1.0)
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about the loaded template"""
        if not self.template:
            return {"template_loaded": False}
        
        return {
            "template_loaded": True,
            "template_path": self.config.template_path,
            "metadata": self.template.metadata,
            "validation_passed": self.validation_result.valid if self.validation_result else None,
            "validation_issues": len(self.validation_result.issues) if self.validation_result else 0,
            "performance_metrics": self.performance_metrics
        }

    def __str__(self) -> str:
        return f"{self.name}(domain={self.domain}, capabilities={len(self.capabilities)}, status={self.status.value})"

    def __repr__(self) -> str:
        template_info = f", template={bool(self.template)}" if hasattr(self, 'template') else ""
        return f"DomainAgent(domain='{self.domain}', name='{self.name}', agent_id='{self.agent_id}'{template_info})"