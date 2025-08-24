#!/usr/bin/env python3
"""
Domain Agent Base Class - Issue #70
Complete template-driven base class for all domain-specific agents in the RIF system.
"""

import json
import re
import time
import yaml
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import validation components
from claude.commands.template_validator import DomainAgentTemplateValidator, ValidationResult

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Enumeration of possible agent task statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentMetadata:
    """Agent metadata from template"""
    name: str
    version: str
    domain: str
    description: str = ""
    author: str = "RIF System"
    created: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class AgentCapability:
    """Individual capability definition"""
    name: str
    category: str
    complexity: str = "medium"
    prerequisites: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 30
    confidence_threshold: float = 0.8

@dataclass
class CapabilityMatch:
    """Result of capability matching for a task"""
    overall_score: float
    capability_breakdown: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    limitations: List[str] = field(default_factory=list)
    matched_capabilities: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: AgentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    confidence_score: float = 0.0
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.status == AgentStatus.COMPLETED

@dataclass
class AgentConfiguration:
    """Agent configuration from template"""
    priority_weight: float = 1.0
    parallel_execution: bool = True
    timeout_overrides: Dict[str, int] = field(default_factory=dict)
    resource_limits: Dict[str, int] = field(default_factory=dict)

class AgentTemplate:
    """Agent template loaded from YAML configuration"""
    
    def __init__(self, template_data: Dict[str, Any]):
        """Initialize template from parsed YAML data"""
        template_section = template_data.get("domain_agent_template", {})
        
        # Extract metadata
        metadata_section = template_section.get("metadata", {})
        self.metadata = AgentMetadata(
            name=metadata_section.get("name", "unknown"),
            version=metadata_section.get("version", "1.0.0"),
            domain=metadata_section.get("domain", "general"),
            description=metadata_section.get("description", ""),
            author=metadata_section.get("author", "RIF System"),
            created=metadata_section.get("created", ""),
            tags=metadata_section.get("tags", [])
        )
        
        # Extract capabilities
        capabilities_section = template_section.get("capabilities", {})
        self.capabilities = {
            "primary": capabilities_section.get("primary", []),
            "secondary": capabilities_section.get("secondary", []),
            "cross_domain": capabilities_section.get("cross_domain", [])
        }
        
        # Extract expertise
        self.expertise = template_section.get("expertise", {})
        
        # Extract tools
        self.tools = template_section.get("tools", {})
        
        # Extract validation rules
        self.validation_rules = template_section.get("validation_rules", {})
        
        # Extract configuration
        config_section = template_section.get("configuration", {})
        self.configuration = AgentConfiguration(
            priority_weight=config_section.get("priority_weight", 1.0),
            parallel_execution=config_section.get("parallel_execution", True),
            timeout_overrides=config_section.get("timeout_overrides", {}),
            resource_limits=config_section.get("resource_limits", {})
        )
        
        # Extract integration settings
        self.integration = template_section.get("integration", {})
        
        # Extract quality gates
        self.quality_gates = template_section.get("quality_gates", {})
        
        # Store raw template data
        self.raw_data = template_data
    
    @classmethod
    def from_yaml(cls, template_path: Union[str, Path]) -> 'AgentTemplate':
        """Load template from YAML file"""
        template_path = Path(template_path)
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        try:
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
                
            return cls(template_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in template file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading template: {e}")
    
    def get_all_capabilities(self) -> List[str]:
        """Get all capabilities from all categories"""
        all_caps = []
        for cap_list in self.capabilities.values():
            all_caps.extend(cap_list)
        return list(set(all_caps))  # Remove duplicates
    
    def get_primary_capabilities(self) -> List[str]:
        """Get primary capabilities"""
        return self.capabilities.get("primary", [])
    
    def get_domain(self) -> str:
        """Get domain from metadata"""
        return self.metadata.domain
    
    def get_name(self) -> str:
        """Get name from metadata"""
        return self.metadata.name

class DomainAgent(ABC):
    """Enhanced base class for all domain-specific agents with template support"""
    
    def __init__(self, template_path: Optional[Union[str, Path]] = None, 
                 domain: Optional[str] = None, capabilities: Optional[List[str]] = None, 
                 name: Optional[str] = None):
        """
        Initialize domain agent with template or fallback parameters
        
        Args:
            template_path: Path to YAML template file (preferred)
            domain: Domain specialization (fallback)
            capabilities: List of capabilities (fallback) 
            name: Agent name (fallback)
        """
        self.template = None
        self.validation_result = None
        
        # Template-based initialization (preferred)
        if template_path:
            self._initialize_from_template(template_path)
        # Fallback initialization
        elif domain and capabilities:
            self._initialize_from_params(domain, capabilities, name)
        else:
            raise ValueError("Must provide either template_path or (domain + capabilities)")
        
        # Common initialization
        self.agent_id = f"{self.domain}_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = datetime.now()
        self.task_history = []
        self.analysis_history = []  # Backward compatibility
        
        # Initialize knowledge paths
        self.knowledge_base_path = Path("knowledge")
        self.patterns_path = self.knowledge_base_path / "patterns"
        self.decisions_path = self.knowledge_base_path / "decisions"
        
        # Load domain capabilities configuration
        self.capability_config = self._load_capability_config()
        
        logger.info(f"Initialized {self.__class__.__name__} with domain: {self.domain}, template: {template_path is not None}")
    
    def _initialize_from_template(self, template_path: Union[str, Path]):
        """Initialize agent from template file"""
        # Validate template first
        validator = DomainAgentTemplateValidator()
        self.validation_result = validator.validate_template(template_path)
        
        if not self.validation_result.valid:
            error_summary = ", ".join([issue.message for issue in self.validation_result.errors[:3]])
            raise ValueError(f"Template validation failed: {error_summary}")
        
        # Load template
        self.template = AgentTemplate.from_yaml(template_path)
        
        # Set agent properties from template
        self.domain = self.template.get_domain()
        self.name = self.template.get_name()
        self.capabilities = self.template.get_all_capabilities()
        self.configuration = self.template.configuration
        
        logger.info(f"Loaded template: {self.name} v{self.template.metadata.version}")
    
    def _initialize_from_params(self, domain: str, capabilities: List[str], name: Optional[str]):
        """Initialize agent from direct parameters (fallback)"""
        self.domain = domain
        self.capabilities = capabilities
        self.name = name or f"{domain}-agent"
        self.configuration = AgentConfiguration()  # Default configuration
        
        logger.info(f"Initialized without template: {self.name}")
    
    def _load_capability_config(self) -> Dict[str, Any]:
        """Load capability configuration from config file"""
        config_path = Path("config") / "domain-capabilities.yaml"
        
        if not config_path.exists():
            logger.warning(f"Capability config not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading capability config: {e}")
            return {}
    
    @abstractmethod
    def execute_primary_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """
        Execute the primary task for this agent
        
        Args:
            task_data: Task data and context
            
        Returns:
            TaskResult with execution details
        """
        pass
    
    def can_handle_task(self, task_description: str, context: Optional[Dict] = None) -> CapabilityMatch:
        """
        Determine if this agent can handle a specific task
        
        Args:
            task_description: Description of the task
            context: Optional context information
            
        Returns:
            CapabilityMatch indicating how well this agent can handle the task
        """
        # Extract required capabilities from task description
        required_capabilities = self._extract_required_capabilities(task_description)
        
        # Calculate capability match scores
        capability_breakdown = {}
        matched_capabilities = []
        missing_capabilities = []
        
        total_score = 0.0
        capability_count = max(len(required_capabilities), 1)
        
        for required_cap in required_capabilities:
            if required_cap in self.capabilities:
                # Agent has this capability
                capability_score = self._calculate_capability_strength(required_cap)
                capability_breakdown[required_cap] = capability_score
                matched_capabilities.append(required_cap)
                total_score += capability_score
            else:
                # Agent doesn't have this capability
                capability_breakdown[required_cap] = 0.0
                missing_capabilities.append(required_cap)
        
        # Calculate overall score (0.0 to 1.0)
        overall_score = total_score / capability_count
        
        # Calculate confidence based on template validation and capability coverage
        confidence = self._calculate_confidence(overall_score, len(matched_capabilities), len(required_capabilities))
        
        # Identify limitations
        limitations = []
        if missing_capabilities:
            limitations.append(f"Missing capabilities: {', '.join(missing_capabilities)}")
        if overall_score < 0.7:
            limitations.append("Low capability match for this task")
        
        return CapabilityMatch(
            overall_score=overall_score,
            capability_breakdown=capability_breakdown,
            confidence=confidence,
            limitations=limitations,
            matched_capabilities=matched_capabilities,
            missing_capabilities=missing_capabilities
        )
    
    def _extract_required_capabilities(self, task_description: str) -> List[str]:
        """Extract likely required capabilities from task description"""
        task_lower = task_description.lower()
        required_caps = []
        
        # Map keywords to capabilities
        capability_keywords = {
            "analyze": "analysis",
            "analysis": "analysis", 
            "review": "analysis",
            "examine": "analysis",
            "implement": "implementation",
            "develop": "implementation",
            "create": "implementation",
            "build": "implementation",
            "code": "implementation",
            "test": "testing",
            "testing": "testing",
            "validate": "validation",
            "verify": "validation",
            "check": "validation",
            "optimize": "optimization",
            "improve": "optimization",
            "performance": "performance_optimization",
            "ui": "ui_development",
            "interface": "ui_development",
            "frontend": "ui_development",
            "api": "api_development",
            "backend": "api_development",
            "database": "database_design",
            "security": "security_audit",
            "secure": "security_audit",
            "document": "documentation",
            "docs": "documentation"
        }
        
        # Find matching capabilities
        for keyword, capability in capability_keywords.items():
            if keyword in task_lower:
                if capability not in required_caps:
                    required_caps.append(capability)
        
        # If no specific capabilities found, assume general analysis
        if not required_caps:
            required_caps.append("analysis")
        
        return required_caps
    
    def _calculate_capability_strength(self, capability: str) -> float:
        """Calculate how strong this agent is at a specific capability"""
        if not self.template:
            # Without template, assume medium strength
            return 0.7
        
        # Check if it's a primary capability
        if capability in self.template.capabilities.get("primary", []):
            return 1.0
        
        # Check if it's a secondary capability 
        if capability in self.template.capabilities.get("secondary", []):
            return 0.8
        
        # Check if it's a cross-domain capability
        if capability in self.template.capabilities.get("cross_domain", []):
            return 0.6
        
        # Not found in capabilities
        return 0.0
    
    def _calculate_confidence(self, overall_score: float, matched_count: int, required_count: int) -> float:
        """Calculate confidence in the capability match"""
        # Base confidence on overall score
        base_confidence = overall_score
        
        # Adjust for coverage ratio
        coverage_ratio = matched_count / max(required_count, 1)
        coverage_bonus = coverage_ratio * 0.2
        
        # Template validation bonus
        validation_bonus = 0.1 if self.template and self.validation_result and self.validation_result.valid else 0.0
        
        # Calculate final confidence (capped at 1.0)
        confidence = min(base_confidence + coverage_bonus + validation_bonus, 1.0)
        
        return confidence
    
    def execute_task(self, task_id: str, task_data: Dict[str, Any]) -> TaskResult:
        """Execute a task and return results"""
        start_time = datetime.now()
        
        # Create task result
        result = TaskResult(
            task_id=task_id,
            status=AgentStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Check if agent can handle this task
            task_description = task_data.get("description", "")
            capability_match = self.can_handle_task(task_description, task_data)
            
            # Set confidence score from capability match
            result.confidence_score = capability_match.confidence
            
            if capability_match.overall_score < 0.5:
                result.status = AgentStatus.FAILED
                result.error_message = f"Insufficient capability match: {capability_match.overall_score:.2f}"
                return result
            
            # Execute primary task (implemented by subclass)
            primary_result = self.execute_primary_task(task_data)
            
            # Update result with primary task outcome
            result.status = primary_result.status
            result.result_data = primary_result.result_data
            result.confidence_score = primary_result.confidence_score
            result.error_message = primary_result.error_message
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            result.status = AgentStatus.FAILED
            result.error_message = str(e)
        
        finally:
            # Finalize timing
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
            
            # Record task in history
            self.task_history.append(result)
            
            # Maintain history size
            if len(self.task_history) > 100:
                self.task_history = self.task_history[-100:]
        
        return result
    
    def has_capability(self, capability: str) -> bool:
        """Check if this agent has a specific capability"""
        return capability in self.capabilities
    
    def get_capability_strength(self, capability: str) -> float:
        """Get the strength of a specific capability (0.0 to 1.0)"""
        return self._calculate_capability_strength(capability)
    
    def get_domain_info(self) -> Dict[str, Any]:
        """Get information about this agent's domain and capabilities"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'domain': self.domain,
            'capabilities': self.capabilities,
            'created_at': self.created_at.isoformat(),
            'tasks_executed': len(self.task_history),
            'analyses_performed': len(self.analysis_history),  # Backward compatibility
            'template_loaded': self.template is not None,
            'validation_passed': self.validation_result and self.validation_result.valid if self.validation_result else None
        }
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about the loaded template"""
        if not self.template:
            return {
                'template_loaded': False,
                'template_path': None,
                'metadata': {},
                'validation_result': None
            }
        
        return {
            'template_loaded': True,
            'metadata': {
                'name': self.template.metadata.name,
                'version': self.template.metadata.version,
                'domain': self.template.metadata.domain,
                'description': self.template.metadata.description,
                'author': self.template.metadata.author,
                'tags': self.template.metadata.tags
            },
            'capabilities_count': len(self.template.get_all_capabilities()),
            'primary_capabilities': self.template.get_primary_capabilities(),
            'validation_passed': self.validation_result.valid if self.validation_result else None,
            'performance_metrics': {
                'validation_time_ms': self.validation_result.validation_time_ms if self.validation_result else None,
                'template_load_time_ms': getattr(self.validation_result.performance_validation, 'load_time_ms', None) if self.validation_result and self.validation_result.performance_validation else None
            }
        }
    
    def record_analysis(self, analysis_type: str, results: Dict[str, Any]) -> None:
        """Record an analysis in the agent's history (backward compatibility)"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'results_summary': {
                'issues_found': len(results.get('issues', [])),
                'suggestions_made': len(results.get('suggestions', [])),
                'confidence_score': results.get('confidence', 0.0)
            }
        }
        self.analysis_history.append(record)
        
        # Limit history size
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
    
    def get_task_history(self) -> List[TaskResult]:
        """Get the task execution history"""
        return self.task_history.copy()
    
    def get_recent_task_results(self, limit: int = 10) -> List[TaskResult]:
        """Get recent task results"""
        return self.task_history[-limit:] if self.task_history else []
    
    def load_domain_patterns(self) -> Dict[str, Any]:
        """Load patterns specific to this domain from knowledge base"""
        patterns = {}
        pattern_file = self.patterns_path / f"{self.domain}-patterns.json"
        
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r') as f:
                    patterns = json.load(f)
                logger.info(f"Loaded {len(patterns)} patterns for {self.domain} domain")
            except Exception as e:
                logger.error(f"Error loading patterns for {self.domain}: {e}")
        
        return patterns
    
    def save_domain_learning(self, learning_data: Dict[str, Any]) -> None:
        """Save learning data to the knowledge base"""
        learning_file = self.knowledge_base_path / "learning" / f"{self.domain}-learnings.json"
        learning_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            existing_learnings = []
            if learning_file.exists():
                with open(learning_file, 'r') as f:
                    existing_learnings = json.load(f)
            
            learning_entry = {
                'timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id,
                **learning_data
            }
            existing_learnings.append(learning_entry)
            
            with open(learning_file, 'w') as f:
                json.dump(existing_learnings, f, indent=2)
            
            logger.info(f"Saved learning data for {self.domain} domain")
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        if not self.task_history and not self.analysis_history:
            return {'tasks_executed': 0, 'analyses_performed': 0}
        
        # Analysis metrics (backward compatibility)
        recent_analyses = [
            a for a in self.analysis_history 
            if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(days=7)
        ]
        
        # Task performance metrics
        recent_tasks = [
            task for task in self.task_history
            if task.end_time and task.end_time > datetime.now() - timedelta(days=7)
        ]
        
        successful_tasks = [task for task in self.task_history if task.success]
        failed_tasks = [task for task in self.task_history if task.status == AgentStatus.FAILED]
        
        metrics = {
            # Task metrics
            'total_tasks_executed': len(self.task_history),
            'recent_tasks': len(recent_tasks),
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(successful_tasks) / max(len(self.task_history), 1),
            
            # Performance metrics
            'avg_task_duration_seconds': sum(
                task.duration_seconds for task in self.task_history if task.duration_seconds > 0
            ) / max(len([task for task in self.task_history if task.duration_seconds > 0]), 1),
            
            'avg_confidence_score': sum(
                task.confidence_score for task in self.task_history if task.confidence_score > 0
            ) / max(len([task for task in self.task_history if task.confidence_score > 0]), 1),
        }
        
        # Backward compatibility - analysis metrics
        if self.analysis_history:
            metrics.update({
                'total_analyses': len(self.analysis_history),
                'recent_analyses': len(recent_analyses),
                'avg_issues_per_analysis': sum(
                    a['results_summary']['issues_found'] for a in self.analysis_history
                ) / len(self.analysis_history),
                'avg_suggestions_per_analysis': sum(
                    a['results_summary']['suggestions_made'] for a in self.analysis_history
                ) / len(self.analysis_history)
            })
        
        return metrics