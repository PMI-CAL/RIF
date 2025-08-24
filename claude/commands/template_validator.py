#!/usr/bin/env python3
"""
Domain Agent Template Validator - Issue #70
Comprehensive validation framework for domain agent templates
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
from jsonschema import validate, ValidationError as JsonSchemaError, Draft7Validator
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"
    WARNING = "warning" 
    INFO = "info"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    location: str = ""
    suggestion: Optional[str] = None
    rule_name: str = ""

@dataclass  
class ValidationResult:
    """Complete validation result for a template"""
    template_path: str
    valid: bool
    validation_time_ms: float
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    schema_validation: Optional['SchemaValidationResult'] = None
    semantic_validation: Optional['SemanticValidationResult'] = None
    performance_validation: Optional['PerformanceValidationResult'] = None
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues"""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
        
    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors"""
        return len(self.errors) > 0
        
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        return {
            "template_path": self.template_path,
            "valid": self.valid,
            "validation_time_ms": self.validation_time_ms,
            "total_issues": len(self.issues),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "info": len(self.info)
        }

@dataclass
class SchemaValidationResult:
    """Schema validation specific result"""
    valid: bool
    yaml_valid: bool = True
    structure_valid: bool = True
    required_sections: List[str] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    invalid_sections: List[str] = field(default_factory=list)

@dataclass
class SemanticValidationResult:
    """Semantic validation specific result"""
    valid: bool
    capabilities_valid: bool = True
    expertise_valid: bool = True
    tools_valid: bool = True
    validation_rules_valid: bool = True
    unknown_capabilities: List[str] = field(default_factory=list)
    unknown_expertise: List[str] = field(default_factory=list)

@dataclass
class PerformanceValidationResult:
    """Performance validation specific result"""
    valid: bool
    load_time_ms: float = 0.0
    parse_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

class CapabilityRegistry:
    """Registry of known capabilities and their definitions"""
    
    def __init__(self):
        self.capabilities = self._load_default_capabilities()
        self.similarity_threshold = 0.3
        
    def _load_default_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Load default capability definitions"""
        return {
            # Development capabilities
            "analysis": {
                "description": "Analyze code, requirements, and system components",
                "category": "development",
                "complexity": "medium",
                "prerequisites": [],
                "outputs": ["analysis_report", "findings", "recommendations"]
            },
            "implementation": {
                "description": "Implement solutions, features, and fixes",
                "category": "development", 
                "complexity": "high",
                "prerequisites": ["analysis"],
                "outputs": ["code", "implementation_artifacts"]
            },
            "validation": {
                "description": "Validate implementations against requirements",
                "category": "quality",
                "complexity": "medium",
                "prerequisites": ["implementation"],
                "outputs": ["validation_report", "test_results"]
            },
            "testing": {
                "description": "Design and execute comprehensive testing",
                "category": "quality",
                "complexity": "medium", 
                "prerequisites": [],
                "outputs": ["test_suite", "test_reports"]
            },
            "optimization": {
                "description": "Optimize performance and resource usage",
                "category": "performance",
                "complexity": "high",
                "prerequisites": ["analysis", "implementation"],
                "outputs": ["optimization_plan", "performance_improvements"]
            },
            "documentation": {
                "description": "Create comprehensive technical documentation",
                "category": "communication",
                "complexity": "low",
                "prerequisites": [],
                "outputs": ["documentation", "guides", "examples"]
            },
            "code_review": {
                "description": "Review code quality and adherence to standards",
                "category": "quality",
                "complexity": "medium",
                "prerequisites": [],
                "outputs": ["review_comments", "quality_assessment"]
            },
            "knowledge_sharing": {
                "description": "Share knowledge and best practices",
                "category": "communication",
                "complexity": "low",
                "prerequisites": [],
                "outputs": ["knowledge_artifacts", "training_materials"]
            },
            # UI/Frontend capabilities
            "ui_development": {
                "description": "Develop user interfaces and frontend components", 
                "category": "frontend",
                "complexity": "high",
                "prerequisites": ["analysis"],
                "outputs": ["ui_components", "frontend_code"]
            },
            "component_architecture": {
                "description": "Design reusable component systems",
                "category": "frontend", 
                "complexity": "high",
                "prerequisites": ["ui_development"],
                "outputs": ["component_library", "architecture_design"]
            },
            "responsive_design": {
                "description": "Create responsive and adaptive interfaces",
                "category": "frontend",
                "complexity": "medium",
                "prerequisites": ["ui_development"],
                "outputs": ["responsive_components", "css_frameworks"]
            },
            "accessibility_testing": {
                "description": "Test and ensure accessibility compliance",
                "category": "quality",
                "complexity": "medium",
                "prerequisites": ["ui_development"],
                "outputs": ["accessibility_reports", "compliance_certificates"]
            },
            "performance_optimization": {
                "description": "Optimize application performance",
                "category": "performance", 
                "complexity": "high",
                "prerequisites": ["implementation"],
                "outputs": ["performance_analysis", "optimization_recommendations"]
            },
            # Backend capabilities
            "api_development": {
                "description": "Design and implement REST/GraphQL APIs",
                "category": "backend",
                "complexity": "high",
                "prerequisites": ["analysis"],
                "outputs": ["api_endpoints", "api_documentation"]
            },
            "database_design": {
                "description": "Design and optimize database schemas",
                "category": "backend",
                "complexity": "high", 
                "prerequisites": ["analysis"],
                "outputs": ["database_schema", "migration_scripts"]
            },
            "microservices": {
                "description": "Design and implement microservice architectures",
                "category": "backend",
                "complexity": "very_high",
                "prerequisites": ["api_development", "database_design"],
                "outputs": ["service_architecture", "deployment_configs"]
            },
            # Security capabilities
            "security_audit": {
                "description": "Conduct comprehensive security audits",
                "category": "security",
                "complexity": "high",
                "prerequisites": ["analysis"],
                "outputs": ["security_report", "vulnerability_assessment"]
            },
            "penetration_testing": {
                "description": "Perform penetration testing and vulnerability assessment",
                "category": "security",
                "complexity": "very_high",
                "prerequisites": ["security_audit"],
                "outputs": ["pentest_report", "remediation_plan"]
            },
            "compliance_checking": {
                "description": "Verify compliance with security standards",
                "category": "security",
                "complexity": "medium",
                "prerequisites": ["security_audit"],
                "outputs": ["compliance_report", "certification_artifacts"]
            }
        }
        
    def is_valid_capability(self, capability: str) -> bool:
        """Check if capability is recognized"""
        return capability in self.capabilities
        
    def get_capability_info(self, capability: str) -> Optional[Dict[str, Any]]:
        """Get information about a capability"""
        return self.capabilities.get(capability)
        
    def suggest_similar_capabilities(self, capability: str) -> List[str]:
        """Suggest similar capabilities for unknown ones"""
        suggestions = []
        
        for known_cap in self.capabilities.keys():
            # Simple similarity check based on string matching
            similarity = self._calculate_similarity(capability.lower(), known_cap.lower())
            if similarity >= self.similarity_threshold:
                suggestions.append(known_cap)
                
        return suggestions[:3]  # Return top 3 suggestions
        
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple Jaccard similarity based on character n-grams
        def get_ngrams(s: str, n: int = 2) -> Set[str]:
            return set([s[i:i+n] for i in range(len(s)-n+1)])
            
        ngrams1 = get_ngrams(s1)
        ngrams2 = get_ngrams(s2)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0

class ExpertiseValidator:
    """Validator for expertise areas"""
    
    def __init__(self):
        self.known_technologies = self._load_known_technologies()
        self.known_patterns = self._load_known_patterns()
        self.known_domains = self._load_known_domains()
        
    def _load_known_technologies(self) -> Set[str]:
        """Load known technologies"""
        return {
            # Frontend technologies
            "react", "vue", "angular", "svelte", "typescript", "javascript", 
            "html", "css", "sass", "less", "webpack", "vite", "rollup",
            "jest", "cypress", "playwright", "testing-library",
            # Backend technologies  
            "python", "node.js", "java", "go", "rust", "c#", "php",
            "django", "flask", "fastapi", "express", "spring", "gin",
            "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            # Cloud and DevOps
            "docker", "kubernetes", "aws", "gcp", "azure", "terraform",
            "ansible", "jenkins", "gitlab-ci", "github-actions",
            # Security
            "oauth", "jwt", "ssl", "tls", "encryption", "authentication",
            "authorization", "owasp", "security-scanning"
        }
        
    def _load_known_patterns(self) -> Set[str]:
        """Load known patterns"""
        return {
            # Architectural patterns
            "mvc", "mvp", "mvvm", "clean_architecture", "hexagonal_architecture",
            "microservices", "monolith", "serverless", "event_driven", "cqrs",
            # Design patterns
            "singleton", "factory", "observer", "strategy", "decorator",
            "adapter", "facade", "proxy", "command", "state",
            "component_composition", "higher_order_components", "render_props",
            # Testing patterns
            "unit_testing", "integration_testing", "e2e_testing", "tdd", "bdd",
            "mocking", "stubbing", "test_doubles", "property_based_testing"
        }
        
    def _load_known_domains(self) -> Set[str]:
        """Load known domains"""
        return {
            "frontend", "backend", "fullstack", "mobile", "desktop",
            "web_development", "api_development", "database_design",
            "devops", "cloud_architecture", "security", "testing",
            "ui_design", "ux_design", "performance_optimization",
            "data_engineering", "machine_learning", "ai"
        }
        
    def validate_technologies(self, technologies: List[str]) -> List[str]:
        """Validate technology list, return unknown ones"""
        unknown = []
        for tech in technologies:
            if tech.lower() not in self.known_technologies:
                unknown.append(tech)
        return unknown
        
    def validate_patterns(self, patterns: List[str]) -> List[str]:
        """Validate pattern list, return unknown ones"""
        unknown = []
        for pattern in patterns:
            if pattern.lower() not in self.known_patterns:
                unknown.append(pattern)
        return unknown
        
    def validate_domains(self, domains: List[str]) -> List[str]:
        """Validate domain list, return unknown ones"""
        unknown = []
        for domain in domains:
            if domain.lower() not in self.known_domains:
                unknown.append(domain)
        return unknown

class DomainAgentTemplateValidator:
    """Main template validator class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize validator with optional configuration"""
        self.config = self._load_config(config_path)
        self.capability_registry = CapabilityRegistry()
        self.expertise_validator = ExpertiseValidator()
        self.template_schema = self._load_template_schema()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validator configuration"""
        default_config = {
            "strict_mode": True,
            "performance_checks": True,
            "max_load_time_ms": 100,
            "max_parse_time_ms": 50,
            "max_validation_time_ms": 200,
            "enable_semantic_validation": True,
            "enable_performance_validation": True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
        
    def _load_template_schema(self) -> Dict[str, Any]:
        """Load JSON schema for template validation"""
        # Define JSON schema for domain agent templates
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Domain Agent Template",
            "type": "object",
            "required": ["domain_agent_template"],
            "properties": {
                "domain_agent_template": {
                    "type": "object",
                    "required": ["metadata", "capabilities", "expertise", "tools", "validation_rules"],
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "required": ["name", "version", "domain"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                                "domain": {"type": "string", "minLength": 1},
                                "description": {"type": "string"},
                                "author": {"type": "string"},
                                "created": {"type": "string", "format": "date-time"},
                                "tags": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "capabilities": {
                            "type": "object",
                            "required": ["primary"],
                            "properties": {
                                "primary": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "secondary": {"type": "array", "items": {"type": "string"}},
                                "cross_domain": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "expertise": {
                            "type": "object",
                            "required": ["technologies", "patterns", "domains"],
                            "properties": {
                                "technologies": {
                                    "type": "object",
                                    "required": ["primary"],
                                    "properties": {
                                        "primary": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                        "secondary": {"type": "array", "items": {"type": "string"}},
                                        "emerging": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "patterns": {
                                    "type": "object",
                                    "properties": {
                                        "architectural": {"type": "array", "items": {"type": "string"}},
                                        "design": {"type": "array", "items": {"type": "string"}},
                                        "testing": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "domains": {
                                    "type": "object",
                                    "required": ["core"],
                                    "properties": {
                                        "core": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                        "adjacent": {"type": "array", "items": {"type": "string"}},
                                        "supporting": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        },
                        "tools": {
                            "type": "object",
                            "required": ["required"],
                            "properties": {
                                "required": {
                                    "type": "object",
                                    "properties": {
                                        "development": {"type": "array", "items": {"type": "string"}},
                                        "testing": {"type": "array", "items": {"type": "string"}},
                                        "build": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "recommended": {
                                    "type": "object",
                                    "properties": {
                                        "quality": {"type": "array", "items": {"type": "string"}},
                                        "performance": {"type": "array", "items": {"type": "string"}},
                                        "deployment": {"type": "array", "items": {"type": "string"}}
                                    }
                                },
                                "optional": {
                                    "type": "object",
                                    "properties": {
                                        "advanced": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        },
                        "validation_rules": {
                            "type": "object",
                            "required": ["code_quality"],
                            "properties": {
                                "code_quality": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "testing": {"type": "array", "items": {"type": "string"}},
                                "security": {"type": "array", "items": {"type": "string"}},
                                "performance": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "configuration": {
                            "type": "object",
                            "properties": {
                                "priority_weight": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                                "parallel_execution": {"type": "boolean"},
                                "timeout_overrides": {
                                    "type": "object",
                                    "properties": {
                                        "analysis": {"type": "integer", "minimum": 1},
                                        "implementation": {"type": "integer", "minimum": 1},
                                        "testing": {"type": "integer", "minimum": 1},
                                        "validation": {"type": "integer", "minimum": 1}
                                    }
                                },
                                "resource_limits": {
                                    "type": "object",
                                    "properties": {
                                        "memory_mb": {"type": "integer", "minimum": 1},
                                        "cpu_cores": {"type": "integer", "minimum": 1},
                                        "disk_space_mb": {"type": "integer", "minimum": 1}
                                    }
                                }
                            }
                        },
                        "integration": {
                            "type": "object",
                            "properties": {
                                "rif_workflow": {"type": "boolean"},
                                "knowledge_base": {"type": "boolean"},
                                "pattern_matching": {"type": "boolean"},
                                "learning_system": {"type": "boolean"},
                                "github_integration": {"type": "boolean"}
                            }
                        },
                        "quality_gates": {
                            "type": "object",
                            "properties": {
                                "functional": {"type": "array", "items": {"type": "string"}},
                                "performance": {"type": "array", "items": {"type": "string"}},
                                "integration": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    }
                }
            }
        }
    
    def validate_template(self, template_path: Union[str, Path]) -> ValidationResult:
        """Validate a complete domain agent template"""
        start_time = time.time()
        template_path = Path(template_path)
        
        result = ValidationResult(
            template_path=str(template_path),
            valid=True,
            validation_time_ms=0.0
        )
        
        # Phase 1: Basic file and YAML validation
        try:
            schema_result = self._validate_schema(template_path)
            result.schema_validation = schema_result
            
            if not schema_result.valid:
                result.valid = False
                result.issues.extend(self._convert_schema_issues(schema_result))
                
        except Exception as e:
            result.valid = False
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="schema",
                message=f"Schema validation failed: {e}",
                location="template",
                rule_name="schema_validation"
            ))
            
        # Phase 2: Semantic validation (if schema is valid)
        if result.schema_validation and result.schema_validation.valid and self.config.get("enable_semantic_validation", True):
            try:
                template_data = self._load_template_data(template_path)
                semantic_result = self._validate_semantic(template_data)
                result.semantic_validation = semantic_result
                
                if not semantic_result.valid:
                    result.valid = False
                    result.issues.extend(self._convert_semantic_issues(semantic_result))
                    
            except Exception as e:
                result.valid = False
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="semantic",
                    message=f"Semantic validation failed: {e}",
                    location="template",
                    rule_name="semantic_validation"
                ))
        
        # Phase 3: Performance validation
        if self.config.get("enable_performance_validation", True):
            try:
                perf_result = self._validate_performance(template_path)
                result.performance_validation = perf_result
                
                if not perf_result.valid:
                    result.warnings.extend(self._convert_performance_issues(perf_result))
                    
            except Exception as e:
                result.warnings.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="performance",
                    message=f"Performance validation failed: {e}",
                    location="template",
                    rule_name="performance_validation"
                ))
        
        # Calculate total validation time
        end_time = time.time()
        result.validation_time_ms = (end_time - start_time) * 1000
        
        # Log validation summary
        summary = result.get_summary()
        logger.info(f"Template validation completed: {summary}")
        
        return result
    
    def _validate_schema(self, template_path: Path) -> SchemaValidationResult:
        """Validate template against JSON schema"""
        result = SchemaValidationResult(valid=True)
        
        # Check if file exists
        if not template_path.exists():
            result.valid = False
            result.yaml_valid = False
            return result
            
        # Load and parse YAML
        try:
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.valid = False
            result.yaml_valid = False
            return result
        except Exception as e:
            result.valid = False
            result.yaml_valid = False
            return result
            
        # Validate against JSON schema
        try:
            validate(template_data, self.template_schema)
            result.structure_valid = True
            
            # Check for required sections
            template_section = template_data.get("domain_agent_template", {})
            required_sections = ["metadata", "capabilities", "expertise", "tools", "validation_rules"]
            
            for section in required_sections:
                if section in template_section:
                    result.required_sections.append(section)
                else:
                    result.missing_sections.append(section)
                    
            if result.missing_sections:
                result.valid = False
                result.structure_valid = False
                
        except JsonSchemaError as e:
            result.valid = False
            result.structure_valid = False
            result.invalid_sections.append(str(e.absolute_path))
            
        return result
    
    def _validate_semantic(self, template_data: Dict[str, Any]) -> SemanticValidationResult:
        """Validate semantic correctness of template data"""
        result = SemanticValidationResult(valid=True)
        
        template_section = template_data.get("domain_agent_template", {})
        
        # Validate capabilities
        capabilities = template_section.get("capabilities", {})
        all_capabilities = []
        
        for cap_type in ["primary", "secondary", "cross_domain"]:
            cap_list = capabilities.get(cap_type, [])
            all_capabilities.extend(cap_list)
            
        for capability in all_capabilities:
            if not self.capability_registry.is_valid_capability(capability):
                result.unknown_capabilities.append(capability)
                
        if result.unknown_capabilities:
            result.capabilities_valid = False
            result.valid = False
            
        # Validate expertise areas
        expertise = template_section.get("expertise", {})
        
        # Check technologies
        tech_section = expertise.get("technologies", {})
        all_technologies = []
        for tech_type in ["primary", "secondary", "emerging"]:
            tech_list = tech_section.get(tech_type, [])
            all_technologies.extend(tech_list)
            
        unknown_tech = self.expertise_validator.validate_technologies(all_technologies)
        result.unknown_expertise.extend([f"technology: {tech}" for tech in unknown_tech])
        
        # Check patterns
        patterns_section = expertise.get("patterns", {})
        all_patterns = []
        for pattern_type in ["architectural", "design", "testing"]:
            pattern_list = patterns_section.get(pattern_type, [])
            all_patterns.extend(pattern_list)
            
        unknown_patterns = self.expertise_validator.validate_patterns(all_patterns)
        result.unknown_expertise.extend([f"pattern: {pattern}" for pattern in unknown_patterns])
        
        # Check domains
        domains_section = expertise.get("domains", {})
        all_domains = []
        for domain_type in ["core", "adjacent", "supporting"]:
            domain_list = domains_section.get(domain_type, [])
            all_domains.extend(domain_list)
            
        unknown_domains = self.expertise_validator.validate_domains(all_domains)
        result.unknown_expertise.extend([f"domain: {domain}" for domain in unknown_domains])
        
        if result.unknown_expertise:
            result.expertise_valid = False
            if self.config.get("strict_mode", True):
                result.valid = False
                
        return result
    
    def _validate_performance(self, template_path: Path) -> PerformanceValidationResult:
        """Validate template loading and processing performance"""
        result = PerformanceValidationResult(valid=True)
        
        # Measure load time
        load_start = time.time()
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            load_end = time.time()
            result.load_time_ms = (load_end - load_start) * 1000
        except Exception:
            result.valid = False
            return result
            
        # Measure parse time
        parse_start = time.time()
        try:
            yaml.safe_load(content)
            parse_end = time.time()
            result.parse_time_ms = (parse_end - parse_start) * 1000
        except Exception:
            result.valid = False
            return result
            
        # Check against performance thresholds
        max_load_time = self.config.get("max_load_time_ms", 100)
        max_parse_time = self.config.get("max_parse_time_ms", 50)
        
        if result.load_time_ms > max_load_time:
            result.valid = False
            
        if result.parse_time_ms > max_parse_time:
            result.valid = False
            
        return result
    
    def _load_template_data(self, template_path: Path) -> Dict[str, Any]:
        """Load template data from file"""
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _convert_schema_issues(self, schema_result: SchemaValidationResult) -> List[ValidationIssue]:
        """Convert schema validation results to issues"""
        issues = []
        
        if not schema_result.yaml_valid:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="schema",
                message="Invalid YAML syntax",
                location="template",
                rule_name="yaml_syntax"
            ))
            
        if not schema_result.structure_valid:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="schema",
                message="Template structure does not match required schema",
                location="template",
                rule_name="schema_structure"
            ))
            
        for section in schema_result.missing_sections:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="schema",
                message=f"Missing required section: {section}",
                location=f"domain_agent_template.{section}",
                rule_name="required_sections"
            ))
            
        return issues
    
    def _convert_semantic_issues(self, semantic_result: SemanticValidationResult) -> List[ValidationIssue]:
        """Convert semantic validation results to issues"""
        issues = []
        
        for capability in semantic_result.unknown_capabilities:
            suggestions = self.capability_registry.suggest_similar_capabilities(capability)
            suggestion_text = f"Similar capabilities: {', '.join(suggestions)}" if suggestions else None
            
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR if self.config.get("strict_mode", True) else ValidationSeverity.WARNING,
                category="semantic",
                message=f"Unknown capability: {capability}",
                location="capabilities",
                suggestion=suggestion_text,
                rule_name="capability_validation"
            ))
            
        for expertise in semantic_result.unknown_expertise:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="semantic",
                message=f"Unknown expertise area: {expertise}",
                location="expertise",
                rule_name="expertise_validation"
            ))
            
        return issues
    
    def _convert_performance_issues(self, perf_result: PerformanceValidationResult) -> List[ValidationIssue]:
        """Convert performance validation results to issues"""
        issues = []
        
        max_load_time = self.config.get("max_load_time_ms", 100)
        max_parse_time = self.config.get("max_parse_time_ms", 50)
        
        if perf_result.load_time_ms > max_load_time:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance", 
                message=f"Template load time ({perf_result.load_time_ms:.1f}ms) exceeds recommended threshold ({max_load_time}ms)",
                location="template",
                rule_name="load_performance"
            ))
            
        if perf_result.parse_time_ms > max_parse_time:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                message=f"Template parse time ({perf_result.parse_time_ms:.1f}ms) exceeds recommended threshold ({max_parse_time}ms)",
                location="template",
                rule_name="parse_performance"
            ))
            
        return issues
    
    def validate_capability_mapping(self, capabilities: List[str]) -> ValidationResult:
        """Validate capability mapping for a list of capabilities"""
        start_time = time.time()
        
        result = ValidationResult(
            template_path="capability_mapping",
            valid=True,
            validation_time_ms=0.0
        )
        
        unknown_capabilities = []
        for capability in capabilities:
            if not self.capability_registry.is_valid_capability(capability):
                unknown_capabilities.append(capability)
                
        if unknown_capabilities:
            result.valid = False
            for capability in unknown_capabilities:
                suggestions = self.capability_registry.suggest_similar_capabilities(capability)
                suggestion_text = f"Similar capabilities: {', '.join(suggestions)}" if suggestions else None
                
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="capability_mapping",
                    message=f"Unknown capability: {capability}",
                    location="capabilities",
                    suggestion=suggestion_text,
                    rule_name="capability_validation"
                ))
        
        end_time = time.time()
        result.validation_time_ms = (end_time - start_time) * 1000
        
        return result
    
    def get_validation_report(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        return {
            "template_path": result.template_path,
            "validation_summary": result.get_summary(),
            "validation_status": "PASSED" if result.valid else "FAILED",
            "issues_by_severity": {
                "errors": len(result.errors),
                "warnings": len(result.warnings), 
                "info": len(result.info)
            },
            "detailed_issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion,
                    "rule": issue.rule_name
                }
                for issue in result.issues + result.warnings + result.info
            ],
            "performance_metrics": {
                "validation_time_ms": result.validation_time_ms,
                "load_time_ms": result.performance_validation.load_time_ms if result.performance_validation else None,
                "parse_time_ms": result.performance_validation.parse_time_ms if result.performance_validation else None
            },
            "schema_validation": {
                "yaml_valid": result.schema_validation.yaml_valid if result.schema_validation else None,
                "structure_valid": result.schema_validation.structure_valid if result.schema_validation else None,
                "missing_sections": result.schema_validation.missing_sections if result.schema_validation else []
            },
            "semantic_validation": {
                "capabilities_valid": result.semantic_validation.capabilities_valid if result.semantic_validation else None,
                "expertise_valid": result.semantic_validation.expertise_valid if result.semantic_validation else None,
                "unknown_capabilities": result.semantic_validation.unknown_capabilities if result.semantic_validation else [],
                "unknown_expertise": result.semantic_validation.unknown_expertise if result.semantic_validation else []
            }
        }


# CLI interface for standalone usage
def main():
    """Command-line interface for template validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate domain agent templates")
    parser.add_argument("template_path", help="Path to template file to validate")
    parser.add_argument("--config", help="Path to validator configuration file")
    parser.add_argument("--report", help="Path to save validation report")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    
    args = parser.parse_args()
    
    # Create validator
    validator = DomainAgentTemplateValidator(args.config)
    if args.strict:
        validator.config["strict_mode"] = True
    
    # Validate template
    result = validator.validate_template(args.template_path)
    
    # Print summary
    print(f"\n=== Template Validation Results ===")
    print(f"Template: {result.template_path}")
    print(f"Status: {'PASSED' if result.valid else 'FAILED'}")
    print(f"Validation time: {result.validation_time_ms:.1f}ms")
    print(f"Issues: {len(result.errors)} errors, {len(result.warnings)} warnings")
    
    # Print detailed issues
    if result.issues or result.warnings:
        print(f"\n=== Issues Found ===")
        for issue in result.issues + result.warnings:
            icon = "❌" if issue.severity == ValidationSeverity.ERROR else "⚠️"
            print(f"{icon} [{issue.category}] {issue.message}")
            if issue.location:
                print(f"   Location: {issue.location}")
            if issue.suggestion:
                print(f"   Suggestion: {issue.suggestion}")
            print()
    
    # Save report if requested
    if args.report:
        report = validator.get_validation_report(result)
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.report}")
    
    # Exit with appropriate code
    exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()