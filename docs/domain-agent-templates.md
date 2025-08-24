# Domain Agent Templates - Complete Guide

## Overview

The RIF Domain Agent Template System provides a comprehensive framework for creating, validating, and managing specialized domain agents through YAML-based templates. This system enables:

- **Template-driven agent configuration** with comprehensive validation
- **Capability mapping and expertise definition** for precise task matching
- **Specialized domain templates** for frontend, backend, and security agents
- **Factory integration** for seamless agent creation and management
- **Quality gates and validation rules** for enterprise-grade reliability

## Architecture

### Core Components

```
templates/                          # Agent template definitions
├── domain-agent-template.yaml     # Master template schema
├── frontend-agent-template.yaml   # Frontend specialist template
├── backend-agent-template.yaml    # Backend specialist template  
└── security-agent-template.yaml   # Security specialist template

claude/commands/
├── template_validator.py          # Template validation framework
├── domain_agent_base.py           # Template-driven base agent class
└── domain_agent_factory.py        # Factory with template integration

config/
└── domain-capabilities.yaml       # Capability and expertise definitions

tests/
└── test_domain_template_system.py # Comprehensive test suite
```

### Template Schema Structure

```yaml
domain_agent_template:
  metadata:              # Agent identification and versioning
  capabilities:          # What the agent can do (primary, secondary, cross-domain)
  expertise:            # What the agent knows (technologies, patterns, domains)
  tools:               # Required, recommended, and optional tools
  validation_rules:    # Quality gates and validation requirements
  configuration:       # Agent behavior and resource limits
  integration:         # RIF system integration points
  quality_gates:       # Success metrics and performance thresholds
```

## Quick Start

### 1. Creating an Agent from Template

```python
from claude.commands.domain_agent_base import DomainAgent
from pathlib import Path

# Create a frontend specialist agent
agent = DomainAgent(template_path="templates/frontend-agent-template.yaml")

print(f"Agent: {agent.name}")
print(f"Domain: {agent.domain}")
print(f"Capabilities: {[cap.name for cap in agent.capabilities]}")
```

### 2. Validating a Template

```python
from claude.commands.template_validator import DomainAgentTemplateValidator

validator = DomainAgentTemplateValidator()
result = validator.validate_template("templates/frontend-agent-template.yaml")

if result.valid:
    print("Template is valid!")
else:
    for error in result.errors:
        print(f"Error: {error.message}")
```

### 3. Using the Factory

```python
from claude.commands.domain_agent_factory import DomainAgentFactory

factory = DomainAgentFactory()

# Create specialized agents
success, frontend_agent, message = factory.create_specialized_agent("frontend", "MyFrontendAgent")
success, backend_agent, message = factory.create_specialized_agent("backend", "MyBackendAgent")

# List available templates
templates = factory.list_available_templates()
for name, info in templates.items():
    print(f"{name}: {info['domain']} - {len(info['capabilities'])} capabilities")
```

## Template Development

### Creating a Custom Template

1. **Start with the base template**:
   ```bash
   cp templates/domain-agent-template.yaml templates/my-custom-template.yaml
   ```

2. **Customize the metadata**:
   ```yaml
   metadata:
     name: "my-custom-agent"
     version: "1.0.0"
     domain: "custom"
     description: "Custom domain agent for specific use case"
   ```

3. **Define capabilities**:
   ```yaml
   capabilities:
     primary:
       - "custom_analysis"      # Your primary capabilities
       - "specialized_processing"
     secondary:
       - "reporting"           # Supporting capabilities
   ```

4. **Specify expertise**:
   ```yaml
   expertise:
     technologies:
       primary: ["python", "custom-framework"]
       secondary: ["docker", "kubernetes"]
     patterns:
       architectural: ["custom_pattern", "microservices"]
     domains:
       core: ["custom_domain"]
   ```

5. **Validate the template**:
   ```python
   validator = DomainAgentTemplateValidator()
   result = validator.validate_template("templates/my-custom-template.yaml")
   ```

### Template Inheritance

Templates support inheritance from the base template:

```yaml
template_inheritance:
  base_template: "domain-agent-template.yaml"
  specialization: "custom"
  customization_points:
    - "custom_frameworks"
    - "specialized_tools"
  protected_sections:
    - "security_requirements"
    - "performance_thresholds"
```

## Validation Framework

### Validation Levels

1. **Schema Validation**: YAML structure and required fields
2. **Semantic Validation**: Capability and expertise recognition
3. **Performance Validation**: Template loading and processing speed
4. **Integration Validation**: Compatibility with RIF workflow

### Custom Validation Rules

Add custom validation in `config/domain-capabilities.yaml`:

```yaml
validation_rules:
  custom_quality:
    custom_rule_name:
      description: "Custom validation rule"
      automated_tools: ["custom-tool"]
      manual_checks: ["custom-manual-check"]
```

### Validation CLI

Use the template validator from command line:

```bash
python claude/commands/template_validator.py templates/my-template.yaml --strict --report validation-report.json
```

## Capability System

### Capability Categories

- **Development**: `analysis`, `implementation`, `optimization`, `refactoring`
- **Frontend**: `ui_development`, `component_architecture`, `responsive_design`
- **Backend**: `api_development`, `database_design`, `microservices`
- **Security**: `security_audit`, `penetration_testing`, `compliance_checking`
- **Testing**: `unit_testing`, `integration_testing`, `e2e_testing`
- **Quality**: `code_review`, `documentation`, `quality_assurance`

### Adding New Capabilities

1. **Define in capabilities config**:
   ```yaml
   capabilities:
     your_category:
       new_capability:
         description: "Description of the capability"
         complexity: "medium"
         estimated_time_minutes: 45
         prerequisites: ["prerequisite_capability"]
         outputs: ["expected_output1", "expected_output2"]
   ```

2. **Use in templates**:
   ```yaml
   capabilities:
     primary:
       - "new_capability"
   ```

## Task Matching

### Capability Matching

Agents can automatically match their capabilities to tasks:

```python
# Task matching
match = agent.can_handle_task(
    "Implement a React component with accessibility features",
    context={"framework": "react", "requirements": ["wcag", "responsive"]}
)

print(f"Match score: {match.overall_score}")
print(f"Confidence: {match.confidence}")
print(f"Limitations: {match.limitations}")
```

### Factory-Level Task Matching

```python
# Find best agents for a task
matches = factory.match_agent_to_task(
    "Build a secure REST API with authentication",
    context={"security_level": "high", "auth_method": "oauth"}
)

for agent_id, score in matches[:3]:  # Top 3 matches
    agent = factory.registry.get_agent(agent_id)
    print(f"{agent.name}: {score:.2f}")
```

## Specialized Templates

### Frontend Agent Template

Specialized for UI development with:
- **Primary capabilities**: `ui_development`, `component_architecture`, `responsive_design`
- **Technologies**: React, Vue, TypeScript, CSS
- **Validation rules**: WCAG 2.1 AA compliance, Core Web Vitals, browser compatibility
- **Quality gates**: Lighthouse scores, accessibility testing, performance benchmarks

```python
# Create frontend specialist
agent = DomainAgent(template_path="templates/frontend-agent-template.yaml")
```

### Backend Agent Template

Specialized for server-side development with:
- **Primary capabilities**: `api_development`, `database_design`, `data_processing`
- **Technologies**: Python, Node.js, PostgreSQL, Redis
- **Validation rules**: API documentation, security compliance, performance thresholds
- **Quality gates**: Response times, database optimization, concurrent load handling

```python
# Create backend specialist
agent = DomainAgent(template_path="templates/backend-agent-template.yaml")
```

### Security Agent Template

Specialized for security analysis with:
- **Primary capabilities**: `security_audit`, `threat_modeling`, `secure_coding`
- **Technologies**: OWASP tools, penetration testing frameworks
- **Validation rules**: OWASP Top 10 coverage, compliance frameworks
- **Quality gates**: Vulnerability detection, risk assessment, remediation guidance

```python
# Create security specialist
agent = DomainAgent(template_path="templates/security-agent-template.yaml")
```

## Integration with RIF Workflow

### Workflow Integration Points

Templates specify integration with RIF components:

```yaml
integration:
  rif_workflow: true          # Integrates with RIF workflow engine
  knowledge_base: true        # Uses knowledge base for learning
  pattern_matching: true      # Applies successful patterns
  learning_system: true       # Contributes to system learning
  github_integration: true    # GitHub issue integration
```

### Agent Orchestration

Agents created from templates work seamlessly with RIF orchestration:

```python
# Orchestrator can automatically select appropriate agents
task = {
    "description": "Implement user authentication with security audit",
    "context": {"security_level": "high", "compliance": "SOC2"}
}

# System automatically matches frontend + backend + security agents
selected_agents = orchestrator.select_agents_for_task(task)
```

## Performance and Monitoring

### Template Performance Metrics

Templates include performance tracking:

```yaml
quality_gates:
  performance:
    - "template_processing_under_100ms"
    - "validation_under_50ms"
    - "initialization_under_200ms"
```

### Agent Performance Monitoring

```python
# Get performance metrics
metrics = agent.get_template_info()
print(f"Template load time: {metrics['performance_metrics']['template_load_time_ms']}ms")
print(f"Validation time: {metrics['performance_metrics']['validation_time_ms']}ms")
```

### Factory Metrics

```python
# Factory performance tracking
factory_metrics = factory.get_factory_metrics()
print(f"Agents created: {factory_metrics['agents_created']}")
print(f"Average creation time: {factory_metrics['average_creation_time']}")
```

## Testing

### Running Template Tests

```bash
# Run all template system tests
python -m pytest tests/test_domain_template_system.py -v

# Run specific test categories
python -m pytest tests/test_domain_template_system.py::TestDomainAgentTemplateValidator -v
python -m pytest tests/test_domain_template_system.py::TestDomainAgent -v
```

### Test Categories

- **Template Loading**: YAML parsing, schema validation
- **Capability Registry**: Capability recognition, similarity matching
- **Expertise Validation**: Technology/pattern/domain validation
- **Agent Creation**: Template-driven agent initialization
- **Task Matching**: Capability-to-task matching accuracy
- **Performance**: Template loading speed, validation performance
- **Integration**: End-to-end workflow testing

## Advanced Usage

### Custom Capability Definitions

Extend the capability system:

```python
from claude.commands.template_validator import CapabilityRegistry

registry = CapabilityRegistry()

# Add custom capability
registry.capabilities["custom_domain"] = {
    "custom_capability": {
        "description": "Custom domain-specific capability",
        "complexity": "high",
        "prerequisites": ["analysis"],
        "outputs": ["custom_output"]
    }
}
```

### Template Validation Hooks

Add custom validation logic:

```python
class CustomTemplateValidator(DomainAgentTemplateValidator):
    def validate_custom_rules(self, template_data):
        """Add custom validation logic"""
        # Custom validation implementation
        pass
```

### Factory Extensions

Extend factory functionality:

```python
class CustomAgentFactory(DomainAgentFactory):
    def create_project_team(self, project_spec):
        """Create a complete project team based on specifications"""
        # Custom team creation logic
        pass
```

## Troubleshooting

### Common Issues

1. **Template validation fails**:
   - Check YAML syntax with a validator
   - Verify all required sections are present
   - Ensure capabilities are defined in `domain-capabilities.yaml`

2. **Agent initialization errors**:
   - Confirm template file exists and is readable
   - Check that all dependencies are installed
   - Verify configuration file paths

3. **Capability matching issues**:
   - Update capability definitions
   - Check similarity thresholds
   - Review task description keywords

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all template operations will show detailed logs
```

### Validation Reports

Generate detailed validation reports:

```python
validator = DomainAgentTemplateValidator()
result = validator.validate_template("templates/my-template.yaml")
report = validator.get_validation_report(result)

# Save detailed report
with open("validation-report.json", "w") as f:
    json.dump(report, f, indent=2)
```

## Best Practices

### Template Design

1. **Keep templates focused**: Each template should serve a specific domain
2. **Use inheritance**: Extend base template rather than duplicating
3. **Document capabilities**: Provide clear descriptions and examples
4. **Define quality gates**: Specify measurable success criteria
5. **Version templates**: Use semantic versioning for template evolution

### Capability Definition

1. **Be specific**: Use descriptive capability names
2. **Define prerequisites**: Clearly specify capability dependencies
3. **Estimate complexity**: Provide realistic time and resource estimates
4. **Group logically**: Organize capabilities by domain and function

### Validation Rules

1. **Make rules testable**: Ensure validation rules can be automated
2. **Set realistic thresholds**: Base performance targets on actual metrics
3. **Document exceptions**: Clearly specify when rules don't apply
4. **Regular review**: Update validation rules based on experience

### Performance Optimization

1. **Cache templates**: Reuse loaded and validated templates
2. **Lazy loading**: Load capabilities and expertise on demand
3. **Batch operations**: Process multiple templates together when possible
4. **Monitor metrics**: Track performance and optimize bottlenecks

## Migration Guide

### From Legacy Agents

1. **Analyze existing agent structure**
2. **Create template matching capabilities**
3. **Define expertise areas based on existing functionality**
4. **Add validation rules for quality gates**
5. **Test template-driven agent behavior**
6. **Migrate incrementally with fallback support**

### Template Versioning

Handle template evolution:

```yaml
metadata:
  version: "2.0.0"          # Semantic versioning
  
compatibility:
  min_rif_version: "1.2.0"  # Minimum RIF system version
  breaking_changes:         # Document breaking changes
    - "capability_structure_changed"
    - "validation_rules_updated"
```

## API Reference

### Core Classes

- **`DomainAgent`**: Template-driven base agent class
- **`AgentTemplate`**: Template data container
- **`DomainAgentTemplateValidator`**: Template validation framework
- **`CapabilityRegistry`**: Capability management system
- **`DomainAgentFactory`**: Agent creation and management factory

### Key Methods

- **`validate_template(path)`**: Validate template file
- **`can_handle_task(description, context)`**: Task capability matching
- **`create_agent_from_template(path, name, config)`**: Create template-based agent
- **`match_agent_to_task(description, context)`**: Find best agents for task

## Contributing

### Adding New Capabilities

1. Define capability in `config/domain-capabilities.yaml`
2. Add validation rules if needed
3. Update documentation
4. Add tests for new capability
5. Submit PR with examples

### Creating New Templates

1. Copy base template as starting point
2. Customize for specific domain
3. Add comprehensive validation rules
4. Include usage examples
5. Add specialized tests

### Extending Validation

1. Add validation logic to `template_validator.py`
2. Create test cases for new validation
3. Update schema documentation
4. Ensure backward compatibility

## Support

For questions, issues, or contributions:
- Check existing GitHub issues
- Review test cases for examples
- Consult API documentation
- Submit detailed bug reports with template files and error messages

The Domain Agent Template System provides a robust, extensible foundation for creating and managing specialized agents in the RIF ecosystem. With comprehensive validation, performance monitoring, and seamless integration, it enables scalable, maintainable agent development for any domain.