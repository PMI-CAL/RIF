# Development Framework Setup Guide

This guide provides detailed instructions for setting up and configuring the development framework for projects in any technology stack.

## Prerequisites

### System Requirements
- **Git**: Version control and hooks integration
- **GitHub CLI** (optional): Enhanced GitHub integration
- **Claude Code**: AI-powered development assistant
- **Technology Stack**: Whatever languages/frameworks your project requires

### Framework Requirements
The framework is technology-agnostic and adapts to your project's needs automatically.

## Installation Methods

### Method 1: Direct Framework Setup (Recommended)

1. **Download Framework**
   ```bash
   # Clone or download the framework
   git clone <framework-repo> dev-framework
   cd dev-framework
   ```

2. **Copy Framework to Your Project**
   ```bash
   # Copy to existing project
   cp -r claude/ /path/to/your/project/.claude
   cp -r config/ /path/to/your/project/framework-config
   cp -r docs/ /path/to/your/project/framework-docs
   
   # Navigate to your project
   cd /path/to/your/project
   ```

3. **Initialize Framework**
   ```bash
   # Initialize git if not already done
   git init
   
   # Add framework files
   git add .claude/ framework-config/ framework-docs/
   git commit -m "Add development framework integration"
   ```

### Method 2: New Project Setup

If starting a new project:

1. **Create Project Structure**
   ```bash
   mkdir my-new-project
   cd my-new-project
   
   # Copy framework components
   cp -r /path/to/framework/claude .claude
   cp -r /path/to/framework/config framework-config
   cp -r /path/to/framework/docs framework-docs
   ```

2. **Customize Configuration**
   - Edit `.claude/rules/` for project-specific rules
   - Update `.claude/context/` for technology stack context
   - Customize development commands in `.claude/commands/`

## Framework Components

### Claude Code Integration

The framework integrates with Claude Code for intelligent development assistance.

#### Core Files:
- **`.claude/rules/github-workflow.md`**: GitHub integration rules and workflow automation
- **`.claude/rules/code-quality.md`**: Quality standards and development principles
- **`.claude/commands/development.md`**: Technology-agnostic development commands

#### Development Agent System:
- **Project Manager**: Sprint planning, timeline management
- **System Architect**: Design decisions, technical reviews
- **Developer**: Implementation, testing, documentation
- **Quality Assurance**: Quality gates, testing strategies
- **Business Analyst**: Requirements analysis, business logic
- **Scrum Master**: Process improvement, team coordination
- **Context Server Discovery**: Automatic context server integration

#### Customization:
```bash
# Edit core development rules
nano .claude/rules/code-quality.md

# Customize project-specific context
nano .claude/context/core-module-processing.md

# Configure development commands
nano .claude/commands/development.md
```

### Quality Standards Framework

#### Development Principles:
- **Clean Code**: Maintainable, readable, and well-structured code
- **Design Patterns**: Appropriate architectural patterns for your stack
- **Security Best Practices**: Security-first development approach
- **Documentation**: Comprehensive and up-to-date documentation

#### Technology-Agnostic Quality Gates:
- **Code Complexity**: Maintain reasonable complexity metrics
- **Security Scanning**: Automated vulnerability detection (varies by stack)
- **Documentation**: Completeness and quality validation
- **Testing**: Comprehensive test coverage appropriate for your technology

### GitHub Integration

#### Issue Management:
- **Automatic Categorization**: Component-based issue classification
- **Intelligent Prioritization**: Business impact and technical complexity analysis
- **Progress Tracking**: Development velocity and milestone tracking
- **Workflow Automation**: Status updates and agent assignments

#### Workflow Automation:
- **Branch Naming**: Consistent naming conventions
- **Commit Messages**: Standardized commit message formats
- **PR Templates**: Automated pull request templates
- **Issue Lifecycle**: Automatic status updates and assignments

## Technology Stack Detection

The framework automatically detects your technology stack and adapts accordingly:

### Supported Detection Patterns:
- **JavaScript/TypeScript**: package.json, tsconfig.json, yarn.lock
- **Python**: requirements.txt, pyproject.toml, setup.py
- **Java**: pom.xml, build.gradle, gradle.properties
- **C#/.NET**: *.csproj, *.sln, packages.config
- **Go**: go.mod, go.sum
- **Rust**: Cargo.toml, Cargo.lock
- **Ruby**: Gemfile, gemspec
- **PHP**: composer.json, composer.lock

### Automatic Context Configuration:
The framework analyzes your project and configures:
- Language-specific quality standards
- Technology-appropriate testing frameworks
- Relevant development tools and integrations
- Context servers for enhanced development experience

## Context Server Integration

The framework includes automatic context server discovery:

### Detected Context Servers:
- **Git Server**: Enhanced Git operations and history analysis
- **Database Servers**: PostgreSQL, MongoDB, Redis integration
- **Cloud Provider Servers**: AWS, GCP, Azure services
- **Language-Specific Servers**: NPM, PyPI, Cargo, Maven
- **Documentation Servers**: Confluence, Notion, team wikis

### Configuration:
Context servers are automatically configured in `.claude/mcp_settings.json` based on your project's technology stack.

## Advanced Configuration

### Custom Development Rules

Create project-specific development rules:

```markdown
<!-- .claude/rules/project-specific.md -->
# Project-Specific Development Rules

## Technology Stack: [Your Stack]
- Framework guidelines specific to your technology
- Coding standards for your team
- Integration patterns for your architecture

## Quality Gates
- Technology-specific quality metrics
- Custom validation rules
- Performance benchmarks
```

### Context Customization

```markdown
<!-- .claude/context/technology-context.md -->
# Technology Context: [Your Stack]

## Architecture Overview
- System architecture description
- Key components and their interactions
- External dependencies and integrations

## Development Patterns
- Common patterns used in the codebase
- Recommended approaches for new features
- Anti-patterns to avoid
```

### Agent Workflow Customization

Customize agent triggers and workflows:

```yaml
# .claude/workflow-config.yaml
workflow:
  triggers:
    - label: "workflow-state:planning"
      agent: "project-manager"
    - label: "workflow-state:implementing" 
      agent: "developer"
    - label: "workflow-state:testing"
      agent: "quality-assurance"
```

## Testing Framework Integration

### Technology-Agnostic Testing:
The framework adapts to your testing tools:

- **JavaScript**: Jest, Mocha, Cypress
- **Python**: pytest, unittest, coverage
- **Java**: JUnit, TestNG, Mockito
- **C#**: NUnit, xUnit, MSTest
- **Go**: go test, testify
- **Rust**: cargo test, proptest

### Framework Validation:
```bash
# Validate framework integration
# (Commands adapt based on detected technology)

# For Node.js projects:
npm test

# For Python projects:
pytest

# For Java projects:
mvn test

# For Go projects:
go test ./...
```

## Troubleshooting

### Common Issues

#### 1. Technology Stack Not Detected
**Problem**: Framework doesn't recognize your technology stack

**Solution**:
```bash
# Check for configuration files
ls -la | grep -E "(package\.json|requirements\.txt|pom\.xml|Cargo\.toml|go\.mod)"

# Manually specify technology in .claude/context/
echo "Technology Stack: [Your Technology]" > .claude/context/tech-stack.md
```

#### 2. Context Servers Not Found
**Problem**: No relevant context servers detected

**Solution**:
```bash
# Trigger manual context server discovery
# Add label to any GitHub issue:
gh issue edit <issue-number> --add-label "workflow-state:context-discovery"

# Or create manual configuration:
nano .claude/mcp_settings.json
```

#### 3. Quality Gates Not Configured
**Problem**: No quality standards applied

**Solution**:
```bash
# Check for technology-specific quality tools in your project
# Framework will provide generic quality guidelines if specific tools aren't found

# Add technology-specific quality rules:
nano .claude/rules/code-quality.md
```

### Validation Commands

The framework provides technology-agnostic validation:

```bash
# Check framework integration
find .claude -name "*.md" | wc -l  # Should show multiple files

# Verify GitHub integration
gh issue list --label "workflow-state:*" || echo "No workflow labels found (normal for new setup)"

# Test agent communication
grep -r "workflow-agent:" .claude/agents/ | head -5
```

## Performance Optimization

### Large Projects

For large projects with many files:

```yaml
# .claude/performance-config.yaml
quality_checks:
  parallel_processing: true
  technology_specific: true
  
caching:
  enabled: true
  cache_duration: 3600
  
github_integration:
  batch_size: 50
  rate_limiting: true
```

### Resource Management

```bash
# Monitor framework integration health
# (Commands adapt to your technology stack)

# Check Claude Code integration
ls -la .claude/

# Verify context servers
cat .claude/mcp_settings.json 2>/dev/null || echo "No context servers configured yet"
```

## Next Steps

After completing the setup:

1. **Review Generated Files**: Check all framework configuration files
2. **Test Technology Detection**: Verify your stack is properly detected
3. **Configure Context Servers**: Set up relevant context servers for your technology
4. **Test Agent Workflows**: Create a test issue and verify agent automation
5. **Customize Rules**: Adapt quality and workflow rules to your team's needs
6. **Team Onboarding**: Share setup instructions with your team

## Support and Resources

- **Documentation**: Check `framework-docs/` directory for detailed guides
- **GitHub Integration**: Use issue labels to trigger agent workflows
- **Community**: GitHub Discussions for questions and ideas
- **Issues**: GitHub Issues for bugs and feature requests

The framework adapts to your technology stack and development practices, providing intelligent assistance regardless of your chosen tools and languages.