# RIF Deployment Configuration System

This guide explains how to configure and deploy the Reactive Intelligence Framework (RIF) in any environment using the portable configuration system.

## Overview

The RIF deployment configuration system provides:

- **Environment-agnostic path resolution** using variable substitution
- **Configurable feature flags** for different deployment modes  
- **Secure path handling** with validation and sanitization
- **Migration tools** for updating existing hard-coded paths
- **Validation tools** for ensuring configuration correctness

## Quick Start

1. **Initialize the deployment configuration:**
   ```bash
   ./scripts/deploy.sh
   ```

2. **Customize your environment:**
   ```bash
   cp .env.template .env
   # Edit .env file with your settings
   ```

3. **Validate the configuration:**
   ```bash
   python3 scripts/validate_configuration.py
   ```

4. **Migrate existing hard-coded paths (optional):**
   ```bash
   python3 scripts/migrate_paths.py --dry-run
   python3 scripts/migrate_paths.py --migrate
   ```

## Configuration Files

### deploy.config.json

The main configuration file that defines deployment settings:

```json
{
  "version": "1.0.0",
  "deployment_mode": "project",
  "paths": {
    "rif_home": "${PROJECT_ROOT}/.rif",
    "knowledge_base": "${PROJECT_ROOT}/.rif/knowledge",
    "agents": "${PROJECT_ROOT}/.rif/agents",
    "commands": "${PROJECT_ROOT}/.rif/commands",
    "docs": "${PROJECT_ROOT}/docs",
    "config": "${PROJECT_ROOT}/config",
    "scripts": "${PROJECT_ROOT}/scripts",
    "templates": "${PROJECT_ROOT}/templates",
    "systems": "${PROJECT_ROOT}/systems"
  },
  "features": {
    "self_development_checks": false,
    "audit_logging": false,
    "development_telemetry": false,
    "shadow_mode": false,
    "quality_gates": true,
    "pattern_learning": true
  },
  "knowledge": {
    "preserve_patterns": true,
    "preserve_decisions": false,
    "clean_on_init": true,
    "backup_existing": true
  },
  "environment": {
    "github_integration": true,
    "claude_code_hooks": true,
    "mcp_servers": false,
    "lightrag_backend": true
  },
  "security": {
    "sanitize_paths": true,
    "validate_templates": true,
    "restrict_file_access": true
  }
}
```

### .env File

Environment-specific variables:

```bash
# Project root directory (automatically detected if not set)
PROJECT_ROOT=/path/to/your/project

# RIF home directory
RIF_HOME=${PROJECT_ROOT}/.rif

# Deployment mode: project, development, or production
RIF_DEPLOYMENT_MODE=project

# Feature flags
RIF_QUALITY_GATES=true
RIF_PATTERN_LEARNING=true

# GitHub integration
GITHUB_TOKEN=your_github_token_here
```

## Path Resolution

### Using PathResolver in Python Code

```python
from claude.commands.path_resolver import PathResolver

# Initialize resolver
resolver = PathResolver()

# Resolve paths
knowledge_path = resolver.resolve("knowledge_base")
config_path = resolver.resolve("config")
docs_path = resolver.resolve("docs")

# Check features
if resolver.is_feature_enabled("quality_gates"):
    # Enable quality gates
    pass

# Get deployment mode
mode = resolver.get_deployment_mode()
if mode == "development":
    # Development-specific settings
    pass
```

### Available Path Keys

| Path Key | Description | Default Value |
|----------|-------------|---------------|
| `rif_home` | RIF home directory | `${PROJECT_ROOT}/.rif` |
| `knowledge_base` | Knowledge base storage | `${PROJECT_ROOT}/.rif/knowledge` |
| `agents` | Agent definitions | `${PROJECT_ROOT}/.rif/agents` |
| `commands` | Command implementations | `${PROJECT_ROOT}/.rif/commands` |
| `docs` | Documentation | `${PROJECT_ROOT}/docs` |
| `config` | Configuration files | `${PROJECT_ROOT}/config` |
| `scripts` | Deployment/utility scripts | `${PROJECT_ROOT}/scripts` |
| `templates` | Project templates | `${PROJECT_ROOT}/templates` |
| `systems` | System implementations | `${PROJECT_ROOT}/systems` |

### Variable Substitution

The following variables are supported in path templates:

- `${PROJECT_ROOT}` - Automatically detected project root
- `${HOME}` - User's home directory
- `${RIF_HOME}` - RIF home directory
- `${USER}` - Current username

## Deployment Modes

### Project Mode (Default)

Suitable for using RIF as a project template or in a specific project:

- Minimal logging and telemetry
- Quality gates enabled
- Pattern learning enabled
- No development-specific features

### Development Mode

For RIF development and testing:

- Enhanced logging and debugging
- Development telemetry enabled
- All validation checks active
- Shadow mode for testing

### Production Mode

For production deployments:

- Minimal resource usage
- Security features maximized
- Comprehensive audit logging
- Performance optimizations

## Feature Flags

Configure features through the `features` section:

| Feature | Description | Default |
|---------|-------------|---------|
| `self_development_checks` | Enable RIF self-development validation | `false` |
| `audit_logging` | Enable comprehensive audit logging | `false` |
| `development_telemetry` | Enable development metrics collection | `false` |
| `shadow_mode` | Enable shadow mode for testing | `false` |
| `quality_gates` | Enable quality validation gates | `true` |
| `pattern_learning` | Enable pattern learning and storage | `true` |

## Security Configuration

Security settings in the `security` section:

| Setting | Description | Default |
|---------|-------------|---------|
| `sanitize_paths` | Sanitize and validate all path inputs | `true` |
| `validate_templates` | Validate template syntax and safety | `true` |
| `restrict_file_access` | Restrict file access to project boundaries | `true` |

## Migration Guide

### Updating Existing Code

1. **Replace hard-coded paths:**
   ```python
   # Before
   knowledge_path = "/Users/cal/DEV/RIF/knowledge"
   
   # After
   from claude.commands.path_resolver import PathResolver
   resolver = PathResolver()
   knowledge_path = resolver.resolve("knowledge_base")
   ```

2. **Use automatic migration:**
   ```bash
   python3 scripts/migrate_paths.py --dry-run  # Preview changes
   python3 scripts/migrate_paths.py --migrate  # Apply changes
   ```

3. **Restore from backups if needed:**
   ```bash
   python3 scripts/migrate_paths.py --restore
   ```

### Testing After Migration

1. **Validate configuration:**
   ```bash
   python3 scripts/validate_configuration.py
   ```

2. **Test path resolution:**
   ```bash
   python3 claude/commands/path_resolver.py
   ```

3. **Run your application tests to ensure everything works**

## Command-Line Tools

### Deployment Script

```bash
./scripts/deploy.sh [OPTIONS]

Options:
  --mode MODE         Set deployment mode (project, development, production)
  --project-root DIR  Set project root directory
  --help             Show help message
```

### Configuration Validation

```bash
python3 scripts/validate_configuration.py [OPTIONS]

Options:
  --project-root DIR  Project root directory
  --verbose          Enable verbose logging
  --json             Output results in JSON format
```

### Path Migration

```bash
python3 scripts/migrate_paths.py [OPTIONS]

Options:
  --dry-run          Show changes without modifying files (default)
  --migrate          Actually perform the migration
  --restore          Restore files from backups
  --project-root DIR Project root directory
```

### Path Resolution Testing

```bash
python3 claude/commands/path_resolver.py [PATH_KEY]

# Examples:
python3 claude/commands/path_resolver.py knowledge_base
python3 claude/commands/path_resolver.py  # Show all paths
```

## Best Practices

1. **Always use PathResolver** instead of hard-coded paths
2. **Test configuration** after any changes using validation tools
3. **Use feature flags** to control functionality per environment
4. **Keep sensitive data** in `.env` file, not in configuration
5. **Backup before migration** (automatic with migration tools)
6. **Validate paths exist** before using them in critical operations
7. **Use relative paths** within the project structure when possible

## Troubleshooting

### Common Issues

**Configuration not found:**
```bash
# Ensure deploy.config.json exists in project root
ls -la deploy.config.json
```

**Path resolution fails:**
```bash
# Check PROJECT_ROOT is set correctly
echo $PROJECT_ROOT

# Validate configuration
python3 scripts/validate_configuration.py
```

**Permission errors:**
```bash
# Check directory permissions
ls -la .rif/
chmod -R u+w .rif/
```

**Import errors:**
```bash
# Ensure path_resolver.py is in correct location
ls -la claude/commands/path_resolver.py

# Check Python path includes project
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### Getting Help

1. Run configuration validation to identify issues
2. Check the deployment report: `cat deployment-report.md`
3. Review logs for specific error messages
4. Test individual components using the command-line tools

## Integration with CI/CD

For automated deployment environments:

1. **Set environment variables** in your CI/CD system:
   ```bash
   export PROJECT_ROOT=/path/to/deployment
   export RIF_DEPLOYMENT_MODE=production
   ```

2. **Run deployment script** in your deployment pipeline:
   ```bash
   ./scripts/deploy.sh --mode production
   ```

3. **Validate configuration** as part of deployment:
   ```bash
   python3 scripts/validate_configuration.py --json > validation-results.json
   ```

4. **Use deployment report** for documentation:
   ```bash
   cat deployment-report.md >> deployment-summary.md
   ```

This configuration system ensures RIF can be deployed consistently across different environments while maintaining security and reliability.