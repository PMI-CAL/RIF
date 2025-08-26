# RIF Deployment Gitignore Guide

This guide explains how to use the RIF deployment gitignore system to cleanly separate RIF framework files from your project code when deploying RIF as a project template.

## Overview

The deployment gitignore system provides:
- **Clean separation** between RIF framework and project files
- **Flexible mode switching** between development and deployment
- **Selective knowledge preservation** for project-specific patterns
- **Comprehensive coverage** of all development artifacts

## Files Overview

### Core Files

| File | Purpose |
|------|---------|
| `.gitignore.deployment` | Main deployment gitignore template |
| `scripts/manage_gitignore.py` | Python script for mode management |
| `.rif/knowledge-config.yaml` | Knowledge preservation configuration |
| `docs/deployment-gitignore-guide.md` | This documentation |

## Quick Start

### 1. Switch to Deployment Mode

```bash
# Using the management script
python scripts/manage_gitignore.py deploy

# Or manually
cp .gitignore.deployment .gitignore
```

### 2. Verify Deployment Mode

```bash
python scripts/manage_gitignore.py status
```

### 3. Switch Back to Development Mode

```bash
python scripts/manage_gitignore.py develop
```

## Detailed Usage

### GitignoreManager Script

The `manage_gitignore.py` script provides comprehensive gitignore management:

#### Commands

```bash
# Show current status
python scripts/manage_gitignore.py status

# Switch to deployment mode (exclude RIF framework)
python scripts/manage_gitignore.py deploy

# Switch to development mode (track all files)  
python scripts/manage_gitignore.py develop

# Create backup of current gitignore
python scripts/manage_gitignore.py backup

# Restore from backup
python scripts/manage_gitignore.py restore
python scripts/manage_gitignore.py restore --backup-name .gitignore.backup.20231201_143022

# List available backups
python scripts/manage_gitignore.py list-backups
```

#### Options

```bash
# Specify project root directory
python scripts/manage_gitignore.py --project-root /path/to/project deploy
```

### What Gets Excluded in Deployment Mode

#### RIF Framework Files
- `claude/agents/` - RIF agent definitions
- `claude/commands/` - RIF command implementations  
- `claude/rules/` - Code quality rules
- `systems/` - RIF system implementations
- `knowledge/audits/` - Audit files
- `knowledge/checkpoints/` - Recovery checkpoints
- `validation/` - Validation artifacts
- `incidents/` - Incident reports

#### Development Artifacts
- `demo_*.py` - Demo and example files
- `test_*.py` - Test files
- `validate_*.py` - Validation scripts
- `*_validation_results_*.json` - Test results
- `*.duckdb*` - Database files
- `*.log` - Log files

#### RIF Configuration
- `config/rif-*.yaml` - RIF-specific configuration
- `config/framework-*.yaml` - Framework configuration
- `rif-init.sh` - RIF initialization script

### What Gets Preserved

#### Project Files
- `src/`, `lib/`, `app/` - Source code directories
- `package.json`, `requirements.txt` - Dependency files
- `README.md`, `docs/` - Documentation
- `docker-compose.yml`, `Dockerfile` - Deployment files
- `.github/workflows/` - CI/CD configuration

#### Project-Specific Knowledge
- `knowledge/patterns/project-*.json` - Project patterns
- `knowledge/decisions/project-*.json` - Project decisions  
- `knowledge/learning/project-*.json` - Project learnings
- `config/project-*.yaml` - Project configuration

## Knowledge Preservation

### Configuration

The `.rif/knowledge-config.yaml` file controls which knowledge files are preserved:

```yaml
preserve:
  patterns:
    - "project-*.json"
    - "custom-*.json"
  decisions:
    - "project-*.json"
    - "architecture-*.json"
  learning:
    - "project-*.json"
    - "domain-*.json"

exclude:
  - "rif-*.json"
  - "issue-*.json"
  - "audit-*.json"
```

### Selective Preservation

Some directories use selective preservation:

- **`knowledge/patterns/`**: Preserves `project-*` patterns, excludes `rif-*` patterns
- **`knowledge/decisions/`**: Preserves project decisions, excludes system decisions
- **`knowledge/learning/`**: Preserves domain learnings, excludes issue-specific learnings

## Customization

### Adding Project-Specific Patterns

Add custom ignore patterns at the end of `.gitignore.deployment`:

```gitignore
# ==================================================
# Custom Project Additions
# ==================================================
# My custom build directory
/my-build/

# Project-specific temp files
*.mytemp

# Custom configuration
local.config.js
```

### Auto-Detection

The management script automatically detects project types and adds appropriate patterns:

- **Node.js**: Detects `package.json`, adds `node_modules/`
- **Python**: Detects `requirements.txt`, adds `__pycache__/`
- **Rust**: Detects `Cargo.toml`, adds `target/`
- **Go**: Detects `go.mod`, adds `vendor/`
- **Java**: Detects `pom.xml`, adds `target/`, `.m2/`

## Best Practices

### 1. Use Deployment Mode for Projects

When using RIF as a project template:
```bash
# Set up new project
cp -r rif-template my-project
cd my-project
python scripts/manage_gitignore.py deploy
git init
git add .
git commit -m "Initial project setup"
```

### 2. Use Development Mode for RIF Work

When working on RIF framework itself:
```bash
cd rif-framework
python scripts/manage_gitignore.py develop
# Now all RIF files are tracked
```

### 3. Regular Backups

The script automatically creates backups before switching modes, but you can create manual backups:

```bash
python scripts/manage_gitignore.py backup
```

### 4. Review Before Committing

Always review what files are being tracked:
```bash
git status
git ls-files
```

## Troubleshooting

### Issue: Deployment Template Missing

**Error**: `.gitignore.deployment template not found`

**Solution**: 
```bash
# Check if file exists
ls -la .gitignore.deployment

# If missing, copy from RIF template or recreate
cp /path/to/rif-template/.gitignore.deployment .
```

### Issue: Wrong Files Being Tracked

**Symptoms**: RIF framework files appear in `git status`

**Solutions**:
1. Verify deployment mode is active:
   ```bash
   python scripts/manage_gitignore.py status
   ```

2. If in wrong mode, switch to deployment:
   ```bash
   python scripts/manage_gitignore.py deploy
   ```

3. Remove already-tracked RIF files:
   ```bash
   git rm -r --cached claude/
   git rm -r --cached knowledge/audits/
   git commit -m "Remove RIF framework files from tracking"
   ```

### Issue: Important Project Files Being Ignored

**Symptoms**: Project files not appearing in `git status`

**Solutions**:
1. Check if files match ignore patterns:
   ```bash
   git check-ignore -v path/to/file
   ```

2. Add exception to `.gitignore`:
   ```gitignore
   # Exclude general pattern
   *.log
   # But include important project logs
   !project.log
   !logs/important.log
   ```

3. Force add important files:
   ```bash
   git add -f important-file.ext
   ```

## Mode Comparison

| Aspect | Development Mode | Deployment Mode |
|--------|------------------|-----------------|
| **RIF Framework Files** | ✅ Tracked | ❌ Ignored |
| **Project Source Code** | ✅ Tracked | ✅ Tracked |
| **RIF Configuration** | ✅ Tracked | ❌ Ignored |
| **Project Configuration** | ✅ Tracked | ✅ Tracked |
| **Knowledge Base** | ✅ All Tracked | 🔄 Selective |
| **Development Tools** | ✅ Tracked | ❌ Ignored |
| **Test/Demo Files** | ✅ Tracked | ❌ Ignored |

## Migration Guide

### From Manual Gitignore

If you have a manual `.gitignore`:

1. **Backup current file**:
   ```bash
   python scripts/manage_gitignore.py backup
   ```

2. **Switch to deployment mode**:
   ```bash
   python scripts/manage_gitignore.py deploy
   ```

3. **Review and merge custom patterns**:
   - Compare your backup with new deployment gitignore
   - Add any project-specific patterns to the custom section

### From Development to Deployment

When transitioning a RIF development project to a regular project:

1. **Clean up tracking**:
   ```bash
   # Remove RIF files from git tracking
   git rm -r --cached claude/
   git rm -r --cached knowledge/audits/
   git rm -r --cached validation/
   git commit -m "Prepare for deployment mode"
   ```

2. **Switch mode**:
   ```bash
   python scripts/manage_gitignore.py deploy
   ```

3. **Verify clean state**:
   ```bash
   git status  # Should only show project files
   ```

## Advanced Usage

### Programmatic Usage

You can use the GitignoreManager class in your own scripts:

```python
from scripts.manage_gitignore import GitignoreManager

manager = GitignoreManager("/path/to/project")

# Check current mode
status = manager.get_mode_status()
print(f"Current mode: {status['current_mode']}")

# Switch modes
if status['current_mode'] != 'deployment':
    manager.setup_for_deployment()

# Detect project type
patterns = manager.detect_project_patterns()
print(f"Detected patterns: {patterns}")
```

### Integration with CI/CD

You can integrate mode switching into your build pipeline:

```yaml
# .github/workflows/deploy.yml
- name: Switch to deployment mode
  run: python scripts/manage_gitignore.py deploy

- name: Verify clean deployment
  run: |
    if git ls-files | grep -q "claude/"; then
      echo "Error: RIF framework files still tracked"
      exit 1
    fi
```

## Support

For issues with the deployment gitignore system:

1. **Check the status**: `python scripts/manage_gitignore.py status`
2. **Review backups**: `python scripts/manage_gitignore.py list-backups`  
3. **Restore if needed**: `python scripts/manage_gitignore.py restore`
4. **File an issue** with the RIF project including your status output

## Related Documentation

- [RIF Project Generation Guide](rif-projectgen-guide.md)
- [Knowledge Base Management](knowledge-management.md)
- [Development Workflow Guide](development-workflow.md)