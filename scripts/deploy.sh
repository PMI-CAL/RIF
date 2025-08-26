#!/bin/bash
# RIF Deployment Script
# Configures RIF for deployment in any environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_CONFIG="$PROJECT_ROOT/deploy.config.json"
ENV_TEMPLATE="$PROJECT_ROOT/.env.template"
ENV_FILE="$PROJECT_ROOT/.env"

# Default deployment mode
DEPLOYMENT_MODE="${RIF_DEPLOYMENT_MODE:-project}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --help)
            echo "RIF Deployment Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE         Set deployment mode (project, development, production)"
            echo "  --project-root DIR  Set project root directory"
            echo "  --help             Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  RIF_DEPLOYMENT_MODE   Default deployment mode"
            echo "  PROJECT_ROOT          Project root directory"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

log_info "Starting RIF deployment configuration..."
log_info "Project root: $PROJECT_ROOT"
log_info "Deployment mode: $DEPLOYMENT_MODE"

# Validate deployment mode
case $DEPLOYMENT_MODE in
    project|development|production)
        log_info "Valid deployment mode: $DEPLOYMENT_MODE"
        ;;
    *)
        log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
        log_error "Valid modes: project, development, production"
        exit 1
        ;;
esac

# Check if deploy.config.json exists
if [[ ! -f "$DEPLOY_CONFIG" ]]; then
    log_warning "deploy.config.json not found, creating default configuration..."
    
    cat > "$DEPLOY_CONFIG" << EOF
{
  "version": "1.0.0",
  "deployment_mode": "$DEPLOYMENT_MODE",
  "paths": {
    "rif_home": "\${PROJECT_ROOT}/.rif",
    "knowledge_base": "\${PROJECT_ROOT}/.rif/knowledge",
    "agents": "\${PROJECT_ROOT}/.rif/agents",
    "commands": "\${PROJECT_ROOT}/.rif/commands",
    "docs": "\${PROJECT_ROOT}/docs",
    "config": "\${PROJECT_ROOT}/config",
    "scripts": "\${PROJECT_ROOT}/scripts",
    "templates": "\${PROJECT_ROOT}/templates",
    "systems": "\${PROJECT_ROOT}/systems"
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
EOF
    
    log_success "Created default deploy.config.json"
fi

# Update deployment mode in config if different
if command -v python3 &> /dev/null; then
    python3 -c "
import json
config_path = '$DEPLOY_CONFIG'
mode = '$DEPLOYMENT_MODE'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if config.get('deployment_mode') != mode:
        config['deployment_mode'] = mode
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f'Updated deployment mode to: {mode}')
except Exception as e:
    print(f'Error updating config: {e}')
    exit(1)
"
else
    log_warning "Python3 not found, skipping config update"
fi

# Create .env file if it doesn't exist
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_TEMPLATE" ]]; then
        log_info "Creating .env file from template..."
        cp "$ENV_TEMPLATE" "$ENV_FILE"
        
        # Replace PROJECT_ROOT in .env file
        if command -v sed &> /dev/null; then
            sed -i.bak "s|\${PWD}|$PROJECT_ROOT|g" "$ENV_FILE"
            rm -f "$ENV_FILE.bak"
        fi
        
        # Set deployment mode
        if command -v sed &> /dev/null; then
            sed -i.bak "s|RIF_DEPLOYMENT_MODE=project|RIF_DEPLOYMENT_MODE=$DEPLOYMENT_MODE|g" "$ENV_FILE"
            rm -f "$ENV_FILE.bak"
        fi
        
        log_success "Created .env file"
        log_warning "Please edit .env file to customize your configuration"
    else
        log_warning ".env.template not found, skipping .env creation"
    fi
else
    log_info ".env file already exists"
fi

# Create necessary directories
log_info "Creating directory structure..."

# Read paths from configuration and create directories
if command -v python3 &> /dev/null; then
    PATHS_TO_CREATE=$(python3 -c "
import json
import os
from pathlib import Path

config_path = '$DEPLOY_CONFIG'
project_root = Path('$PROJECT_ROOT')

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    paths = config.get('paths', {})
    
    for key, template in paths.items():
        # Simple variable replacement
        expanded = template.replace('\${PROJECT_ROOT}', str(project_root))
        expanded = os.path.expandvars(expanded)
        
        path = Path(expanded)
        print(str(path))
        
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    exit(1)
")
    
    # Create directories
    echo "$PATHS_TO_CREATE" | while read -r dir; do
        if [[ -n "$dir" && "$dir" != "Error:"* ]]; then
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        fi
    done
else
    # Fallback: create common directories
    log_warning "Python3 not available, creating default directories"
    mkdir -p "$PROJECT_ROOT/.rif/knowledge"
    mkdir -p "$PROJECT_ROOT/.rif/agents" 
    mkdir -p "$PROJECT_ROOT/.rif/commands"
    mkdir -p "$PROJECT_ROOT/docs"
    mkdir -p "$PROJECT_ROOT/config"
    mkdir -p "$PROJECT_ROOT/scripts"
    mkdir -p "$PROJECT_ROOT/templates"
    mkdir -p "$PROJECT_ROOT/systems"
fi

# Validate configuration
log_info "Validating configuration..."

if command -v python3 &> /dev/null && [[ -f "$PROJECT_ROOT/claude/commands/path_resolver.py" ]]; then
    cd "$PROJECT_ROOT"
    python3 -c "
import sys
sys.path.insert(0, 'claude/commands')
from path_resolver import validate_configuration

is_valid, errors = validate_configuration()

if is_valid:
    print('✅ Configuration validation passed')
else:
    print('❌ Configuration validation failed:')
    for error in errors:
        print(f'  - {error}')
    sys.exit(1)
"
    if [[ $? -eq 0 ]]; then
        log_success "Configuration validation passed"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
else
    log_warning "Path resolver not available, skipping validation"
fi

# Set up git hooks if in a git repository
if [[ -d "$PROJECT_ROOT/.git" ]]; then
    log_info "Setting up git hooks..."
    
    HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
    
    # Create post-checkout hook for environment setup
    cat > "$HOOKS_DIR/post-checkout" << 'EOF'
#!/bin/bash
# RIF post-checkout hook
# Ensures environment is properly configured after checkout

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [[ -f "$PROJECT_ROOT/scripts/deploy.sh" ]]; then
    echo "🔧 Running RIF deployment configuration..."
    "$PROJECT_ROOT/scripts/deploy.sh" --mode "${RIF_DEPLOYMENT_MODE:-project}"
fi
EOF
    
    chmod +x "$HOOKS_DIR/post-checkout"
    log_success "Created git post-checkout hook"
fi

# Generate deployment report
log_info "Generating deployment report..."

REPORT_FILE="$PROJECT_ROOT/deployment-report.md"

cat > "$REPORT_FILE" << EOF
# RIF Deployment Report

**Generated**: $(date)
**Deployment Mode**: $DEPLOYMENT_MODE
**Project Root**: $PROJECT_ROOT

## Configuration Summary

- Deploy Config: $(test -f "$DEPLOY_CONFIG" && echo "✅ Present" || echo "❌ Missing")
- Environment File: $(test -f "$ENV_FILE" && echo "✅ Present" || echo "❌ Missing") 
- Path Resolver: $(test -f "$PROJECT_ROOT/claude/commands/path_resolver.py" && echo "✅ Present" || echo "❌ Missing")

## Directory Structure

EOF

# Add directory listing to report
if command -v tree &> /dev/null; then
    echo '```' >> "$REPORT_FILE"
    tree -L 3 "$PROJECT_ROOT" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
else
    echo "Directory tree not available (install 'tree' command)" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Next Steps

1. Review and customize the \`.env\` file for your environment
2. Test path resolution: \`python3 claude/commands/path_resolver.py\`
3. Update any hard-coded paths in your custom scripts
4. Configure GitHub token in \`.env\` if using GitHub integration
5. Run validation: \`python3 claude/commands/path_resolver.py\`

## Migration Checklist

- [ ] Review deploy.config.json settings
- [ ] Customize .env file  
- [ ] Update hard-coded paths in custom code
- [ ] Test configuration with path resolver
- [ ] Validate all paths are accessible
- [ ] Update CI/CD scripts if applicable
- [ ] Document any environment-specific requirements

EOF

log_success "Deployment report generated: $REPORT_FILE"

# Final summary
echo ""
log_success "🎉 RIF deployment configuration complete!"
echo ""
echo "Configuration files:"
echo "  📄 deploy.config.json - Main deployment configuration"  
echo "  📄 .env - Environment variables"
echo "  📄 deployment-report.md - Deployment summary"
echo ""
echo "Next steps:"
echo "  1. Review and customize .env file"
echo "  2. Test configuration: python3 claude/commands/path_resolver.py"
echo "  3. Update any remaining hard-coded paths"
echo ""
echo "For help: $0 --help"