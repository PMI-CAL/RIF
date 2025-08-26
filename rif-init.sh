#!/bin/bash

# RIF - Reactive Intelligence Framework Initialization Script
# Main entry point for initializing RIF in a project

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIF_VERSION="1.0.0"
DEFAULT_MODE="development"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Show usage information
show_usage() {
    echo "RIF - Reactive Intelligence Framework Initialization"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -m, --mode MODE           Set initialization mode: development|production"
    echo "                           Default: $DEFAULT_MODE"
    echo "  -v, --version             Show RIF version"
    echo "  --interactive             Enable interactive configuration (default)"
    echo "  --non-interactive         Disable interactive prompts"
    echo "  --force                   Force initialization even if already configured"
    echo ""
    echo "Modes:"
    echo "  development              Full RIF development environment"
    echo "  production               Clean production template for projects"
    echo ""
    echo "Examples:"
    echo "  $0                       Initialize with development mode"
    echo "  $0 --mode production     Initialize for production use"
    echo "  $0 --non-interactive     Initialize without prompts"
}

# Parse command line arguments
parse_arguments() {
    MODE="$DEFAULT_MODE"
    INTERACTIVE=true
    FORCE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -v|--version)
                echo "RIF version $RIF_VERSION"
                exit 0
                ;;
            --interactive)
                INTERACTIVE=true
                shift
                ;;
            --non-interactive)
                INTERACTIVE=false
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate mode
    if [[ "$MODE" != "development" ]] && [[ "$MODE" != "production" ]]; then
        print_error "Invalid mode: $MODE"
        print_info "Valid modes: development, production"
        exit 1
    fi
}

# Detect if we're in development or deploy branch
detect_branch_mode() {
    local deploy_config="$SCRIPT_DIR/deploy.config.json"
    
    if [[ -f "$deploy_config" ]]; then
        local deploy_mode=$(python3 -c "import json; print(json.load(open('$deploy_config')).get('deployment_mode', 'unknown'))" 2>/dev/null || echo "unknown")
        if [[ "$deploy_mode" == "template" ]]; then
            print_info "Detected: Production template branch"
            return 0  # Production template
        fi
    fi
    
    # Check if we have development artifacts
    if [[ -d "$SCRIPT_DIR/knowledge/audits" ]] || [[ -d "$SCRIPT_DIR/validation" ]]; then
        print_info "Detected: Development branch"
        return 1  # Development branch
    fi
    
    print_info "Detected: Unknown branch type"
    return 2  # Unknown
}

# Initialize RIF based on mode
initialize_rif() {
    print_header "RIF Initialization"
    
    print_info "Mode: $MODE"
    print_info "Interactive: $INTERACTIVE"
    print_info "RIF Directory: $SCRIPT_DIR"
    
    # Check if already initialized
    if [[ -f "$SCRIPT_DIR/.rif-initialized" ]] && [[ "$FORCE" != "true" ]]; then
        print_warning "RIF already initialized in this directory"
        print_info "Use --force to re-initialize"
        
        if [[ "$INTERACTIVE" == "true" ]]; then
            read -p "Re-initialize anyway? (y/N): " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_info "Initialization aborted"
                exit 0
            fi
        else
            exit 0
        fi
    fi
    
    # Detect branch type and warn about mismatches
    detect_branch_mode
    local branch_type=$?
    
    if [[ $branch_type -eq 0 ]] && [[ "$MODE" == "development" ]]; then
        print_warning "You're on a production template branch but requesting development mode"
        print_info "Consider using: git clone -b main <repo> for development"
    elif [[ $branch_type -eq 1 ]] && [[ "$MODE" == "production" ]]; then
        print_warning "You're on a development branch but requesting production mode"
        print_info "Consider using: git clone -b deploy <repo> for production"
    fi
    
    # Run appropriate setup script
    if [[ -f "$SCRIPT_DIR/setup.sh" ]]; then
        print_info "Running framework setup..."
        if [[ "$INTERACTIVE" == "true" ]]; then
            "$SCRIPT_DIR/setup.sh" "$SCRIPT_DIR" 
        else
            "$SCRIPT_DIR/setup.sh" --non-interactive "$SCRIPT_DIR"
        fi
    else
        print_warning "Setup script not found, continuing with basic initialization"
    fi
    
    # Create mode-specific configuration
    create_mode_configuration
    
    # Set up development environment if needed
    if [[ "$MODE" == "development" ]]; then
        setup_development_environment
    else
        setup_production_environment
    fi
    
    # Mark as initialized
    echo "{\"mode\": \"$MODE\", \"initialized\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\", \"version\": \"$RIF_VERSION\"}" > "$SCRIPT_DIR/.rif-initialized"
    
    print_success "RIF initialization completed!"
    show_next_steps
}

# Create mode-specific configuration
create_mode_configuration() {
    print_info "Configuring for $MODE mode..."
    
    local config_dir="$SCRIPT_DIR/.rif"
    mkdir -p "$config_dir"
    
    # Create mode configuration file
    cat > "$config_dir/mode.json" << EOF
{
  "mode": "$MODE",
  "configured": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "features": {
    "self_development": $(if [[ "$MODE" == "development" ]]; then echo "true"; else echo "false"; fi),
    "audit_logging": $(if [[ "$MODE" == "development" ]]; then echo "true"; else echo "false"; fi),
    "quality_gates": true,
    "pattern_learning": true
  },
  "paths": {
    "knowledge": "$config_dir/knowledge",
    "agents": "$SCRIPT_DIR/claude/agents",
    "commands": "$SCRIPT_DIR/claude/commands"
  }
}
EOF
    
    print_success "Mode configuration created"
}

# Setup development environment
setup_development_environment() {
    print_info "Setting up development environment..."
    
    local config_dir="$SCRIPT_DIR/.rif"
    
    # Initialize knowledge base for development
    if [[ -d "$SCRIPT_DIR/knowledge" ]]; then
        ln -sf "$SCRIPT_DIR/knowledge" "$config_dir/knowledge"
        print_info "✓ Linked to full knowledge base"
    else
        mkdir -p "$config_dir/knowledge"
        print_info "✓ Created development knowledge base"
    fi
    
    # Enable Claude Code hooks for development
    if [[ ! -d "$SCRIPT_DIR/.claude" ]]; then
        mkdir -p "$SCRIPT_DIR/.claude"
        cat > "$SCRIPT_DIR/.claude/settings.json" << 'EOF'
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "echo 'RIF Development Mode Active'",
        "output": "context"
      }
    ]
  }
}
EOF
        print_info "✓ Created Claude Code development hooks"
    fi
    
    print_success "Development environment configured"
}

# Setup production environment  
setup_production_environment() {
    print_info "Setting up production environment..."
    
    local config_dir="$SCRIPT_DIR/.rif"
    
    # Initialize clean knowledge base for production
    mkdir -p "$config_dir/knowledge"
    
    # Copy essential patterns only
    if [[ -d "$SCRIPT_DIR/knowledge/patterns" ]]; then
        mkdir -p "$config_dir/knowledge/patterns"
        # Copy only generic patterns, skip issue-specific ones
        find "$SCRIPT_DIR/knowledge/patterns" -name "*.json" -not -name "*issue*" -exec cp {} "$config_dir/knowledge/patterns/" \; 2>/dev/null || true
        print_info "✓ Copied essential patterns"
    fi
    
    # Basic Claude Code hooks for production
    if [[ ! -d "$SCRIPT_DIR/.claude" ]]; then
        mkdir -p "$SCRIPT_DIR/.claude"
        cat > "$SCRIPT_DIR/.claude/settings.json" << 'EOF'
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command", 
        "command": "echo 'RIF Production Mode - Ready for Development'",
        "output": "context"
      }
    ]
  }
}
EOF
        print_info "✓ Created Claude Code production hooks"
    fi
    
    print_success "Production environment configured"
}

# Show next steps based on mode
show_next_steps() {
    print_header "Next Steps"
    
    if [[ "$MODE" == "development" ]]; then
        echo "🔧 Development Mode - RIF Framework Development"
        echo ""
        echo "You can now:"
        echo "1. Create GitHub issues to trigger RIF agents"
        echo "2. Use Claude Code with full RIF development features"
        echo "3. Develop and test RIF framework enhancements"
        echo "4. Access full knowledge base and audit logs"
        echo ""
        echo "Quick start:"
        echo "  gh issue create --title 'Test RIF Development' --body 'Testing RIF in development mode'"
        
    else
        echo "🚀 Production Mode - Project Development with RIF"
        echo ""
        echo "You can now:"
        echo "1. Create GitHub issues for your project features"
        echo "2. Use RIF agents for development tasks"
        echo "3. Leverage patterns and templates"
        echo "4. Build your project with RIF assistance"
        echo ""
        echo "Quick start:"
        echo "  gh issue create --title 'Initialize Project Architecture' --body 'Set up initial project structure'"
    fi
    
    echo ""
    echo "📚 Documentation:"
    echo "  - Framework Guide: docs/"
    echo "  - Agent Documentation: claude/agents/"
    echo "  - Configuration: config/"
    echo ""
    echo "🎯 Need Help?"
    echo "  - Check CLAUDE.md for full RIF capabilities"
    echo "  - Review agent templates in claude/agents/"
    echo "  - Use GitHub issues for RIF assistance"
}

# Main execution
main() {
    print_header "RIF Framework Initialization"
    
    parse_arguments "$@"
    initialize_rif
}

# Trap to handle script interruption
trap 'print_error "Initialization interrupted"; exit 1' INT TERM

# Execute main function
main "$@"