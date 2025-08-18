#!/bin/bash

# Development Framework Setup Script
# Technology-agnostic setup for any project type

set -e  # Exit on any error

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=""
INTERACTIVE=true
FORCE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Display usage information
show_usage() {
    echo "Development Framework Setup"
    echo ""
    echo "Usage: $0 [OPTIONS] <project_directory>"
    echo ""
    echo "Arguments:"
    echo "  project_directory    Target directory for framework integration"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -n, --non-interactive  Run in non-interactive mode"
    echo "  -f, --force         Overwrite existing files"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/my-project"
    echo "  $0 --non-interactive --force ./existing-project"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -n|--non-interactive)
                INTERACTIVE=false
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$PROJECT_DIR" ]]; then
                    PROJECT_DIR="$1"
                else
                    print_error "Multiple project directories specified"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$PROJECT_DIR" ]]; then
        print_error "Project directory is required"
        show_usage
        exit 1
    fi
}

# Detect project technology stack
detect_technology() {
    local project_dir="$1"
    local tech_stack=""
    
    print_info "Detecting technology stack in $project_dir..."
    
    # Check for different technology indicators
    if [[ -f "$project_dir/package.json" ]]; then
        tech_stack="JavaScript/Node.js"
    elif [[ -f "$project_dir/requirements.txt" ]] || [[ -f "$project_dir/pyproject.toml" ]] || [[ -f "$project_dir/setup.py" ]]; then
        tech_stack="Python"
    elif [[ -f "$project_dir/pom.xml" ]] || [[ -f "$project_dir/build.gradle" ]]; then
        tech_stack="Java"
    elif [[ -f "$project_dir/Cargo.toml" ]]; then
        tech_stack="Rust"
    elif [[ -f "$project_dir/go.mod" ]]; then
        tech_stack="Go"
    elif [[ -f "$project_dir/composer.json" ]]; then
        tech_stack="PHP"
    elif [[ -f "$project_dir/Gemfile" ]]; then
        tech_stack="Ruby"
    elif [[ -f "$project_dir"/*.csproj ]] || [[ -f "$project_dir"/*.sln ]]; then
        tech_stack=".NET/C#"
    else
        tech_stack="Unknown/Generic"
    fi
    
    print_info "Detected technology stack: $tech_stack"
    echo "$tech_stack"
}

# Check if directory exists and handle accordingly
check_project_directory() {
    if [[ ! -d "$PROJECT_DIR" ]]; then
        if [[ "$INTERACTIVE" == true ]]; then
            read -p "Directory $PROJECT_DIR does not exist. Create it? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                mkdir -p "$PROJECT_DIR"
                print_success "Created directory: $PROJECT_DIR"
            else
                print_error "Cannot proceed without target directory"
                exit 1
            fi
        else
            mkdir -p "$PROJECT_DIR"
            print_success "Created directory: $PROJECT_DIR"
        fi
    fi
    
    # Convert to absolute path
    PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
}

# Check if framework files already exist
check_existing_files() {
    local existing_files=()
    
    if [[ -d "$PROJECT_DIR/.claude" ]]; then
        existing_files+=(".claude/")
    fi
    
    if [[ -d "$PROJECT_DIR/framework-config" ]]; then
        existing_files+=("framework-config/")
    fi
    
    if [[ -d "$PROJECT_DIR/framework-docs" ]]; then
        existing_files+=("framework-docs/")
    fi
    
    if [[ ${#existing_files[@]} -gt 0 ]]; then
        print_warning "Framework files already exist: ${existing_files[*]}"
        
        if [[ "$FORCE" == false ]]; then
            if [[ "$INTERACTIVE" == true ]]; then
                read -p "Overwrite existing framework files? [y/N]: " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_error "Cannot proceed without overwriting existing files"
                    exit 1
                fi
            else
                print_error "Existing framework files found. Use --force to overwrite"
                exit 1
            fi
        fi
        
        print_info "Removing existing framework files..."
        rm -rf "$PROJECT_DIR/.claude" "$PROJECT_DIR/framework-config" "$PROJECT_DIR/framework-docs"
    fi
}

# Copy framework files to project directory
copy_framework_files() {
    print_info "Copying framework files to $PROJECT_DIR..."
    
    # Copy Claude Code integration
    if [[ -d "$FRAMEWORK_DIR/claude" ]]; then
        cp -r "$FRAMEWORK_DIR/claude" "$PROJECT_DIR/.claude"
        print_success "Copied Claude Code integration files"
    fi
    
    # Copy configuration
    if [[ -d "$FRAMEWORK_DIR/config" ]]; then
        cp -r "$FRAMEWORK_DIR/config" "$PROJECT_DIR/framework-config"
        print_success "Copied configuration files"
    fi
    
    # Copy documentation
    if [[ -d "$FRAMEWORK_DIR/docs" ]]; then
        cp -r "$FRAMEWORK_DIR/docs" "$PROJECT_DIR/framework-docs"
        print_success "Copied documentation files"
    fi
    
    # Copy templates
    if [[ -d "$FRAMEWORK_DIR/templates" ]]; then
        cp -r "$FRAMEWORK_DIR/templates" "$PROJECT_DIR/framework-templates"
        print_success "Copied template files"
    fi
}

# Initialize git repository if needed
init_git_repo() {
    if [[ ! -d "$PROJECT_DIR/.git" ]]; then
        if [[ "$INTERACTIVE" == true ]]; then
            read -p "Initialize git repository? [Y/n]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                return
            fi
        fi
        
        print_info "Initializing git repository..."
        cd "$PROJECT_DIR"
        git init
        print_success "Initialized git repository"
    fi
}

# Add framework files to git
add_to_git() {
    if [[ -d "$PROJECT_DIR/.git" ]]; then
        print_info "Adding framework files to git..."
        cd "$PROJECT_DIR"
        git add .claude/ framework-config/ framework-docs/ framework-templates/ 2>/dev/null || true
        
        if git diff --cached --quiet; then
            print_info "No new files to commit"
        else
            if [[ "$INTERACTIVE" == true ]]; then
                read -p "Commit framework integration? [Y/n]: " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    git commit -m "Add development framework integration"
                    print_success "Committed framework integration"
                fi
            else
                git commit -m "Add development framework integration"
                print_success "Committed framework integration"
            fi
        fi
    fi
}

# Trigger context server discovery
trigger_context_discovery() {
    if [[ -d "$PROJECT_DIR/.git" ]] && command -v gh &> /dev/null; then
        if [[ "$INTERACTIVE" == true ]]; then
            read -p "Create GitHub issue for context server discovery? [Y/n]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                print_info "Creating context server discovery issue..."
                cd "$PROJECT_DIR"
                
                # Create issue for context server discovery
                local issue_url
                issue_url=$(gh issue create \
                    --title "Context Server Discovery and Integration" \
                    --body "**Automated Request**: Context server discovery

The development framework has been integrated and is ready for context server discovery and configuration.

## Technology Stack Detected
- **Stack**: $1

## Requested Actions
1. Analyze project structure and technology stack
2. Identify relevant context servers for integration  
3. Generate configuration files for recommended servers
4. Provide installation and setup instructions

This issue will be automatically processed by the Context Server Discovery Agent." \
                    --label "workflow-state:context-discovery" 2>/dev/null)
                
                if [[ $? -eq 0 ]]; then
                    print_success "Created context server discovery issue: $issue_url"
                else
                    print_warning "Could not create GitHub issue (ensure you're in a Git repository with GitHub remote)"
                fi
            fi
        fi
    fi
}

# Display next steps
show_next_steps() {
    local tech_stack="$1"
    
    echo ""
    print_success "Framework integration complete!"
    echo ""
    echo "Next Steps:"
    echo "1. Review framework configuration in framework-config/"
    echo "2. Customize Claude Code rules in .claude/rules/"
    echo "3. Add project-specific context in .claude/context/"
    echo "4. Context server discovery will enhance development experience"
    echo ""
    
    case "$tech_stack" in
        "JavaScript/Node.js")
            echo "Technology-specific recommendations:"
            echo "- Review .claude/commands/development.md for Node.js commands"
            echo "- Configure package.json scripts for quality gates"
            echo "- Set up ESLint/Prettier integration"
            ;;
        "Python")
            echo "Technology-specific recommendations:"
            echo "- Set up virtual environment and requirements.txt"
            echo "- Configure pytest and pre-commit hooks"
            echo "- Review Python-specific quality standards"
            ;;
        "Java")
            echo "Technology-specific recommendations:"
            echo "- Configure Maven/Gradle quality plugins"
            echo "- Set up JUnit and code coverage"
            echo "- Review Java-specific coding standards"
            ;;
        *)
            echo "For your $tech_stack project:"
            echo "- Adapt quality standards to your technology stack"
            echo "- Configure appropriate testing frameworks"
            echo "- Customize development commands for your tools"
            ;;
    esac
    
    echo ""
    echo "Framework Documentation: framework-docs/setup-guide.md"
    echo "Agent Workflows: .claude/agents/"
    echo ""
    print_info "Happy coding with AI assistance! 🚀"
}

# Main execution
main() {
    print_info "Development Framework Setup"
    print_info "Framework directory: $FRAMEWORK_DIR"
    
    parse_args "$@"
    check_project_directory
    
    local tech_stack
    tech_stack=$(detect_technology "$PROJECT_DIR")
    
    check_existing_files
    copy_framework_files
    init_git_repo
    add_to_git
    trigger_context_discovery "$tech_stack"
    show_next_steps "$tech_stack"
}

# Run main function with all arguments
main "$@"