#!/bin/bash

# RIF-ProjectGen - Intelligent Project Creation System
# This script orchestrates the creation of new RIF-enabled projects

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIF_HOME="$(dirname "$SCRIPT_DIR")"
ENGINES_DIR="$RIF_HOME/engines"
GENERATORS_DIR="$RIF_HOME/generators"
AUTOMATION_DIR="$RIF_HOME/automation"
TEMPLATES_DIR="$RIF_HOME/templates/projects"
KNOWLEDGE_DIR="$RIF_HOME/knowledge"
CONFIG_FILE="$RIF_HOME/config/projectgen-config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
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

log_phase() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════${NC}"
    echo ""
}

# Banner
show_banner() {
    cat << "EOF"
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║     RIF-ProjectGen - Intelligent Project Creator     ║
║     Powered by Reactive Intelligence Framework       ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
EOF
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=0
    
    # Check for required commands
    for cmd in git gh jq; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed"
            missing=1
        fi
    done
    
    # Check GitHub authentication
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI not authenticated. Run: gh auth login"
        missing=1
    fi
    
    # Check RIF structure
    if [ ! -d "$RIF_HOME/claude/agents" ]; then
        log_error "RIF framework not properly initialized"
        missing=1
    fi
    
    if [ $missing -eq 1 ]; then
        log_error "Prerequisites not met. Please fix the issues above."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Phase 1: Discovery
discovery_phase() {
    log_phase "PHASE 1: PROJECT DISCOVERY"
    
    # Project name
    echo -e "${BOLD}What would you like to name your project?${NC}"
    read -p "Project name: " PROJECT_NAME
    
    # Sanitize project name
    PROJECT_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g')
    
    # Project type
    echo ""
    echo -e "${BOLD}What type of application are you building?${NC}"
    echo "1) Web Application (Frontend + Backend)"
    echo "2) API Service (REST/GraphQL)"
    echo "3) Command Line Tool"
    echo "4) Library/Package"
    echo "5) Mobile Application"
    echo "6) Custom/Other"
    read -p "Choice [1-6]: " PROJECT_TYPE_CHOICE
    
    case $PROJECT_TYPE_CHOICE in
        1) PROJECT_TYPE="web-app" ;;
        2) PROJECT_TYPE="api-service" ;;
        3) PROJECT_TYPE="cli-tool" ;;
        4) PROJECT_TYPE="library" ;;
        5) PROJECT_TYPE="mobile-app" ;;
        6) PROJECT_TYPE="custom" ;;
        *) PROJECT_TYPE="custom" ;;
    esac
    
    # Project description
    echo ""
    echo -e "${BOLD}Briefly describe your project:${NC}"
    read -p "Description: " PROJECT_DESCRIPTION
    
    # Target users
    echo ""
    echo -e "${BOLD}Who are the target users?${NC}"
    read -p "Target users: " TARGET_USERS
    
    # Core features
    echo ""
    echo -e "${BOLD}What are the core features? (comma-separated)${NC}"
    read -p "Core features: " CORE_FEATURES
    
    # Technology preferences
    echo ""
    echo -e "${BOLD}Any specific technology preferences? (or press Enter to skip)${NC}"
    read -p "Technologies: " TECH_STACK
    
    # Create project brief
    log_info "Generating project brief..."
    
    mkdir -p "$RIF_HOME/temp"
    cat > "$RIF_HOME/temp/project-brief.json" << EOF
{
    "name": "$PROJECT_NAME",
    "type": "$PROJECT_TYPE",
    "description": "$PROJECT_DESCRIPTION",
    "target_users": "$TARGET_USERS",
    "core_features": "$CORE_FEATURES",
    "tech_stack": "$TECH_STACK",
    "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    log_success "Project brief created"
}

# Phase 2: Documentation Generation
documentation_phase() {
    log_phase "PHASE 2: DOCUMENTATION GENERATION"
    
    log_info "Generating Product Requirements Document..."
    
    # Create PRD from project brief
    cat > "$RIF_HOME/temp/PRD.md" << EOF
# Product Requirements Document
## $PROJECT_NAME

### Executive Summary
$PROJECT_DESCRIPTION

### Target Users
$TARGET_USERS

### Core Features
$(echo "$CORE_FEATURES" | tr ',' '\n' | sed 's/^/- /')

### Technical Requirements
- Project Type: $PROJECT_TYPE
- Technology Stack: ${TECH_STACK:-"To be determined"}

### User Stories

#### Epic 1: Core Functionality
$(echo "$CORE_FEATURES" | tr ',' '\n' | head -1 | sed 's/^/As a user, I want to /')

##### Story 1.1: Initial Setup
- **As a** user
- **I want to** set up the application
- **So that** I can start using the features

##### Story 1.2: Basic Operations
- **As a** user  
- **I want to** perform basic operations
- **So that** I can accomplish my tasks

#### Epic 2: User Management
- User authentication
- User profiles
- Permission management

### Non-Functional Requirements
- Performance: Response time <2 seconds
- Security: Industry-standard encryption
- Scalability: Support 1000+ concurrent users
- Availability: 99.9% uptime

### Success Criteria
- All core features implemented
- Comprehensive test coverage
- Documentation complete
- Performance benchmarks met

### Timeline
- Phase 1: Foundation (Week 1-2)
- Phase 2: Core Features (Week 3-4)
- Phase 3: Polish & Testing (Week 5)
- Phase 4: Deployment (Week 6)

---
*Generated by RIF-ProjectGen on $(date -u +"%Y-%m-%dT%H:%M:%SZ")*
EOF
    
    log_success "PRD generated"
    
    # Optionally generate architecture document
    log_info "Generating architecture document..."
    
    cat > "$RIF_HOME/temp/architecture.md" << EOF
# System Architecture
## $PROJECT_NAME

### Overview
$PROJECT_DESCRIPTION

### Architecture Style
$(case "$PROJECT_TYPE" in
    "web-app") echo "Multi-tier web architecture" ;;
    "api-service") echo "Microservices architecture" ;;
    "cli-tool") echo "Command-pattern architecture" ;;
    "library") echo "Modular library architecture" ;;
    *) echo "Custom architecture" ;;
esac)

### Components
$(case "$PROJECT_TYPE" in
    "web-app") cat << WEBARCH
- **Frontend**: User interface layer
- **Backend**: Business logic layer
- **Database**: Data persistence layer
- **Cache**: Performance optimization layer
WEBARCH
    ;;
    "api-service") cat << APIARCH
- **API Gateway**: Request routing
- **Services**: Business logic
- **Data Store**: Persistence
- **Message Queue**: Async processing
APIARCH
    ;;
    *) echo "- **Core**: Main functionality"
       echo "- **Utilities**: Helper functions"
       echo "- **Tests**: Test suite"
    ;;
esac)

### Technology Stack
${TECH_STACK:-"- To be determined based on requirements"}

### Deployment Architecture
- Development environment
- Staging environment
- Production environment

### Security Considerations
- Authentication & Authorization
- Data encryption
- Input validation
- Rate limiting

---
*Generated by RIF-ProjectGen on $(date -u +"%Y-%m-%dT%H:%M:%SZ")*
EOF
    
    log_success "Architecture document generated"
}

# Phase 3: Project Setup
setup_phase() {
    log_phase "PHASE 3: PROJECT SETUP"
    
    # Determine project directory
    PROJECT_DIR="$HOME/DEV/$PROJECT_NAME"
    
    log_info "Setting up project at: $PROJECT_DIR"
    
    # Create project directory
    if [ -d "$PROJECT_DIR" ]; then
        log_warning "Directory already exists. Using timestamp suffix."
        PROJECT_DIR="${PROJECT_DIR}-$(date +%s)"
    fi
    
    # Clone RIF framework
    log_info "Cloning RIF framework..."
    cp -r "$RIF_HOME" "$PROJECT_DIR"
    
    # Clean up unnecessary files
    rm -rf "$PROJECT_DIR/.git"
    rm -rf "$PROJECT_DIR/temp"
    
    # Move documentation
    mkdir -p "$PROJECT_DIR/docs"
    cp "$RIF_HOME/temp/PRD.md" "$PROJECT_DIR/docs/"
    cp "$RIF_HOME/temp/architecture.md" "$PROJECT_DIR/docs/"
    cp "$RIF_HOME/temp/project-brief.json" "$PROJECT_DIR/"
    
    # Initialize git repository
    log_info "Initializing git repository..."
    cd "$PROJECT_DIR"
    git init
    
    # Create initial commit
    git add -A
    git commit -m "Initial commit: $PROJECT_NAME - RIF-enabled project

Project type: $PROJECT_TYPE
Description: $PROJECT_DESCRIPTION

Generated with RIF-ProjectGen
Includes RIF framework for intelligent development"
    
    log_success "Local repository created"
}

# Phase 4: GitHub Integration
github_phase() {
    log_phase "PHASE 4: GITHUB INTEGRATION"
    
    cd "$PROJECT_DIR"
    
    # Create GitHub repository
    log_info "Creating GitHub repository..."
    
    GITHUB_USER=$(gh api user -q .login)
    
    gh repo create "$PROJECT_NAME" \
        --private \
        --description "$PROJECT_DESCRIPTION" \
        --clone=false \
        2>/dev/null || {
            log_warning "Repository might already exist"
            PROJECT_NAME="${PROJECT_NAME}-$(date +%s)"
            gh repo create "$PROJECT_NAME" \
                --private \
                --description "$PROJECT_DESCRIPTION" \
                --clone=false
        }
    
    # Add remote and push
    git remote add origin "https://github.com/$GITHUB_USER/$PROJECT_NAME.git"
    git push -u origin main
    
    log_success "GitHub repository created and synced"
    
    # Create labels
    log_info "Setting up GitHub labels..."
    
    gh label create "state:new" --description "New issue" --color "0E8A16" 2>/dev/null || true
    gh label create "state:analyzing" --description "Being analyzed" --color "FBCA04" 2>/dev/null || true
    gh label create "state:planning" --description "Being planned" --color "1D76DB" 2>/dev/null || true
    gh label create "state:implementing" --description "Being implemented" --color "5319E7" 2>/dev/null || true
    gh label create "state:validating" --description "Being validated" --color "E99695" 2>/dev/null || true
    gh label create "state:complete" --description "Completed" --color "0E8A16" 2>/dev/null || true
    
    log_success "Labels configured"
}

# Phase 5: Issue Generation
issue_generation_phase() {
    log_phase "PHASE 5: ISSUE GENERATION"
    
    cd "$PROJECT_DIR"
    
    log_info "Creating GitHub issues from PRD..."
    
    # Create setup issue
    gh issue create \
        --title "Initial Project Setup" \
        --body "Set up the basic project structure and dependencies for $PROJECT_NAME.

## Tasks
- [ ] Set up development environment
- [ ] Install dependencies
- [ ] Configure build tools
- [ ] Set up testing framework
- [ ] Configure CI/CD

## Acceptance Criteria
- Project builds successfully
- Tests can be run
- Documentation is accessible" \
        --label "state:new"
    
    # Create feature issues from core features
    IFS=',' read -ra FEATURES <<< "$CORE_FEATURES"
    for feature in "${FEATURES[@]}"; do
        feature=$(echo "$feature" | xargs) # Trim whitespace
        gh issue create \
            --title "Implement: $feature" \
            --body "Implement the $feature functionality.

## Description
Add support for $feature as described in the PRD.

## Tasks
- [ ] Design implementation approach
- [ ] Write core functionality
- [ ] Add tests
- [ ] Update documentation

## Acceptance Criteria
- Feature works as specified
- Tests pass
- Documentation updated" \
            --label "state:new"
    done
    
    log_success "Issues created from PRD"
}

# Phase 6: Kickoff
kickoff_phase() {
    log_phase "PHASE 6: DEVELOPMENT KICKOFF"
    
    cd "$PROJECT_DIR"
    
    # Create kickoff summary
    cat << EOF

${GREEN}╔═══════════════════════════════════════════════════════╗${NC}
${GREEN}║         PROJECT CREATED SUCCESSFULLY! 🚀             ║${NC}
${GREEN}╚═══════════════════════════════════════════════════════╝${NC}

${BOLD}Project Details:${NC}
  Name:        $PROJECT_NAME
  Type:        $PROJECT_TYPE
  Location:    $PROJECT_DIR
  Repository:  https://github.com/$GITHUB_USER/$PROJECT_NAME

${BOLD}Generated Documents:${NC}
  ✅ Project Brief (project-brief.json)
  ✅ Product Requirements Document (docs/PRD.md)
  ✅ Architecture Document (docs/architecture.md)

${BOLD}GitHub Setup:${NC}
  ✅ Repository created
  ✅ RIF framework integrated
  ✅ Issues created from PRD
  ✅ Labels configured

${BOLD}Next Steps:${NC}
  1. cd $PROJECT_DIR
  2. gh issue list              # View created issues
  3. gh issue view 1            # Start with first issue
  4. RIF agents will help implement each issue

${BOLD}Track Progress:${NC}
  gh issue list --label "state:*"

${CYAN}RIF agents are ready to help with implementation!${NC}
${CYAN}Happy coding! 🎉${NC}

EOF
    
    # Update knowledge base
    log_info "Updating knowledge base..."
    
    mkdir -p "$KNOWLEDGE_DIR/patterns/projects"
    cat > "$KNOWLEDGE_DIR/patterns/projects/${PROJECT_NAME}.json" << EOF
{
    "project": "$PROJECT_NAME",
    "type": "$PROJECT_TYPE",
    "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "features": "$CORE_FEATURES",
    "tech_stack": "$TECH_STACK",
    "repository": "https://github.com/$GITHUB_USER/$PROJECT_NAME"
}
EOF
    
    log_success "Knowledge base updated"
}

# Main execution
main() {
    show_banner
    check_prerequisites
    
    # Create temp directory
    mkdir -p "$RIF_HOME/temp"
    
    # Execute phases
    discovery_phase
    documentation_phase
    setup_phase
    github_phase
    issue_generation_phase
    kickoff_phase
    
    # Cleanup
    rm -rf "$RIF_HOME/temp"
    
    log_success "RIF-ProjectGen completed successfully!"
}

# Handle interrupts
trap 'echo ""; log_error "Process interrupted"; exit 1' INT TERM

# Run main function
main "$@"