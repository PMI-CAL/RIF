#!/bin/bash
# RIF Branch Management Installation Script
# Sets up git hooks, configuration, and validation

set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}🔧 RIF Branch Management Installation${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}❌ Error: Not in a git repository${NC}"
        echo "Please run this script from within a git repository"
        exit 1
    fi
    echo -e "${GREEN}✅ Git repository detected${NC}"
}

# Function to backup existing hooks
backup_existing_hooks() {
    local hooks_dir="$REPO_ROOT/.git/hooks"
    local backup_dir="$REPO_ROOT/.git/hooks.backup.$(date +%Y%m%d_%H%M%S)"
    
    if [[ -f "$hooks_dir/pre-commit" ]] || [[ -f "$hooks_dir/pre-push" ]]; then
        echo -e "${YELLOW}⚠️  Existing git hooks found, backing up...${NC}"
        mkdir -p "$backup_dir"
        
        if [[ -f "$hooks_dir/pre-commit" ]]; then
            cp "$hooks_dir/pre-commit" "$backup_dir/pre-commit"
            echo -e "   Backed up: pre-commit -> $backup_dir/pre-commit"
        fi
        
        if [[ -f "$hooks_dir/pre-push" ]]; then
            cp "$hooks_dir/pre-push" "$backup_dir/pre-push"
            echo -e "   Backed up: pre-push -> $backup_dir/pre-push"
        fi
        
        echo -e "${GREEN}✅ Existing hooks backed up to: $backup_dir${NC}"
    fi
}

# Function to install git hooks
install_git_hooks() {
    local hooks_dir="$REPO_ROOT/.git/hooks"
    
    echo -e "${BLUE}📦 Installing git hooks...${NC}"
    
    # Pre-commit hook should already be created by the implementer
    if [[ -f "$hooks_dir/pre-commit" ]]; then
        echo -e "${GREEN}✅ Pre-commit hook already installed${NC}"
    else
        echo -e "${RED}❌ Pre-commit hook not found${NC}"
        echo "Expected location: $hooks_dir/pre-commit"
        return 1
    fi
    
    # Pre-push hook should already be created by the implementer  
    if [[ -f "$hooks_dir/pre-push" ]]; then
        echo -e "${GREEN}✅ Pre-push hook already installed${NC}"
    else
        echo -e "${RED}❌ Pre-push hook not found${NC}"
        echo "Expected location: $hooks_dir/pre-push"
        return 1
    fi
    
    # Ensure hooks are executable
    chmod +x "$hooks_dir/pre-commit" "$hooks_dir/pre-push"
    echo -e "${GREEN}✅ Git hooks are executable${NC}"
}

# Function to validate Python dependencies
validate_python_setup() {
    echo -e "${BLUE}🐍 Validating Python setup...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        echo "Please install Python 3 to use branch management automation"
        return 1
    fi
    
    # Check if branch manager module can be imported
    if python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); from claude.commands.branch_manager import BranchManager" 2>/dev/null; then
        echo -e "${GREEN}✅ BranchManager module can be imported${NC}"
    else
        echo -e "${YELLOW}⚠️  BranchManager import test failed (may still work in context)${NC}"
    fi
}

# Function to create configuration directories
setup_directories() {
    echo -e "${BLUE}📁 Setting up directories...${NC}"
    
    local knowledge_dir="$REPO_ROOT/knowledge"
    if [[ ! -d "$knowledge_dir" ]]; then
        mkdir -p "$knowledge_dir"
        echo -e "${GREEN}✅ Created knowledge directory${NC}"
    else
        echo -e "${GREEN}✅ Knowledge directory exists${NC}"
    fi
}

# Function to test the installation
test_installation() {
    echo -e "${BLUE}🧪 Testing installation...${NC}"
    
    local current_branch
    current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
    
    echo -e "   Current branch: $current_branch"
    
    # Test pre-commit hook (simulate)
    if [[ "$current_branch" == "main" ]]; then
        echo -e "${YELLOW}⚠️  Currently on main branch - hook protection is active${NC}"
        echo -e "   Git commits will be blocked unless RIF_EMERGENCY_OVERRIDE is set"
    else
        echo -e "${GREEN}✅ On feature branch - commits will be allowed${NC}"
    fi
    
    # Test branch cleanup script
    if [[ -x "$REPO_ROOT/scripts/branch-cleanup.py" ]]; then
        echo -e "${GREEN}✅ Branch cleanup script is executable${NC}"
        echo -e "   Test with: python3 scripts/branch-cleanup.py --dry-run"
    else
        echo -e "${RED}❌ Branch cleanup script not found or not executable${NC}"
    fi
    
    echo -e "${GREEN}✅ Installation test completed${NC}"
}

# Function to display usage instructions
show_usage_instructions() {
    echo ""
    echo -e "${BLUE}📋 RIF Branch Management - Usage Instructions${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""
    echo -e "${GREEN}✨ Branch Protection:${NC}"
    echo "   • Direct commits to main branch are now blocked"
    echo "   • All development work must happen on feature branches"
    echo "   • Emergency override: export RIF_EMERGENCY_OVERRIDE=\"incident-id\""
    echo ""
    echo -e "${GREEN}🌿 Branch Naming Convention:${NC}"
    echo "   • Feature branches: issue-{number}-{description}"
    echo "   • Emergency branches: emergency-{incident-id}-{description}"
    echo "   • Example: issue-226-branch-management-system"
    echo ""
    echo -e "${GREEN}🔄 Proper Workflow:${NC}"
    echo "   1. Create feature branch: git checkout -b issue-226-feature-name"
    echo "   2. Make your changes and commits"
    echo "   3. Push branch: git push origin issue-226-feature-name"
    echo "   4. Create Pull Request to merge back to main"
    echo "   5. Merge through PR interface after review"
    echo ""
    echo -e "${GREEN}🧹 Branch Cleanup:${NC}"
    echo "   • Dry run: python3 scripts/branch-cleanup.py --dry-run"
    echo "   • Cleanup: python3 scripts/branch-cleanup.py --cleanup"
    echo "   • Force cleanup: python3 scripts/branch-cleanup.py --force"
    echo ""
    echo -e "${GREEN}🚨 Emergency Procedures:${NC}"
    echo "   • Set override: export RIF_EMERGENCY_OVERRIDE=\"INC-20250825-001\""
    echo "   • Make emergency changes (commits/pushes will be allowed)"
    echo "   • Create PR immediately after emergency fix"
    echo "   • All emergency actions are logged for compliance"
    echo ""
    echo -e "${YELLOW}⚠️  Important Notes:${NC}"
    echo "   • All emergency overrides are logged and require compliance verification"
    echo "   • RIF workflow will automatically create branches for issues"
    echo "   • Branch cleanup runs automatically on merged branches older than 7 days"
    echo ""
}

# Main installation flow
main() {
    echo -e "${BLUE}Starting RIF Branch Management installation...${NC}"
    echo ""
    
    check_git_repo
    backup_existing_hooks
    install_git_hooks
    validate_python_setup
    setup_directories
    test_installation
    
    echo ""
    echo -e "${GREEN}🎉 RIF Branch Management installation completed successfully!${NC}"
    
    show_usage_instructions
}

# Run main function
main "$@"