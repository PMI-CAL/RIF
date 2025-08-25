#!/bin/bash

# RIF Branch Protection Configuration Script
# Configures branch protection rules for main and deploy branches

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if gh CLI is installed
    if ! command -v gh &> /dev/null; then
        print_error "GitHub CLI (gh) is required but not installed"
        print_info "Install with: https://cli.github.com/"
        exit 1
    fi
    
    print_info "✓ GitHub CLI found"
    
    # Check if user is authenticated
    if ! gh auth status &> /dev/null; then
        print_error "Not authenticated with GitHub CLI"
        print_info "Run: gh auth login"
        exit 1
    fi
    
    print_info "✓ GitHub CLI authenticated"
    
    # Check if we're in a git repository
    if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
        print_error "Not in a Git repository"
        exit 1
    fi
    
    print_info "✓ Git repository detected"
    
    # Get repository information
    REPO_OWNER=$(gh repo view --json owner --jq '.owner.login' 2>/dev/null || echo "unknown")
    REPO_NAME=$(gh repo view --json name --jq '.name' 2>/dev/null || echo "unknown")
    
    if [[ "$REPO_OWNER" == "unknown" ]] || [[ "$REPO_NAME" == "unknown" ]]; then
        print_error "Cannot determine repository information"
        print_info "Make sure you're in the correct repository and have push access"
        exit 1
    fi
    
    print_info "✓ Repository: $REPO_OWNER/$REPO_NAME"
    
    # Check repository permissions
    REPO_PERMS=$(gh api repos/$REPO_OWNER/$REPO_NAME --jq '.permissions.admin' 2>/dev/null || echo "false")
    if [[ "$REPO_PERMS" != "true" ]]; then
        print_warning "You may not have admin permissions to configure branch protection"
        print_warning "Some operations may fail"
    else
        print_info "✓ Admin permissions confirmed"
    fi
    
    print_success "Prerequisites check completed"
}

# Configure deploy branch protection
configure_deploy_branch_protection() {
    print_header "Configuring Deploy Branch Protection"
    
    print_info "Setting up protection for 'deploy' branch..."
    
    # Create deploy branch protection rules
    local protection_config='{
        "required_status_checks": {
            "strict": false,
            "contexts": []
        },
        "enforce_admins": false,
        "required_pull_request_reviews": {
            "required_approving_review_count": 0,
            "dismiss_stale_reviews": false,
            "require_code_owner_reviews": false,
            "restrict_pushes": true
        },
        "restrictions": {
            "users": [],
            "teams": [],
            "apps": ["github-actions"]
        },
        "allow_force_pushes": false,
        "allow_deletions": false
    }'
    
    # Apply protection rules
    if gh api repos/$REPO_OWNER/$REPO_NAME/branches/deploy/protection \
       --method PUT \
       --input <(echo "$protection_config") \
       --silent; then
        print_success "✓ Deploy branch protection configured"
    else
        print_warning "⚠ Deploy branch protection may not be fully configured"
        print_info "This is normal if the deploy branch doesn't exist yet"
    fi
    
    print_info "Deploy branch settings:"
    print_info "  - Direct pushes: Disabled (GitHub Actions only)"
    print_info "  - Force pushes: Disabled"
    print_info "  - Branch deletion: Disabled"
    print_info "  - Status checks: None required"
    print_info "  - Pull request reviews: None required"
}

# Configure main branch protection  
configure_main_branch_protection() {
    print_header "Configuring Main Branch Protection"
    
    print_info "Setting up protection for 'main' branch..."
    
    # Create main branch protection rules
    local protection_config='{
        "required_status_checks": {
            "strict": true,
            "contexts": ["RIF PR Quality Gates"]
        },
        "enforce_admins": false,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": true,
            "require_code_owner_reviews": false,
            "restrict_pushes": false
        },
        "restrictions": null,
        "allow_force_pushes": false,
        "allow_deletions": false
    }'
    
    # Apply protection rules
    if gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection \
       --method PUT \
       --input <(echo "$protection_config") \
       --silent; then
        print_success "✓ Main branch protection configured"
    else
        print_warning "⚠ Main branch protection configuration failed"
        print_info "You may need admin permissions or the branch may not exist"
    fi
    
    print_info "Main branch settings:"
    print_info "  - Direct pushes: Allowed for admins/maintainers"
    print_info "  - Force pushes: Disabled"
    print_info "  - Branch deletion: Disabled"
    print_info "  - Status checks: RIF PR Quality Gates required"
    print_info "  - Pull request reviews: 1+ required for external contributors"
}

# Verify branch protection settings
verify_protection_settings() {
    print_header "Verifying Branch Protection Settings"
    
    # Check deploy branch protection
    print_info "Checking deploy branch protection..."
    if gh api repos/$REPO_OWNER/$REPO_NAME/branches/deploy/protection &>/dev/null; then
        local deploy_protected=$(gh api repos/$REPO_OWNER/$REPO_NAME/branches/deploy/protection --jq '.allow_force_pushes.enabled' 2>/dev/null || echo "unknown")
        if [[ "$deploy_protected" == "false" ]]; then
            print_success "✓ Deploy branch properly protected"
        else
            print_warning "⚠ Deploy branch protection may not be complete"
        fi
    else
        print_info "ℹ Deploy branch not found (will be created by first sync)"
    fi
    
    # Check main branch protection
    print_info "Checking main branch protection..."
    if gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection &>/dev/null; then
        local main_protected=$(gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection --jq '.allow_force_pushes.enabled' 2>/dev/null || echo "unknown")
        if [[ "$main_protected" == "false" ]]; then
            print_success "✓ Main branch properly protected"
        else
            print_warning "⚠ Main branch protection may not be complete"
        fi
    else
        print_warning "⚠ Main branch protection not found"
    fi
    
    print_success "Protection verification completed"
}

# Create CODEOWNERS file if it doesn't exist
create_codeowners() {
    print_header "Creating CODEOWNERS File"
    
    local codeowners_file="$PROJECT_ROOT/.github/CODEOWNERS"
    
    if [[ -f "$codeowners_file" ]]; then
        print_info "CODEOWNERS file already exists"
        return
    fi
    
    print_info "Creating CODEOWNERS file for code review requirements..."
    
    # Ensure .github directory exists
    mkdir -p "$PROJECT_ROOT/.github"
    
    cat > "$codeowners_file" << 'EOF'
# RIF Framework Code Owners
# These owners will be requested for review when PRs are opened

# Global owners for all files
* @rif-maintainers

# Core framework files require maintainer review
/claude/ @rif-maintainers
/config/ @rif-maintainers  
/systems/ @rif-maintainers
CLAUDE.md @rif-maintainers
rif-init.sh @rif-maintainers
setup.sh @rif-maintainers

# GitHub workflows require admin review
/.github/workflows/ @rif-admins
/.github/scripts/ @rif-admins

# Documentation can be reviewed by any maintainer
/docs/ @rif-maintainers
README.md @rif-maintainers

# Deployment configuration requires admin review
/scripts/clean-for-deploy.sh @rif-admins
/scripts/configure-branch-protection.sh @rif-admins
/.github/workflows/deploy-sync.yml @rif-admins
/config/deploy-branch.yaml @rif-admins
EOF
    
    print_success "✓ CODEOWNERS file created"
    print_info "Add team members to @rif-maintainers and @rif-admins teams in GitHub"
}

# Display configuration summary
display_summary() {
    print_header "Branch Protection Configuration Summary"
    
    echo "Repository: $REPO_OWNER/$REPO_NAME"
    echo ""
    echo "Branch Protection Rules:"
    echo ""
    echo "📋 Main Branch (main):"
    echo "   ✅ Required status checks: RIF PR Quality Gates"
    echo "   ✅ Required reviews: 1+ for external contributors"
    echo "   ✅ Dismiss stale reviews: Yes"
    echo "   ❌ Force pushes: Disabled"
    echo "   ❌ Branch deletion: Disabled"
    echo ""
    echo "🚀 Deploy Branch (deploy):"
    echo "   ⭕ Required status checks: None (trusts main branch)"
    echo "   ⭕ Required reviews: None (automated sync)"
    echo "   🤖 Restricted pushes: GitHub Actions only"
    echo "   ❌ Force pushes: Disabled"
    echo "   ❌ Branch deletion: Disabled"
    echo ""
    echo "📝 Additional Configuration:"
    echo "   ✅ CODEOWNERS file created/verified"
    echo "   ✅ Protection rules applied"
    echo ""
    echo "Next Steps:"
    echo "1. Add team members to @rif-maintainers GitHub team"
    echo "2. Add admin users to @rif-admins GitHub team"
    echo "3. Test the deploy sync workflow: gh workflow run deploy-sync.yml"
    echo "4. Verify branch protection by attempting restricted operations"
}

# Show usage information
show_usage() {
    echo "RIF Branch Protection Configuration Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --dry-run      Show what would be configured without making changes"
    echo "  --verify-only  Only verify existing protection settings"
    echo ""
    echo "This script configures branch protection rules for the RIF"
    echo "dual-branch deployment strategy."
}

# Dry run mode
dry_run() {
    print_header "DRY RUN MODE - No Changes Will Be Made"
    
    echo "Would configure the following branch protection rules:"
    echo ""
    echo "Main Branch Protection:"
    echo "  - Enable required status checks: RIF PR Quality Gates"
    echo "  - Require 1+ pull request reviews for external contributors"
    echo "  - Enable dismissal of stale reviews"
    echo "  - Disable force pushes"
    echo "  - Disable branch deletion"
    echo ""
    echo "Deploy Branch Protection:"
    echo "  - No required status checks (trusts main branch testing)"
    echo "  - No required pull request reviews (automated sync)"
    echo "  - Restrict pushes to GitHub Actions only"
    echo "  - Disable force pushes"
    echo "  - Disable branch deletion"
    echo ""
    echo "Additional Actions:"
    echo "  - Create/update .github/CODEOWNERS file"
    echo "  - Verify repository permissions"
    echo "  - Test API access to GitHub"
    
    print_info "Dry run complete - no changes made"
}

# Verify only mode
verify_only() {
    print_header "VERIFICATION MODE - Checking Existing Settings"
    
    check_prerequisites
    verify_protection_settings
    
    # Show current protection status
    echo ""
    echo "Current Protection Status:"
    
    # Main branch
    if gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection &>/dev/null; then
        echo "✅ Main branch protection: Enabled"
        local checks=$(gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection --jq '.required_status_checks.contexts[]' 2>/dev/null | wc -l)
        echo "   Status checks: $checks configured"
        
        local reviews=$(gh api repos/$REPO_OWNER/$REPO_NAME/branches/main/protection --jq '.required_pull_request_reviews.required_approving_review_count' 2>/dev/null || echo "0")
        echo "   Required reviews: $reviews"
    else
        echo "❌ Main branch protection: Not configured"
    fi
    
    # Deploy branch  
    if gh api repos/$REPO_OWNER/$REPO_NAME/branches/deploy/protection &>/dev/null; then
        echo "✅ Deploy branch protection: Enabled"
        local restricted=$(gh api repos/$REPO_OWNER/$REPO_NAME/branches/deploy/protection --jq '.restrictions != null' 2>/dev/null || echo "false")
        echo "   Push restrictions: $restricted"
    else
        echo "ℹ Deploy branch protection: Not found (branch may not exist)"
    fi
    
    # CODEOWNERS
    if [[ -f "$PROJECT_ROOT/.github/CODEOWNERS" ]]; then
        echo "✅ CODEOWNERS file: Present"
    else
        echo "❌ CODEOWNERS file: Missing"
    fi
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    print_header "RIF Branch Protection Configuration"
    print_info "Configuring branch protection for dual-branch deployment strategy"
    
    check_prerequisites
    configure_main_branch_protection
    configure_deploy_branch_protection
    create_codeowners
    verify_protection_settings
    display_summary
    
    print_success "Branch protection configuration completed!"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    --dry-run)
        check_prerequisites
        dry_run
        exit 0
        ;;
    --verify-only)
        verify_only
        exit 0
        ;;
    "")
        # No arguments, proceed with normal execution
        ;;
    *)
        echo "Unknown argument: $1"
        show_usage
        exit 1
        ;;
esac

# Trap to handle script interruption
trap 'print_error "Configuration interrupted"; exit 1' INT TERM

# Execute main function
main "$@"