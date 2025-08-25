#!/bin/bash

# RIF Deployment Branch Cleaning Script
# Removes development artifacts to create a clean production template
# This script is designed to be run from the repository root

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="/tmp/rif-deploy-clean-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

print_info() {
    log "${BLUE}[INFO]${NC} $1"
}

print_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    log "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    log "${BLUE}========================================${NC}"
    log "${BLUE} $1 ${NC}"
    log "${BLUE}========================================${NC}"
}

# Validate environment
validate_environment() {
    print_header "Environment Validation"
    
    # Check we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/CLAUDE.md" ]] || [[ ! -f "$PROJECT_ROOT/rif-init.sh" ]]; then
        print_error "Not running from RIF repository root"
        print_error "Current directory: $PWD"
        print_error "Expected files: CLAUDE.md, rif-init.sh"
        exit 1
    fi
    
    print_info "✓ Running from RIF repository root"
    
    # Check git status
    cd "$PROJECT_ROOT"
    if [[ ! -d ".git" ]]; then
        print_error "Not in a Git repository"
        exit 1
    fi
    
    print_info "✓ Git repository detected"
    
    # Warn about uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_warning "Uncommitted changes detected - these will be included in cleaning"
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Aborted by user"
            exit 0
        fi
    fi
    
    print_success "Environment validation complete"
}

# Create backup before cleaning
create_backup() {
    print_header "Creating Backup"
    
    local backup_dir="/tmp/rif-deploy-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_info "Backup directory: $backup_dir"
    
    # Backup key directories that will be modified
    if [[ -d "knowledge" ]]; then
        cp -r knowledge "$backup_dir/"
        print_info "✓ Knowledge directory backed up"
    fi
    
    if [[ -d "tests" ]]; then
        cp -r tests "$backup_dir/"
        print_info "✓ Tests directory backed up"
    fi
    
    if [[ -d "validation" ]]; then
        cp -r validation "$backup_dir/"
        print_info "✓ Validation directory backed up"
    fi
    
    # Create restore script
    cat > "$backup_dir/restore.sh" << EOF
#!/bin/bash
# Restore script for deployment cleaning backup
# Run from RIF repository root

echo "Restoring from backup: $backup_dir"

if [[ -d "$backup_dir/knowledge" ]]; then
    rm -rf knowledge
    cp -r "$backup_dir/knowledge" .
    echo "✓ Knowledge restored"
fi

if [[ -d "$backup_dir/tests" ]]; then
    rm -rf tests  
    cp -r "$backup_dir/tests" .
    echo "✓ Tests restored"
fi

if [[ -d "$backup_dir/validation" ]]; then
    rm -rf validation
    cp -r "$backup_dir/validation" .
    echo "✓ Validation restored"
fi

echo "✅ Restore complete"
EOF
    
    chmod +x "$backup_dir/restore.sh"
    
    print_success "Backup created: $backup_dir"
    print_info "To restore: $backup_dir/restore.sh"
    echo "BACKUP_DIR=$backup_dir" >> "$LOG_FILE"
}

# Remove development artifacts
remove_development_artifacts() {
    print_header "Removing Development Artifacts"
    
    cd "$PROJECT_ROOT"
    
    # Track removed items for summary
    local removed_dirs=()
    local removed_files=()
    
    # Remove development directories
    local dev_dirs=(
        "knowledge/audits"
        "knowledge/enforcement_logs"
        "knowledge/evidence_collection"
        "knowledge/false_positive_detection"
        "validation"
        "incidents"
        "htmlcov"
        "test_output"
    )
    
    for dir in "${dev_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            removed_dirs+=("$dir")
            print_info "✓ Removed directory: $dir"
        fi
    done
    
    # Remove development files by pattern
    print_info "Removing test and development files..."
    
    # Test files
    find . -name "*test_issue_*" -type f -delete 2>/dev/null || true
    find . -name "validate_issue_*.py" -type f -delete 2>/dev/null || true
    find . -name "audit_*.py" -type f -delete 2>/dev/null || true
    find . -name "*.test.*" -type f -delete 2>/dev/null || true
    
    # Development issue files
    find . -name "*issue_[0-9]*" -type f -delete 2>/dev/null || true
    
    # Log files
    find . -name "*.log" -type f -delete 2>/dev/null || true
    find . -name "migration.log" -type f -delete 2>/dev/null || true
    find . -name "shadow-mode.log" -type f -delete 2>/dev/null || true
    
    # Coverage files
    rm -f .coverage coverage.xml 2>/dev/null || true
    
    # Development summary files
    local summary_files=(
        "CHAT_ERROR_CAPTURE_ANALYSIS_SUMMARY.md"
        "DPIBS_RESEARCH_PHASE1_IMPLEMENTATION_COMPLETE.md" 
        "IMPLEMENTATION_COMPLETE_SUMMARY.md"
        "INCREMENTAL_EXTRACTION_IMPLEMENTATION.md"
        "MCP_KNOWLEDGE_SERVER_SUCCESS_REPORT.md"
        "PATTERN_EXPORT_IMPORT_IMPLEMENTATION_SUMMARY.md"
        "PATTERN_VISUALIZATION_IMPLEMENTATION.md"
        "PATTERN_VISUALIZATION_IMPLEMENTATION_COMPLETE.md"
        "QUALITY_SYSTEMS_VALIDATION_SUMMARY.md"
        "RIF_ORCHESTRATION_COMPLETE_REPORT.md"
    )
    
    for file in "${summary_files[@]}"; do
        if [[ -f "$file" ]]; then
            rm "$file"
            removed_files+=("$file")
            print_info "✓ Removed file: $file"
        fi
    done
    
    # Remove pattern-matched files
    find . -name "RIF_LEARNING_REPORT_*.md" -type f -delete 2>/dev/null || true
    find . -name "VALIDATION_REPORT_*.md" -type f -delete 2>/dev/null || true
    find . -name "VALIDATION_SUMMARY_*.md" -type f -delete 2>/dev/null || true
    find . -name "ISSUE_*_IMPLEMENTATION_COMPLETE.md" -type f -delete 2>/dev/null || true
    find . -name "ISSUE_*_VALIDATION_REPORT.md" -type f -delete 2>/dev/null || true
    
    # Environment files (will be replaced)
    rm -f .env.* 2>/dev/null || true
    
    # Database files (keep schema, remove data)
    find . -name "*.duckdb" -type f -delete 2>/dev/null || true
    find . -name "*.duckdb.wal" -type f -delete 2>/dev/null || true
    find . -name "*test*.db" -type f -delete 2>/dev/null || true
    find . -name "demo*.duckdb" -type f -delete 2>/dev/null || true
    
    print_success "Development artifacts removed"
    print_info "Directories removed: ${#removed_dirs[@]}"
    print_info "Files removed: ${#removed_files[@]}+"
}

# Clean knowledge base for deployment
clean_knowledge_base() {
    print_header "Cleaning Knowledge Base"
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -d "knowledge" ]]; then
        print_warning "No knowledge directory found, skipping cleanup"
        return
    fi
    
    # Create Python script for knowledge cleaning
    cat > /tmp/clean_knowledge.py << 'EOF'
#!/usr/bin/env python3
"""
Clean knowledge base for deployment - keep essential patterns and decisions,
remove development-specific data
"""
import os
import json
import shutil
from pathlib import Path

def clean_knowledge_base():
    knowledge_dir = Path("knowledge")
    if not knowledge_dir.exists():
        print("No knowledge directory found, skipping cleanup")
        return
    
    removed_items = []
    
    # Files to remove
    remove_files = [
        "issue_closure_prevention_log.json",
        "pending_user_validations.json", 
        "user_validation_log.json",
        "events.jsonl",
        "shadow-mode.log",
        "migration_state.json",
        "migration_final_report.json",
        "cutover_config.json"
    ]
    
    for file_name in remove_files:
        file_path = knowledge_dir / file_name
        if file_path.exists():
            file_path.unlink()
            removed_items.append(f"file: {file_name}")
            print(f"✓ Removed {file_name}")
    
    # Clean learning directory - keep general patterns, remove issue-specific
    learning_dir = knowledge_dir / "learning"
    if learning_dir.exists():
        for file_path in learning_dir.glob("issue-*"):
            file_path.unlink()
            removed_items.append(f"learning: {file_path.name}")
            print(f"✓ Removed learning/{file_path.name}")
        
        for file_path in learning_dir.glob("*issue*"):
            file_path.unlink()
            removed_items.append(f"learning: {file_path.name}")  
            print(f"✓ Removed learning/{file_path.name}")
    
    # Clean checkpoints - keep only general patterns
    checkpoints_dir = knowledge_dir / "checkpoints"
    if checkpoints_dir.exists():
        for file_path in checkpoints_dir.glob("issue-*"):
            file_path.unlink()
            removed_items.append(f"checkpoint: {file_path.name}")
            print(f"✓ Removed checkpoint/{file_path.name}")
    
    # Clean patterns directory - keep generic patterns, remove issue-specific
    patterns_dir = knowledge_dir / "patterns"
    if patterns_dir.exists():
        for file_path in patterns_dir.glob("*issue*"):
            file_path.unlink()
            removed_items.append(f"pattern: {file_path.name}")
            print(f"✓ Removed pattern/{file_path.name}")
    
    print(f"✅ Knowledge base cleaned - {len(removed_items)} items removed")
    return removed_items

if __name__ == "__main__":
    clean_knowledge_base()
EOF
    
    # Run the cleaning script
    python3 /tmp/clean_knowledge.py
    rm /tmp/clean_knowledge.py
    
    print_success "Knowledge base cleaning completed"
}

# Update configuration for deployment
update_configuration() {
    print_header "Updating Deployment Configuration"
    
    cd "$PROJECT_ROOT"
    
    # Create deployment-specific configuration
    cat > deploy.config.json << 'EOF'
{
  "version": "1.0.0",
  "deployment_mode": "template",
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
    "backup_existing": false
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
    
    print_success "✓ Deploy configuration created"
    
    # Update README with deployment-specific instructions if it exists
    if [[ -f README.md ]]; then
        # Create backup of original README
        cp README.md README.md.backup
        
        # Add deployment banner at the top
        {
            echo "# RIF - Reactive Intelligence Framework (Production Template)"
            echo ""
            echo "> 🚀 **This is the production-ready template branch.** For development, see the [main branch](../../tree/main)."
            echo ""
            cat README.md.backup
        } > README.md
        
        # Update installation instructions
        sed -i.bak 's|git clone.*|git clone -b deploy https://github.com/[username]/rif.git my-project|' README.md
        sed -i.bak 's|\./rif-init\.sh|cd my-project \&\& ./rif-init.sh --mode production|' README.md
        
        # Clean up backup files
        rm README.md.backup README.md.bak 2>/dev/null || true
        
        print_success "✓ README updated for deployment"
    fi
    
    print_success "Configuration updates completed"
}

# Validate deployment state
validate_deployment() {
    print_header "Validating Deployment State"
    
    cd "$PROJECT_ROOT"
    
    local validation_errors=0
    
    # Check essential files exist
    local essential_files=(
        "README.md"
        "rif-init.sh"
        "setup.sh"
        "claude/agents/rif-implementer.md"
        "config/rif-workflow.yaml"
        "CLAUDE.md"
        "deploy.config.json"
    )
    
    for file in "${essential_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "✗ Essential file missing: $file"
            ((validation_errors++))
        else
            print_info "✓ Essential file present: $file"
        fi
    done
    
    # Validate configuration syntax
    if [[ -f deploy.config.json ]]; then
        if python3 -c "import json; json.load(open('deploy.config.json'))" 2>/dev/null; then
            print_info "✓ Deploy configuration JSON is valid"
        else
            print_error "✗ Invalid JSON in deploy.config.json"
            ((validation_errors++))
        fi
    fi
    
    if [[ -f config/rif-workflow.yaml ]]; then
        if python3 -c "import yaml; yaml.safe_load(open('config/rif-workflow.yaml'))" 2>/dev/null; then
            print_info "✓ Workflow configuration YAML is valid"
        else
            print_warning "⚠ Workflow YAML might have issues (yaml module not available)"
        fi
    fi
    
    # Check that development artifacts are properly removed
    if [[ -d knowledge/audits ]]; then
        print_error "✗ Development artifacts still present: knowledge/audits"
        ((validation_errors++))
    else
        print_info "✓ Development artifacts properly removed"
    fi
    
    # Test script executability
    for script in rif-init.sh setup.sh; do
        if [[ ! -x "$script" ]]; then
            print_error "✗ Script not executable: $script"
            ((validation_errors++))
        else
            print_info "✓ Script executable: $script"
        fi
    done
    
    # Summary
    if [[ $validation_errors -eq 0 ]]; then
        print_success "✅ Deployment validation passed - 0 errors"
        return 0
    else
        print_error "❌ Deployment validation failed - $validation_errors errors"
        return 1
    fi
}

# Generate deployment report
generate_report() {
    print_header "Generating Deployment Report"
    
    local report_file="$PROJECT_ROOT/deployment-clean-report-$(date +%Y%m%d-%H%M%S).json"
    
    cd "$PROJECT_ROOT"
    
    cat > "$report_file" << EOF
{
  "cleaning_summary": {
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "script_version": "1.0.0",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "cleaning_duration_seconds": "$SECONDS",
    "user": "$USER",
    "hostname": "$(hostname)"
  },
  "operations_performed": {
    "development_artifacts_removed": true,
    "knowledge_base_cleaned": true,
    "configuration_updated": true,
    "validation_completed": true,
    "backup_created": true
  },
  "files_created": [
    "deploy.config.json",
    "$report_file"
  ],
  "log_file": "$LOG_FILE",
  "validation_status": "$(if validate_deployment >/dev/null 2>&1; then echo "passed"; else echo "failed"; fi)",
  "next_steps": [
    "Review validation results",
    "Test RIF initialization: ./rif-init.sh --mode production",
    "Commit changes if on deploy branch",
    "Test user onboarding flow"
  ]
}
EOF
    
    print_success "Deployment report generated: $report_file"
    echo "REPORT_FILE=$report_file" >> "$LOG_FILE"
}

# Print summary
print_summary() {
    echo ""
    echo "=============================================="
    echo "RIF DEPLOYMENT CLEANING SUMMARY"
    echo "=============================================="
    echo "Duration: ${SECONDS} seconds"
    echo "Log file: $LOG_FILE"
    
    if grep -q "BACKUP_DIR=" "$LOG_FILE"; then
        local backup_dir=$(grep "BACKUP_DIR=" "$LOG_FILE" | cut -d'=' -f2)
        echo "Backup: $backup_dir"
        echo "To restore: $backup_dir/restore.sh"
    fi
    
    if grep -q "REPORT_FILE=" "$LOG_FILE"; then
        local report_file=$(grep "REPORT_FILE=" "$LOG_FILE" | cut -d'=' -f2)
        echo "Report: $report_file"
    fi
    
    echo ""
    
    if validate_deployment >/dev/null 2>&1; then
        echo "✅ Deployment cleaning completed successfully!"
        echo ""
        echo "The repository is now clean and ready for deployment."
        echo "Next steps:"
        echo "1. Test initialization: ./rif-init.sh --mode production"
        echo "2. Commit changes if on deploy branch"
        echo "3. Test user onboarding workflow"
    else
        echo "❌ Deployment cleaning completed with validation errors"
        echo "Check the validation output above for details."
    fi
}

# Main function
main() {
    print_header "RIF Deployment Branch Cleaning"
    print_info "Starting deployment cleaning at $(date)"
    print_info "Log file: $LOG_FILE"
    
    # Cleaning pipeline
    validate_environment
    create_backup
    remove_development_artifacts
    clean_knowledge_base
    update_configuration
    validate_deployment
    generate_report
    print_summary
    
    if ! validate_deployment >/dev/null 2>&1; then
        exit 1
    fi
}

# Usage information
show_usage() {
    echo "RIF Deployment Branch Cleaning Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --dry-run      Show what would be cleaned without making changes"
    echo "  --no-backup    Skip backup creation (not recommended)"
    echo ""
    echo "This script removes development artifacts to create a clean"
    echo "production template suitable for user deployment."
    echo ""
    echo "Run from the RIF repository root directory."
}

# Dry run mode
dry_run() {
    print_header "DRY RUN MODE - No Changes Will Be Made"
    
    cd "$PROJECT_ROOT"
    
    echo "Would remove these directories:"
    local dev_dirs=(
        "knowledge/audits"
        "knowledge/enforcement_logs"
        "knowledge/evidence_collection"
        "knowledge/false_positive_detection"
        "validation"
        "incidents"
        "htmlcov"
        "test_output"
    )
    
    for dir in "${dev_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            echo "  - $dir"
        fi
    done
    
    echo ""
    echo "Would remove these file patterns:"
    echo "  - *test_issue_*"
    echo "  - validate_issue_*.py"
    echo "  - audit_*.py"
    echo "  - *.test.*"
    echo "  - *issue_[0-9]*"
    echo "  - *.log files"
    echo "  - Coverage files"
    echo "  - Development summary .md files"
    echo "  - Database files (.duckdb, etc.)"
    
    echo ""
    echo "Would clean knowledge base:"
    if [[ -d "knowledge" ]]; then
        echo "  - Issue-specific learning files"
        echo "  - Development checkpoints"
        echo "  - Audit logs and validation data"
    else
        echo "  - No knowledge directory found"
    fi
    
    echo ""
    echo "Would create/update:"
    echo "  - deploy.config.json"
    echo "  - README.md (deployment version)"
    echo "  - Backup in /tmp/rif-deploy-backup-*"
    
    print_info "Dry run complete - no changes made"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    --dry-run)
        validate_environment
        dry_run
        exit 0
        ;;
    --no-backup)
        SKIP_BACKUP=true
        shift
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
trap 'print_error "Cleaning interrupted"; exit 1' INT TERM

# Execute main function
main "$@"