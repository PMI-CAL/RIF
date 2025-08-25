#!/bin/bash
# RIF Knowledge Base Deployment Script
#
# This script orchestrates the complete knowledge base cleanup and deployment process:
# 1. Creates backup
# 2. Analyzes current state  
# 3. Performs cleanup
# 4. Validates results
# 5. Prepares for deployment
#
# Usage: ./scripts/deploy_knowledge.sh [--dry-run] [--skip-backup]

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KNOWLEDGE_DIR="$PROJECT_ROOT/knowledge"
BACKUP_DIR="$PROJECT_ROOT/knowledge_backup"
LOG_FILE="$BACKUP_DIR/deployment_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
DRY_RUN=false
SKIP_BACKUP=false
VERBOSE=false

# Function to log messages
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    case $level in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        *)
            echo -e "$message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check knowledge directory exists
    if [ ! -d "$KNOWLEDGE_DIR" ]; then
        log "ERROR" "Knowledge directory not found: $KNOWLEDGE_DIR"
        exit 1
    fi
    
    # Check scripts exist
    local required_scripts=(
        "clean_knowledge_for_deploy.py"
        "analyze_knowledge_size.py" 
        "test_cleaned_knowledge.py"
        "init_project_knowledge.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$script" ]; then
            log "ERROR" "Required script not found: $script"
            exit 1
        fi
    done
    
    log "SUCCESS" "Prerequisites check passed"
}

# Function to analyze current state
analyze_current_state() {
    log "INFO" "Analyzing current knowledge base state..."
    
    local analysis_file="$BACKUP_DIR/pre_cleanup_analysis.txt"
    
    python3 "$SCRIPT_DIR/analyze_knowledge_size.py" \
        --knowledge-dir "$KNOWLEDGE_DIR" \
        --detailed \
        --output-report "$analysis_file"
    
    if [ $? -eq 0 ]; then
        log "SUCCESS" "Analysis complete - report saved to $analysis_file"
        
        # Show summary
        local total_size=$(du -sh "$KNOWLEDGE_DIR" | cut -f1)
        local file_count=$(find "$KNOWLEDGE_DIR" -type f | wc -l)
        log "INFO" "Current state: $total_size total, $file_count files"
    else
        log "ERROR" "Analysis failed"
        exit 1
    fi
}

# Function to perform backup
create_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        log "WARN" "Skipping backup (--skip-backup specified)"
        return
    fi
    
    log "INFO" "Creating backup of knowledge base..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    local backup_file="$BACKUP_DIR/knowledge_pre_cleanup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # Create compressed backup
    tar -czf "$backup_file" -C "$PROJECT_ROOT" knowledge/
    
    if [ $? -eq 0 ]; then
        local backup_size=$(du -sh "$backup_file" | cut -f1)
        log "SUCCESS" "Backup created: $backup_file ($backup_size)"
        
        # Create rollback script
        local rollback_script="$BACKUP_DIR/rollback_$(date +%Y%m%d_%H%M%S).sh"
        cat > "$rollback_script" << EOF
#!/bin/bash
# Rollback script for knowledge base cleanup
# Generated: $(date)

set -e

echo "Rolling back knowledge base from backup..."
echo "Backup file: $backup_file"

# Remove current knowledge directory
if [ -d "$KNOWLEDGE_DIR" ]; then
    echo "Removing current knowledge directory..."
    rm -rf "$KNOWLEDGE_DIR"
fi

# Extract backup
echo "Restoring from backup..."
tar -xzf "$backup_file" -C "$PROJECT_ROOT"

echo "Rollback complete!"
echo "Knowledge base restored from $backup_file"
EOF
        chmod +x "$rollback_script"
        log "INFO" "Rollback script created: $rollback_script"
        
    else
        log "ERROR" "Backup creation failed"
        exit 1
    fi
}

# Function to perform cleanup
perform_cleanup() {
    log "INFO" "Performing knowledge base cleanup..."
    
    local cleanup_args=(
        "--knowledge-dir" "$KNOWLEDGE_DIR"
        "--backup-dir" "$BACKUP_DIR"
    )
    
    if [ "$DRY_RUN" = true ]; then
        cleanup_args+=("--dry-run")
        log "INFO" "Running in DRY RUN mode - no changes will be made"
    fi
    
    if [ "$SKIP_BACKUP" = true ]; then
        cleanup_args+=("--skip-backup")
    fi
    
    python3 "$SCRIPT_DIR/clean_knowledge_for_deploy.py" "${cleanup_args[@]}"
    
    if [ $? -eq 0 ]; then
        if [ "$DRY_RUN" = true ]; then
            log "SUCCESS" "Dry run completed successfully"
        else
            log "SUCCESS" "Cleanup completed successfully"
        fi
    else
        log "ERROR" "Cleanup failed"
        exit 1
    fi
}

# Function to validate results
validate_results() {
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "Skipping validation in dry run mode"
        return
    fi
    
    log "INFO" "Validating cleaned knowledge base..."
    
    local validation_report="$BACKUP_DIR/validation_report_$(date +%Y%m%d_%H%M%S).md"
    local validation_json="$BACKUP_DIR/validation_results_$(date +%Y%m%d_%H%M%S).json"
    
    python3 "$SCRIPT_DIR/test_cleaned_knowledge.py" \
        --knowledge-dir "$KNOWLEDGE_DIR" \
        --output-report "$validation_report" \
        --json-output "$validation_json"
    
    local validation_exit_code=$?
    
    if [ $validation_exit_code -eq 0 ]; then
        log "SUCCESS" "Validation passed - knowledge base ready for deployment"
        log "INFO" "Validation report: $validation_report"
    else
        log "ERROR" "Validation failed - knowledge base needs additional cleanup"
        log "INFO" "Check validation report: $validation_report"
        exit 1
    fi
}

# Function to generate deployment package
create_deployment_package() {
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "Skipping deployment package creation in dry run mode"
        return
    fi
    
    log "INFO" "Creating deployment package..."
    
    local package_file="$BACKUP_DIR/rif_knowledge_deployment_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # Create deployment package
    tar -czf "$package_file" \
        -C "$PROJECT_ROOT" \
        knowledge/ \
        scripts/init_project_knowledge.py \
        scripts/test_cleaned_knowledge.py \
        --exclude="*.log" \
        --exclude="*.tmp"
    
    if [ $? -eq 0 ]; then
        local package_size=$(du -sh "$package_file" | cut -f1)
        log "SUCCESS" "Deployment package created: $package_file ($package_size)"
        
        # Create deployment instructions
        local instructions_file="$BACKUP_DIR/DEPLOYMENT_INSTRUCTIONS.md"
        cat > "$instructions_file" << EOF
# RIF Knowledge Base Deployment Instructions

## Package Information
- **Package File**: $(basename "$package_file")
- **Created**: $(date)
- **Size**: $package_size

## Deployment Steps

### 1. Extract Package
\`\`\`bash
tar -xzf $package_file -C /target/project/directory
\`\`\`

### 2. Initialize for New Project
\`\`\`bash
cd /target/project/directory
python3 scripts/init_project_knowledge.py \\
  --project-name "Your Project Name" \\
  --project-type "web-app"
\`\`\`

### 3. Validate Installation
\`\`\`bash
python3 scripts/test_cleaned_knowledge.py --knowledge-dir knowledge
\`\`\`

### 4. Complete RIF Setup
\`\`\`bash
./rif-init.sh
\`\`\`

## Project Types
- web-app, mobile-app, desktop-app
- library, framework, api, microservices
- enterprise, prototype, poc, experimental

## Support
- Validation should pass with 80%+ score
- Knowledge base should be under 10MB
- Core framework components preserved
- No RIF-specific development artifacts

## Rollback
If needed, use the rollback script created during cleanup.
EOF
        
        log "SUCCESS" "Deployment instructions created: $instructions_file"
    else
        log "ERROR" "Deployment package creation failed"
        exit 1
    fi
}

# Function to show deployment summary
show_deployment_summary() {
    log "" ""
    log "" "=========================================="
    log "" "   KNOWLEDGE BASE DEPLOYMENT SUMMARY"
    log "" "=========================================="
    log "" ""
    
    if [ "$DRY_RUN" = true ]; then
        log "INFO" "DRY RUN MODE - No actual changes made"
        log "INFO" "Remove --dry-run flag to perform actual deployment"
        log "" ""
        return
    fi
    
    # Show final size
    local final_size=$(du -sh "$KNOWLEDGE_DIR" 2>/dev/null | cut -f1 || echo "N/A")
    local final_files=$(find "$KNOWLEDGE_DIR" -type f 2>/dev/null | wc -l || echo "N/A")
    
    log "SUCCESS" "Final knowledge base size: $final_size ($final_files files)"
    
    # Show backup location
    if [ "$SKIP_BACKUP" != true ]; then
        log "INFO" "Backup location: $BACKUP_DIR"
    fi
    
    # Show deployment package
    local latest_package=$(find "$BACKUP_DIR" -name "rif_knowledge_deployment_*.tar.gz" -type f | sort | tail -1)
    if [ -n "$latest_package" ]; then
        log "SUCCESS" "Deployment package: $(basename "$latest_package")"
    fi
    
    log "" ""
    log "SUCCESS" "Knowledge base deployment preparation complete!"
    log "" ""
    log "INFO" "Next steps:"
    log "INFO" "1. Use deployment package for new projects"
    log "INFO" "2. Run init_project_knowledge.py for project setup"  
    log "INFO" "3. Initialize RIF with ./rif-init.sh"
    log "INFO" "4. Begin development using GitHub issues"
    log "" ""
}

# Function to show usage
show_usage() {
    cat << EOF
RIF Knowledge Base Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    --dry-run        Perform dry run without making changes
    --skip-backup    Skip backup creation (not recommended)
    --verbose        Enable verbose logging
    --help           Show this help message

EXAMPLES:
    # Perform complete deployment preparation
    $0
    
    # Test what would happen without making changes
    $0 --dry-run
    
    # Quick deployment without backup (risky)
    $0 --skip-backup

The script will:
1. Analyze current knowledge base
2. Create backup (unless --skip-backup)
3. Clean knowledge base for deployment
4. Validate cleaning results  
5. Create deployment package
6. Generate deployment instructions

All outputs are saved to: $BACKUP_DIR
EOF
}

# Main execution function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Show header
    log "" ""
    log "" "=========================================="
    log "" "   RIF Knowledge Base Deployment"
    log "" "=========================================="
    log "" ""
    
    # Execute deployment pipeline
    check_prerequisites
    analyze_current_state
    create_backup
    perform_cleanup
    validate_results
    create_deployment_package
    show_deployment_summary
    
    exit 0
}

# Handle script interruption
trap 'log "ERROR" "Deployment interrupted"; exit 130' INT TERM

# Run main function
main "$@"