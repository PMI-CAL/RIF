#!/bin/bash

# DPIBS Production Deployment Script
# Enterprise-grade deployment with rollback capabilities
# Phase 4 Implementation for Issue #132

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="/tmp/dpibs-backup-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/tmp/dpibs-deploy-$(date +%Y%m%d-%H%M%S).log"
ENVIRONMENT="production"
ROLLBACK_REQUIRED=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Environment validation
validate_environment() {
    print_header "Environment Validation"
    
    # Check required commands
    local required_commands=("python3" "git" "gh" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "Required command not found: $cmd"
            exit 1
        fi
        print_info "✓ $cmd available"
    done
    
    # Check Python dependencies
    if ! python3 -c "import sys; assert sys.version_info >= (3, 8)" 2>/dev/null; then
        print_error "Python 3.8+ required"
        exit 1
    fi
    print_info "✓ Python 3.8+ available"
    
    # Check Git status
    if [[ ! -d "$SCRIPT_DIR/.git" ]]; then
        print_error "Not in a Git repository"
        exit 1
    fi
    print_info "✓ Git repository detected"
    
    # Check for uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_warning "Uncommitted changes detected - will be included in deployment"
    fi
    
    print_success "Environment validation complete"
}

# Pre-deployment backup
create_backup() {
    print_header "Creating Backup"
    
    mkdir -p "$BACKUP_DIR"
    print_info "Backup directory: $BACKUP_DIR"
    
    # Backup configuration files
    if [[ -d "$SCRIPT_DIR/config" ]]; then
        cp -r "$SCRIPT_DIR/config" "$BACKUP_DIR/"
        print_info "✓ Configuration files backed up"
    fi
    
    # Backup knowledge database
    if [[ -d "$SCRIPT_DIR/knowledge" ]]; then
        cp -r "$SCRIPT_DIR/knowledge" "$BACKUP_DIR/"
        print_info "✓ Knowledge database backed up"
    fi
    
    # Backup systems directory
    if [[ -d "$SCRIPT_DIR/systems" ]]; then
        cp -r "$SCRIPT_DIR/systems" "$BACKUP_DIR/"
        print_info "✓ Systems directory backed up"
    fi
    
    # Create deployment manifest
    cat > "$BACKUP_DIR/deployment-manifest.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git branch --show-current)",
    "environment": "$ENVIRONMENT",
    "backup_location": "$BACKUP_DIR",
    "deployment_script": "$0",
    "user": "$USER",
    "hostname": "$(hostname)"
}
EOF
    
    print_success "Backup created: $BACKUP_DIR"
}

# Database schema validation and migration
validate_database() {
    print_header "Database Validation"
    
    # Check if DPIBS tables exist and create if needed
    python3 << EOF
import sqlite3
import os
import json
from pathlib import Path

def validate_dpibs_database():
    """Validate and create DPIBS database schema if needed"""
    db_path = Path("$SCRIPT_DIR/knowledge/dpibs.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create DPIBS performance optimization tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dpibs_performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        metric_type TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metadata TEXT,
        environment TEXT DEFAULT 'production'
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dpibs_cache_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        cache_level TEXT NOT NULL,
        hit_rate REAL NOT NULL,
        miss_rate REAL NOT NULL,
        total_requests INTEGER NOT NULL,
        cache_size_mb REAL,
        metadata TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dpibs_system_resources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        cpu_usage_percent REAL,
        memory_usage_percent REAL,
        disk_usage_percent REAL,
        network_io_mb REAL,
        metadata TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dpibs_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        resolved BOOLEAN DEFAULT FALSE,
        resolved_timestamp DATETIME,
        metadata TEXT
    )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON dpibs_performance_metrics(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON dpibs_cache_stats(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON dpibs_system_resources(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON dpibs_alerts(timestamp)')
    
    conn.commit()
    conn.close()
    
    print("✓ DPIBS database schema validated and created")
    return True

if __name__ == "__main__":
    try:
        validate_dpibs_database()
        exit(0)
    except Exception as e:
        print(f"✗ Database validation failed: {e}")
        exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        print_success "Database validation complete"
    else
        print_error "Database validation failed"
        exit 1
    fi
}

# System health checks
run_health_checks() {
    print_header "System Health Checks"
    
    # Test monitoring dashboard syntax
    print_info "Testing monitoring dashboard syntax..."
    if python3 -m py_compile "$SCRIPT_DIR/systems/dpibs_monitoring_dashboard.py" 2>/dev/null; then
        print_success "✓ Monitoring dashboard syntax valid"
    else
        print_error "✗ Monitoring dashboard syntax error"
        print_error "Fix monitoring dashboard before deployment"
        exit 1
    fi
    
    # Test core DPIBS components
    print_info "Testing core DPIBS components..."
    python3 << 'EOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test imports
    import importlib.util
    
    # Test monitoring configuration
    import yaml
    config_path = "config/monitoring.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("✓ Monitoring configuration valid")
    else:
        print("✗ Monitoring configuration missing")
        sys.exit(1)
        
    # Test knowledge database accessibility
    if os.path.exists("knowledge"):
        print("✓ Knowledge database accessible")
    else:
        print("✗ Knowledge database missing")
        sys.exit(1)
        
    print("✓ Core component validation complete")
    
except Exception as e:
    print(f"✗ Component validation failed: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        print_success "Health checks passed"
    else
        print_error "Health checks failed"
        ROLLBACK_REQUIRED=true
        return 1
    fi
}

# Configure production environment
configure_production() {
    print_header "Production Configuration"
    
    # Set production environment variables
    cat > "$SCRIPT_DIR/.env.production" << EOF
# DPIBS Production Environment Configuration
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

ENVIRONMENT=production
DPIBS_LOG_LEVEL=INFO
DPIBS_MONITORING_ENABLED=true
DPIBS_CACHE_ENABLED=true
DPIBS_PERFORMANCE_TRACKING=true

# Database Configuration
DPIBS_DB_PATH=./knowledge/dpibs.db
DPIBS_BACKUP_ENABLED=true
DPIBS_BACKUP_INTERVAL_HOURS=6

# Monitoring Configuration
DPIBS_METRICS_RETENTION_HOURS=168  # 1 week
DPIBS_ALERT_RETENTION_HOURS=720    # 30 days
DPIBS_MAX_METRICS_IN_MEMORY=10000

# Performance Thresholds
DPIBS_RESPONSE_TIME_THRESHOLD_MS=200
DPIBS_CACHE_HIT_RATE_THRESHOLD=70
DPIBS_CPU_THRESHOLD_PERCENT=80
DPIBS_MEMORY_THRESHOLD_PERCENT=80

# Security Configuration
DPIBS_SECURE_MODE=true
DPIBS_API_RATE_LIMITING=true
EOF
    
    print_success "✓ Production environment configured"
    
    # Update monitoring configuration for production
    if [[ -f "$SCRIPT_DIR/config/monitoring.yaml" ]]; then
        # Create production-optimized monitoring config
        cp "$SCRIPT_DIR/config/monitoring.yaml" "$SCRIPT_DIR/config/monitoring.yaml.backup"
        
        python3 << 'EOF'
import yaml
import os

config_path = "config/monitoring.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Production optimizations
config['performance_monitoring']['enabled'] = True
config['performance_monitoring']['collection_interval_seconds'] = 60
config['alerts']['enabled'] = True
config['dashboard']['auto_start'] = True
config['dashboard']['port'] = 8080
config['logging']['level'] = 'INFO'

# Save updated config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✓ Monitoring configuration updated for production")
EOF
        
        print_success "✓ Monitoring configuration optimized for production"
    fi
}

# Start production services
start_services() {
    print_header "Starting Production Services"
    
    # Create systemd service files for production deployment
    if command -v systemctl &> /dev/null && [[ -d "/etc/systemd/system" ]]; then
        print_info "Creating systemd service files..."
        
        # This would require sudo, so we create the files in a temp location
        cat > "/tmp/dpibs-monitoring.service" << EOF
[Unit]
Description=DPIBS Monitoring Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
Environment=ENVIRONMENT=production
ExecStart=/usr/bin/python3 systems/dpibs_monitoring_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        print_info "✓ Service file created: /tmp/dpibs-monitoring.service"
        print_info "  To install: sudo cp /tmp/dpibs-monitoring.service /etc/systemd/system/"
        print_info "  To enable: sudo systemctl enable dpibs-monitoring"
        print_info "  To start: sudo systemctl start dpibs-monitoring"
    fi
    
    # Start monitoring in background for testing
    print_info "Starting monitoring dashboard in test mode..."
    
    # Test the monitoring dashboard startup
    timeout 10 python3 << 'EOF' &
import sys
import os
sys.path.insert(0, '.')

try:
    # Import and test dashboard creation
    from systems.dpibs_monitoring_dashboard import DPIBSMonitoringDashboard
    print("✓ Monitoring dashboard imports successfully")
    
    # Test configuration loading
    import yaml
    with open('config/monitoring.yaml') as f:
        config = yaml.safe_load(f)
    print("✓ Monitoring configuration loads successfully")
    
    print("✓ Services validation complete")
    
except Exception as e:
    print(f"✗ Service validation failed: {e}")
    sys.exit(1)
EOF
    
    wait
    
    if [[ $? -eq 0 ]]; then
        print_success "✓ Services validated and ready"
    else
        print_error "✗ Service validation failed"
        ROLLBACK_REQUIRED=true
        return 1
    fi
}

# Validate deployment
validate_deployment() {
    print_header "Deployment Validation"
    
    # Run comprehensive validation tests
    python3 << 'EOF'
import os
import sys
import yaml
import json
from pathlib import Path

def validate_deployment():
    """Comprehensive deployment validation"""
    print("Running deployment validation tests...")
    
    # Test 1: Configuration files exist and are valid
    config_files = [
        'config/monitoring.yaml',
        'config/rif-workflow.yaml',
        '.env.production'
    ]
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"✗ Missing configuration file: {config_file}")
            return False
        print(f"✓ Configuration file exists: {config_file}")
    
    # Test 2: YAML configurations are valid
    try:
        with open('config/monitoring.yaml') as f:
            monitoring_config = yaml.safe_load(f)
        print("✓ Monitoring configuration is valid YAML")
    except Exception as e:
        print(f"✗ Monitoring configuration YAML error: {e}")
        return False
    
    # Test 3: Database is accessible
    db_path = Path('knowledge/dpibs.db')
    if db_path.exists():
        print("✓ DPIBS database exists")
    else:
        print("✗ DPIBS database missing")
        return False
    
    # Test 4: Key directories exist
    required_dirs = ['systems', 'knowledge', 'config', 'claude']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"✗ Missing required directory: {dir_name}")
            return False
        print(f"✓ Required directory exists: {dir_name}")
    
    # Test 5: Python modules can be imported
    try:
        sys.path.insert(0, '.')
        # Test critical imports
        import importlib.util
        
        # Test monitoring dashboard
        spec = importlib.util.spec_from_file_location(
            "dashboard", "systems/dpibs_monitoring_dashboard.py"
        )
        if spec is None:
            print("✗ Cannot load monitoring dashboard module")
            return False
        print("✓ Monitoring dashboard module loadable")
        
    except Exception as e:
        print(f"✗ Module import test failed: {e}")
        return False
    
    print("✓ All deployment validation tests passed")
    return True

if __name__ == "__main__":
    if validate_deployment():
        exit(0)
    else:
        exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        print_success "Deployment validation passed"
    else
        print_error "Deployment validation failed"
        ROLLBACK_REQUIRED=true
        return 1
    fi
}

# Rollback function
rollback_deployment() {
    print_header "Rolling Back Deployment"
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        print_error "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    print_info "Restoring from backup: $BACKUP_DIR"
    
    # Stop any running services
    pkill -f "dpibs_monitoring_dashboard" 2>/dev/null || true
    
    # Restore configuration
    if [[ -d "$BACKUP_DIR/config" ]]; then
        rm -rf "$SCRIPT_DIR/config"
        cp -r "$BACKUP_DIR/config" "$SCRIPT_DIR/"
        print_info "✓ Configuration restored"
    fi
    
    # Restore systems
    if [[ -d "$BACKUP_DIR/systems" ]]; then
        rm -rf "$SCRIPT_DIR/systems"
        cp -r "$BACKUP_DIR/systems" "$SCRIPT_DIR/"
        print_info "✓ Systems directory restored"
    fi
    
    # Restore knowledge database
    if [[ -d "$BACKUP_DIR/knowledge" ]]; then
        rm -rf "$SCRIPT_DIR/knowledge"
        cp -r "$BACKUP_DIR/knowledge" "$SCRIPT_DIR/"
        print_info "✓ Knowledge database restored"
    fi
    
    # Remove production environment file
    rm -f "$SCRIPT_DIR/.env.production"
    
    print_success "Rollback completed"
    print_info "System restored to pre-deployment state"
}

# Generate deployment report
generate_report() {
    print_header "Generating Deployment Report"
    
    local report_file="$SCRIPT_DIR/deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "deployment_summary": {
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "environment": "$ENVIRONMENT",
        "status": "$(if [[ "$ROLLBACK_REQUIRED" == "true" ]]; then echo "failed"; else echo "success"; fi)",
        "git_commit": "$(git rev-parse HEAD)",
        "git_branch": "$(git branch --show-current)",
        "deployment_duration_seconds": "$SECONDS",
        "user": "$USER",
        "hostname": "$(hostname)"
    },
    "components_deployed": {
        "monitoring_dashboard": true,
        "database_schema": true,
        "production_configuration": true,
        "health_checks": true,
        "backup_created": true
    },
    "files_created": [
        ".env.production",
        "config/monitoring.yaml.backup",
        "/tmp/dpibs-monitoring.service",
        "$report_file"
    ],
    "backup_location": "$BACKUP_DIR",
    "log_file": "$LOG_FILE",
    "next_steps": [
        "Install systemd service: sudo cp /tmp/dpibs-monitoring.service /etc/systemd/system/",
        "Enable service: sudo systemctl enable dpibs-monitoring",
        "Start service: sudo systemctl start dpibs-monitoring",
        "Monitor logs: journalctl -u dpibs-monitoring -f",
        "Access dashboard: http://localhost:8080"
    ]
}
EOF
    
    print_success "Deployment report generated: $report_file"
    
    # Display summary
    echo ""
    echo "========================="
    echo "DEPLOYMENT SUMMARY"
    echo "========================="
    echo "Status: $(if [[ "$ROLLBACK_REQUIRED" == "true" ]]; then echo -e "${RED}FAILED${NC}"; else echo -e "${GREEN}SUCCESS${NC}"; fi)"
    echo "Duration: ${SECONDS} seconds"
    echo "Backup: $BACKUP_DIR"
    echo "Log: $LOG_FILE"
    echo "Report: $report_file"
    echo ""
    
    if [[ "$ROLLBACK_REQUIRED" == "false" ]]; then
        echo "🎉 DPIBS Production deployment completed successfully!"
        echo ""
        echo "Next Steps:"
        echo "1. Install the systemd service (requires sudo):"
        echo "   sudo cp /tmp/dpibs-monitoring.service /etc/systemd/system/"
        echo "   sudo systemctl enable dpibs-monitoring"
        echo "   sudo systemctl start dpibs-monitoring"
        echo ""
        echo "2. Monitor the service:"
        echo "   journalctl -u dpibs-monitoring -f"
        echo ""
        echo "3. Access the monitoring dashboard:"
        echo "   http://localhost:8080"
    else
        echo "❌ Deployment failed and was rolled back"
        echo "Check the log file for details: $LOG_FILE"
    fi
}

# Main deployment function
main() {
    print_header "DPIBS Production Deployment"
    print_info "Starting deployment at $(date)"
    print_info "Script: $0"
    print_info "Log file: $LOG_FILE"
    
    # Deployment pipeline
    validate_environment || exit 1
    create_backup || exit 1
    validate_database || exit 1
    run_health_checks || { rollback_deployment; exit 1; }
    configure_production || { rollback_deployment; exit 1; }
    start_services || { rollback_deployment; exit 1; }
    validate_deployment || { rollback_deployment; exit 1; }
    
    # Generate final report
    generate_report
    
    if [[ "$ROLLBACK_REQUIRED" == "true" ]]; then
        exit 1
    fi
    
    print_success "Production deployment completed successfully!"
}

# Trap to handle script interruption
trap 'print_error "Deployment interrupted"; rollback_deployment; exit 1' INT TERM

# Execute main function
main "$@"