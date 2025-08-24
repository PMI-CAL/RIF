#!/bin/bash
#
# Claude Code Knowledge MCP Server Startup Script
#
# This script starts the Claude Code Knowledge MCP Server with proper environment
# configuration and monitoring. It handles initialization, health checks, and
# graceful shutdown.
#

set -euo pipefail

# Configuration
SERVER_NAME="claude-code-knowledge"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIF_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/.server.pid"
CONFIG_FILE="${SCRIPT_DIR}/mcp.json"

# Logging setup
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server.log"
ERROR_LOG="${LOG_DIR}/error.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "${ERROR_LOG}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "${LOG_FILE}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "${LOG_FILE}"
}

# Check if server is already running
check_running() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid=$(cat "${PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            return 0  # Running
        else
            rm -f "${PID_FILE}"
            return 1  # Not running
        fi
    fi
    return 1  # Not running
}

# Health check function
health_check() {
    local max_attempts=5
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Health check attempt ${attempt}/${max_attempts}"
        
        # Simple Python import test
        if python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
sys.path.insert(0, '${RIF_ROOT}')

try:
    from server import ClaudeCodeKnowledgeServer
    from config import load_server_config
    print('✓ Imports successful')
    
    config = load_server_config()
    server = ClaudeCodeKnowledgeServer(config.__dict__)
    print('✓ Server creation successful')
    
    exit(0)
except Exception as e:
    print(f'✗ Health check failed: {e}')
    exit(1)
" 2>>"${ERROR_LOG}"; then
            success "Health check passed"
            return 0
        else
            warn "Health check failed, attempt ${attempt}/${max_attempts}"
            sleep 2
            ((attempt++))
        fi
    done
    
    error "Health check failed after ${max_attempts} attempts"
    return 1
}

# Prerequisites check
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        return 1
    fi
    
    # Check Python version
    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log "Python version: ${python_version}"
    
    # Check if we can import required modules
    if ! python3 -c "import asyncio, json, logging, dataclasses" 2>/dev/null; then
        error "Required Python modules not available"
        return 1
    fi
    
    # Check RIF root directory
    if [[ ! -d "${RIF_ROOT}" ]]; then
        error "RIF root directory not found: ${RIF_ROOT}"
        return 1
    fi
    
    success "Prerequisites check passed"
    return 0
}

# Start the server
start_server() {
    log "Starting ${SERVER_NAME} MCP Server..."
    
    if check_running; then
        warn "Server is already running (PID: $(cat "${PID_FILE}"))"
        return 0
    fi
    
    # Check prerequisites
    if ! check_prerequisites; then
        error "Prerequisites check failed"
        return 1
    fi
    
    # Set environment variables
    export PYTHONPATH="${SCRIPT_DIR}:${RIF_ROOT}:${PYTHONPATH:-}"
    export MCP_LOG_LEVEL="${MCP_LOG_LEVEL:-INFO}"
    export MCP_CACHE_ENABLED="${MCP_CACHE_ENABLED:-true}"
    export MCP_DEBUG_MODE="${MCP_DEBUG_MODE:-false}"
    
    # Start server in background
    log "Starting server with environment:"
    log "  PYTHONPATH: ${PYTHONPATH}"
    log "  MCP_LOG_LEVEL: ${MCP_LOG_LEVEL}"
    log "  Working directory: ${SCRIPT_DIR}"
    
    # Run the server
    cd "${SCRIPT_DIR}"
    python3 -u server.py >> "${LOG_FILE}" 2>> "${ERROR_LOG}" &
    local server_pid=$!
    
    # Save PID
    echo "${server_pid}" > "${PID_FILE}"
    
    # Wait a moment and check if it started
    sleep 2
    if kill -0 "${server_pid}" 2>/dev/null; then
        success "Server started successfully (PID: ${server_pid})"
        
        # Run health check
        if health_check; then
            success "${SERVER_NAME} MCP Server is running and healthy"
            log "Server logs: ${LOG_FILE}"
            log "Error logs: ${ERROR_LOG}"
            return 0
        else
            error "Server started but health check failed"
            stop_server
            return 1
        fi
    else
        error "Server failed to start"
        rm -f "${PID_FILE}"
        return 1
    fi
}

# Stop the server
stop_server() {
    log "Stopping ${SERVER_NAME} MCP Server..."
    
    if ! check_running; then
        warn "Server is not running"
        return 0
    fi
    
    local pid=$(cat "${PID_FILE}")
    log "Sending SIGTERM to PID ${pid}"
    
    if kill "${pid}" 2>/dev/null; then
        # Wait for graceful shutdown
        local wait_time=0
        local max_wait=10
        
        while kill -0 "${pid}" 2>/dev/null && [[ $wait_time -lt $max_wait ]]; do
            sleep 1
            ((wait_time++))
        done
        
        # Force kill if still running
        if kill -0 "${pid}" 2>/dev/null; then
            warn "Server didn't stop gracefully, forcing shutdown"
            kill -9 "${pid}" 2>/dev/null || true
        fi
        
        rm -f "${PID_FILE}"
        success "Server stopped successfully"
    else
        error "Failed to stop server (PID: ${pid})"
        rm -f "${PID_FILE}"
        return 1
    fi
}

# Show server status
show_status() {
    if check_running; then
        local pid=$(cat "${PID_FILE}")
        success "${SERVER_NAME} MCP Server is running (PID: ${pid})"
        
        # Show recent logs
        if [[ -f "${LOG_FILE}" ]]; then
            echo ""
            log "Recent log entries:"
            tail -10 "${LOG_FILE}"
        fi
        
        return 0
    else
        warn "${SERVER_NAME} MCP Server is not running"
        return 1
    fi
}

# Show logs
show_logs() {
    local log_type="${1:-server}"
    
    case "${log_type}" in
        "error"|"errors")
            if [[ -f "${ERROR_LOG}" ]]; then
                tail -f "${ERROR_LOG}"
            else
                error "Error log file not found"
                return 1
            fi
            ;;
        "server"|"")
            if [[ -f "${LOG_FILE}" ]]; then
                tail -f "${LOG_FILE}"
            else
                error "Server log file not found"  
                return 1
            fi
            ;;
        *)
            error "Unknown log type: ${log_type}. Use 'server' or 'error'"
            return 1
            ;;
    esac
}

# Show help
show_help() {
    cat << EOF
${SERVER_NAME} MCP Server Management Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    start         Start the MCP server
    stop          Stop the MCP server  
    restart       Restart the MCP server
    status        Show server status
    logs [type]   Follow server logs (type: server|error, default: server)
    health        Run health check
    help          Show this help message

ENVIRONMENT VARIABLES:
    MCP_LOG_LEVEL     Logging level (DEBUG|INFO|WARNING|ERROR, default: INFO)
    MCP_CACHE_ENABLED Enable caching (true|false, default: true)
    MCP_DEBUG_MODE    Enable debug mode (true|false, default: false)

EXAMPLES:
    $0 start                    # Start the server
    $0 status                   # Check if server is running
    $0 logs                     # Follow server logs
    $0 logs error              # Follow error logs
    MCP_LOG_LEVEL=DEBUG $0 start # Start with debug logging

FILES:
    Configuration: ${CONFIG_FILE}
    Server logs:   ${LOG_FILE}
    Error logs:    ${ERROR_LOG}
    PID file:      ${PID_FILE}

EOF
}

# Main command processing
case "${1:-help}" in
    "start")
        start_server
        ;;
    "stop")
        stop_server
        ;;
    "restart")
        log "Restarting ${SERVER_NAME} MCP Server..."
        stop_server
        sleep 1
        start_server
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "${2:-server}"
        ;;
    "health")
        health_check
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac