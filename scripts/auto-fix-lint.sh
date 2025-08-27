#!/bin/bash

# Auto-fix Lint Issues Script
# Part of GitHub-First Automation System (Issue #283)
# Automatically fixes common linting issues across multiple technologies

set -euo pipefail

echo "🔧 Auto-fix Lint: Starting lint issue resolution..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Counter for fixes applied
FIXES_APPLIED=0

# JavaScript/Node.js lint fixes
if [ -f "package.json" ]; then
    log "Detected Node.js project - applying JavaScript lint fixes"
    
    # Check for ESLint
    if command -v npx &> /dev/null && npx eslint --version &> /dev/null; then
        log "Running ESLint auto-fix..."
        if npx eslint . --fix --ext .js,.jsx,.ts,.tsx 2>/dev/null; then
            log "✅ ESLint fixes applied successfully"
            ((FIXES_APPLIED++))
        else
            warn "ESLint auto-fix completed with some issues remaining"
        fi
    fi
    
    # Check for Prettier
    if command -v npx &> /dev/null && npx prettier --version &> /dev/null; then
        log "Running Prettier auto-format..."
        if npx prettier --write "**/*.{js,jsx,ts,tsx,json,css,md}" 2>/dev/null; then
            log "✅ Prettier formatting applied successfully"
            ((FIXES_APPLIED++))
        else
            warn "Prettier formatting completed with some issues"
        fi
    fi
    
    # Check for specific npm scripts
    if npm run lint:fix --if-present &> /dev/null; then
        log "✅ npm run lint:fix executed successfully"
        ((FIXES_APPLIED++))
    fi
fi

# Python lint fixes
if [ -f "requirements.txt" ] || [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    log "Detected Python project - applying Python lint fixes"
    
    # Check for Black formatter
    if command -v black &> /dev/null; then
        log "Running Black formatter..."
        if black . 2>/dev/null; then
            log "✅ Black formatting applied successfully"
            ((FIXES_APPLIED++))
        else
            warn "Black formatting completed with some issues"
        fi
    fi
    
    # Check for isort
    if command -v isort &> /dev/null; then
        log "Running isort import sorting..."
        if isort . 2>/dev/null; then
            log "✅ Import sorting applied successfully"
            ((FIXES_APPLIED++))
        else
            warn "Import sorting completed with some issues"
        fi
    fi
    
    # Check for autopep8
    if command -v autopep8 &> /dev/null; then
        log "Running autopep8 fixes..."
        find . -name "*.py" -exec autopep8 --in-place --aggressive --aggressive {} \; 2>/dev/null
        log "✅ autopep8 fixes applied"
        ((FIXES_APPLIED++))
    fi
fi

# YAML/JSON lint fixes
log "Applying YAML and JSON formatting fixes..."

# Fix YAML files
if command -v yamllint &> /dev/null; then
    find . -name "*.yml" -o -name "*.yaml" | while read -r file; do
        if yamllint "$file" &> /dev/null; then
            log "✅ YAML file $file is valid"
        else
            warn "YAML file $file has issues that couldn't be auto-fixed"
        fi
    done
fi

# Fix JSON files
find . -name "*.json" | while read -r file; do
    if command -v jq &> /dev/null; then
        if jq . "$file" > /dev/null 2>&1; then
            # Pretty-print JSON file
            jq . "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
            log "✅ JSON file $file formatted"
        else
            warn "JSON file $file has syntax errors that couldn't be auto-fixed"
        fi
    fi
done

# Shell script lint fixes
if command -v shellcheck &> /dev/null; then
    log "Running shellcheck validation..."
    find . -name "*.sh" -exec shellcheck {} \; &> /dev/null || warn "Some shell scripts have issues"
fi

# Markdown lint fixes
if command -v markdownlint &> /dev/null; then
    log "Running markdown lint fixes..."
    if markdownlint --fix "**/*.md" 2>/dev/null; then
        log "✅ Markdown formatting applied"
        ((FIXES_APPLIED++))
    else
        warn "Markdown linting completed with some remaining issues"
    fi
fi

# Common file permission fixes
log "Fixing common file permission issues..."
find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
log "✅ Shell script permissions fixed"
((FIXES_APPLIED++))

# Remove trailing whitespace
log "Removing trailing whitespace..."
find . -type f \( -name "*.js" -o -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i 's/[[:space:]]*$//' {} \; 2>/dev/null || true
log "✅ Trailing whitespace removed"
((FIXES_APPLIED++))

# Fix line endings (convert CRLF to LF)
log "Normalizing line endings..."
find . -type f \( -name "*.js" -o -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" \) -exec dos2unix {} \; 2>/dev/null || true
log "✅ Line endings normalized"

# Summary
echo ""
log "🎯 Auto-fix Lint Summary:"
log "   Fixes Applied: $FIXES_APPLIED categories"
log "   Status: Auto-fix process completed"

if [ $FIXES_APPLIED -gt 0 ]; then
    log "✅ Lint auto-fixes have been applied successfully!"
    echo "LINT_FIXES_APPLIED=true" >> "$GITHUB_OUTPUT" 2>/dev/null || true
else
    warn "⚠️ No lint fixes were applied - either no issues found or tools not available"
    echo "LINT_FIXES_APPLIED=false" >> "$GITHUB_OUTPUT" 2>/dev/null || true
fi

exit 0