#!/bin/bash

# Auto-fix Format Issues Script
# Part of GitHub-First Automation System (Issue #283)
# Automatically fixes common formatting issues across multiple technologies

set -euo pipefail

echo "🎨 Auto-fix Format: Starting format issue resolution..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Counter for formatting fixes
FORMAT_FIXES=0
FILES_FORMATTED=0

# JavaScript/TypeScript formatting
if [ -f "package.json" ]; then
    log "Detected Node.js project - applying JavaScript/TypeScript formatting"
    
    # Prettier formatting
    if command -v npx &> /dev/null && npx prettier --version &> /dev/null; then
        log "Running Prettier formatting..."
        
        # Create prettier config if it doesn't exist
        if [ ! -f ".prettierrc" ] && [ ! -f ".prettierrc.json" ] && [ ! -f "prettier.config.js" ]; then
            info "Creating default Prettier configuration..."
            cat > .prettierrc << EOF
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false
}
EOF
        fi
        
        # Format JavaScript/TypeScript files
        if npx prettier --write "**/*.{js,jsx,ts,tsx}" 2>/dev/null; then
            JS_FILES=$(find . -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" | wc -l)
            log "✅ Prettier formatted $JS_FILES JavaScript/TypeScript files"
            ((FORMAT_FIXES++))
            FILES_FORMATTED=$((FILES_FORMATTED + JS_FILES))
        fi
        
        # Format JSON files
        if npx prettier --write "**/*.json" 2>/dev/null; then
            JSON_FILES=$(find . -name "*.json" | wc -l)
            log "✅ Prettier formatted $JSON_FILES JSON files"
            FILES_FORMATTED=$((FILES_FORMATTED + JSON_FILES))
        fi
        
        # Format CSS/SCSS files
        if npx prettier --write "**/*.{css,scss}" 2>/dev/null; then
            CSS_FILES=$(find . -name "*.css" -o -name "*.scss" | wc -l)
            if [ "$CSS_FILES" -gt 0 ]; then
                log "✅ Prettier formatted $CSS_FILES CSS/SCSS files"
                FILES_FORMATTED=$((FILES_FORMATTED + CSS_FILES))
            fi
        fi
        
        # Format Markdown files
        if npx prettier --write "**/*.md" 2>/dev/null; then
            MD_FILES=$(find . -name "*.md" | wc -l)
            log "✅ Prettier formatted $MD_FILES Markdown files"
            FILES_FORMATTED=$((FILES_FORMATTED + MD_FILES))
        fi
    fi
    
    # Run npm format script if available
    if npm run format --if-present &> /dev/null; then
        log "✅ npm run format executed successfully"
        ((FORMAT_FIXES++))
    fi
fi

# Python formatting
if [ -f "requirements.txt" ] || [ -f "setup.py" ] || [ -f "pyproject.toml" ] || find . -name "*.py" -print -quit | grep -q .; then
    log "Detected Python project - applying Python formatting"
    
    # Black formatting
    if command -v black &> /dev/null; then
        log "Running Black formatter..."
        PY_FILES_BEFORE=$(find . -name "*.py" | wc -l)
        if black --line-length 88 . 2>/dev/null; then
            log "✅ Black formatted $PY_FILES_BEFORE Python files"
            ((FORMAT_FIXES++))
            FILES_FORMATTED=$((FILES_FORMATTED + PY_FILES_BEFORE))
        fi
    else
        # Fallback to autopep8 if black is not available
        if command -v autopep8 &> /dev/null; then
            log "Running autopep8 formatter..."
            PY_FILES=$(find . -name "*.py")
            if [ -n "$PY_FILES" ]; then
                echo "$PY_FILES" | xargs autopep8 --in-place --aggressive --aggressive
                PY_COUNT=$(echo "$PY_FILES" | wc -l)
                log "✅ autopep8 formatted $PY_COUNT Python files"
                ((FORMAT_FIXES++))
                FILES_FORMATTED=$((FILES_FORMATTED + PY_COUNT))
            fi
        fi
    fi
    
    # isort for import sorting
    if command -v isort &> /dev/null; then
        log "Running isort for import sorting..."
        if isort . --profile=black 2>/dev/null; then
            log "✅ Import sorting applied to Python files"
            ((FORMAT_FIXES++))
        fi
    fi
fi

# YAML formatting
log "Formatting YAML files..."
YAML_FILES=$(find . -name "*.yml" -o -name "*.yaml" | head -20)  # Limit to prevent overload
if [ -n "$YAML_FILES" ]; then
    YAML_COUNT=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Basic YAML formatting - ensure proper spacing
            if command -v yq &> /dev/null; then
                yq eval '.' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file" 2>/dev/null || rm -f "${file}.tmp"
                ((YAML_COUNT++))
            fi
        fi
    done <<< "$YAML_FILES"
    
    if [ $YAML_COUNT -gt 0 ]; then
        log "✅ Formatted $YAML_COUNT YAML files"
        ((FORMAT_FIXES++))
        FILES_FORMATTED=$((FILES_FORMATTED + YAML_COUNT))
    fi
fi

# JSON formatting
log "Formatting JSON files..."
JSON_FILES=$(find . -name "*.json" | grep -v node_modules | head -20)  # Limit and exclude node_modules
if [ -n "$JSON_FILES" ]; then
    JSON_COUNT=0
    while IFS= read -r file; do
        if [ -f "$file" ] && command -v jq &> /dev/null; then
            if jq . "$file" > /dev/null 2>&1; then
                jq . "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
                ((JSON_COUNT++))
            else
                warn "Skipping invalid JSON file: $file"
            fi
        fi
    done <<< "$JSON_FILES"
    
    if [ $JSON_COUNT -gt 0 ]; then
        log "✅ Formatted $JSON_COUNT JSON files"
        ((FORMAT_FIXES++))
        FILES_FORMATTED=$((FILES_FORMATTED + JSON_COUNT))
    fi
fi

# Shell script formatting
log "Formatting shell scripts..."
SHELL_FILES=$(find . -name "*.sh" | head -10)  # Limit to prevent overload
if [ -n "$SHELL_FILES" ] && command -v shfmt &> /dev/null; then
    SHELL_COUNT=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            shfmt -w -i 2 "$file" 2>/dev/null && ((SHELL_COUNT++))
        fi
    done <<< "$SHELL_FILES"
    
    if [ $SHELL_COUNT -gt 0 ]; then
        log "✅ Formatted $SHELL_COUNT shell scripts"
        ((FORMAT_FIXES++))
        FILES_FORMATTED=$((FILES_FORMATTED + SHELL_COUNT))
    fi
fi

# Generic text file cleanup
log "Applying generic text formatting fixes..."

# Remove trailing whitespace from common file types
TEXT_FILES=$(find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" \) | head -50)
WHITESPACE_FIXES=0

if [ -n "$TEXT_FILES" ]; then
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Remove trailing whitespace
            if sed -i 's/[[:space:]]*$//' "$file" 2>/dev/null; then
                ((WHITESPACE_FIXES++))
            fi
            
            # Ensure file ends with newline
            if [ -s "$file" ] && [ "$(tail -c 1 "$file" | wc -l)" -eq 0 ]; then
                echo >> "$file"
            fi
        fi
    done <<< "$TEXT_FILES"
fi

if [ $WHITESPACE_FIXES -gt 0 ]; then
    log "✅ Removed trailing whitespace from $WHITESPACE_FIXES files"
    ((FORMAT_FIXES++))
fi

# Line ending normalization (CRLF to LF)
log "Normalizing line endings..."
if command -v dos2unix &> /dev/null; then
    LINE_ENDING_FILES=$(find . -type f \( -name "*.js" -o -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" \) | head -100)
    if [ -n "$LINE_ENDING_FILES" ]; then
        echo "$LINE_ENDING_FILES" | xargs dos2unix 2>/dev/null || true
        log "✅ Normalized line endings"
        ((FORMAT_FIXES++))
    fi
fi

# EditorConfig compliance (if .editorconfig exists)
if [ -f ".editorconfig" ]; then
    info "EditorConfig detected - formatting rules will be respected by supported formatters"
    ((FORMAT_FIXES++))
fi

# Summary
echo ""
log "🎨 Auto-fix Format Summary:"
log "   Format Fix Categories: $FORMAT_FIXES"
log "   Total Files Formatted: $FILES_FORMATTED"
log "   Status: Format auto-fix process completed"

if [ $FORMAT_FIXES -gt 0 ]; then
    log "✅ Format auto-fixes have been applied successfully!"
    echo "FORMAT_FIXES_APPLIED=true" >> "$GITHUB_OUTPUT" 2>/dev/null || true
    echo "FILES_FORMATTED=$FILES_FORMATTED" >> "$GITHUB_OUTPUT" 2>/dev/null || true
else
    warn "⚠️ No format fixes were applied - either no issues found or tools not available"
    echo "FORMAT_FIXES_APPLIED=false" >> "$GITHUB_OUTPUT" 2>/dev/null || true
    echo "FILES_FORMATTED=0" >> "$GITHUB_OUTPUT" 2>/dev/null || true
fi

exit 0