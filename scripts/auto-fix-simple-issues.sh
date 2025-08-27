#!/bin/bash

# Auto-fix Simple Issues Script
# Part of GitHub-First Automation System (Issue #283)
# Automatically fixes simple, non-risky issues that are commonly found in code reviews

set -euo pipefail

echo "🔨 Auto-fix Simple Issues: Starting simple issue resolution..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

debug() {
    echo -e "${PURPLE}[$(date '+%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}"
}

# Counter for fixes
SIMPLE_FIXES=0

# Fix common typos in comments and documentation
log "Fixing common typos in documentation and comments..."
TYPO_FIXES=0

# Define common typos and their corrections
declare -A TYPOS=(
    ["teh"]="the"
    ["recieve"]="receive"
    ["seperate"]="separate"
    ["occured"]="occurred"
    ["neccessary"]="necessary"
    ["existant"]="existent"
    ["beleive"]="believe"
    ["acheive"]="achieve"
    ["occassion"]="occasion"
    ["accomodate"]="accommodate"
    ["definately"]="definitely"
    ["enviroment"]="environment"
    ["performace"]="performance"
    ["fucntion"]="function"
    ["paramter"]="parameter"
    ["retrun"]="return"
    ["lenght"]="length"
    ["widht"]="width"
    ["hieght"]="height"
    ["strign"]="string"
)

# Fix typos in markdown, text, and comment files
for ext in "*.md" "*.txt" "*.py" "*.js" "*.ts" "*.yml" "*.yaml"; do
    while IFS= read -r -d '' file; do
        for typo in "${!TYPOS[@]}"; do
            correct="${TYPOS[$typo]}"
            if grep -q "\\b$typo\\b" "$file" 2>/dev/null; then
                sed -i "s/\\b$typo\\b/$correct/g" "$file" 2>/dev/null || continue
                debug "Fixed typo '$typo' -> '$correct' in $file"
                ((TYPO_FIXES++))
            fi
        done
    done < <(find . -name "$ext" -type f -print0 2>/dev/null | head -z -50)
done

if [ $TYPO_FIXES -gt 0 ]; then
    log "✅ Fixed $TYPO_FIXES typos in documentation and comments"
    ((SIMPLE_FIXES++))
fi

# Fix missing file extensions in imports (Python)
log "Fixing Python import issues..."
PYTHON_FIXES=0

find . -name "*.py" -type f | head -20 | while read -r pyfile; do
    # Remove unused imports (simple cases)
    if command -v autoflake &> /dev/null; then
        if autoflake --remove-all-unused-imports --in-place "$pyfile" 2>/dev/null; then
            debug "Removed unused imports from $pyfile"
            ((PYTHON_FIXES++))
        fi
    fi
done

if [ $PYTHON_FIXES -gt 0 ]; then
    log "✅ Fixed Python import issues"
    ((SIMPLE_FIXES++))
fi

# Fix package.json issues
if [ -f "package.json" ]; then
    log "Fixing package.json issues..."
    PKG_FIXES=0
    
    # Sort package.json scripts alphabetically
    if command -v jq &> /dev/null; then
        if jq '.scripts = (.scripts | to_entries | sort_by(.key) | from_entries)' package.json > package.json.tmp; then
            if ! cmp -s package.json package.json.tmp; then
                mv package.json.tmp package.json
                log "✅ Sorted package.json scripts alphabetically"
                ((PKG_FIXES++))
            else
                rm package.json.tmp
            fi
        fi
        
        # Sort dependencies alphabetically
        if jq '.dependencies = (.dependencies | to_entries | sort_by(.key) | from_entries) | .devDependencies = (.devDependencies | to_entries | sort_by(.key) | from_entries)' package.json > package.json.tmp; then
            if ! cmp -s package.json package.json.tmp; then
                mv package.json.tmp package.json
                log "✅ Sorted package.json dependencies alphabetically"
                ((PKG_FIXES++))
            else
                rm package.json.tmp
            fi
        fi
    fi
    
    if [ $PKG_FIXES -gt 0 ]; then
        ((SIMPLE_FIXES++))
    fi
fi

# Fix common README issues
if [ -f "README.md" ]; then
    log "Fixing README.md issues..."
    README_FIXES=0
    
    # Ensure README has proper title structure
    if ! grep -q "^# " README.md; then
        # Add project title if missing
        PROJECT_NAME=$(basename "$(pwd)")
        sed -i "1i # $PROJECT_NAME" README.md 2>/dev/null
        log "✅ Added missing title to README.md"
        ((README_FIXES++))
    fi
    
    # Fix common markdown formatting issues
    # Fix unordered list spacing
    sed -i 's/^\*\([^ ]\)/\* \1/g' README.md 2>/dev/null
    sed -i 's/^-\([^ ]\)/- \1/g' README.md 2>/dev/null
    
    # Fix heading spacing
    sed -i 's/^#\([^ #]\)/# \1/g' README.md 2>/dev/null
    sed -i 's/^##\([^ #]\)/## \1/g' README.md 2>/dev/null
    sed -i 's/^###\([^ #]\)/### \1/g' README.md 2>/dev/null
    
    if [ $README_FIXES -gt 0 ]; then
        log "✅ Fixed README.md formatting issues"
        ((SIMPLE_FIXES++))
    fi
fi

# Fix .gitignore issues
if [ -f ".gitignore" ]; then
    log "Optimizing .gitignore..."
    GITIGNORE_FIXES=0
    
    # Remove duplicate entries
    if sort .gitignore | uniq > .gitignore.tmp; then
        if ! cmp -s .gitignore .gitignore.tmp; then
            mv .gitignore.tmp .gitignore
            log "✅ Removed duplicate .gitignore entries"
            ((GITIGNORE_FIXES++))
        else
            rm .gitignore.tmp
        fi
    fi
    
    # Add common missing entries for detected project types
    ADDITIONS=""
    
    if [ -f "package.json" ] && ! grep -q "node_modules" .gitignore; then
        ADDITIONS="$ADDITIONS\nnode_modules/"
    fi
    
    if [ -f "requirements.txt" ] || [ -f "setup.py" ]; then
        if ! grep -q "__pycache__" .gitignore; then
            ADDITIONS="$ADDITIONS\n__pycache__/"
        fi
        if ! grep -q "*.pyc" .gitignore; then
            ADDITIONS="$ADDITIONS\n*.pyc"
        fi
    fi
    
    if [ ! -z "$ADDITIONS" ]; then
        echo -e "$ADDITIONS" >> .gitignore
        log "✅ Added missing common .gitignore entries"
        ((GITIGNORE_FIXES++))
    fi
    
    if [ $GITIGNORE_FIXES -gt 0 ]; then
        ((SIMPLE_FIXES++))
    fi
fi

# Fix file permissions
log "Fixing file permissions..."
PERM_FIXES=0

# Make shell scripts executable
find . -name "*.sh" -type f ! -perm -111 | while read -r script; do
    chmod +x "$script"
    debug "Made $script executable"
    ((PERM_FIXES++))
done

# Remove execute permission from non-executable files
find . -type f \( -name "*.md" -o -name "*.txt" -o -name "*.json" -o -name "*.yml" -o -name "*.yaml" \) -perm -111 | while read -r file; do
    chmod -x "$file"
    debug "Removed execute permission from $file"
    ((PERM_FIXES++))
done

if [ $PERM_FIXES -gt 0 ]; then
    log "✅ Fixed file permissions"
    ((SIMPLE_FIXES++))
fi

# Fix common YAML issues
log "Fixing YAML formatting issues..."
YAML_FIXES=0

find . -name "*.yml" -o -name "*.yaml" | head -10 | while read -r yamlfile; do
    # Fix common indentation issues (basic)
    if sed -i 's/	/  /g' "$yamlfile" 2>/dev/null; then  # Replace tabs with spaces
        debug "Fixed tab indentation in $yamlfile"
        ((YAML_FIXES++))
    fi
    
    # Fix trailing spaces in YAML
    if sed -i 's/[[:space:]]*$//' "$yamlfile" 2>/dev/null; then
        debug "Fixed trailing spaces in $yamlfile"
    fi
done

if [ $YAML_FIXES -gt 0 ]; then
    log "✅ Fixed YAML formatting issues"
    ((SIMPLE_FIXES++))
fi

# Security dependency updates (safe updates only)
if [ -f "package.json" ]; then
    log "Applying safe security updates..."
    SECURITY_FIXES=0
    
    # Run npm audit fix for non-breaking changes only
    if command -v npm &> /dev/null; then
        if npm audit fix --only=prod 2>/dev/null; then
            log "✅ Applied safe security updates"
            ((SECURITY_FIXES++))
        fi
    fi
    
    if [ $SECURITY_FIXES -gt 0 ]; then
        ((SIMPLE_FIXES++))
    fi
fi

# Fix common code style issues (non-risky)
log "Fixing simple code style issues..."
STYLE_FIXES=0

# Remove double blank lines
find . -name "*.js" -o -name "*.py" -o -name "*.ts" -o -name "*.md" | head -20 | while read -r file; do
    if sed -i '/^$/N;/^\n$/d' "$file" 2>/dev/null; then
        debug "Removed double blank lines from $file"
        ((STYLE_FIXES++))
    fi
done

# Ensure files end with newline
find . -type f \( -name "*.js" -o -name "*.py" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" \) | head -30 | while read -r file; do
    if [ -s "$file" ] && [ "$(tail -c 1 "$file" | wc -l)" -eq 0 ]; then
        echo >> "$file"
        debug "Added newline at end of $file"
        ((STYLE_FIXES++))
    fi
done

if [ $STYLE_FIXES -gt 0 ]; then
    log "✅ Fixed simple code style issues"
    ((SIMPLE_FIXES++))
fi

# Create or update basic project files if missing
log "Ensuring basic project files exist..."
PROJECT_FILE_FIXES=0

# Create basic .editorconfig if missing
if [ ! -f ".editorconfig" ]; then
    cat > .editorconfig << 'EOF'
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2

[*.py]
indent_size = 4

[*.md]
trim_trailing_whitespace = false
EOF
    log "✅ Created basic .editorconfig"
    ((PROJECT_FILE_FIXES++))
fi

# Create basic .gitattributes if missing and needed
if [ ! -f ".gitattributes" ] && ([ -f "package.json" ] || find . -name "*.py" -print -quit | grep -q .); then
    cat > .gitattributes << 'EOF'
# Ensure consistent line endings
* text=auto

# Ensure binary files are not converted
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.mov binary
*.mp4 binary
*.mp3 binary
*.flv binary
*.fla binary
*.swf binary
*.gz binary
*.zip binary
*.7z binary
*.ttf binary
*.eot binary
*.woff binary
*.pyc binary
*.pdf binary
*.ez binary
*.bz2 binary
*.swp binary
*.image binary
*.hprof binary
EOF
    log "✅ Created basic .gitattributes"
    ((PROJECT_FILE_FIXES++))
fi

if [ $PROJECT_FILE_FIXES -gt 0 ]; then
    ((SIMPLE_FIXES++))
fi

# Summary
echo ""
log "🔨 Auto-fix Simple Issues Summary:"
log "   Simple Fix Categories Applied: $SIMPLE_FIXES"
log "   Status: Simple issues auto-fix process completed"

if [ $SIMPLE_FIXES -gt 0 ]; then
    log "✅ Simple issue auto-fixes have been applied successfully!"
    log "   🎯 Fixed: Typos, formatting, permissions, project structure"
    echo "SIMPLE_FIXES_APPLIED=true" >> "$GITHUB_OUTPUT" 2>/dev/null || true
    echo "SIMPLE_FIXES_COUNT=$SIMPLE_FIXES" >> "$GITHUB_OUTPUT" 2>/dev/null || true
else
    info "ℹ️ No simple fixes were needed - project appears to be well-maintained!"
    echo "SIMPLE_FIXES_APPLIED=false" >> "$GITHUB_OUTPUT" 2>/dev/null || true
    echo "SIMPLE_FIXES_COUNT=0" >> "$GITHUB_OUTPUT" 2>/dev/null || true
fi

exit 0