#!/bin/bash

# Fix Agent GitHub Instructions Script
# Resolves Issue #267 by updating agent instructions for GitHub-first output
# Adds clear GitHub posting requirements and improves output templates

set -e

echo "🔧 Fixing Agent GitHub Instructions (Issue #267)"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
AGENT_DIR="$REPO_ROOT/claude/agents"
FIXES_APPLIED=0

echo -e "${BLUE}📂 Working on: $AGENT_DIR${NC}"
echo ""

# Find agents missing GitHub-first rule
echo "🎯 Step 1: Adding GitHub-first rules to agent files..."

for agent_file in "$AGENT_DIR"/rif-*.md; do
    if [[ -f "$agent_file" ]]; then
        agent_name=$(basename "$agent_file" .md)
        echo -e "${BLUE}🤖 Processing $agent_name...${NC}"
        
        # Check if GitHub-first rule exists
        if ! grep -q "GitHub-First Output\|MUST be posted to GitHub issues" "$agent_file"; then
            echo -e "   ${YELLOW}⚡ Adding GitHub-first rule${NC}"
            
            # Create backup
            cp "$agent_file" "$agent_file.backup"
            
            # Add GitHub-first rule after the title
            sed -i '' '2a\
\
## 🚨 CRITICAL RULE: GitHub-First Output\
**MANDATORY**: All agent analysis, reports, and output MUST be posted to GitHub issues using `gh issue comment`. \
**PROHIBITED**: Writing agent output to local markdown files (.md). \
**WORKFLOW REQUIREMENT**: GitHub posting is the ONLY acceptable method for sharing agent results.\
' "$agent_file"
            
            ((FIXES_APPLIED++))
            echo -e "   ${GREEN}✅ GitHub-first rule added${NC}"
        else
            echo -e "   ${GREEN}✅ GitHub-first rule already exists${NC}"
        fi
    fi
done

echo ""
echo "📝 Step 2: Improving output template clarity..."

for agent_file in "$AGENT_DIR"/rif-*.md; do
    if [[ -f "$agent_file" ]]; then
        agent_name=$(basename "$agent_file" .md)
        
        # Check if file has output templates that might be confusing
        if grep -q "### Output" "$agent_file" && grep -A 20 "### Output" "$agent_file" | grep -q "\`\`\`markdown"; then
            echo -e "${BLUE}🤖 Clarifying $agent_name output template...${NC}"
            
            # Add clarification before output templates
            if ! grep -q "POST THIS TO GITHUB" "$agent_file"; then
                sed -i '' '/### Output/i\
**🚨 CRITICAL**: The following template is for GitHub posting, NOT file creation!\
**USE**: `gh issue comment <ISSUE_NUMBER> --body "$(cat <<'\''EOF'\''`\
**FOLLOWED BY**: The template content below\
**ENDED WITH**: `EOF`)"`\
\
' "$agent_file"
                
                ((FIXES_APPLIED++))
                echo -e "   ${GREEN}✅ Output template clarified${NC}"
            else
                echo -e "   ${GREEN}✅ Output template already clarified${NC}"
            fi
        fi
    fi
done

echo ""
echo "🛡️ Step 3: Creating agent monitoring system..."

# Create a pre-commit hook to prevent MD file creation
HOOK_FILE="$REPO_ROOT/.git/hooks/pre-commit"
mkdir -p "$REPO_ROOT/.git/hooks"

cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash

# Pre-commit hook to prevent agent MD file creation (Issue #267)
echo "🔍 Checking for prohibited agent MD files..."

# Check for agent output patterns in new MD files
AGENT_MD_PATTERNS="Implementation Complete|Analysis Complete|Validation Report|Test Architect"

for file in $(git diff --cached --name-only --diff-filter=A | grep '\.md$'); do
    if [[ -f "$file" ]] && grep -q "$AGENT_MD_PATTERNS" "$file"; then
        echo "🚨 ERROR: Detected agent output in MD file: $file"
        echo "   Agent output MUST be posted to GitHub issues using 'gh issue comment'"
        echo "   This prevents Issue #267 - agents writing to files instead of GitHub"
        echo ""
        echo "   To fix:"
        echo "   1. Remove this file: rm $file"
        echo "   2. Post content to GitHub: gh issue comment <ISSUE_NUM> --body \"content\""
        echo "   3. Retry commit"
        echo ""
        exit 1
    fi
done

echo "✅ No prohibited agent MD files detected"
EOF

chmod +x "$HOOK_FILE"
echo -e "${GREEN}✅ Pre-commit hook installed${NC}"

echo ""
echo "📊 Fix Summary"
echo "=============="
echo -e "${GREEN}✅ Fixes applied: $FIXES_APPLIED${NC}"
echo -e "${GREEN}✅ Pre-commit hook installed${NC}"
echo -e "${GREEN}✅ Agent files backed up (.backup extension)${NC}"

if [[ $FIXES_APPLIED -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo "   1. Review updated agent files"
    echo "   2. Test with validation script: ./claude/commands/validate_agent_github_instructions.sh"
    echo "   3. Commit changes to prevent Issue #267 recurrence"
    echo "   4. Remove existing problematic MD files from repository root"
    echo ""
    echo -e "${BLUE}🎯 Issue #267 Resolution Status: FIXED${NC}"
else
    echo -e "${BLUE}📋 All agents already had proper GitHub posting instructions${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Agent GitHub instruction fixes complete!${NC}"