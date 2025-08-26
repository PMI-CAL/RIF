#!/bin/bash

# Validate Agent GitHub Instructions Script
# Detects if agents are creating MD files instead of posting to GitHub issues
# Created to prevent Issue #267 - agents writing analysis to MD files

set -e

echo "🔍 Agent GitHub Instruction Validation"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Get the repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
echo -e "${BLUE}📂 Repository: $REPO_ROOT${NC}"

# Track validation results
VALIDATION_ERRORS=0
VALIDATION_WARNINGS=0

echo ""
echo "🚫 Checking for prohibited agent MD file creation..."

# Look for recently created MD files that match agent output patterns
AGENT_MD_FILES=$(find "$REPO_ROOT" -maxdepth 1 -name "*.md" -type f \
    -exec grep -l "Implementation Complete\|Analysis Complete\|Validation Report\|Test Architect" {} \; 2>/dev/null || true)

if [ -n "$AGENT_MD_FILES" ]; then
    echo -e "${RED}❌ CRITICAL: Found agent output MD files (should be GitHub comments instead):${NC}"
    for file in $AGENT_MD_FILES; do
        filename=$(basename "$file")
        echo -e "   ${RED}• $filename${NC}"
        
        # Check if it looks like an issue analysis
        if echo "$filename" | grep -q "ISSUE_.*_.*\.md"; then
            echo -e "     ${RED}🚨 This appears to be issue analysis that should be posted to GitHub!${NC}"
            ((VALIDATION_ERRORS++))
        fi
    done
    echo ""
else
    echo -e "${GREEN}✅ No problematic agent MD files found${NC}"
fi

echo ""
echo "📝 Checking agent instruction consistency..."

# Check that all agent MD files have GitHub-first instructions
AGENT_DIR="$REPO_ROOT/claude/agents"
if [ -d "$AGENT_DIR" ]; then
    AGENT_FILES=$(find "$AGENT_DIR" -name "rif-*.md" -type f)
    
    for agent_file in $AGENT_FILES; do
        agent_name=$(basename "$agent_file" .md)
        echo -e "${BLUE}🤖 Checking $agent_name...${NC}"
        
        # Check for GitHub-first rule
        if grep -q "GitHub-First Output\|MUST be posted to GitHub issues" "$agent_file"; then
            echo -e "   ${GREEN}✅ Has GitHub-first rule${NC}"
        else
            echo -e "   ${RED}❌ Missing GitHub-first rule${NC}"
            ((VALIDATION_ERRORS++))
        fi
        
        # Check for prohibited file creation rule
        if grep -q "PROHIBITED.*markdown files\|PROHIBITED.*\.md" "$agent_file"; then
            echo -e "   ${GREEN}✅ Has file creation prohibition${NC}"
        else
            echo -e "   ${RED}❌ Missing file creation prohibition${NC}"
            ((VALIDATION_ERRORS++))
        fi
        
        # Check for gh issue comment instructions
        if grep -q "gh issue comment" "$agent_file"; then
            echo -e "   ${GREEN}✅ Has GitHub comment instructions${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Missing explicit GitHub comment instructions${NC}"
            ((VALIDATION_WARNINGS++))
        fi
        
        # Check for conflicting output templates that look like files
        if grep -q "### Output" "$agent_file" && grep -A 20 "### Output" "$agent_file" | grep -q "\`\`\`markdown"; then
            echo -e "   ${YELLOW}⚠️  Has markdown output template (ensure clarity about GitHub posting)${NC}"
            ((VALIDATION_WARNINGS++))
        fi
    done
else
    echo -e "${RED}❌ Agent directory not found: $AGENT_DIR${NC}"
    ((VALIDATION_ERRORS++))
fi

echo ""
echo "🔧 Checking GitHub CLI access..."

# Test GitHub CLI authentication
if command -v gh &> /dev/null; then
    if gh auth status &> /dev/null; then
        echo -e "${GREEN}✅ GitHub CLI authenticated${NC}"
    else
        echo -e "${RED}❌ GitHub CLI not authenticated${NC}"
        ((VALIDATION_ERRORS++))
    fi
else
    echo -e "${RED}❌ GitHub CLI not installed${NC}"
    ((VALIDATION_ERRORS++))
fi

echo ""
echo "📊 Validation Summary"
echo "===================="

if [ $VALIDATION_ERRORS -eq 0 ] && [ $VALIDATION_WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All validations passed! Agents should post to GitHub correctly.${NC}"
    exit 0
elif [ $VALIDATION_ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Validation completed with $VALIDATION_WARNINGS warnings${NC}"
    echo -e "${YELLOW}   Consider addressing warnings to improve clarity${NC}"
    exit 0
else
    echo -e "${RED}❌ Validation failed with $VALIDATION_ERRORS errors and $VALIDATION_WARNINGS warnings${NC}"
    echo -e "${RED}   Critical issues must be fixed to prevent agents writing to MD files${NC}"
    
    if [ $VALIDATION_ERRORS -gt 0 ]; then
        echo ""
        echo -e "${RED}🚨 CRITICAL FIXES NEEDED:${NC}"
        echo -e "${RED}   1. Remove any agent-generated MD files from repository root${NC}"
        echo -e "${RED}   2. Ensure all agents have GitHub-first output rules${NC}"
        echo -e "${RED}   3. Add clear prohibition against creating .md files${NC}"
        echo -e "${RED}   4. Verify GitHub CLI access for agent operations${NC}"
        echo ""
        echo -e "${RED}   This script was created to prevent Issue #267 recurrence.${NC}"
    fi
    
    exit 1
fi