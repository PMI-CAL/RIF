#!/bin/bash

# RIF - Reactive Intelligence Framework
# Initialization and Setup Script

set -e

echo "=========================================="
echo "   RIF - Reactive Intelligence Framework"
echo "   Automatic Intelligent Development System"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check for git
if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ Git is not installed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Git found${NC}"
fi

# Check for GitHub CLI
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠ GitHub CLI not found - installing...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    else
        echo "Please install GitHub CLI: https://cli.github.com/"
        exit 1
    fi
else
    echo -e "${GREEN}✓ GitHub CLI found${NC}"
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠ jq not found - installing...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        sudo apt-get install -y jq || echo "Please install jq manually"
    fi
else
    echo -e "${GREEN}✓ jq found${NC}"
fi

# Check for Claude Code
if [ -d "$HOME/.claude" ] || command -v claude &> /dev/null; then
    echo -e "${GREEN}✓ Claude Code detected${NC}"
else
    echo -e "${YELLOW}⚠ Claude Code not detected - RIF works best with Claude Code${NC}"
fi

echo ""
echo -e "${BLUE}Setting up RIF in current directory...${NC}"

# Create directory structure
echo "Creating RIF directories..."
mkdir -p knowledge/{patterns,decisions,issues,metrics,learning,checkpoints}
mkdir -p .claude
mkdir -p .github/ISSUE_TEMPLATE

# Initialize knowledge base
echo "Initializing knowledge base..."
cat > knowledge/patterns/recent.json << 'EOF'
{
  "patterns": [],
  "version": "1.0.0",
  "updated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

cat > knowledge/issues/resolved.json << 'EOF'
{
  "issues": [],
  "version": "1.0.0",
  "updated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

cat > knowledge/metrics/quality.json << 'EOF'
{
  "gates": {
    "coverage": {"threshold": 80, "current": 0},
    "security": {"threshold": "no_critical", "current": "unknown"},
    "performance": {"threshold": "baseline", "current": "unknown"}
  },
  "version": "1.0.0",
  "updated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

# Create GitHub issue templates
echo "Creating GitHub issue templates..."
cat > .github/ISSUE_TEMPLATE/rif-task.md << 'EOF'
---
name: RIF Task
about: Create a task for RIF to automatically handle
title: ''
labels: 'state:new'
assignees: ''
---

## Description
[Describe what needs to be done]

## Context
[Any relevant context or background]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Notes
[Any additional notes or constraints]
EOF

cat > .github/ISSUE_TEMPLATE/rif-bug.md << 'EOF'
---
name: RIF Bug Fix
about: Report a bug for RIF to automatically fix
title: '[BUG] '
labels: 'state:new, type:bug'
assignees: ''
---

## Bug Description
[Clear description of the bug]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [e.g., macOS, Linux]
- Version: [e.g., 1.0.0]
- Dependencies: [relevant versions]
EOF

# Check if hooks are already configured
if [ -f ".claude/settings.json" ]; then
    echo -e "${YELLOW}⚠ .claude/settings.json already exists - backing up...${NC}"
    cp .claude/settings.json .claude/settings.json.backup.$(date +%s)
fi

# Copy Claude settings if not exists
if [ ! -f ".claude/settings.json" ]; then
    echo "Configuring Claude Code hooks..."
    cp "${BASH_SOURCE%/*}/.claude/settings.json" .claude/settings.json
    echo -e "${GREEN}✓ Claude Code hooks configured${NC}"
else
    echo -e "${YELLOW}⚠ Claude settings exist - please merge manually${NC}"
fi

# Setup git hooks (optional)
echo ""
read -p "Do you want to set up git hooks for automatic RIF triggers? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# RIF post-commit hook
echo '{
  "event": "commit",
  "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
  "hash": "'$(git rev-parse HEAD)'",
  "message": "'$(git log -1 --pretty=%B | head -1)'"
}' >> knowledge/events.jsonl
EOF
    chmod +x .git/hooks/post-commit
    echo -e "${GREEN}✓ Git hooks configured${NC}"
fi

# Check GitHub authentication
echo ""
echo -e "${BLUE}Checking GitHub authentication...${NC}"
if gh auth status &> /dev/null; then
    echo -e "${GREEN}✓ GitHub authenticated${NC}"
    REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "")
    if [ -n "$REPO" ]; then
        echo -e "${GREEN}✓ Repository detected: $REPO${NC}"
    else
        echo -e "${YELLOW}⚠ Not in a GitHub repository${NC}"
        read -p "Create a GitHub repository? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "Repository name: " REPO_NAME
            gh repo create "$REPO_NAME" --private --clone=false
            git remote add origin "https://github.com/$(gh api user -q .login)/$REPO_NAME.git"
            echo -e "${GREEN}✓ Repository created${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ GitHub not authenticated${NC}"
    echo "Run: gh auth login"
fi

# Create initial RIF documentation
echo ""
echo "Creating RIF documentation..."
cat > RIF-README.md << 'EOF'
# RIF-Enabled Project

This project uses RIF (Reactive Intelligence Framework) for automatic intelligent development.

## How It Works

1. **Create an Issue**: Describe what you need in a GitHub issue
2. **RIF Activates**: Agents automatically analyze and implement
3. **Review Results**: Check issue comments for progress and results

## Quick Start

### Create Your First RIF Task

```bash
gh issue create --title "Your task description" --body "Details about what you need"
```

RIF will automatically:
- Analyze the requirements
- Plan the implementation
- Write the code
- Test the solution
- Update documentation
- Learn from the experience

## RIF Agents

- **RIF-Analyst**: Analyzes requirements and patterns
- **RIF-Planner**: Creates execution plans
- **RIF-Architect**: Designs solutions
- **RIF-Implementer**: Writes code
- **RIF-Validator**: Tests and validates
- **RIF-Learner**: Updates knowledge base

## Knowledge Base

RIF learns from every task:
- `knowledge/patterns/` - Successful patterns
- `knowledge/issues/` - Resolved issues
- `knowledge/decisions/` - Design decisions
- `knowledge/metrics/` - Performance data

## Monitoring

Check RIF status:
```bash
gh issue list --label "state:*"
```

View recent events:
```bash
tail -f knowledge/events.jsonl | jq .
```

## Troubleshooting

If RIF isn't responding:
1. Check issue has `state:new` label
2. Verify Claude Code hooks are configured
3. Ensure GitHub CLI is authenticated
4. Check knowledge directory exists

## Learn More

See the main [CLAUDE.md](CLAUDE.md) file for complete documentation.
EOF

echo -e "${GREEN}✓ Documentation created${NC}"

# Final setup summary
echo ""
echo "=========================================="
echo -e "${GREEN}RIF Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review and commit RIF files:"
echo "   git add ."
echo "   git commit -m 'Add RIF - Reactive Intelligence Framework'"
echo ""
echo "2. Create your first RIF issue:"
echo "   gh issue create --title 'Test RIF' --body 'Test that RIF is working'"
echo ""
echo "3. Watch RIF work automatically:"
echo "   gh issue view <number> --comments"
echo ""
echo "RIF is now ready to intelligently handle your development tasks!"
echo ""
echo "Remember: RIF works AUTOMATICALLY - just create issues and review results!"