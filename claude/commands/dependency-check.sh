#!/bin/bash
# RIF Dependency Management System
# Analyzes issue dependencies and identifies blocking relationships

set -e

# Function to extract issue numbers from text
extract_deps() {
    echo "$1" | grep -oE "(depends on|blocked by|requires|waiting for|needs) #[0-9]+" -i | grep -oE "#[0-9]+" | grep -oE "[0-9]+"
}

# Function to check if issue is open
is_open() {
    gh issue view "$1" --json state --jq '.state' 2>/dev/null | grep -q "OPEN"
}

echo "# RIF Dependency Analysis"
echo "========================"
echo ""

# Get all open issues
ISSUES=$(gh issue list --state open --json number,title,body,labels --limit 100)

# Build dependency graph
echo "## Dependency Graph"
echo ""

declare -A deps
declare -A blocked_by

# Parse dependencies from issue bodies
echo "$ISSUES" | jq -r '.[] | "\(.number)|\(.body)"' | while IFS='|' read -r issue_num body; do
    dep_nums=$(extract_deps "$body")
    if [ -n "$dep_nums" ]; then
        for dep in $dep_nums; do
            if is_open "$dep"; then
                echo "Issue #$issue_num → blocked by → Issue #$dep"
            fi
        done
    fi
done

echo ""
echo "## Execution Order"
echo ""

# Find issues with no dependencies (can run immediately)
echo "### Ready to Execute (No Dependencies):"
echo "$ISSUES" | jq -r '.[] | "\(.number)|\(.body)"' | while IFS='|' read -r issue_num body; do
    dep_nums=$(extract_deps "$body")
    if [ -z "$dep_nums" ]; then
        # Check current state
        state=$(echo "$ISSUES" | jq -r ".[] | select(.number == $issue_num) | .labels[] | select(.name | startswith(\"state:\")) | .name" | head -1)
        if [ -n "$state" ] && [ "$state" != "state:complete" ] && [ "$state" != "state:error" ]; then
            echo "- Issue #$issue_num (${state})"
        fi
    fi
done

echo ""
echo "### Blocked Issues:"
echo "$ISSUES" | jq -r '.[] | "\(.number)|\(.body)"' | while IFS='|' read -r issue_num body; do
    dep_nums=$(extract_deps "$body")
    if [ -n "$dep_nums" ]; then
        open_deps=""
        for dep in $dep_nums; do
            if is_open "$dep"; then
                open_deps="$open_deps #$dep"
            fi
        done
        if [ -n "$open_deps" ]; then
            echo "- Issue #$issue_num blocked by:$open_deps"
        fi
    fi
done

echo ""
echo "## Parallel Execution Groups"
echo ""

# Identify groups that can run in parallel (no shared dependencies or conflicts)
echo "### Group 1 (Can run in parallel):"
ready_issues=$(echo "$ISSUES" | jq -r '.[] | "\(.number)|\(.body)"' | while IFS='|' read -r issue_num body; do
    dep_nums=$(extract_deps "$body")
    if [ -z "$dep_nums" ]; then
        state=$(echo "$ISSUES" | jq -r ".[] | select(.number == $issue_num) | .labels[] | select(.name | startswith(\"state:\")) | .name" | head -1)
        if [ -n "$state" ] && [ "$state" != "state:complete" ] && [ "$state" != "state:error" ]; then
            echo "$issue_num"
        fi
    fi
done | head -4)

for issue in $ready_issues; do
    title=$(gh issue view "$issue" --json title --jq '.title')
    echo "- Issue #$issue: $title"
done

echo ""
echo "## Recommendations"
echo ""
echo "1. Process issues in Group 1 first (up to 4 in parallel)"
echo "2. As blocked issues become unblocked, add them to next batch"
echo "3. Re-evaluate dependencies after each batch completion"
echo "4. Use RIF-Orchestrator agent for detailed conflict analysis"

# Save to knowledge base
mkdir -p knowledge/dependencies
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
{
    echo "{"
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"ready_issues\": ["
    first=true
    for issue in $ready_issues; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        echo -n "    $issue"
    done
    echo ""
    echo "  ],"
    echo "  \"analysis_complete\": true"
    echo "}"
} > knowledge/dependencies/analysis-$(date +%s).json

echo ""
echo "Analysis saved to knowledge/dependencies/"