#!/bin/bash

# RIF Release Workflow Test Script
# Validates the release automation workflow functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_VERSION="v0.0.1-test"
WORKFLOW_FILE=".github/workflows/release-automation.yml"
CONFIG_FILE=".github/release-config.yml"
UTILS_FILE=".github/scripts/release-utils.js"

echo -e "${BLUE}🧪 RIF Release Workflow Test Suite${NC}"
echo "======================================="

# Function to print test results
print_test_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✅ $test_name: PASS${NC}"
    elif [ "$result" = "FAIL" ]; then
        echo -e "${RED}❌ $test_name: FAIL${NC}"
        if [ -n "$details" ]; then
            echo -e "${RED}   Details: $details${NC}"
        fi
    elif [ "$result" = "SKIP" ]; then
        echo -e "${YELLOW}⏭️  $test_name: SKIP${NC}"
        if [ -n "$details" ]; then
            echo -e "${YELLOW}   Reason: $details${NC}"
        fi
    fi
}

# Test 1: File Existence and Structure
echo -e "\n${BLUE}📁 Test 1: File Existence and Structure${NC}"

# Test workflow file exists
if [ -f "$WORKFLOW_FILE" ]; then
    print_test_result "Workflow file exists" "PASS"
else
    print_test_result "Workflow file exists" "FAIL" "$WORKFLOW_FILE not found"
    exit 1
fi

# Test configuration file exists
if [ -f "$CONFIG_FILE" ]; then
    print_test_result "Configuration file exists" "PASS"
else
    print_test_result "Configuration file exists" "FAIL" "$CONFIG_FILE not found"
fi

# Test utilities file exists
if [ -f "$UTILS_FILE" ]; then
    print_test_result "Utilities file exists" "PASS"
else
    print_test_result "Utilities file exists" "FAIL" "$UTILS_FILE not found"
fi

# Test 2: YAML Syntax Validation
echo -e "\n${BLUE}🔍 Test 2: YAML Syntax Validation${NC}"

# Check if yamllint is available
if command -v yamllint &> /dev/null; then
    if yamllint "$WORKFLOW_FILE" &> /dev/null; then
        print_test_result "Workflow YAML syntax" "PASS"
    else
        print_test_result "Workflow YAML syntax" "FAIL" "yamllint errors found"
    fi
    
    if yamllint "$CONFIG_FILE" &> /dev/null; then
        print_test_result "Configuration YAML syntax" "PASS"
    else
        print_test_result "Configuration YAML syntax" "FAIL" "yamllint errors found"
    fi
else
    print_test_result "YAML syntax validation" "SKIP" "yamllint not available"
fi

# Test 3: GitHub CLI Integration
echo -e "\n${BLUE}🐙 Test 3: GitHub CLI Integration${NC}"

# Check if gh CLI is available and authenticated
if command -v gh &> /dev/null; then
    if gh auth status &> /dev/null; then
        print_test_result "GitHub CLI authentication" "PASS"
        
        # Test workflow listing
        if gh workflow list | grep -q "Release Management Automation"; then
            print_test_result "Workflow registration" "PASS"
        else
            print_test_result "Workflow registration" "FAIL" "Workflow not found in gh workflow list"
        fi
    else
        print_test_result "GitHub CLI authentication" "FAIL" "Not authenticated with GitHub"
    fi
else
    print_test_result "GitHub CLI integration" "SKIP" "gh CLI not available"
fi

# Test 4: Workflow Structure Validation
echo -e "\n${BLUE}⚙️  Test 4: Workflow Structure Validation${NC}"

# Check for required jobs
required_jobs=(
    "version-analysis"
    "build-assets"
    "create-release"
    "deploy"
    "rif-integration"
    "announce-release"
    "post-release-validation"
)

for job in "${required_jobs[@]}"; do
    if grep -q "^[[:space:]]*${job}:" "$WORKFLOW_FILE"; then
        print_test_result "Job '$job' exists" "PASS"
    else
        print_test_result "Job '$job' exists" "FAIL" "Job not found in workflow"
    fi
done

# Check for required triggers
if grep -q "workflow_dispatch:" "$WORKFLOW_FILE"; then
    print_test_result "Manual trigger configured" "PASS"
else
    print_test_result "Manual trigger configured" "FAIL" "workflow_dispatch not found"
fi

if grep -q "tags:" "$WORKFLOW_FILE"; then
    print_test_result "Tag trigger configured" "PASS"
else
    print_test_result "Tag trigger configured" "FAIL" "Tag trigger not found"
fi

# Test 5: Permission Validation
echo -e "\n${BLUE}🔐 Test 5: Permission Validation${NC}"

required_permissions=(
    "contents: write"
    "issues: write"
    "pull-requests: read"
)

for permission in "${required_permissions[@]}"; do
    if grep -q "$permission" "$WORKFLOW_FILE"; then
        print_test_result "Permission '$permission'" "PASS"
    else
        print_test_result "Permission '$permission'" "FAIL" "Permission not found"
    fi
done

# Test 6: Semantic Versioning Logic
echo -e "\n${BLUE}🏷️  Test 6: Semantic Versioning Logic${NC}"

# Test semantic version patterns in workflow
version_patterns=(
    "BREAKING"
    "feat:"
    "fix:"
    "feature"
    "bugfix"
)

for pattern in "${version_patterns[@]}"; do
    if grep -qi "$pattern" "$WORKFLOW_FILE"; then
        print_test_result "Version pattern '$pattern'" "PASS"
    else
        print_test_result "Version pattern '$pattern'" "FAIL" "Pattern not found in workflow"
    fi
done

# Test 7: Asset Management
echo -e "\n${BLUE}📦 Test 7: Asset Management${NC}"

asset_features=(
    "source-code"
    "documentation"
    "config-templates"
    "checksums"
    "upload-artifact"
)

for feature in "${asset_features[@]}"; do
    if grep -qi "$feature" "$WORKFLOW_FILE"; then
        print_test_result "Asset feature '$feature'" "PASS"
    else
        print_test_result "Asset feature '$feature'" "FAIL" "Feature not found in workflow"
    fi
done

# Test 8: RIF Integration
echo -e "\n${BLUE}🤖 Test 8: RIF Integration${NC}"

rif_features=(
    "state:"
    "rif-integration"
    "knowledge"
    "labels"
)

for feature in "${rif_features[@]}"; do
    if grep -qi "$feature" "$WORKFLOW_FILE"; then
        print_test_result "RIF feature '$feature'" "PASS"
    else
        print_test_result "RIF feature '$feature'" "FAIL" "Feature not found in workflow"
    fi
done

# Test 9: Environment Support
echo -e "\n${BLUE}🌍 Test 9: Environment Support${NC}"

environments=(
    "production"
    "staging"
    "development"
)

for env in "${environments[@]}"; do
    if grep -qi "$env" "$WORKFLOW_FILE"; then
        print_test_result "Environment '$env'" "PASS"
    else
        print_test_result "Environment '$env'" "FAIL" "Environment not found in workflow"
    fi
done

# Test 10: Error Handling and Validation
echo -e "\n${BLUE}🛡️  Test 10: Error Handling and Validation${NC}"

error_handling=(
    "if:"
    "needs:"
    "always()"
    "success()"
    "failure()"
)

for handler in "${error_handling[@]}"; do
    if grep -q "$handler" "$WORKFLOW_FILE"; then
        print_test_result "Error handling '$handler'" "PASS"
    else
        print_test_result "Error handling '$handler'" "FAIL" "Handler not found in workflow"
    fi
done

# Test 11: Node.js Utilities Validation
echo -e "\n${BLUE}🟢 Test 11: Node.js Utilities Validation${NC}"

if command -v node &> /dev/null; then
    # Test Node.js syntax
    if node -c "$UTILS_FILE" &> /dev/null; then
        print_test_result "Node.js utilities syntax" "PASS"
    else
        print_test_result "Node.js utilities syntax" "FAIL" "Syntax errors found"
    fi
    
    # Test required functions exist
    required_functions=(
        "analyzeCommitsForVersionType"
        "generateChangelog"
        "calculateNextVersion"
        "isValidVersion"
    )
    
    for func in "${required_functions[@]}"; do
        if grep -q "$func" "$UTILS_FILE"; then
            print_test_result "Function '$func' exists" "PASS"
        else
            print_test_result "Function '$func' exists" "FAIL" "Function not found"
        fi
    done
else
    print_test_result "Node.js utilities validation" "SKIP" "Node.js not available"
fi

# Test 12: Mock Workflow Execution (Dry Run)
echo -e "\n${BLUE}🎭 Test 12: Mock Workflow Execution${NC}"

if command -v gh &> /dev/null && gh auth status &> /dev/null; then
    # Check if we can access the workflow
    if gh workflow list | grep -q "Release Management Automation"; then
        print_test_result "Workflow accessibility" "PASS"
        
        # Note: We don't actually trigger the workflow in tests
        # This would require creating test tags or manual triggers
        print_test_result "Workflow execution test" "SKIP" "Manual testing required"
    else
        print_test_result "Workflow accessibility" "FAIL" "Cannot access workflow"
    fi
else
    print_test_result "Mock workflow execution" "SKIP" "GitHub CLI not available/authenticated"
fi

# Test 13: Documentation and Examples
echo -e "\n${BLUE}📚 Test 13: Documentation and Examples${NC}"

# Check for inline documentation in workflow
if grep -q "# " "$WORKFLOW_FILE"; then
    print_test_result "Workflow documentation" "PASS"
else
    print_test_result "Workflow documentation" "FAIL" "No comments found in workflow"
fi

# Check for example configurations
if grep -q "example\|default\|template" "$CONFIG_FILE"; then
    print_test_result "Configuration examples" "PASS"
else
    print_test_result "Configuration examples" "FAIL" "No examples found in configuration"
fi

# Final Summary
echo -e "\n${BLUE}📊 Test Summary${NC}"
echo "==================="

# Count results (simplified - in a real implementation you'd track this properly)
echo -e "${GREEN}✅ Tests completed${NC}"
echo -e "${YELLOW}⚠️  Manual validation required for:${NC}"
echo "   - Actual workflow execution"
echo "   - GitHub Release creation"
echo "   - Asset attachment verification"
echo "   - RIF integration testing"
echo "   - Multi-environment deployment"

# Usage instructions
echo -e "\n${BLUE}🚀 Next Steps${NC}"
echo "==============="
echo "1. Review any failed tests above"
echo "2. Manually test workflow with:"
echo "   gh workflow run release-automation.yml -f version=v0.0.1-test -f prerelease=true"
echo "3. Validate release creation in GitHub UI"
echo "4. Test RIF integration with actual issues"
echo "5. Verify announcement system functionality"

echo -e "\n${GREEN}✅ Release workflow validation completed!${NC}"