#!/usr/bin/env bash
# Test script for deploy_cloud_run.sh
# This script validates the deploy script works correctly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="${SCRIPT_DIR}/deploy_cloud_run.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to print test results
print_test_result() {
    local test_name="$1"
    local result="$2"
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if [ "$result" == "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

echo "=================================="
echo "Testing deploy_cloud_run.sh"
echo "=================================="
echo ""

# Test 1: Script exists and is executable
echo "Test 1: Verify deploy script exists and is executable"
if [ -f "$DEPLOY_SCRIPT" ] && [ -x "$DEPLOY_SCRIPT" ]; then
    print_test_result "deploy_cloud_run.sh exists and is executable" "PASS"
else
    print_test_result "deploy_cloud_run.sh exists and is executable" "FAIL"
    echo "  Error: Script not found or not executable at $DEPLOY_SCRIPT"
fi

# Test 2: Script requires PROJECT_ID
echo ""
echo "Test 2: Verify script requires PROJECT_ID environment variable"
OUTPUT=$(bash -c "bash '$DEPLOY_SCRIPT'" 2>&1 || true)
if echo "$OUTPUT" | grep -q "PROJECT_ID"; then
    print_test_result "Script requires PROJECT_ID" "PASS"
else
    print_test_result "Script requires PROJECT_ID" "FAIL"
    echo "  Error: Script should fail when PROJECT_ID is not set"
fi

# Test 3: Script uses correct defaults
echo ""
echo "Test 3: Verify script uses correct default values"
SCRIPT_CONTENT=$(cat "$DEPLOY_SCRIPT")

if echo "$SCRIPT_CONTENT" | grep -q 'REGION="${REGION:-us-central1}"'; then
    print_test_result "Default REGION is us-central1" "PASS"
else
    print_test_result "Default REGION is us-central1" "FAIL"
fi

if echo "$SCRIPT_CONTENT" | grep -q 'REPO="${REPO:-markettech}"'; then
    print_test_result "Default REPO is markettech" "PASS"
else
    print_test_result "Default REPO is markettech" "FAIL"
fi

if echo "$SCRIPT_CONTENT" | grep -q 'SERVICE="${SERVICE:-markettech-truth-engine}"'; then
    print_test_result "Default SERVICE is markettech-truth-engine" "PASS"
else
    print_test_result "Default SERVICE is markettech-truth-engine" "FAIL"
fi

if echo "$SCRIPT_CONTENT" | grep -q 'TAG="${TAG:-v1}"'; then
    print_test_result "Default TAG is v1" "PASS"
else
    print_test_result "Default TAG is v1" "FAIL"
fi

# Test 4: Script doesn't reference non-existent terraform directory
echo ""
echo "Test 4: Verify script doesn't reference non-existent 'terraform' subdirectory"
if grep -q "pushd terraform" "$DEPLOY_SCRIPT"; then
    print_test_result "Script doesn't use 'pushd terraform'" "FAIL"
    echo "  Error: Script should not try to change to 'terraform' directory"
else
    print_test_result "Script doesn't use 'pushd terraform'" "PASS"
fi

# Test 5: Script contains terraform apply command
echo ""
echo "Test 5: Verify script contains terraform apply command"
if grep -q "terraform apply" "$DEPLOY_SCRIPT"; then
    print_test_result "Script contains terraform apply command" "PASS"
else
    print_test_result "Script contains terraform apply command" "FAIL"
    echo "  Error: Script should contain 'terraform apply' command"
fi

# Test 6: Script contains terraform init command
echo ""
echo "Test 6: Verify script contains terraform init command"
if grep -q "terraform init" "$DEPLOY_SCRIPT"; then
    print_test_result "Script contains terraform init command" "PASS"
else
    print_test_result "Script contains terraform init command" "FAIL"
    echo "  Error: Script should contain 'terraform init' command"
fi

# Test 7: Script contains gcloud builds submit command
echo ""
echo "Test 7: Verify script contains gcloud builds submit command"
if grep -q "gcloud builds submit" "$DEPLOY_SCRIPT"; then
    print_test_result "Script contains gcloud builds submit command" "PASS"
else
    print_test_result "Script contains gcloud builds submit command" "FAIL"
    echo "  Error: Script should contain 'gcloud builds submit' command"
fi

# Test 8: Terraform files exist in root directory
echo ""
echo "Test 8: Verify Terraform files exist in repository root"
for tf_file in main.tf variables.tf versions.tf; do
    if [ -f "${SCRIPT_DIR}/${tf_file}" ]; then
        print_test_result "File ${tf_file} exists in root" "PASS"
    else
        print_test_result "File ${tf_file} exists in root" "FAIL"
        echo "  Error: ${tf_file} not found in repository root"
    fi
done

# Test 9: Verify script runs terraform commands from repository root
echo ""
echo "Test 9: Verify script runs terraform from current directory"
if grep -B1 -A1 "terraform" "$DEPLOY_SCRIPT" | grep -q "pushd"; then
    print_test_result "Script runs terraform from current directory" "FAIL"
    echo "  Error: Script changes directory before running terraform"
else
    print_test_result "Script runs terraform from current directory" "PASS"
fi

# Test 10: Script has proper error handling
echo ""
echo "Test 10: Verify script has proper error handling (set -euo pipefail)"
if head -5 "$DEPLOY_SCRIPT" | grep -q "set -euo pipefail"; then
    print_test_result "Script has proper error handling" "PASS"
else
    print_test_result "Script has proper error handling" "FAIL"
    echo "  Error: Script should have 'set -euo pipefail' near the top"
fi

# Test 11: Verify deploy script documentation
echo ""
echo "Test 11: Verify script contains informative output messages"
if grep -q "Applying Terraform" "$DEPLOY_SCRIPT" && grep -q "Building and pushing image" "$DEPLOY_SCRIPT"; then
    print_test_result "Script contains deployment progress messages" "PASS"
else
    print_test_result "Script contains deployment progress messages" "FAIL"
fi

# Summary
echo ""
echo "=================================="
echo "Test Summary"
echo "=================================="
echo "Tests run: $TESTS_RUN"
echo -e "${GREEN}Tests passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests failed: $TESTS_FAILED${NC}"
else
    echo -e "${GREEN}Tests failed: $TESTS_FAILED${NC}"
fi
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
    echo "The deploy_cloud_run.sh script is properly configured to:"
    echo "  • Run from the repository root directory"
    echo "  • Access Terraform files in the current directory"
    echo "  • Build and push container images"
    echo "  • Deploy Terraform-managed resources"
    echo "  • Validate required environment variables"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the errors above.${NC}"
    exit 1
fi
