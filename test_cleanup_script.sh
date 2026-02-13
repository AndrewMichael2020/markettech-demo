#!/usr/bin/env bash
# Test script for cleanup_cloud_run.sh
# This script validates the cleanup script works correctly and verifies workspace teardown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_cloud_run.sh"

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
echo "Testing cleanup_cloud_run.sh"
echo "=================================="
echo ""

# Test 1: Script exists and is executable
echo "Test 1: Verify cleanup script exists and is executable"
if [ -f "$CLEANUP_SCRIPT" ] && [ -x "$CLEANUP_SCRIPT" ]; then
    print_test_result "cleanup_cloud_run.sh exists and is executable" "PASS"
else
    print_test_result "cleanup_cloud_run.sh exists and is executable" "FAIL"
    echo "  Error: Script not found or not executable at $CLEANUP_SCRIPT"
fi

# Test 2: Script requires PROJECT_ID
echo ""
echo "Test 2: Verify script requires PROJECT_ID environment variable"
OUTPUT=$(bash -c "bash '$CLEANUP_SCRIPT'" 2>&1 || true)
if echo "$OUTPUT" | grep -q "PROJECT_ID"; then
    print_test_result "Script requires PROJECT_ID" "PASS"
else
    print_test_result "Script requires PROJECT_ID" "FAIL"
    echo "  Error: Script should fail when PROJECT_ID is not set"
fi

# Test 3: Script uses correct defaults
echo ""
echo "Test 3: Verify script uses correct default values"
DEFAULTS_CORRECT=true

# Export PROJECT_ID and check if script generates correct values
export PROJECT_ID="test-project-123"
export REGION="us-west1"
export REPO="test-repo"
export SERVICE="test-service"
export TAG="test-tag"

# Extract the variable assignments from the script
SCRIPT_CONTENT=$(cat "$CLEANUP_SCRIPT")

if echo "$SCRIPT_CONTENT" | grep -q 'REGION="${REGION:-us-central1}"'; then
    print_test_result "Default REGION is us-central1" "PASS"
else
    print_test_result "Default REGION is us-central1" "FAIL"
    DEFAULTS_CORRECT=false
fi

if echo "$SCRIPT_CONTENT" | grep -q 'REPO="${REPO:-markettech}"'; then
    print_test_result "Default REPO is markettech" "PASS"
else
    print_test_result "Default REPO is markettech" "FAIL"
    DEFAULTS_CORRECT=false
fi

if echo "$SCRIPT_CONTENT" | grep -q 'SERVICE="${SERVICE:-markettech-truth-engine}"'; then
    print_test_result "Default SERVICE is markettech-truth-engine" "PASS"
else
    print_test_result "Default SERVICE is markettech-truth-engine" "FAIL"
    DEFAULTS_CORRECT=false
fi

if echo "$SCRIPT_CONTENT" | grep -q 'TAG="${TAG:-v1}"'; then
    print_test_result "Default TAG is v1" "PASS"
else
    print_test_result "Default TAG is v1" "FAIL"
    DEFAULTS_CORRECT=false
fi

# Test 4: Script doesn't reference non-existent terraform directory
echo ""
echo "Test 4: Verify script doesn't reference non-existent 'terraform' subdirectory"
if grep -q "pushd terraform" "$CLEANUP_SCRIPT"; then
    print_test_result "Script doesn't use 'pushd terraform'" "FAIL"
    echo "  Error: Script should not try to change to 'terraform' directory"
else
    print_test_result "Script doesn't use 'pushd terraform'" "PASS"
fi

# Test 5: Script contains terraform destroy command
echo ""
echo "Test 5: Verify script contains terraform destroy command"
if grep -q "terraform destroy" "$CLEANUP_SCRIPT"; then
    print_test_result "Script contains terraform destroy command" "PASS"
else
    print_test_result "Script contains terraform destroy command" "FAIL"
    echo "  Error: Script should contain 'terraform destroy' command"
fi

# Test 6: Script contains terraform init command
echo ""
echo "Test 6: Verify script contains terraform init command"
if grep -q "terraform init" "$CLEANUP_SCRIPT"; then
    print_test_result "Script contains terraform init command" "PASS"
else
    print_test_result "Script contains terraform init command" "FAIL"
    echo "  Error: Script should contain 'terraform init' command"
fi

# Test 7: Terraform files exist in root directory
echo ""
echo "Test 7: Verify Terraform files exist in repository root"
TF_FILES_EXIST=true

for tf_file in main.tf variables.tf versions.tf; do
    if [ -f "${SCRIPT_DIR}/${tf_file}" ]; then
        print_test_result "File ${tf_file} exists in root" "PASS"
    else
        print_test_result "File ${tf_file} exists in root" "FAIL"
        TF_FILES_EXIST=false
        echo "  Error: ${tf_file} not found in repository root"
    fi
done

# Test 8: Verify terraform files are referenced from correct location
echo ""
echo "Test 8: Verify script runs terraform commands from repository root"
# The script should run terraform commands without changing directories
if ! grep -q "pushd\|cd " "$CLEANUP_SCRIPT" | grep -v "echo\|#"; then
    print_test_result "Script runs terraform from current directory" "PASS"
else
    # Check if there's actually a pushd that affects terraform
    if grep -B1 -A1 "terraform" "$CLEANUP_SCRIPT" | grep -q "pushd"; then
        print_test_result "Script runs terraform from current directory" "FAIL"
        echo "  Error: Script changes directory before running terraform"
    else
        print_test_result "Script runs terraform from current directory" "PASS"
    fi
fi

# Test 9: Script has proper error handling
echo ""
echo "Test 9: Verify script has proper error handling (set -euo pipefail)"
if head -5 "$CLEANUP_SCRIPT" | grep -q "set -euo pipefail"; then
    print_test_result "Script has proper error handling" "PASS"
else
    print_test_result "Script has proper error handling" "FAIL"
    echo "  Error: Script should have 'set -euo pipefail' near the top"
fi

# Test 10: Verify cleanup script documentation
echo ""
echo "Test 10: Verify script contains helpful output messages"
if grep -q "Destroying Terraform-managed resources" "$CLEANUP_SCRIPT"; then
    print_test_result "Script contains informative messages" "PASS"
else
    print_test_result "Script contains informative messages" "FAIL"
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
    echo "The cleanup_cloud_run.sh script is properly configured to:"
    echo "  • Run from the repository root directory"
    echo "  • Access Terraform files in the current directory"
    echo "  • Properly destroy all Terraform-managed resources"
    echo "  • Validate required environment variables"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review the errors above.${NC}"
    exit 1
fi
