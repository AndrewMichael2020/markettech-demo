#!/usr/bin/env bash
# Master test runner for all deployment/teardown scripts
# This script runs all infrastructure script tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Infrastructure Scripts Test Suite${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Track overall success
OVERALL_SUCCESS=true

# Test 1: Run cleanup script tests
echo -e "${YELLOW}Running cleanup_cloud_run.sh tests...${NC}"
echo ""
if "${SCRIPT_DIR}/test_cleanup_script.sh"; then
    echo ""
    echo -e "${GREEN}✓ Cleanup script tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}✗ Cleanup script tests FAILED${NC}"
    OVERALL_SUCCESS=false
fi

echo ""
echo "============================================"
echo ""

# Test 2: Run deploy script tests
echo -e "${YELLOW}Running deploy_cloud_run.sh tests...${NC}"
echo ""
if "${SCRIPT_DIR}/test_deploy_script.sh"; then
    echo ""
    echo -e "${GREEN}✓ Deploy script tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}✗ Deploy script tests FAILED${NC}"
    OVERALL_SUCCESS=false
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Overall Test Summary${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "${GREEN}✓ All infrastructure script tests PASSED${NC}"
    echo ""
    echo "The infrastructure scripts are properly configured:"
    echo "  • Scripts run from repository root"
    echo "  • Terraform files are correctly referenced"
    echo "  • Environment variables are validated"
    echo "  • Error handling is in place"
    echo "  • All commands are present and correct"
    echo ""
    echo -e "${GREEN}The workspace can be safely deployed and torn down.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some infrastructure script tests FAILED${NC}"
    echo ""
    echo "Please review the test output above to identify and fix issues."
    exit 1
fi
