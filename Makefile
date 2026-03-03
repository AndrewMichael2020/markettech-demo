.PHONY: setup test test-all test-infrastructure clean run lint help

# System Strategy: Detect OS to set the correct python command
PYTHON := python3
PIP := $(PYTHON) -m pip

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "Dependencies installed."

# The "CI-Safe" Test Command (excludes AI/integration tests)
test: ## Run test suite (CI-safe, excludes AI tests)
	$(PYTHON) -m pytest -m "not ai"

# The "Full System" Test Command (includes AI tests)
test-all: ## Run all tests including AI tests
	$(PYTHON) -m pytest

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=src --cov-report=html --cov-report=term -m "not ai"

test-infrastructure: ## Test infrastructure deployment and cleanup scripts
	@echo "Testing infrastructure scripts..."
	@./test_infrastructure_scripts.sh

run: ## Run local Streamlit application
	$(PYTHON) -m streamlit run app.py

lint: ## Run code quality checks
	@echo "Note: Install flake8, black, mypy for full linting support"
	@command -v flake8 >/dev/null 2>&1 && flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics || echo "flake8 not installed, skipping"
	@command -v black >/dev/null 2>&1 && black --check . || echo "black not installed, skipping"
	@command -v mypy >/dev/null 2>&1 && mypy src --ignore-missing-imports || echo "mypy not installed, skipping"

clean: ## Clean up temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete

