.PHONY: setup test run clean lint help

# Default Python version
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

test: ## Run test suite
	$(PYTHON) -m pytest -v

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term

run: ## Run local Streamlit application
	$(PYTHON) -m streamlit run app.py

clean: ## Clean up temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete

lint: ## Run code quality checks (optional - requires additional tools)
	@echo "Note: Install flake8, black, mypy for full linting support"
	@command -v flake8 >/dev/null 2>&1 && flake8 . --max-line-length=120 --exclude=.venv,venv,__pycache__ || echo "flake8 not installed, skipping"
	@command -v black >/dev/null 2>&1 && black --check . || echo "black not installed, skipping"
	@command -v mypy >/dev/null 2>&1 && mypy . --ignore-missing-imports || echo "mypy not installed, skipping"
