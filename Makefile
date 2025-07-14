.PHONY: install install-dev lint test docs clean build help

# Default target
help:
	@echo "Available targets:"
	@echo "  install    - Install package in development mode with all dependencies"
	@echo "  install-dev - Install package with development dependencies"
	@echo "  lint       - Run code formatting and linting (black, flake8, isort)"
	@echo "  test       - Run tests with coverage (pytest --cov)"
	@echo "  docs       - Build documentation (mkdocs build)"
	@echo "  clean      - Remove build artifacts, cache files, and compiled Python files"
	@echo "  build      - Build the package"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,docs]"

lint:
	isort .
	black .
	flake8 .

test:
	pytest

build:
	python -m build

docs:
	mkdocs build

clean:
	@echo "Cleaning build artifacts and cache files..."
	@if exist build rmdir /s /q build 2>nul || echo "No build directory found"
	@if exist dist rmdir /s /q dist 2>nul || echo "No dist directory found"
	@if exist *.egg-info rmdir /s /q *.egg-info 2>nul || echo "No egg-info directories found"
	@for /r . %%i in (*.pyc) do @del "%%i" 2>nul || echo ""
	@for /r . %%i in (*.pyo) do @del "%%i" 2>nul || echo ""
	@for /d /r . %%i in (__pycache__) do @rmdir /s /q "%%i" 2>nul || echo ""
	@for /d /r . %%i in (.pytest_cache) do @rmdir /s /q "%%i" 2>nul || echo ""
	@for /d /r . %%i in (*.egg-info) do @rmdir /s /q "%%i" 2>nul || echo ""
	@echo "Clean completed!"