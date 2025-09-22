# Makefile for OFDM Chirp Generator
# Uses UV for dependency management and virtual environments

.PHONY: help install install-dev test test-cov lint format check clean examples docs

# Default target
help:
	@echo "OFDM Chirp Generator - UV-based Development"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install package with basic dependencies"
	@echo "  install-dev  - Install package with all development dependencies"
	@echo "  test         - Run test suite"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  check        - Run all quality checks"
	@echo "  clean        - Clean up generated files"
	@echo "  examples     - Run example scripts"
	@echo "  benchmark    - Run performance benchmarks demo"
	@echo "  benchmark-full - Run comprehensive performance benchmarks"
	@echo "  docs         - Generate documentation"
	@echo ""
	@echo "UV Commands:"
	@echo "  uv sync      - Sync dependencies from lock file"
	@echo "  uv add       - Add new dependency"
	@echo "  uv remove    - Remove dependency"
	@echo "  uv lock      - Update lock file"

# Installation targets
install:
	uv pip install .

install-dev:
	uv sync
	uv pip install -e ".[all]"

# Testing targets
test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=ofdm_chirp_generator --cov-report=html --cov-report=term-missing

test-fast:
	uv run pytest -v -m "not slow"

test-gpu:
	uv run pytest -v -m "gpu"

# Code quality targets
lint:
	uv run flake8 ofdm_chirp_generator/ tests/ examples/

format:
	uv run black ofdm_chirp_generator/ tests/ examples/
	uv run isort ofdm_chirp_generator/ tests/ examples/

check: lint
	uv run black --check ofdm_chirp_generator/ tests/ examples/
	uv run isort --check-only ofdm_chirp_generator/ tests/ examples/
	uv run mypy ofdm_chirp_generator/

# Example targets
examples:
	@echo "Running OFDM Generator examples..."
	uv run examples/basic_usage.py
	uv run examples/ofdm_generator_demo.py

example-basic:
	uv run examples/basic_usage.py

example-chirp:
	uv run examples/chirp_modulator_demo.py

example-ofdm:
	uv run examples/ofdm_structure_demo.py

example-gpu:
	uv run examples/gpu_backend_demo.py

example-generator:
	uv run examples/ofdm_generator_demo.py

# Utility targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	@echo "Documentation generation not yet implemented"

# UV-specific targets
sync:
	uv sync

lock:
	uv lock

add:
	@echo "Usage: make add PACKAGE=<package_name>"
	@echo "Example: make add PACKAGE=matplotlib"
ifdef PACKAGE
	uv add $(PACKAGE)
endif

add-dev:
	@echo "Usage: make add-dev PACKAGE=<package_name>"
	@echo "Example: make add-dev PACKAGE=pytest-mock"
ifdef PACKAGE
	uv add --dev $(PACKAGE)
endif

remove:
	@echo "Usage: make remove PACKAGE=<package_name>"
	@echo "Example: make remove PACKAGE=matplotlib"
ifdef PACKAGE
	uv remove $(PACKAGE)
endif

# Environment management
venv-info:
	uv pip list

venv-clean:
	uv venv --rm
	uv sync

# Build targets
build:
	uv build

publish-test:
	uv publish --repository testpypi

publish:
	uv publish

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Activate with: source .venv/bin/activate"

dev-check: format test check
	@echo "All development checks passed!"

# Performance and profiling targets
profile:
	@echo "Running performance profiling..."
	uv run python -m cProfile -o profile.stats examples/performance_optimization_demo.py
	@echo "Profile saved to profile.stats"

benchmark:
	@echo "Running performance benchmarks demo..."
	uv run python examples/performance_benchmarks_demo.py

benchmark-full:
	@echo "Running comprehensive performance benchmarks..."
	uv run python scripts/run_performance_benchmarks.py --output-dir benchmark_results

benchmark-regression:
	@echo "Running performance regression tests..."
	uv run python scripts/run_performance_benchmarks.py --check-regression --tolerance 0.15

benchmark-baseline:
	@echo "Saving performance baseline..."
	uv run python scripts/run_performance_benchmarks.py --save-baseline

benchmark-pytest:
	@echo "Running pytest performance benchmarks..."
	uv run pytest tests/test_performance_benchmarks.py -v -m performance

# CI/CD targets
ci-test:
	uv run pytest --cov=ofdm_chirp_generator --cov-report=xml

ci-check: check ci-test
	@echo "CI checks completed"