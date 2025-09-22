# Development Guide

This project uses [UV](https://docs.astral.sh/uv/) for Python package management, virtual environment management, and task execution. All development tasks must be performed using UV to ensure consistent environments and reproducible builds.

## Prerequisites

### Install UV

UV is required for all development tasks. Install it using one of these methods:

```bash
# Using pip
pip install uv

# Using pipx (recommended)
pipx install uv

# Using homebrew (macOS)
brew install uv

# Using the installer script
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more installation options, see the [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Project Setup

### 1. Clone and Initialize

```bash
git clone <repository-url>
cd ofdm-chirp-generator

# Initialize UV project and sync dependencies
uv sync
```

### 2. Verify Installation

```bash
# Check UV version
uv --version

# Verify project setup
uv run python -c "from ofdm_chirp_generator import OFDMConfig; print('Setup successful!')"
```

## Development Workflow

### Virtual Environment Management

UV automatically manages the virtual environment. All commands should be run with `uv run`:

```bash
# Activate the UV environment (automatic with uv run)
uv run python --version

# Install additional dependencies
uv add numpy matplotlib
uv add --dev pytest black

# Remove dependencies
uv remove matplotlib
```

### Running Tests

All tests must be executed through UV:

```bash
# Run all tests
uv run test

# Run tests with coverage
uv run test-cov

# Run specific test file
uv run pytest tests/test_models.py -v

# Run tests for specific task
uv run pytest tests/test_chirp_modulator.py tests/test_subcarrier_manager.py -v
```

### Code Quality

Use UV scripts for code formatting and linting:

```bash
# Format code
uv run format

# Check formatting
uv run format-check

# Sort imports
uv run sort-imports

# Run linting
uv run lint

# Type checking
uv run type-check
```

### Running Examples

Execute examples through UV:

```bash
# Run individual demos
uv run demo-basic
uv run demo-gpu
uv run demo-chirp
uv run demo-ofdm
uv run demo-generator

# Run all demos
uv run demo-all

# Or run directly
uv run python examples/basic_usage.py
```

## Task Implementation Guidelines

When implementing tasks from the specification, follow these UV-based practices:

### 1. Environment Setup for Each Task

```bash
# Before starting any task, ensure environment is synced
uv sync

# Add any new dependencies needed for the task
uv add <package-name>
```

### 2. Development Process

```bash
# 1. Implement the feature
# 2. Run tests continuously during development
uv run pytest tests/test_<feature>.py -v

# 3. Run code quality checks
uv run format
uv run lint
uv run type-check

# 4. Run full test suite
uv run test

# 5. Test examples
uv run demo-<feature>
```

### 3. Adding New Dependencies

```bash
# Core dependencies (add to [project.dependencies])
uv add numpy scipy

# Development dependencies (add to [tool.uv.dev-dependencies])
uv add --dev pytest-mock pytest-benchmark

# Optional dependencies (add to [project.optional-dependencies])
uv add --optional gpu cupy-cuda11x
uv add --optional visualization matplotlib seaborn
```

### 4. Creating New Examples

When creating examples for tasks:

```bash
# Create the example file
touch examples/new_feature_demo.py

# Add a UV script to pyproject.toml
# [tool.uv.scripts]
# demo-new-feature = "python examples/new_feature_demo.py"

# Test the example
uv run demo-new-feature
```

## Task-Specific Commands

### Task 1: Project Structure and Data Models
```bash
uv sync
uv run pytest tests/test_models.py tests/test_validation.py -v
uv run demo-basic
```

### Task 2: GPU Backend Implementation
```bash
uv add cupy-cuda11x cupy-cuda12x  # Optional GPU dependencies
uv run pytest tests/test_gpu_backend.py -v
uv run demo-gpu
```

### Task 3: Chirp Signal Generation
```bash
uv run pytest tests/test_chirp_modulator.py -v
uv run demo-chirp
```

### Task 4: OFDM Signal Structure
```bash
uv run pytest tests/test_subcarrier_manager.py -v
uv run demo-ofdm
```

### Task 5: Core Signal Generation Engine
```bash
uv run pytest tests/test_ofdm_generator.py -v
uv run demo-generator
```

## Continuous Integration

The CI pipeline uses UV for all operations:

```yaml
# Example GitHub Actions workflow
- name: Install UV
  run: pip install uv

- name: Setup project
  run: uv sync

- name: Run tests
  run: uv run test-cov

- name: Run linting
  run: uv run lint

- name: Run type checking
  run: uv run type-check
```

## Troubleshooting

### Common Issues

1. **UV not found**: Ensure UV is installed and in your PATH
2. **Dependencies not synced**: Run `uv sync` to update dependencies
3. **Import errors**: Verify you're using `uv run` for all Python commands
4. **GPU dependencies**: CuPy dependencies are optional and may fail on systems without CUDA

### Debugging

```bash
# Check UV environment
uv info

# List installed packages
uv pip list

# Check project configuration
uv run python -c "import sys; print(sys.executable)"

# Verbose dependency resolution
uv sync --verbose
```

## Best Practices

1. **Always use `uv run`** for executing Python commands
2. **Sync before starting work**: `uv sync` ensures consistent environment
3. **Add dependencies properly**: Use `uv add` instead of pip install
4. **Test in UV environment**: All tests must pass with `uv run pytest`
5. **Use UV scripts**: Prefer `uv run test` over `uv run pytest`
6. **Document new scripts**: Add new UV scripts to pyproject.toml
7. **Version lock**: Commit uv.lock for reproducible builds

## Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging with UV](https://docs.astral.sh/uv/guides/projects/)