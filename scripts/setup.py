#!/usr/bin/env python3
"""
Setup script for OFDM Chirp Generator using UV.

This script helps set up the development environment and run common tasks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def setup_environment():
    """Set up the development environment."""
    print("Setting up OFDM Chirp Generator development environment...")
    
    # Check if UV is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✓ UV is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ UV is not installed. Please install UV first:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    
    # Sync dependencies
    if not run_command(["uv", "sync"], "Installing dependencies"):
        return False
    
    # Install package in development mode
    if not run_command(["uv", "pip", "install", "-e", ".[all]"], "Installing package in development mode"):
        return False
    
    print("\n✓ Environment setup complete!")
    print("\nNext steps:")
    print("  1. Activate the virtual environment: source .venv/bin/activate")
    print("  2. Run tests: uv run pytest")
    print("  3. Run examples: uv run examples/ofdm_generator_demo.py")
    
    return True


def run_tests():
    """Run the test suite."""
    print("Running test suite...")
    
    commands = [
        (["uv", "run", "pytest", "-v"], "Running tests"),
        (["uv", "run", "pytest", "--cov=ofdm_chirp_generator", "--cov-report=term-missing"], "Running tests with coverage"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    return True


def run_quality_checks():
    """Run code quality checks."""
    print("Running code quality checks...")
    
    commands = [
        (["uv", "run", "black", "--check", "ofdm_chirp_generator/", "tests/", "examples/"], "Checking code formatting"),
        (["uv", "run", "isort", "--check-only", "ofdm_chirp_generator/", "tests/", "examples/"], "Checking import sorting"),
        (["uv", "run", "flake8", "ofdm_chirp_generator/", "tests/", "examples/"], "Running linter"),
        (["uv", "run", "mypy", "ofdm_chirp_generator/"], "Running type checker"),
    ]
    
    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    return all_passed


def format_code():
    """Format code using black and isort."""
    print("Formatting code...")
    
    commands = [
        (["uv", "run", "black", "ofdm_chirp_generator/", "tests/", "examples/"], "Formatting code with black"),
        (["uv", "run", "isort", "ofdm_chirp_generator/", "tests/", "examples/"], "Sorting imports with isort"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    print("✓ Code formatting complete!")
    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/setup.py <command>")
        print("Commands:")
        print("  setup    - Set up development environment")
        print("  test     - Run test suite")
        print("  check    - Run code quality checks")
        print("  format   - Format code")
        return
    
    command = sys.argv[1]
    
    if command == "setup":
        setup_environment()
    elif command == "test":
        run_tests()
    elif command == "check":
        run_quality_checks()
    elif command == "format":
        format_code()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()