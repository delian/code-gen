#!/usr/bin/env python3
"""
UV-based development script for OFDM Chirp Generator.

This script provides UV-based commands for all development tasks to ensure
consistent environment usage across all project activities.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def run_uv_command(args, description="", check=True):
    """Run a UV command and handle errors."""
    cmd = ["uv"] + args
    print(f"Running: {' '.join(cmd)}")
    if description:
        print(f"  {description}")
    
    try:
        result = subprocess.run(cmd, check=check, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  Error: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("  Error: UV not found. Please install UV first:")
        print("    pip install uv")
        print("    or visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False


def check_uv():
    """Check if UV is available."""
    if not shutil.which("uv"):
        print("UV not found. Please install UV:")
        print("  pip install uv")
        print("  or visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    # Check UV version
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
        print(f"✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("✗ UV installation appears to be broken")
        return False


def setup():
    """Set up the development environment using UV."""
    print("=== Setting up OFDM Chirp Generator with UV ===")
    
    if not check_uv():
        return False
    
    # Sync dependencies
    if not run_uv_command(["sync"], "Syncing dependencies"):
        return False
    
    # Verify installation
    if not run_uv_command(["run", "python", "-c", "from ofdm_chirp_generator import OFDMConfig; print('✓ Package import successful')"], 
                         "Verifying package installation"):
        return False
    
    print("\n✓ Setup completed successfully!")
    print("\nNext steps:")
    print("  uv run test          # Run tests")
    print("  uv run demo-basic    # Run basic demo")
    print("  uv run lint          # Check code quality")
    
    return True


def test(coverage=False, specific_test=None):
    """Run tests using UV."""
    print("=== Running Tests with UV ===")
    
    if specific_test:
        args = ["run", "pytest", specific_test, "-v"]
        desc = f"Running specific test: {specific_test}"
    elif coverage:
        args = ["run", "pytest", "--cov=ofdm_chirp_generator", "--cov-report=html", "--cov-report=term", "-v"]
        desc = "Running tests with coverage"
    else:
        args = ["run", "pytest", "-v"]
        desc = "Running all tests"
    
    return run_uv_command(args, desc)


def lint():
    """Run linting using UV."""
    print("=== Running Code Quality Checks with UV ===")
    
    checks = [
        (["run", "flake8", "ofdm_chirp_generator", "tests", "examples"], "Running flake8"),
        (["run", "black", "--check", "ofdm_chirp_generator", "tests", "examples"], "Checking code formatting"),
        (["run", "isort", "--check-only", "ofdm_chirp_generator", "tests", "examples"], "Checking import sorting"),
        (["run", "mypy", "ofdm_chirp_generator"], "Running type checking")
    ]
    
    all_passed = True
    for args, desc in checks:
        if not run_uv_command(args, desc, check=False):
            all_passed = False
    
    return all_passed


def format_code():
    """Format code using UV."""
    print("=== Formatting Code with UV ===")
    
    formats = [
        (["run", "black", "ofdm_chirp_generator", "tests", "examples"], "Formatting with black"),
        (["run", "isort", "ofdm_chirp_generator", "tests", "examples"], "Sorting imports with isort")
    ]
    
    for args, desc in formats:
        if not run_uv_command(args, desc):
            return False
    
    print("✓ Code formatting completed!")
    return True


def demo(demo_name="all"):
    """Run demos using UV."""
    print(f"=== Running Demo: {demo_name} ===")
    
    demos = {
        "basic": ["run", "python", "examples/basic_usage.py"],
        "gpu": ["run", "python", "examples/gpu_backend_demo.py"],
        "chirp": ["run", "python", "examples/chirp_modulator_demo.py"],
        "ofdm": ["run", "python", "examples/ofdm_structure_demo.py"],
        "generator": ["run", "python", "examples/ofdm_generator_demo.py"]
    }
    
    if demo_name == "all":
        for name, args in demos.items():
            print(f"\n--- Running {name} demo ---")
            if not run_uv_command(args, f"Running {name} demo"):
                return False
        return True
    elif demo_name in demos:
        return run_uv_command(demos[demo_name], f"Running {demo_name} demo")
    else:
        print(f"Unknown demo: {demo_name}")
        print(f"Available demos: {', '.join(demos.keys())}, all")
        return False


def add_dependency(package, dev=False, optional=None):
    """Add a dependency using UV."""
    print(f"=== Adding Dependency: {package} ===")
    
    args = ["add"]
    if dev:
        args.append("--dev")
    if optional:
        args.extend(["--optional", optional])
    args.append(package)
    
    return run_uv_command(args, f"Adding {'dev ' if dev else ''}dependency: {package}")


def remove_dependency(package):
    """Remove a dependency using UV."""
    print(f"=== Removing Dependency: {package} ===")
    
    return run_uv_command(["remove", package], f"Removing dependency: {package}")


def info():
    """Show UV project information."""
    print("=== UV Project Information ===")
    
    commands = [
        (["info"], "Project information"),
        (["pip", "list"], "Installed packages"),
        (["run", "python", "--version"], "Python version"),
        (["run", "python", "-c", "import sys; print(f'Python executable: {sys.executable}')"], "Python executable")
    ]
    
    for args, desc in commands:
        print(f"\n--- {desc} ---")
        run_uv_command(args, check=False)


def task_commands():
    """Show task-specific UV commands."""
    print("=== Task-Specific UV Commands ===")
    
    tasks = {
        "Task 1 - Data Models": [
            "uv run pytest tests/test_models.py tests/test_validation.py -v",
            "uv run demo-basic"
        ],
        "Task 2 - GPU Backend": [
            "uv add cupy-cuda11x cupy-cuda12x  # Optional",
            "uv run pytest tests/test_gpu_backend.py -v",
            "uv run demo-gpu"
        ],
        "Task 3 - Chirp Modulation": [
            "uv run pytest tests/test_chirp_modulator.py -v",
            "uv run demo-chirp"
        ],
        "Task 4 - OFDM Structure": [
            "uv run pytest tests/test_subcarrier_manager.py -v", 
            "uv run demo-ofdm"
        ],
        "Task 5 - Signal Generation": [
            "uv run pytest tests/test_ofdm_generator.py -v",
            "uv run demo-generator"
        ]
    }
    
    for task, commands in tasks.items():
        print(f"\n{task}:")
        for cmd in commands:
            print(f"  {cmd}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("UV Development Script for OFDM Chirp Generator")
        print("=" * 50)
        print("Usage: python scripts/dev.py <command> [args]")
        print("\nCommands:")
        print("  setup                    - Set up development environment")
        print("  test [--cov] [test_path] - Run tests (optionally with coverage or specific test)")
        print("  lint                     - Run code quality checks")
        print("  format                   - Format code")
        print("  demo [name]              - Run demos (basic|gpu|chirp|ofdm|generator|all)")
        print("  add <package> [--dev]    - Add dependency")
        print("  remove <package>         - Remove dependency")
        print("  info                     - Show project information")
        print("  tasks                    - Show task-specific commands")
        print("\nExamples:")
        print("  python scripts/dev.py setup")
        print("  python scripts/dev.py test --cov")
        print("  python scripts/dev.py test tests/test_models.py")
        print("  python scripts/dev.py demo basic")
        print("  python scripts/dev.py add matplotlib --dev")
        return
    
    command = sys.argv[1]
    
    if command == "setup":
        success = setup()
    elif command == "test":
        coverage = "--cov" in sys.argv
        specific_test = None
        for arg in sys.argv[2:]:
            if not arg.startswith("--"):
                specific_test = arg
                break
        success = test(coverage=coverage, specific_test=specific_test)
    elif command == "lint":
        success = lint()
    elif command == "format":
        success = format_code()
    elif command == "demo":
        demo_name = sys.argv[2] if len(sys.argv) > 2 else "all"
        success = demo(demo_name)
    elif command == "add":
        if len(sys.argv) < 3:
            print("Error: Package name required")
            sys.exit(1)
        package = sys.argv[2]
        dev = "--dev" in sys.argv
        success = add_dependency(package, dev=dev)
    elif command == "remove":
        if len(sys.argv) < 3:
            print("Error: Package name required")
            sys.exit(1)
        package = sys.argv[2]
        success = remove_dependency(package)
    elif command == "info":
        info()
        success = True
    elif command == "tasks":
        task_commands()
        success = True
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
    
    if not success:
        print(f"\n✗ Command '{command}' failed")
        sys.exit(1)
    else:
        print(f"\n✓ Command '{command}' completed successfully")


if __name__ == "__main__":
    main()