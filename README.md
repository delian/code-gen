# OFDM Chirp Signal Generator

A GPU-accelerated OFDM (Orthogonal Frequency Division Multiplexing) signal generator that creates signals with chirp-modulated subcarriers. The system generates multiple orthogonal OFDM signals that can be transmitted simultaneously and later separated through phase analysis.

## Features

- **GPU Acceleration**: Uses CuPy for high-performance GPU computation with automatic CPU fallback
- **Chirp Modulation**: Each OFDM subcarrier carries a linear frequency modulated (chirp) signal
- **Orthogonal Signals**: Generates multiple orthogonal OFDM signals for simultaneous transmission
- **Configurable Parameters**: Flexible configuration of subcarriers, spacing, bandwidth, and frequencies
- **Signal Analysis**: Built-in tools for signal analysis and orthogonality verification
- **Memory Management**: Efficient GPU memory management with automatic cleanup

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for dependency management and virtual environments.

### Prerequisites

First, install UV:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Project Setup

Clone the repository and set up the development environment:

```bash
git clone <repository-url>
cd ofdm-chirp-generator

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Installation Options

Install with different feature sets:

```bash
# Basic installation (CPU only)
uv pip install .

# With GPU support
uv pip install ".[gpu]"

# With visualization support
uv pip install ".[visualization]"

# Development installation with all features
uv pip install ".[all]"

# Or install in development mode
uv pip install -e ".[all]"
```

## Quick Start

```python
from ofdm_chirp_generator import OFDMConfig, OFDMGenerator
import numpy as np

# Create configuration
config = OFDMConfig(
    num_subcarriers=16,
    subcarrier_spacing=1000.0,  # Hz
    bandwidth_per_subcarrier=800.0,  # Hz
    center_frequency=10000.0,  # Hz
    sampling_rate=50000.0,  # Hz
    signal_duration=0.01  # seconds
)

# Generate OFDM signal
with OFDMGenerator(config) as generator:
    # Create phase array for subcarriers
    phases = np.linspace(0, 2*np.pi, config.num_subcarriers, endpoint=False)
    
    # Generate signal
    signal = generator.generate_single_signal(phases)
    
    # Analyze signal properties
    analysis = generator.analyze_generated_signal(signal)
    print(f"Signal power: {analysis['signal_power']:.4f}")
    print(f"PAPR: {analysis['papr_db']:.2f} dB")
```

## Examples

Run the included examples using UV:

```bash
# Run examples directly
uv run examples/basic_usage.py
uv run examples/chirp_modulator_demo.py
uv run examples/ofdm_structure_demo.py
uv run examples/gpu_backend_demo.py
uv run examples/ofdm_generator_demo.py

# Or use the installed script
uv run ofdm-demo
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ofdm_chirp_generator --cov-report=html

# Run specific test categories
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m "not gpu"   # Skip GPU tests
```

### Code Quality

```bash
# Format code
uv run black ofdm_chirp_generator/ tests/ examples/

# Sort imports
uv run isort ofdm_chirp_generator/ tests/ examples/

# Lint code
uv run flake8 ofdm_chirp_generator/ tests/ examples/

# Type checking
uv run mypy ofdm_chirp_generator/
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add numpy

# Add development dependency
uv add --dev pytest

# Add optional dependency group
uv add --optional gpu cupy-cuda12x
```

### Virtual Environment Management

```bash
# Create/sync environment
uv sync

# Update dependencies
uv lock --upgrade

# Remove environment
uv venv --rm

# Show environment info
uv pip list
```

## Architecture

The system consists of several key components:

- **OFDMGenerator**: Main orchestrator for signal generation
- **ChirpModulator**: Generates chirp signals for individual subcarriers
- **SubcarrierManager**: Manages frequency allocation and OFDM structure
- **GPUBackend**: Handles GPU acceleration and memory management
- **ConfigValidator**: Validates system parameters and configurations

## Requirements

- Python 3.9+
- NumPy
- CuPy (optional, for GPU acceleration)
- Matplotlib (optional, for visualization)

## UV Workflow

This project is designed to work seamlessly with UV:

1. **Environment Management**: UV automatically creates and manages virtual environments
2. **Dependency Resolution**: Fast dependency resolution and installation
3. **Lock Files**: Reproducible builds with `uv.lock`
4. **Script Running**: Direct script execution with `uv run`
5. **Development Tools**: Integrated development dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.