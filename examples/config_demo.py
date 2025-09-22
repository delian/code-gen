#!/usr/bin/env python3
"""
Configuration Management Demo

This script demonstrates the centralized configuration management system
using Dynaconf and TOML files for the OFDM chirp generator.
"""

import os
import tempfile
from pathlib import Path

# Import configuration management
from ofdm_chirp_generator import (
    ConfigurationError,
    ConfigurationManager,
    GPUBackend,
    OFDMGenerator,
    PhaseOptimizer,
    get_config,
)


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("OFDM CHIRP GENERATOR - CONFIGURATION MANAGEMENT DEMO")
    print("=" * 60)

    # Demonstrate default configuration creation
    print("1. DEFAULT CONFIGURATION CREATION:")
    print("-" * 40)

    # Use the existing config.toml or create a new one
    config_file = "config.toml"
    if not Path(config_file).exists():
        print(f"Creating default configuration file: {config_file}")
    else:
        print(f"Using existing configuration file: {config_file}")

    try:
        config_manager = ConfigurationManager(config_file)
        print(f"✓ Configuration loaded successfully from {config_file}")
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        return
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print("Note: This demo requires Dynaconf. Install with: uv add dynaconf")
        return

    print()

    # Display configuration sections
    print("2. CONFIGURATION SECTIONS:")
    print("-" * 40)

    sections = [
        ("OFDM", config_manager.get_ofdm_config),
        ("Chirp", config_manager.get_chirp_config),
        ("Optimization", config_manager.get_optimization_config),
        ("GPU", config_manager.get_gpu_config),
        ("Orthogonality", config_manager.get_orthogonality_config),
        ("Validation", config_manager.get_validation_config),
        ("Logging", config_manager.get_logging_config),
        ("Defaults", config_manager.get_defaults_config),
    ]

    for section_name, get_config_func in sections:
        try:
            section_config = get_config_func()
            print(f"{section_name} Configuration:")
            for key, value in section_config.items():
                print(f"  {key}: {value}")
            print()
        except Exception as e:
            print(f"✗ Error loading {section_name} configuration: {e}")

    # Demonstrate configuration access methods
    print("3. CONFIGURATION ACCESS METHODS:")
    print("-" * 40)

    # Direct access
    num_subcarriers = config_manager.get("ofdm.num_subcarriers")
    print(f"Direct access - ofdm.num_subcarriers: {num_subcarriers}")

    # Access with default
    custom_param = config_manager.get("custom.parameter", "default_value")
    print(f"Access with default - custom.parameter: {custom_param}")

    # Set and get
    config_manager.set("demo.test_parameter", 42)
    test_param = config_manager.get("demo.test_parameter")
    print(f"Set and get - demo.test_parameter: {test_param}")
    print()

    # Demonstrate configuration validation
    print("4. CONFIGURATION VALIDATION:")
    print("-" * 40)

    # Test signal length validation
    test_lengths = [25, 100, 500, 2000000]
    for length in test_lengths:
        is_valid = config_manager.validate_signal_length(length)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"Signal length {length}: {status}")

    # Memory limit
    memory_limit = config_manager.get_memory_limit_bytes()
    print(f"GPU memory limit: {memory_limit:,} bytes ({memory_limit / (1024**3):.1f} GB)")
    print()

    # Demonstrate integration with OFDM components
    print("5. INTEGRATION WITH OFDM COMPONENTS:")
    print("-" * 40)

    try:
        # Create OFDM configuration from config manager
        ofdm_config = config_manager.create_ofdm_config_object()
        print(
            f"✓ Created OFDMConfig: {ofdm_config.num_subcarriers} subcarriers, {ofdm_config.sampling_rate} Hz"
        )

        # Create optimization configuration
        opt_config = config_manager.create_optimization_config_object()
        print(
            f"✓ Created OptimizationConfig: {opt_config.max_iterations} max iterations, target {opt_config.orthogonality_target}"
        )

        # Initialize GPU backend
        gpu_backend = GPUBackend()
        print(f"✓ GPU Backend: {'GPU' if gpu_backend.is_gpu_available else 'CPU'}")

        # Create OFDM generator with configuration
        generator = OFDMGenerator(ofdm_config, gpu_backend)
        print(f"✓ Created OFDMGenerator with {generator.ofdm_config.num_subcarriers} subcarriers")

        # Create phase optimizer with configuration
        optimizer = PhaseOptimizer(ofdm_config, gpu_backend)
        print(f"✓ Created PhaseOptimizer")

        print()

        # Demonstrate signal generation with configuration
        print("6. SIGNAL GENERATION WITH CONFIGURATION:")
        print("-" * 40)

        # Generate signals using configuration parameters
        num_signals = config_manager.get_defaults_config()["default_num_signals"]
        method = config_manager.get_defaults_config()["default_optimization_method"]

        print(f"Generating {num_signals} signals using {method} optimization...")

        # Use a smaller configuration for demo
        demo_opt_config = config_manager.create_optimization_config_object()
        demo_opt_config.max_iterations = 10  # Reduce for demo speed

        result = optimizer.find_orthogonal_phases(num_signals, demo_opt_config, method=method)

        print(f"✓ Optimization completed:")
        print(f"  Method: {result.method_used}")
        print(f"  Orthogonality score: {result.orthogonality_score:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Time: {result.optimization_time:.3f}s")
        print(f"  Converged: {'Yes' if result.converged else 'No'}")

        # Generate signal set
        signal_set = generator.create_signal_set(result.optimal_phases, result.orthogonality_score)
        print(f"✓ Generated signal set with {signal_set.num_signals} signals")
        print(f"  Signal length: {signal_set.signal_length} samples")
        print(f"  Generation time: {signal_set.generation_timestamp}")

        print()

    except Exception as e:
        print(f"✗ Integration error: {e}")
        print("This may be due to missing dependencies or configuration issues.")

    # Demonstrate configuration modification and reload
    print("7. CONFIGURATION MODIFICATION:")
    print("-" * 40)

    try:
        # Show current value
        current_iterations = config_manager.get("optimization.max_iterations")
        print(f"Current max_iterations: {current_iterations}")

        # Modify configuration
        new_iterations = current_iterations + 100
        config_manager.set("optimization.max_iterations", new_iterations)
        updated_iterations = config_manager.get("optimization.max_iterations")
        print(f"Updated max_iterations: {updated_iterations}")

        # Convert to dictionary
        config_dict = config_manager.to_dict()
        print(f"✓ Configuration converted to dictionary with {len(config_dict)} sections")

        print()

    except Exception as e:
        print(f"✗ Modification error: {e}")

    # Demonstrate global configuration
    print("8. GLOBAL CONFIGURATION:")
    print("-" * 40)

    try:
        # Get global configuration instance
        global_config = get_config()
        print(f"✓ Global configuration loaded from: {global_config.config_file}")

        # Show that it's the same instance
        is_same = global_config is config_manager
        print(f"Same as local instance: {'Yes' if is_same else 'No'}")

        # Access global configuration
        global_num_subcarriers = global_config.get("ofdm.num_subcarriers")
        print(f"Global config num_subcarriers: {global_num_subcarriers}")

        print()

    except Exception as e:
        print(f"✗ Global configuration error: {e}")

    # Demonstrate custom configuration file
    print("9. CUSTOM CONFIGURATION FILE:")
    print("-" * 40)

    try:
        # Create a temporary custom configuration
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            custom_config = """
[ofdm]
num_subcarriers = 12
subcarrier_spacing = 2000.0
bandwidth_per_subcarrier = 1600.0
center_frequency = 20000.0
sampling_rate = 100000.0
signal_duration = 0.004

[optimization]
max_iterations = 200
orthogonality_target = 0.98
population_size = 30

[gpu]
enable_gpu = false
memory_limit_gb = 8.0

[defaults]
default_num_signals = 4
default_optimization_method = "genetic"
"""
            f.write(custom_config)
            custom_config_file = f.name

        # Load custom configuration
        custom_config_manager = ConfigurationManager(custom_config_file, create_default=False)

        custom_ofdm = custom_config_manager.get_ofdm_config()
        print(f"✓ Custom configuration loaded:")
        print(f"  Subcarriers: {custom_ofdm['num_subcarriers']}")
        print(f"  Sampling rate: {custom_ofdm['sampling_rate']} Hz")
        print(f"  Signal duration: {custom_ofdm['signal_duration']} s")

        custom_defaults = custom_config_manager.get_defaults_config()
        print(f"  Default signals: {custom_defaults['default_num_signals']}")
        print(f"  Default method: {custom_defaults['default_optimization_method']}")

        # Clean up
        os.unlink(custom_config_file)
        print(f"✓ Custom configuration file cleaned up")

        print()

    except Exception as e:
        print(f"✗ Custom configuration error: {e}")

    print("Configuration management demonstration completed!")
    print()
    print("Key Benefits:")
    print("- Centralized parameter management")
    print("- TOML-based human-readable configuration")
    print("- Automatic validation and error reporting")
    print("- Easy integration with existing components")
    print("- Support for environment variables and multiple config files")
    print("- Runtime configuration modification and reloading")


def demonstrate_error_handling():
    """Demonstrate configuration error handling."""
    print("\nERROR HANDLING DEMONSTRATION:")
    print("=" * 50)

    # Test with invalid configuration
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        invalid_config = """
[ofdm]
num_subcarriers = -5  # Invalid: negative
subcarrier_spacing = 1000.0
bandwidth_per_subcarrier = 800.0
center_frequency = 10000.0
sampling_rate = 1000.0  # Invalid: too low for Nyquist
signal_duration = 0.002

[optimization]
max_iterations = 100
orthogonality_target = 1.5  # Invalid: > 1
population_size = 20
mutation_rate = 0.1
crossover_rate = 0.8
early_stopping_patience = 10
phase_resolution = 16

[gpu]
memory_limit_gb = -2.0  # Invalid: negative

[orthogonality]
default_threshold = 0.1
correlation_method = "fft"
normalize_correlations = true

[validation]
strict_validation = true
allow_parameter_adjustment = false
min_signal_length_samples = 50
max_signal_length_samples = 100000

[logging]
level = "INFO"
enable_gpu_logging = false
log_performance_metrics = true

[defaults]
default_num_signals = 2
default_optimization_method = "hybrid"
default_export_format = "numpy"
"""
        f.write(invalid_config)
        invalid_config_file = f.name

    try:
        print("Attempting to load invalid configuration...")
        ConfigurationManager(invalid_config_file, create_default=False)
        print("✗ Expected validation error but none occurred")
    except ConfigurationError as e:
        print(f"✓ Caught expected configuration error:")
        print(f"  {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        # Clean up
        os.unlink(invalid_config_file)

    print("\nError handling demonstration completed!")


if __name__ == "__main__":
    main()
    demonstrate_error_handling()
