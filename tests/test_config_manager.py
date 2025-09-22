"""
Tests for configuration management using Dynaconf.

This module tests the ConfigurationManager class and its integration
with the OFDM signal generation system.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ofdm_chirp_generator.config_manager import (
    ConfigurationError,
    ConfigurationManager,
    get_config,
    reload_config,
    reset_config,
)


class TestConfigurationManager:
    """Test cases for ConfigurationManager class."""

    def test_initialization_with_default_config(self, tmp_path):
        """Test ConfigurationManager initialization with default config creation."""
        config_file = tmp_path / "test_config.toml"

        # Should create default config if it doesn't exist
        config_manager = ConfigurationManager(str(config_file), create_default=True)

        assert config_file.exists()
        assert config_manager.config_file == str(config_file)
        assert config_manager.settings is not None

    def test_initialization_without_dynaconf(self, tmp_path):
        """Test ConfigurationManager initialization when Dynaconf is not available."""
        config_file = tmp_path / "test_config.toml"

        with patch("ofdm_chirp_generator.config_manager.DYNACONF_AVAILABLE", False):
            with pytest.raises(ConfigurationError, match="Dynaconf is not available"):
                ConfigurationManager(str(config_file))

    def test_initialization_with_existing_config(self, tmp_path):
        """Test ConfigurationManager initialization with existing config file."""
        config_file = tmp_path / "existing_config.toml"

        # Create a minimal config file
        config_content = """
[ofdm]
num_subcarriers = 4
subcarrier_spacing = 1000.0
bandwidth_per_subcarrier = 800.0
center_frequency = 10000.0
sampling_rate = 50000.0
signal_duration = 0.002

[chirp]
amplitude = 1.0
phase_offset_range = [0.0, 6.283185307179586]

[optimization]
max_iterations = 100
convergence_threshold = 1e-6
orthogonality_target = 0.9
population_size = 20
mutation_rate = 0.1
crossover_rate = 0.8
early_stopping_patience = 10
phase_resolution = 16

[gpu]
enable_gpu = true
memory_limit_gb = 2.0
fallback_to_cpu = true
cleanup_memory_after_operations = true

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

        with open(config_file, "w") as f:
            f.write(config_content)

        config_manager = ConfigurationManager(str(config_file), create_default=False)

        assert config_manager.config_file == str(config_file)
        assert config_manager.settings is not None

    def test_get_ofdm_config(self, tmp_path):
        """Test getting OFDM configuration parameters."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        ofdm_config = config_manager.get_ofdm_config()

        assert isinstance(ofdm_config, dict)
        assert "num_subcarriers" in ofdm_config
        assert "subcarrier_spacing" in ofdm_config
        assert "bandwidth_per_subcarrier" in ofdm_config
        assert "center_frequency" in ofdm_config
        assert "sampling_rate" in ofdm_config
        assert "signal_duration" in ofdm_config

        # Check types
        assert isinstance(ofdm_config["num_subcarriers"], int)
        assert isinstance(ofdm_config["subcarrier_spacing"], float)
        assert isinstance(ofdm_config["bandwidth_per_subcarrier"], float)
        assert isinstance(ofdm_config["center_frequency"], float)
        assert isinstance(ofdm_config["sampling_rate"], float)
        assert isinstance(ofdm_config["signal_duration"], float)

    def test_get_optimization_config(self, tmp_path):
        """Test getting optimization configuration parameters."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        opt_config = config_manager.get_optimization_config()

        assert isinstance(opt_config, dict)
        assert "max_iterations" in opt_config
        assert "convergence_threshold" in opt_config
        assert "orthogonality_target" in opt_config
        assert "population_size" in opt_config
        assert "mutation_rate" in opt_config
        assert "crossover_rate" in opt_config
        assert "early_stopping_patience" in opt_config
        assert "phase_resolution" in opt_config

        # Check types and ranges
        assert isinstance(opt_config["max_iterations"], int)
        assert opt_config["max_iterations"] > 0
        assert isinstance(opt_config["orthogonality_target"], float)
        assert 0 < opt_config["orthogonality_target"] <= 1

    def test_get_gpu_config(self, tmp_path):
        """Test getting GPU configuration parameters."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        gpu_config = config_manager.get_gpu_config()

        assert isinstance(gpu_config, dict)
        assert "enable_gpu" in gpu_config
        assert "memory_limit_gb" in gpu_config
        assert "fallback_to_cpu" in gpu_config
        assert "cleanup_memory_after_operations" in gpu_config

        # Check types
        assert isinstance(gpu_config["enable_gpu"], bool)
        assert isinstance(gpu_config["memory_limit_gb"], float)
        assert isinstance(gpu_config["fallback_to_cpu"], bool)
        assert isinstance(gpu_config["cleanup_memory_after_operations"], bool)

    def test_get_orthogonality_config(self, tmp_path):
        """Test getting orthogonality configuration parameters."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        orth_config = config_manager.get_orthogonality_config()

        assert isinstance(orth_config, dict)
        assert "default_threshold" in orth_config
        assert "correlation_method" in orth_config
        assert "normalize_correlations" in orth_config

        # Check types
        assert isinstance(orth_config["default_threshold"], float)
        assert isinstance(orth_config["correlation_method"], str)
        assert isinstance(orth_config["normalize_correlations"], bool)

    def test_get_validation_config(self, tmp_path):
        """Test getting validation configuration parameters."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        val_config = config_manager.get_validation_config()

        assert isinstance(val_config, dict)
        assert "strict_validation" in val_config
        assert "allow_parameter_adjustment" in val_config
        assert "min_signal_length_samples" in val_config
        assert "max_signal_length_samples" in val_config

        # Check types
        assert isinstance(val_config["strict_validation"], bool)
        assert isinstance(val_config["allow_parameter_adjustment"], bool)
        assert isinstance(val_config["min_signal_length_samples"], int)
        assert isinstance(val_config["max_signal_length_samples"], int)

    def test_get_and_set_methods(self, tmp_path):
        """Test get and set methods for configuration values."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        # Test get method
        num_subcarriers = config_manager.get("ofdm.num_subcarriers")
        assert isinstance(num_subcarriers, int)

        # Test get with default
        non_existent = config_manager.get("non.existent.key", "default_value")
        assert non_existent == "default_value"

        # Test set method
        config_manager.set("ofdm.num_subcarriers", 16)
        updated_value = config_manager.get("ofdm.num_subcarriers")
        assert updated_value == 16

    def test_validation_errors(self, tmp_path):
        """Test configuration validation with invalid parameters."""
        config_file = tmp_path / "invalid_config.toml"

        # Create invalid config
        invalid_config = """
[ofdm]
num_subcarriers = -1  # Invalid: negative
subcarrier_spacing = 1000.0
bandwidth_per_subcarrier = 800.0
center_frequency = 10000.0
sampling_rate = 50000.0
signal_duration = 0.002

[chirp]
amplitude = 1.0
phase_offset_range = [0.0, 6.283185307179586]

[optimization]
max_iterations = 100
convergence_threshold = 1e-6
orthogonality_target = 1.5  # Invalid: > 1
population_size = 20
mutation_rate = 0.1
crossover_rate = 0.8
early_stopping_patience = 10
phase_resolution = 16

[gpu]
enable_gpu = true
memory_limit_gb = -1.0  # Invalid: negative
fallback_to_cpu = true
cleanup_memory_after_operations = true

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

        with open(config_file, "w") as f:
            f.write(invalid_config)

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            ConfigurationManager(str(config_file), create_default=False)

    def test_create_ofdm_config_object(self, tmp_path):
        """Test creating OFDMConfig object from configuration."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        ofdm_config = config_manager.create_ofdm_config_object()

        from ofdm_chirp_generator.models import OFDMConfig

        assert isinstance(ofdm_config, OFDMConfig)
        assert ofdm_config.num_subcarriers > 0
        assert ofdm_config.sampling_rate > 0

    def test_create_optimization_config_object(self, tmp_path):
        """Test creating OptimizationConfig object from configuration."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        opt_config = config_manager.create_optimization_config_object()

        from ofdm_chirp_generator.phase_optimizer import OptimizationConfig

        assert isinstance(opt_config, OptimizationConfig)
        assert opt_config.max_iterations > 0
        assert 0 < opt_config.orthogonality_target <= 1

    def test_validate_signal_length(self, tmp_path):
        """Test signal length validation."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        # Valid signal length
        assert config_manager.validate_signal_length(100) is True

        # Invalid signal lengths
        assert config_manager.validate_signal_length(10) is False  # Too short
        assert config_manager.validate_signal_length(2000000) is False  # Too long

    def test_get_memory_limit_bytes(self, tmp_path):
        """Test getting memory limit in bytes."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        memory_limit = config_manager.get_memory_limit_bytes()

        assert isinstance(memory_limit, int)
        assert memory_limit > 0

        # Should be 4GB in bytes by default
        expected_bytes = 4 * 1024 * 1024 * 1024
        assert memory_limit == expected_bytes

    def test_to_dict(self, tmp_path):
        """Test converting configuration to dictionary."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        config_dict = config_manager.to_dict()

        assert isinstance(config_dict, dict)
        assert "ofdm" in config_dict
        assert "chirp" in config_dict
        assert "optimization" in config_dict
        assert "gpu" in config_dict
        assert "orthogonality" in config_dict
        assert "validation" in config_dict
        assert "logging" in config_dict
        assert "defaults" in config_dict

    def test_reload_configuration(self, tmp_path):
        """Test reloading configuration from file."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        # Get initial value
        initial_value = config_manager.get("ofdm.num_subcarriers")

        # Modify the config file
        config_content = config_file.read_text()
        modified_content = config_content.replace(
            f"num_subcarriers = {initial_value}", f"num_subcarriers = {initial_value + 1}"
        )
        config_file.write_text(modified_content)

        # Reload configuration
        config_manager.reload()

        # Check that value was updated
        updated_value = config_manager.get("ofdm.num_subcarriers")
        assert updated_value == initial_value + 1

    def test_repr(self, tmp_path):
        """Test string representation of ConfigurationManager."""
        config_file = tmp_path / "test_config.toml"
        config_manager = ConfigurationManager(str(config_file))

        repr_str = repr(config_manager)

        assert "ConfigurationManager" in repr_str
        assert str(config_file) in repr_str


class TestGlobalConfiguration:
    """Test cases for global configuration functions."""

    def test_get_config_singleton(self, tmp_path):
        """Test that get_config returns singleton instance."""
        config_file = tmp_path / "global_config.toml"

        # Reset global config first
        reset_config()

        # Get config instances
        config1 = get_config(str(config_file))
        config2 = get_config()  # Should return same instance

        assert config1 is config2
        assert config1.config_file == str(config_file)

    def test_reload_config(self, tmp_path):
        """Test reloading global configuration."""
        config_file = tmp_path / "global_config.toml"

        # Reset and get config
        reset_config()
        config = get_config(str(config_file))

        initial_value = config.get("ofdm.num_subcarriers")

        # Modify config file
        config_content = config_file.read_text()
        modified_content = config_content.replace(
            f"num_subcarriers = {initial_value}", f"num_subcarriers = {initial_value + 2}"
        )
        config_file.write_text(modified_content)

        # Reload global config
        reload_config()

        # Check updated value
        updated_value = config.get("ofdm.num_subcarriers")
        assert updated_value == initial_value + 2

    def test_reset_config(self, tmp_path):
        """Test resetting global configuration."""
        config_file = tmp_path / "global_config.toml"

        # Get initial config
        config1 = get_config(str(config_file))

        # Reset config
        reset_config()

        # Get new config - should be different instance
        config2 = get_config(str(config_file))

        # Should be different instances but same config file
        assert config1 is not config2
        assert config1.config_file == config2.config_file


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration with OFDM components."""

    def test_ofdm_generator_with_config(self, tmp_path):
        """Test OFDMGenerator integration with configuration."""
        config_file = tmp_path / "integration_config.toml"

        # Create config manager
        config_manager = ConfigurationManager(str(config_file))

        # Create OFDM config from configuration
        ofdm_config = config_manager.create_ofdm_config_object()

        # Test that OFDMGenerator can be created with config
        from ofdm_chirp_generator import GPUBackend, OFDMGenerator

        gpu_backend = GPUBackend()
        generator = OFDMGenerator(ofdm_config, gpu_backend)

        assert generator.ofdm_config == ofdm_config
        assert generator.ofdm_config.num_subcarriers == config_manager.get("ofdm.num_subcarriers")

    def test_phase_optimizer_with_config(self, tmp_path):
        """Test PhaseOptimizer integration with configuration."""
        config_file = tmp_path / "integration_config.toml"

        # Create config manager
        config_manager = ConfigurationManager(str(config_file))

        # Create configs from configuration
        ofdm_config = config_manager.create_ofdm_config_object()
        opt_config = config_manager.create_optimization_config_object()

        # Test that PhaseOptimizer can use the configs
        from ofdm_chirp_generator import GPUBackend, PhaseOptimizer

        gpu_backend = GPUBackend()
        optimizer = PhaseOptimizer(ofdm_config, gpu_backend)

        # Run optimization with config parameters
        result = optimizer.find_orthogonal_phases(2, opt_config, method="genetic")

        assert result.iterations <= opt_config.max_iterations
        assert result.method_used == "genetic"

    def test_configuration_parameter_propagation(self, tmp_path):
        """Test that configuration parameters properly propagate through the system."""
        config_file = tmp_path / "propagation_config.toml"

        # Create custom config with specific values
        custom_config = """
[ofdm]
num_subcarriers = 6
subcarrier_spacing = 1500.0
bandwidth_per_subcarrier = 1200.0
center_frequency = 15000.0
sampling_rate = 75000.0
signal_duration = 0.003

[chirp]
amplitude = 2.0
phase_offset_range = [0.0, 6.283185307179586]

[optimization]
max_iterations = 50
convergence_threshold = 1e-5
orthogonality_target = 0.85
population_size = 15
mutation_rate = 0.15
crossover_rate = 0.75
early_stopping_patience = 8
phase_resolution = 24

[gpu]
enable_gpu = true
memory_limit_gb = 2.0
fallback_to_cpu = true
cleanup_memory_after_operations = true

[orthogonality]
default_threshold = 0.15
correlation_method = "fft"
normalize_correlations = true

[validation]
strict_validation = true
allow_parameter_adjustment = false
min_signal_length_samples = 100
max_signal_length_samples = 50000

[logging]
level = "DEBUG"
enable_gpu_logging = true
log_performance_metrics = true

[defaults]
default_num_signals = 3
default_optimization_method = "gradient"
default_export_format = "numpy"
"""

        with open(config_file, "w") as f:
            f.write(custom_config)

        # Create config manager
        config_manager = ConfigurationManager(str(config_file), create_default=False)

        # Verify all custom values are loaded correctly
        ofdm_config = config_manager.get_ofdm_config()
        assert ofdm_config["num_subcarriers"] == 6
        assert ofdm_config["subcarrier_spacing"] == 1500.0
        assert ofdm_config["center_frequency"] == 15000.0

        opt_config = config_manager.get_optimization_config()
        assert opt_config["max_iterations"] == 50
        assert opt_config["orthogonality_target"] == 0.85
        assert opt_config["population_size"] == 15

        gpu_config = config_manager.get_gpu_config()
        assert gpu_config["memory_limit_gb"] == 2.0

        defaults_config = config_manager.get_defaults_config()
        assert defaults_config["default_num_signals"] == 3
        assert defaults_config["default_optimization_method"] == "gradient"
