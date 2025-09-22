"""
Configuration management using Dynaconf for centralized parameter handling.

This module provides a centralized configuration system that loads parameters
from TOML files and provides validation and default value management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from dynaconf import Dynaconf

    DYNACONF_AVAILABLE = True
except ImportError:
    DYNACONF_AVAILABLE = False
    Dynaconf = None

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class ConfigurationManager:
    """Centralized configuration manager using Dynaconf.

    This class provides a unified interface for loading and validating
    configuration parameters from TOML files using Dynaconf.
    """

    def __init__(self, config_file: Optional[str] = None, create_default: bool = True):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file (defaults to config.toml)
            create_default: Whether to create default config if file doesn't exist
        """
        if not DYNACONF_AVAILABLE:
            raise ConfigurationError("Dynaconf is not available. Install it with: uv add dynaconf")

        self.config_file = config_file or "config.toml"
        self.config_path = Path(self.config_file)

        # Create default config if it doesn't exist and create_default is True
        if not self.config_path.exists() and create_default:
            self._create_default_config()

        # Initialize Dynaconf
        try:
            self.settings = Dynaconf(
                settings_files=[self.config_file],
                environments=True,
                load_dotenv=True,
                envvar_prefix="OFDM",
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

        # Validate configuration
        self._validate_configuration()

        logger.info(f"Configuration loaded from {self.config_file}")

    def _create_default_config(self) -> None:
        """Create a default configuration file."""
        default_config = """# OFDM Chirp Generator Configuration
# This file contains all configurable parameters for the OFDM signal generation system

[ofdm]
# OFDM signal configuration parameters
num_subcarriers = 8
subcarrier_spacing = 1000.0  # Hz
bandwidth_per_subcarrier = 800.0  # Hz
center_frequency = 10000.0  # Hz
sampling_rate = 50000.0  # Hz
signal_duration = 0.002  # seconds

[chirp]
# Chirp modulation parameters
amplitude = 1.0
phase_offset_range = [0.0, 6.283185307179586]  # [0, 2*pi] radians

[optimization]
# Phase optimization algorithm parameters
max_iterations = 1000
convergence_threshold = 1e-6
orthogonality_target = 0.95
population_size = 50
mutation_rate = 0.1
crossover_rate = 0.8
early_stopping_patience = 50
phase_resolution = 32

[gpu]
# GPU acceleration settings
enable_gpu = true
memory_limit_gb = 4.0
fallback_to_cpu = true
cleanup_memory_after_operations = true

[orthogonality]
# Orthogonality testing parameters
default_threshold = 0.1
correlation_method = "fft"  # "fft" or "direct"
normalize_correlations = true

[separation]
# Signal separation parameters
quality_threshold = 0.8
max_iterations = 100
convergence_tolerance = 1e-6
enable_phase_correction = true
separation_method = "correlation"  # "correlation" or "mmse"

[validation]
# Parameter validation settings
strict_validation = true
allow_parameter_adjustment = false
min_signal_length_samples = 50
max_signal_length_samples = 1000000

[logging]
# Logging configuration
level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
enable_gpu_logging = false
log_performance_metrics = true

[defaults]
# Default values for optional parameters
default_num_signals = 2
default_optimization_method = "hybrid"
default_export_format = "numpy"
"""

        with open(self.config_path, "w") as f:
            f.write(default_config)

        logger.info(f"Created default configuration file: {self.config_file}")

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Validate OFDM parameters
        try:
            ofdm = self.get_ofdm_config()

            if ofdm["num_subcarriers"] <= 0:
                errors.append("ofdm.num_subcarriers must be positive")

            if ofdm["subcarrier_spacing"] <= 0:
                errors.append("ofdm.subcarrier_spacing must be positive")

            if ofdm["bandwidth_per_subcarrier"] <= 0:
                errors.append("ofdm.bandwidth_per_subcarrier must be positive")

            if ofdm["sampling_rate"] <= 0:
                errors.append("ofdm.sampling_rate must be positive")

            if ofdm["signal_duration"] <= 0:
                errors.append("ofdm.signal_duration must be positive")

            # Check Nyquist criterion
            max_frequency = (
                ofdm["center_frequency"]
                + (ofdm["num_subcarriers"] * ofdm["subcarrier_spacing"]) / 2
            )
            if ofdm["sampling_rate"] < 2 * max_frequency:
                errors.append(
                    f"Sampling rate {ofdm['sampling_rate']} violates Nyquist criterion for max frequency {max_frequency}"
                )

        except Exception as e:
            errors.append(f"Error validating OFDM configuration: {e}")

        # Validate optimization parameters
        try:
            opt = self.get_optimization_config()

            if opt["max_iterations"] <= 0:
                errors.append("optimization.max_iterations must be positive")

            if not 0 < opt["orthogonality_target"] <= 1:
                errors.append("optimization.orthogonality_target must be between 0 and 1")

            if opt["population_size"] <= 0:
                errors.append("optimization.population_size must be positive")

            if not 0 <= opt["mutation_rate"] <= 1:
                errors.append("optimization.mutation_rate must be between 0 and 1")

            if not 0 <= opt["crossover_rate"] <= 1:
                errors.append("optimization.crossover_rate must be between 0 and 1")

        except Exception as e:
            errors.append(f"Error validating optimization configuration: {e}")

        # Validate GPU parameters
        try:
            gpu = self.get_gpu_config()

            if gpu["memory_limit_gb"] <= 0:
                errors.append("gpu.memory_limit_gb must be positive")

        except Exception as e:
            errors.append(f"Error validating GPU configuration: {e}")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ConfigurationError(error_msg)

    def get_ofdm_config(self) -> Dict[str, Any]:
        """Get OFDM configuration parameters.

        Returns:
            Dictionary with OFDM configuration
        """
        return {
            "num_subcarriers": self.settings.get("ofdm.num_subcarriers", 8),
            "subcarrier_spacing": float(self.settings.get("ofdm.subcarrier_spacing", 1000.0)),
            "bandwidth_per_subcarrier": float(
                self.settings.get("ofdm.bandwidth_per_subcarrier", 800.0)
            ),
            "center_frequency": float(self.settings.get("ofdm.center_frequency", 10000.0)),
            "sampling_rate": float(self.settings.get("ofdm.sampling_rate", 50000.0)),
            "signal_duration": float(self.settings.get("ofdm.signal_duration", 0.002)),
        }

    def get_chirp_config(self) -> Dict[str, Any]:
        """Get chirp configuration parameters.

        Returns:
            Dictionary with chirp configuration
        """
        return {
            "amplitude": float(self.settings.get("chirp.amplitude", 1.0)),
            "phase_offset_range": list(
                self.settings.get("chirp.phase_offset_range", [0.0, 6.283185307179586])
            ),
        }

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration parameters.

        Returns:
            Dictionary with optimization configuration
        """
        return {
            "max_iterations": int(self.settings.get("optimization.max_iterations", 1000)),
            "convergence_threshold": float(
                self.settings.get("optimization.convergence_threshold", 1e-6)
            ),
            "orthogonality_target": float(
                self.settings.get("optimization.orthogonality_target", 0.95)
            ),
            "population_size": int(self.settings.get("optimization.population_size", 50)),
            "mutation_rate": float(self.settings.get("optimization.mutation_rate", 0.1)),
            "crossover_rate": float(self.settings.get("optimization.crossover_rate", 0.8)),
            "early_stopping_patience": int(
                self.settings.get("optimization.early_stopping_patience", 50)
            ),
            "phase_resolution": int(self.settings.get("optimization.phase_resolution", 32)),
        }

    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration parameters.

        Returns:
            Dictionary with GPU configuration
        """
        return {
            "enable_gpu": bool(self.settings.get("gpu.enable_gpu", True)),
            "memory_limit_gb": float(self.settings.get("gpu.memory_limit_gb", 4.0)),
            "fallback_to_cpu": bool(self.settings.get("gpu.fallback_to_cpu", True)),
            "cleanup_memory_after_operations": bool(
                self.settings.get("gpu.cleanup_memory_after_operations", True)
            ),
        }

    def get_orthogonality_config(self) -> Dict[str, Any]:
        """Get orthogonality testing configuration parameters.

        Returns:
            Dictionary with orthogonality configuration
        """
        return {
            "default_threshold": float(self.settings.get("orthogonality.default_threshold", 0.1)),
            "correlation_method": str(self.settings.get("orthogonality.correlation_method", "fft")),
            "normalize_correlations": bool(
                self.settings.get("orthogonality.normalize_correlations", True)
            ),
        }

    def get_separation_config(self) -> Dict[str, Any]:
        """Get signal separation configuration parameters.

        Returns:
            Dictionary with separation configuration
        """
        return {
            "quality_threshold": float(self.settings.get("separation.quality_threshold", 0.8)),
            "max_iterations": int(self.settings.get("separation.max_iterations", 100)),
            "convergence_tolerance": float(
                self.settings.get("separation.convergence_tolerance", 1e-6)
            ),
            "enable_phase_correction": bool(
                self.settings.get("separation.enable_phase_correction", True)
            ),
            "separation_method": str(
                self.settings.get("separation.separation_method", "correlation")
            ),
        }

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration parameters.

        Returns:
            Dictionary with validation configuration
        """
        return {
            "strict_validation": bool(self.settings.get("validation.strict_validation", True)),
            "allow_parameter_adjustment": bool(
                self.settings.get("validation.allow_parameter_adjustment", False)
            ),
            "min_signal_length_samples": int(
                self.settings.get("validation.min_signal_length_samples", 50)
            ),
            "max_signal_length_samples": int(
                self.settings.get("validation.max_signal_length_samples", 1000000)
            ),
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration parameters.

        Returns:
            Dictionary with logging configuration
        """
        return {
            "level": str(self.settings.get("logging.level", "INFO")),
            "enable_gpu_logging": bool(self.settings.get("logging.enable_gpu_logging", False)),
            "log_performance_metrics": bool(
                self.settings.get("logging.log_performance_metrics", True)
            ),
        }

    def get_defaults_config(self) -> Dict[str, Any]:
        """Get default values configuration.

        Returns:
            Dictionary with default values
        """
        return {
            "default_num_signals": int(self.settings.get("defaults.default_num_signals", 2)),
            "default_optimization_method": str(
                self.settings.get("defaults.default_optimization_method", "hybrid")
            ),
            "default_export_format": str(
                self.settings.get("defaults.default_export_format", "numpy")
            ),
        }

    def get_performance_optimization_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration parameters.

        Returns:
            Dictionary with performance optimization configuration
        """
        # Use existing optimization config as base and add performance-specific settings
        base_config = self.get_optimization_config()

        # Add performance optimization specific settings with defaults
        performance_config = {
            "enable_chunked_processing": True,
            "enable_adaptive_batching": True,
            "enable_memory_monitoring": True,
            "enable_performance_profiling": self.get_logging_config().get(
                "log_performance_metrics", True
            ),
            "chunk_overlap_samples": 0,
            "memory_cleanup_frequency": 10,
            "performance_log_frequency": 50,
            "memory_pressure_threshold": 0.8,
            "auto_fallback_to_cpu": self.get_gpu_config().get("fallback_to_cpu", True),
            "max_retry_attempts": 3,
        }

        # Try to get performance-specific settings from config if they exist
        performance_config.update(
            {
                "enable_chunked_processing": bool(
                    self.settings.get("performance.enable_chunked_processing", True)
                ),
                "enable_adaptive_batching": bool(
                    self.settings.get("performance.enable_adaptive_batching", True)
                ),
                "enable_memory_monitoring": bool(
                    self.settings.get("performance.enable_memory_monitoring", True)
                ),
                "enable_performance_profiling": bool(
                    self.settings.get("performance.enable_performance_profiling", True)
                ),
                "chunk_overlap_samples": int(
                    self.settings.get("performance.chunk_overlap_samples", 0)
                ),
                "memory_cleanup_frequency": int(
                    self.settings.get("performance.memory_cleanup_frequency", 10)
                ),
                "performance_log_frequency": int(
                    self.settings.get("performance.performance_log_frequency", 50)
                ),
                "memory_pressure_threshold": float(
                    self.settings.get("performance.memory_pressure_threshold", 0.8)
                ),
                "auto_fallback_to_cpu": bool(
                    self.settings.get("performance.auto_fallback_to_cpu", True)
                ),
                "max_retry_attempts": int(self.settings.get("performance.max_retry_attempts", 3)),
            }
        )

        return performance_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'ofdm.num_subcarriers')
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        try:
            keys = key.split(".")
            value = self.settings
            for k in keys:
                value = getattr(value, k)
            return value
        except (AttributeError, KeyError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        target = self.settings

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if not hasattr(target, k):
                setattr(target, k, Dynaconf())
            target = getattr(target, k)

        # Set the final value
        setattr(target, keys[-1], value)

    def reload(self) -> None:
        """Reload configuration from file."""
        try:
            self.settings.reload()
            self._validate_configuration()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            raise ConfigurationError(f"Failed to reload configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "ofdm": self.get_ofdm_config(),
            "chirp": self.get_chirp_config(),
            "optimization": self.get_optimization_config(),
            "gpu": self.get_gpu_config(),
            "orthogonality": self.get_orthogonality_config(),
            "separation": self.get_separation_config(),
            "validation": self.get_validation_config(),
            "logging": self.get_logging_config(),
            "defaults": self.get_defaults_config(),
        }

    def create_ofdm_config_object(self):
        """Create OFDMConfig object from configuration.

        Returns:
            OFDMConfig instance
        """
        from .models import OFDMConfig

        ofdm_params = self.get_ofdm_config()
        return OFDMConfig(**ofdm_params)

    def create_optimization_config_object(self):
        """Create OptimizationConfig object from configuration.

        Returns:
            OptimizationConfig instance
        """
        from .phase_optimizer import OptimizationConfig

        opt_params = self.get_optimization_config()
        return OptimizationConfig(**opt_params)

    def validate_signal_length(self, signal_length: int) -> bool:
        """Validate signal length against configuration limits.

        Args:
            signal_length: Signal length in samples

        Returns:
            True if valid, False otherwise
        """
        validation_config = self.get_validation_config()
        min_length = validation_config["min_signal_length_samples"]
        max_length = validation_config["max_signal_length_samples"]

        return min_length <= signal_length <= max_length

    def get_memory_limit_bytes(self) -> int:
        """Get GPU memory limit in bytes.

        Returns:
            Memory limit in bytes
        """
        gpu_config = self.get_gpu_config()
        return int(gpu_config["memory_limit_gb"] * 1024 * 1024 * 1024)

    def __repr__(self) -> str:
        """String representation of ConfigurationManager."""
        return f"ConfigurationManager(config_file='{self.config_file}')"


# Global configuration instance
_global_config: Optional[ConfigurationManager] = None


def get_config(
    config_file: Optional[str] = None, create_default: bool = True
) -> ConfigurationManager:
    """Get global configuration instance.

    Args:
        config_file: Path to configuration file
        create_default: Whether to create default config if file doesn't exist

    Returns:
        ConfigurationManager instance
    """
    global _global_config

    if _global_config is None or config_file is not None:
        _global_config = ConfigurationManager(config_file, create_default)

    return _global_config


def reload_config() -> None:
    """Reload global configuration."""
    global _global_config

    if _global_config is not None:
        _global_config.reload()


def reset_config() -> None:
    """Reset global configuration (force reload on next access)."""
    global _global_config
    _global_config = None
