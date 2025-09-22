"""
Main interface and high-level API for OFDM Chirp Signal Generator.

This module provides the primary entry point and high-level API that integrates
all components of the OFDM chirp signal generation system. It offers convenient
methods for common use cases while abstracting away low-level configuration details.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config_manager import ConfigurationError, ConfigurationManager, get_config
from .error_handling import ErrorHandler, create_error_context
from .gpu_backend import GPUBackend
from .models import ChirpConfig, OFDMConfig, SignalSet
from .ofdm_generator import OFDMGenerator
from .orthogonal_signal_generator import OrthogonalSignalGenerator, PhaseMatrixManager
from .orthogonality_tester import OrthogonalityTester
from .phase_optimizer import OptimizationConfig, PhaseOptimizer
from .signal_export import PhaseConfigurationManager, SignalExporter, SignalVisualizer
from .signal_separator import SeparationQualityMetrics, SignalSeparator

logger = logging.getLogger(__name__)


class OFDMChirpGenerator:
    """Main interface for OFDM chirp signal generation system.

    This class provides a high-level API that integrates all components of the
    OFDM chirp signal generation system. It offers convenient methods for common
    use cases while handling configuration management and resource cleanup automatically.

    Requirements addressed:
    - All requirements integration: Provides unified interface to all system capabilities
    - 1.1-1.4: OFDM signal generation with configurable parameters
    - 2.1-2.4: Chirp modulation with phase control
    - 3.1-3.3: Multiple orthogonal signal generation
    - 4.1-4.4: GPU acceleration with fallback
    - 5.1-5.4: Phase optimization and orthogonality discovery
    - 6.1-6.4: Signal separation capabilities
    - 7.1-7.4: Configuration management
    - 8.1-8.6: UV-based development environment
    - 9.1-9.6: Centralized TOML configuration
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        ofdm_config: Optional[OFDMConfig] = None,
        enable_gpu: Optional[bool] = None,
        create_default_config: bool = True,
    ):
        """Initialize the OFDM chirp generator.

        Args:
            config_file: Path to configuration file (uses config.toml if None)
            ofdm_config: OFDM configuration object (loads from config if None)
            enable_gpu: Force GPU enable/disable (uses config setting if None)
            create_default_config: Create default config file if it doesn't exist

        Raises:
            ConfigurationError: If configuration cannot be loaded
            RuntimeError: If system initialization fails
        """
        self._error_handler = ErrorHandler()

        try:
            # Load configuration
            self.config_manager = get_config(config_file, create_default_config)
            logger.info(f"Loaded configuration from {self.config_manager.config_file}")

            # Set up OFDM configuration
            if ofdm_config is not None:
                self.ofdm_config = ofdm_config
            else:
                self.ofdm_config = self.config_manager.create_ofdm_config_object()

            # Initialize GPU backend
            gpu_config = self.config_manager.get_gpu_config()
            if enable_gpu is not None:
                gpu_config["enable_gpu"] = enable_gpu

            self.gpu_backend = GPUBackend(
                force_cpu=not gpu_config["enable_gpu"] if enable_gpu is None else not enable_gpu,
                config_file=config_file,
            )

            # Initialize core components
            self.ofdm_generator = OFDMGenerator(self.ofdm_config, self.gpu_backend)
            self.orthogonal_generator = OrthogonalSignalGenerator(
                self.ofdm_config, self.gpu_backend, config_file
            )
            self.signal_separator = SignalSeparator(self.ofdm_config, self.gpu_backend, config_file)
            self.phase_optimizer = PhaseOptimizer(self.ofdm_config, self.gpu_backend, config_file)
            self.orthogonality_tester = OrthogonalityTester(self.gpu_backend, config_file)

            # Initialize utility components
            self.signal_exporter = SignalExporter()
            self.phase_manager = PhaseMatrixManager()

            # System state
            self._initialized = True
            self._last_generated_set: Optional[SignalSet] = None

            logger.info(
                f"OFDMChirpGenerator initialized successfully: "
                f"{self.ofdm_config.num_subcarriers} subcarriers, "
                f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}"
            )

        except Exception as e:
            context = create_error_context("system_initialization", "OFDMChirpGenerator")
            self._error_handler.handle_error(e, context)
            raise RuntimeError(f"Failed to initialize OFDM chirp generator: {e}")

    def generate_single_signal(
        self, phases: Optional[np.ndarray] = None, signal_index: int = 0
    ) -> SignalSet:
        """Generate a single OFDM signal with chirp modulation.

        This is a convenience method that generates a single signal and returns
        it wrapped in a SignalSet for consistency with the API.

        Args:
            phases: Phase array for subcarriers (generates example if None)
            signal_index: Index for generating example phases if phases is None

        Returns:
            SignalSet containing the single generated signal

        Raises:
            ValueError: If phases array is invalid
        """
        if phases is None:
            phases = self.ofdm_generator.get_example_phase_array(signal_index)

        # Generate the signal
        signal = self.ofdm_generator.generate_single_signal(phases)

        # Create SignalSet with single signal
        signal_set = SignalSet(
            signals=[self.gpu_backend.to_cpu(signal)],
            phases=phases.reshape(1, -1),  # Make it 2D for consistency
            orthogonality_score=1.0,  # Single signal is perfectly orthogonal to itself
            generation_timestamp=datetime.now(),
            config=self.ofdm_config,
            metadata={
                "generation_method": "single_signal",
                "signal_index": signal_index,
                "backend": "GPU" if self.gpu_backend.is_gpu_available else "CPU",
            },
        )

        self._last_generated_set = signal_set
        logger.info("Generated single OFDM signal")

        return signal_set

    def generate_orthogonal_set(
        self,
        num_signals: int,
        optimization_method: str = "auto",
        orthogonality_threshold: Optional[float] = None,
        force_regenerate: bool = False,
    ) -> SignalSet:
        """Generate a set of orthogonal OFDM signals.

        This is the primary method for generating multiple orthogonal signals
        that can be transmitted simultaneously and later separated.

        Args:
            num_signals: Number of orthogonal signals to generate (2-16)
            optimization_method: Optimization method ('auto', 'genetic', 'brute_force', 'hybrid')
            orthogonality_threshold: Minimum orthogonality score (uses config if None)
            force_regenerate: Force regeneration even if cached result exists

        Returns:
            SignalSet containing orthogonal signals

        Raises:
            ValueError: If parameters are invalid
        """
        if num_signals < 2:
            raise ValueError("Need at least 2 signals for orthogonal set")

        # Use configuration defaults if not specified
        if orthogonality_threshold is None:
            orth_config = self.config_manager.get_orthogonality_config()
            orthogonality_threshold = orth_config.get("default_threshold", 0.95)

        if optimization_method == "auto":
            defaults_config = self.config_manager.get_defaults_config()
            optimization_method = defaults_config.get("default_optimization_method", "hybrid")

        logger.info(
            f"Generating orthogonal set: {num_signals} signals, "
            f"method={optimization_method}, threshold={orthogonality_threshold}"
        )

        # Generate orthogonal signal set
        signal_set = self.orthogonal_generator.generate_orthogonal_signal_set(
            num_signals, force_regenerate
        )

        # Validate orthogonality meets threshold
        if signal_set.orthogonality_score < orthogonality_threshold:
            logger.warning(
                f"Generated set orthogonality {signal_set.orthogonality_score:.4f} "
                f"below threshold {orthogonality_threshold:.4f}"
            )

        self._last_generated_set = signal_set
        logger.info(
            f"Generated orthogonal set: {num_signals} signals, "
            f"score={signal_set.orthogonality_score:.4f}"
        )

        return signal_set

    def separate_signals(
        self,
        combined_signal: Union[np.ndarray, SignalSet],
        reference_set: Optional[SignalSet] = None,
    ) -> Tuple[SignalSet, SeparationQualityMetrics]:
        """Separate overlapping orthogonal signals.

        This method demonstrates the signal separation capabilities by taking
        a combined signal and separating it into individual components.

        Args:
            combined_signal: Combined signal array or SignalSet to separate
            reference_set: Reference SignalSet for separation (uses last generated if None)

        Returns:
            Tuple of (separated_SignalSet, quality_metrics)

        Raises:
            ValueError: If reference signals are not available
        """
        # Handle different input types
        if isinstance(combined_signal, SignalSet):
            # If SignalSet provided, combine its signals first
            combined_array = self.combine_signal_set(combined_signal)
        else:
            combined_array = combined_signal

        # Use reference set
        if reference_set is None:
            if self._last_generated_set is None:
                raise ValueError(
                    "No reference signal set available. Generate signals first or provide reference_set."
                )
            reference_set = self._last_generated_set

        logger.info(
            f"Separating combined signal using {len(reference_set.signals)} reference signals"
        )

        # Perform separation
        separated_set, quality_metrics = self.signal_separator.separate_signal_set(
            combined_array, reference_set
        )

        logger.info(
            f"Signal separation completed with quality score: {quality_metrics.overall_separation_quality:.4f}"
        )

        return separated_set, quality_metrics

    def combine_signal_set(
        self, signal_set: SignalSet, weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Combine multiple signals into a single overlapping signal.

        This is a utility method for creating combined signals that can be
        used to test separation capabilities.

        Args:
            signal_set: SignalSet containing signals to combine
            weights: Weighting factors for each signal (equal weights if None)

        Returns:
            Combined signal array
        """
        if len(signal_set.signals) == 0:
            raise ValueError("SignalSet contains no signals")

        # Set up weights
        if weights is None:
            weights = [1.0] * len(signal_set.signals)
        elif len(weights) != len(signal_set.signals):
            raise ValueError("Number of weights must match number of signals")

        # Combine signals
        combined = np.zeros_like(signal_set.signals[0], dtype=complex)
        for signal, weight in zip(signal_set.signals, weights):
            combined += weight * signal

        logger.info(f"Combined {len(signal_set.signals)} signals with weights {weights}")

        return combined

    def analyze_signal_set(self, signal_set: SignalSet) -> Dict[str, any]:
        """Perform comprehensive analysis of a signal set.

        Args:
            signal_set: SignalSet to analyze

        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Analyzing signal set with {len(signal_set.signals)} signals")

        # Get orthogonal signal analysis
        orthogonal_analysis = self.orthogonal_generator.analyze_orthogonal_set(signal_set)

        # Add system-level analysis
        system_analysis = {
            "system_info": {
                "backend": "GPU" if self.gpu_backend.is_gpu_available else "CPU",
                "gpu_memory_info": self.gpu_backend.get_memory_info(),
                "configuration_file": self.config_manager.config_file,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "signal_parameters": self.ofdm_generator.get_signal_parameters(),
            "orthogonal_analysis": orthogonal_analysis,
        }

        return system_analysis

    def optimize_phases(
        self, num_signals: int, method: str = "auto", max_iterations: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Optimize phase configurations for maximum orthogonality.

        Args:
            num_signals: Number of signals to optimize phases for
            method: Optimization method ('auto', 'genetic', 'brute_force', 'hybrid')
            max_iterations: Maximum optimization iterations (uses config if None)

        Returns:
            Tuple of (optimal_phase_matrix, orthogonality_score)
        """
        if method == "auto":
            defaults_config = self.config_manager.get_defaults_config()
            method = defaults_config.get("default_optimization_method", "hybrid")

        # Set up optimization configuration
        opt_config = self.config_manager.get_optimization_config()
        if max_iterations is not None:
            opt_config["max_iterations"] = max_iterations

        optimization_config = OptimizationConfig(**opt_config)

        logger.info(f"Optimizing phases for {num_signals} signals using {method} method")

        # Perform optimization
        result = self.phase_optimizer.find_orthogonal_phases(
            num_signals, optimization_config, method
        )

        logger.info(
            f"Phase optimization completed: score={result.orthogonality_score:.4f}, "
            f"iterations={result.iterations}, converged={result.converged}"
        )

        return result.optimal_phases, result.orthogonality_score

    def export_signals(
        self,
        signal_set: SignalSet,
        filename: str,
        format: str = "auto",
        include_visualization: bool = False,
    ) -> List[Path]:
        """Export signal set to files.

        Args:
            signal_set: SignalSet to export
            filename: Base filename for export
            format: Export format ('auto', 'numpy', 'csv', 'json', 'pickle')
            include_visualization: Whether to generate visualization plots

        Returns:
            List of paths to exported files
        """
        if format == "auto":
            defaults_config = self.config_manager.get_defaults_config()
            format = defaults_config.get("default_export_format", "numpy")

        exported_files = []

        # Export signal data
        signal_file = self.signal_exporter.export_signal_set(
            signal_set, filename, format, include_metadata=True
        )
        exported_files.append(signal_file)

        # Export phase configuration
        phase_config_manager = PhaseConfigurationManager()
        phase_file = phase_config_manager.save_phase_configuration(
            signal_set.phases,
            f"{filename}_phases",
            {
                "orthogonality_score": signal_set.orthogonality_score,
                "generation_timestamp": signal_set.generation_timestamp.isoformat(),
                "ofdm_config": {
                    "num_subcarriers": signal_set.config.num_subcarriers,
                    "center_frequency": signal_set.config.center_frequency,
                    "sampling_rate": signal_set.config.sampling_rate,
                },
            },
        )
        exported_files.append(phase_file)

        # Generate visualization if requested
        if include_visualization:
            try:
                visualizer = SignalVisualizer()
                viz_file = visualizer.create_signal_set_visualization(
                    signal_set, f"{filename}_visualization"
                )
                exported_files.append(viz_file)
            except Exception as e:
                logger.warning(f"Could not generate visualization: {e}")

        logger.info(f"Exported signal set to {len(exported_files)} files")

        return exported_files

    def get_system_info(self) -> Dict[str, any]:
        """Get comprehensive system information.

        Returns:
            Dictionary with system configuration and status
        """
        return {
            "ofdm_config": {
                "num_subcarriers": self.ofdm_config.num_subcarriers,
                "subcarrier_spacing": self.ofdm_config.subcarrier_spacing,
                "bandwidth_per_subcarrier": self.ofdm_config.bandwidth_per_subcarrier,
                "center_frequency": self.ofdm_config.center_frequency,
                "sampling_rate": self.ofdm_config.sampling_rate,
                "signal_duration": self.ofdm_config.signal_duration,
            },
            "gpu_backend": self.gpu_backend.device_info,
            "configuration": self.config_manager.to_dict(),
            "capabilities": {
                "max_orthogonal_signals": self.orthogonal_generator.orthogonal_set_config.max_signals,
                "gpu_available": self.gpu_backend.is_gpu_available,
                "visualization_available": hasattr(self, "signal_visualizer"),
                "export_formats": ["numpy", "csv", "json", "pickle"],
            },
            "system_status": {
                "initialized": self._initialized,
                "last_generation_time": (
                    self._last_generated_set.generation_timestamp.isoformat()
                    if self._last_generated_set
                    else None
                ),
                "memory_usage": self.gpu_backend.get_memory_info(),
            },
        }

    def validate_configuration(self) -> Dict[str, any]:
        """Validate current system configuration.

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "configuration_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        try:
            # Validate OFDM configuration
            ofdm_validation = self.ofdm_generator.validate_signal_generation(
                self.ofdm_generator.get_example_phase_array(0)
            )

            for key, is_valid in ofdm_validation.items():
                if not is_valid:
                    validation_results["errors"].append(f"OFDM validation failed: {key}")
                    validation_results["configuration_valid"] = False

            # Check GPU status
            if not self.gpu_backend.is_gpu_available:
                validation_results["warnings"].append("GPU not available, using CPU fallback")

            # Check memory requirements
            memory_info = self.gpu_backend.get_memory_info()
            if memory_info.get("usage_percent", 0) > 80:
                validation_results["warnings"].append("High memory usage detected")

            # Provide recommendations
            if validation_results["configuration_valid"]:
                validation_results["recommendations"].append(
                    "Configuration is valid and ready for signal generation"
                )
            else:
                validation_results["recommendations"].append(
                    "Fix configuration errors before proceeding"
                )

        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {e}")
            validation_results["configuration_valid"] = False

        return validation_results

    def cleanup_resources(self) -> None:
        """Clean up system resources and GPU memory."""
        if hasattr(self, "gpu_backend"):
            self.gpu_backend.cleanup_memory()
        if hasattr(self, "ofdm_generator"):
            self.ofdm_generator.cleanup_resources()
        if hasattr(self, "orthogonal_generator"):
            self.orthogonal_generator.cleanup_resources()
        if hasattr(self, "signal_separator"):
            self.signal_separator.cleanup_resources()

        logger.info("System resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()

    def __repr__(self) -> str:
        """String representation of OFDMChirpGenerator."""
        return (
            f"OFDMChirpGenerator(subcarriers={self.ofdm_config.num_subcarriers}, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}, "
            f"config='{self.config_manager.config_file}')"
        )


# Convenience functions for quick access
def create_generator(config_file: Optional[str] = None, **kwargs) -> OFDMChirpGenerator:
    """Create an OFDMChirpGenerator with default settings.

    Args:
        config_file: Path to configuration file
        **kwargs: Additional arguments passed to OFDMChirpGenerator

    Returns:
        Initialized OFDMChirpGenerator instance
    """
    return OFDMChirpGenerator(config_file=config_file, **kwargs)


def quick_generate_orthogonal_signals(
    num_signals: int = 2, config_file: Optional[str] = None
) -> SignalSet:
    """Quickly generate orthogonal signals with default settings.

    Args:
        num_signals: Number of orthogonal signals to generate
        config_file: Path to configuration file

    Returns:
        SignalSet containing orthogonal signals
    """
    with create_generator(config_file) as generator:
        return generator.generate_orthogonal_set(num_signals)


def quick_test_separation(
    num_signals: int = 2, config_file: Optional[str] = None
) -> Tuple[SignalSet, SeparationQualityMetrics]:
    """Quickly test signal separation with default settings.

    Args:
        num_signals: Number of signals to generate and separate
        config_file: Path to configuration file

    Returns:
        Tuple of (separated_signals, quality_metrics)
    """
    with create_generator(config_file) as generator:
        # Generate orthogonal signals
        original_set = generator.generate_orthogonal_set(num_signals)

        # Combine signals
        combined_signal = generator.combine_signal_set(original_set)

        # Separate signals
        separated_set, quality_metrics = generator.separate_signals(combined_signal, original_set)

        return separated_set, quality_metrics
