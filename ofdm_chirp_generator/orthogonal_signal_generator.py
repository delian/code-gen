"""
Orthogonal signal set generation for OFDM chirp signals.

This module implements methods to generate multiple orthogonal OFDM signals
with phase matrix management and batch generation capabilities.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config_manager import ConfigurationError, get_config
from .gpu_backend import GPUBackend
from .models import OFDMConfig, SignalSet
from .ofdm_generator import OFDMGenerator
from .orthogonality_tester import OrthogonalityTester
from .phase_optimizer import OptimizationConfig, PhaseOptimizer

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class OrthogonalSetConfig:
    """Configuration for orthogonal signal set generation."""

    max_signals: int = 16
    orthogonality_threshold: float = 0.95
    optimization_method: str = "hybrid"
    max_optimization_iterations: int = 1000
    batch_size: int = 4
    enable_caching: bool = True
    validation_enabled: bool = True


class PhaseMatrixManager:
    """Manages storage and retrieval of orthogonal phase configurations."""

    def __init__(self):
        """Initialize phase matrix manager."""
        self.stored_configurations = {}
        self.configuration_metadata = {}

    def store_configuration(
        self,
        config_id: str,
        phase_matrix: np.ndarray,
        orthogonality_score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Store a validated orthogonal phase configuration.

        Args:
            config_id: Unique identifier for the configuration
            phase_matrix: Phase matrix [num_signals x num_subcarriers]
            orthogonality_score: Orthogonality score for this configuration
            metadata: Additional metadata about the configuration
        """
        self.stored_configurations[config_id] = phase_matrix.copy()
        self.configuration_metadata[config_id] = {
            "orthogonality_score": orthogonality_score,
            "num_signals": phase_matrix.shape[0],
            "num_subcarriers": phase_matrix.shape[1],
            "storage_timestamp": datetime.now(),
            "metadata": metadata or {},
        }

        logger.info(
            f"Stored phase configuration '{config_id}' with {phase_matrix.shape[0]} signals"
        )

    def retrieve_configuration(self, config_id: str) -> Tuple[np.ndarray, Dict]:
        """Retrieve a stored phase configuration.

        Args:
            config_id: Configuration identifier

        Returns:
            Tuple of (phase_matrix, metadata)

        Raises:
            KeyError: If configuration not found
        """
        if config_id not in self.stored_configurations:
            raise KeyError(f"Configuration '{config_id}' not found")

        return (
            self.stored_configurations[config_id].copy(),
            self.configuration_metadata[config_id],
        )

    def list_configurations(self) -> List[str]:
        """List all stored configuration IDs."""
        return list(self.stored_configurations.keys())

    def get_best_configuration(self, num_signals: int) -> Optional[Tuple[str, np.ndarray, Dict]]:
        """Get the best stored configuration for a given number of signals.

        Args:
            num_signals: Required number of signals

        Returns:
            Tuple of (config_id, phase_matrix, metadata) or None if not found
        """
        best_config = None
        best_score = -1.0

        for config_id, metadata in self.configuration_metadata.items():
            if metadata["num_signals"] == num_signals:
                if metadata["orthogonality_score"] > best_score:
                    best_score = metadata["orthogonality_score"]
                    best_config = config_id

        if best_config:
            phase_matrix, metadata = self.retrieve_configuration(best_config)
            return best_config, phase_matrix, metadata

        return None

    def remove_configuration(self, config_id: str) -> None:
        """Remove a stored configuration.

        Args:
            config_id: Configuration identifier
        """
        if config_id in self.stored_configurations:
            del self.stored_configurations[config_id]
            del self.configuration_metadata[config_id]
            logger.info(f"Removed configuration '{config_id}'")

    def clear_all(self) -> None:
        """Clear all stored configurations."""
        self.stored_configurations.clear()
        self.configuration_metadata.clear()
        logger.info("Cleared all stored configurations")


class OrthogonalSignalGenerator:
    """Generator for multiple orthogonal OFDM signals.

    This class implements methods to generate multiple orthogonal OFDM signals
    with phase matrix management and batch generation capabilities.

    Requirements addressed:
    - 3.1: Generate multiple orthogonal OFDM signals
    - 3.2: Create signals with orthogonal phase relationships
    - 3.3: Support generation of at least 16 orthogonal signals
    - 5.3: Store phase configuration when valid orthogonal set found
    """

    def __init__(
        self,
        ofdm_config: Optional[OFDMConfig] = None,
        gpu_backend: Optional[GPUBackend] = None,
        config_file: Optional[str] = None,
    ):
        """Initialize orthogonal signal generator.

        Args:
            ofdm_config: OFDM configuration parameters (loads from config if None)
            gpu_backend: GPU backend for acceleration (creates new if None)
            config_file: Path to configuration file (uses default if None)
        """
        # Load configuration if not provided
        if ofdm_config is None:
            try:
                config_manager = get_config(config_file)
                self.ofdm_config = config_manager.create_ofdm_config_object()
                logger.info("Loaded OFDM configuration from config file")
            except (ConfigurationError, Exception) as e:
                logger.warning(f"Could not load configuration: {e}")
                raise ValueError(
                    "Either ofdm_config must be provided or configuration file must be available"
                )
        else:
            self.ofdm_config = ofdm_config

        self.gpu_backend = gpu_backend or GPUBackend()

        # Initialize components
        self.ofdm_generator = OFDMGenerator(self.ofdm_config, self.gpu_backend)
        self.phase_optimizer = PhaseOptimizer(self.ofdm_config, self.gpu_backend, config_file)
        self.orthogonality_tester = OrthogonalityTester(self.gpu_backend, config_file)

        # Phase matrix management
        self.phase_matrix_manager = PhaseMatrixManager()

        # Load orthogonal set configuration
        try:
            config_manager = get_config(config_file)
            orth_config = config_manager.get_orthogonality_config()
            self.orthogonal_set_config = OrthogonalSetConfig(
                max_signals=orth_config.get("max_orthogonal_signals", 16),
                orthogonality_threshold=orth_config.get("generation_threshold", 0.95),
                optimization_method=orth_config.get("optimization_method", "hybrid"),
                max_optimization_iterations=orth_config.get("max_iterations", 1000),
                batch_size=orth_config.get("batch_size", 4),
                enable_caching=orth_config.get("enable_caching", True),
                validation_enabled=orth_config.get("validation_enabled", True),
            )
            logger.info("Loaded orthogonal set configuration from config file")
        except (ConfigurationError, Exception) as e:
            self.orthogonal_set_config = OrthogonalSetConfig()
            logger.warning(f"Could not load orthogonal set configuration: {e}. Using defaults.")

        logger.info(
            f"OrthogonalSignalGenerator initialized: max_signals={self.orthogonal_set_config.max_signals}, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}"
        )

    def generate_orthogonal_signal_set(
        self, num_signals: int, force_regenerate: bool = False
    ) -> SignalSet:
        """Generate a set of orthogonal OFDM signals.

        Requirements:
        - 3.1: Generate multiple orthogonal OFDM signals
        - 3.2: Create signals with orthogonal phase relationships
        - 3.3: Support generation of at least 16 orthogonal signals
        - 5.3: Store phase configuration when valid orthogonal set found

        Args:
            num_signals: Number of orthogonal signals to generate
            force_regenerate: Force regeneration even if cached configuration exists

        Returns:
            SignalSet containing orthogonal signals

        Raises:
            ValueError: If num_signals exceeds maximum or is invalid
        """
        if num_signals < 2:
            raise ValueError("Need at least 2 signals for orthogonal set")

        if num_signals > self.orthogonal_set_config.max_signals:
            raise ValueError(
                f"Requested {num_signals} signals exceeds maximum "
                f"{self.orthogonal_set_config.max_signals}"
            )

        logger.info(f"Generating orthogonal signal set with {num_signals} signals")

        # Check for cached configuration
        if self.orthogonal_set_config.enable_caching and not force_regenerate:
            cached_result = self.phase_matrix_manager.get_best_configuration(num_signals)
            if cached_result:
                config_id, phase_matrix, metadata = cached_result
                logger.info(
                    f"Using cached configuration '{config_id}' with score {metadata['orthogonality_score']:.6f}"
                )

                # Generate signals from cached phase matrix
                signal_set = self.ofdm_generator.create_signal_set(
                    phase_matrix, metadata["orthogonality_score"]
                )
                signal_set.metadata.update(
                    {
                        "generation_method": "cached",
                        "cached_config_id": config_id,
                        "original_generation_time": metadata["storage_timestamp"],
                    }
                )
                return signal_set

        # Generate new orthogonal phase configuration
        start_time = time.time()

        # Set up optimization configuration
        optimization_config = OptimizationConfig(
            max_iterations=self.orthogonal_set_config.max_optimization_iterations,
            orthogonality_target=self.orthogonal_set_config.orthogonality_threshold,
        )

        # Find optimal phase configuration
        optimization_result = self.phase_optimizer.find_orthogonal_phases(
            num_signals, optimization_config, self.orthogonal_set_config.optimization_method
        )

        # Validate orthogonality if enabled
        if self.orthogonal_set_config.validation_enabled:
            validation_result = self._validate_orthogonal_configuration(
                optimization_result.optimal_phases
            )

            if not validation_result["is_valid"]:
                logger.warning(
                    f"Generated configuration failed validation: {validation_result['reason']}"
                )

        # Generate signal set
        signal_set = self.ofdm_generator.create_signal_set(
            optimization_result.optimal_phases, optimization_result.orthogonality_score
        )

        # Add generation metadata
        generation_time = time.time() - start_time
        signal_set.metadata.update(
            {
                "generation_method": "optimized",
                "optimization_method": self.orthogonal_set_config.optimization_method,
                "optimization_iterations": optimization_result.iterations,
                "optimization_converged": optimization_result.converged,
                "generation_time_seconds": generation_time,
                "orthogonality_threshold": self.orthogonal_set_config.orthogonality_threshold,
            }
        )

        # Store configuration if caching is enabled and quality is good
        if (
            self.orthogonal_set_config.enable_caching
            and optimization_result.orthogonality_score
            >= self.orthogonal_set_config.orthogonality_threshold * 0.9
        ):

            config_id = f"auto_{num_signals}sig_{int(time.time())}"
            self.phase_matrix_manager.store_configuration(
                config_id,
                optimization_result.optimal_phases,
                optimization_result.orthogonality_score,
                {
                    "generation_method": self.orthogonal_set_config.optimization_method,
                    "generation_time": generation_time,
                    "converged": optimization_result.converged,
                },
            )

        logger.info(
            f"Generated orthogonal signal set: {num_signals} signals, "
            f"score={optimization_result.orthogonality_score:.6f}, "
            f"time={generation_time:.2f}s"
        )

        return signal_set

    def generate_batch_orthogonal_sets(
        self, signal_counts: List[int], parallel_processing: bool = True
    ) -> Dict[int, SignalSet]:
        """Generate multiple orthogonal signal sets efficiently.

        Args:
            signal_counts: List of signal counts to generate sets for
            parallel_processing: Enable parallel processing for batch generation

        Returns:
            Dictionary mapping signal count to SignalSet
        """
        logger.info(f"Generating batch orthogonal sets for {len(signal_counts)} configurations")

        results = {}
        batch_start_time = time.time()

        # Process in batches for memory efficiency
        batch_size = self.orthogonal_set_config.batch_size

        for i in range(0, len(signal_counts), batch_size):
            batch = signal_counts[i : i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}: {batch}")

            for num_signals in batch:
                try:
                    signal_set = self.generate_orthogonal_signal_set(num_signals)
                    results[num_signals] = signal_set

                except Exception as e:
                    logger.error(f"Failed to generate {num_signals} signal set: {e}")
                    continue

            # Clean up GPU memory between batches
            if self.gpu_backend.is_gpu_available:
                self.gpu_backend.cleanup_memory()

        batch_time = time.time() - batch_start_time
        logger.info(
            f"Batch generation completed: {len(results)}/{len(signal_counts)} sets generated "
            f"in {batch_time:.2f}s"
        )

        return results

    def _validate_orthogonal_configuration(self, phase_matrix: np.ndarray) -> Dict[str, any]:
        """Validate an orthogonal phase configuration.

        Args:
            phase_matrix: Phase matrix to validate

        Returns:
            Dictionary with validation results
        """
        try:
            # Generate signals for validation
            signals = []
            for i in range(phase_matrix.shape[0]):
                signal = self.ofdm_generator.generate_single_signal(phase_matrix[i, :])
                signals.append(signal)

            # Test orthogonality
            orthogonality_result = self.orthogonality_tester.test_signal_set_orthogonality(signals)

            is_valid = (
                orthogonality_result["overall_orthogonality_score"]
                >= self.orthogonal_set_config.orthogonality_threshold
            )

            return {
                "is_valid": is_valid,
                "orthogonality_score": orthogonality_result["overall_orthogonality_score"],
                "orthogonal_pairs": orthogonality_result["orthogonal_pairs"],
                "total_pairs": orthogonality_result["total_pairs"],
                "reason": (
                    "Valid"
                    if is_valid
                    else f"Score {orthogonality_result['overall_orthogonality_score']:.6f} below threshold {self.orthogonal_set_config.orthogonality_threshold}"
                ),
            }

        except Exception as e:
            return {"is_valid": False, "reason": f"Validation error: {str(e)}"}

    def analyze_orthogonal_set(self, signal_set: SignalSet) -> Dict[str, any]:
        """Analyze the orthogonality properties of a signal set.

        Args:
            signal_set: SignalSet to analyze

        Returns:
            Dictionary with comprehensive analysis results
        """
        # Test orthogonality
        orthogonality_result = self.orthogonality_tester.test_signal_set_orthogonality(
            signal_set.signals
        )

        # Analyze individual signal properties
        signal_analyses = []
        for i, signal in enumerate(signal_set.signals):
            analysis = self.ofdm_generator.analyze_generated_signal(signal)
            analysis["signal_index"] = i
            signal_analyses.append(analysis)

        # Calculate set-level metrics
        signal_powers = [analysis["signal_power"] for analysis in signal_analyses]
        power_balance = (
            np.std(signal_powers) / np.mean(signal_powers)
            if np.mean(signal_powers) > 0
            else float("inf")
        )

        return {
            "orthogonality_analysis": orthogonality_result,
            "signal_analyses": signal_analyses,
            "set_metrics": {
                "num_signals": signal_set.num_signals,
                "signal_length": signal_set.signal_length,
                "mean_signal_power": np.mean(signal_powers),
                "power_balance_coefficient": power_balance,
                "generation_timestamp": signal_set.generation_timestamp,
                "orthogonality_score": signal_set.orthogonality_score,
            },
            "phase_matrix_properties": {
                "shape": signal_set.phases.shape,
                "phase_range": (np.min(signal_set.phases), np.max(signal_set.phases)),
                "phase_std": np.std(signal_set.phases),
                "phase_mean": np.mean(signal_set.phases),
            },
        }

    def get_maximum_orthogonal_signals(self) -> int:
        """Determine the maximum number of orthogonal signals achievable.

        Returns:
            Maximum number of orthogonal signals that can be generated
        """
        # Start with theoretical maximum and test downward
        max_theoretical = min(
            self.orthogonal_set_config.max_signals, self.ofdm_config.num_subcarriers
        )

        logger.info(f"Testing maximum orthogonal signals (theoretical max: {max_theoretical})")

        # Binary search for maximum achievable
        low, high = 2, max_theoretical
        max_achievable = 2

        while low <= high:
            mid = (low + high) // 2

            try:
                # Quick test with limited iterations
                test_config = OptimizationConfig(
                    max_iterations=100,
                    orthogonality_target=self.orthogonal_set_config.orthogonality_threshold * 0.8,
                )

                result = self.phase_optimizer.find_orthogonal_phases(
                    mid, test_config, "genetic"  # Use faster method for testing
                )

                if result.orthogonality_score >= test_config.orthogonality_target:
                    max_achievable = mid
                    low = mid + 1
                else:
                    high = mid - 1

            except Exception as e:
                logger.debug(f"Failed to generate {mid} signals: {e}")
                high = mid - 1

        logger.info(f"Maximum achievable orthogonal signals: {max_achievable}")
        return max_achievable

    def export_phase_configurations(self) -> Dict[str, any]:
        """Export all stored phase configurations.

        Returns:
            Dictionary with all configurations and metadata
        """
        export_data = {
            "configurations": {},
            "export_timestamp": datetime.now().isoformat(),
            "ofdm_config": {
                "num_subcarriers": self.ofdm_config.num_subcarriers,
                "subcarrier_spacing": self.ofdm_config.subcarrier_spacing,
                "center_frequency": self.ofdm_config.center_frequency,
                "sampling_rate": self.ofdm_config.sampling_rate,
            },
        }

        for config_id in self.phase_matrix_manager.list_configurations():
            phase_matrix, metadata = self.phase_matrix_manager.retrieve_configuration(config_id)
            export_data["configurations"][config_id] = {
                "phase_matrix": phase_matrix.tolist(),
                "metadata": metadata,
            }

        return export_data

    def import_phase_configurations(self, import_data: Dict[str, any]) -> int:
        """Import phase configurations from exported data.

        Args:
            import_data: Dictionary with configuration data

        Returns:
            Number of configurations imported
        """
        imported_count = 0

        for config_id, config_data in import_data.get("configurations", {}).items():
            try:
                phase_matrix = np.array(config_data["phase_matrix"])
                metadata = config_data["metadata"]

                # Validate compatibility with current OFDM config
                if phase_matrix.shape[1] != self.ofdm_config.num_subcarriers:
                    logger.warning(f"Skipping {config_id}: incompatible subcarrier count")
                    continue

                self.phase_matrix_manager.store_configuration(
                    config_id, phase_matrix, metadata["orthogonality_score"], metadata
                )
                imported_count += 1

            except Exception as e:
                logger.error(f"Failed to import configuration {config_id}: {e}")

        logger.info(f"Imported {imported_count} phase configurations")
        return imported_count

    def cleanup_resources(self) -> None:
        """Clean up GPU resources and memory."""
        self.gpu_backend.cleanup_memory()
        self.ofdm_generator.cleanup_resources()
        logger.debug("OrthogonalSignalGenerator resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()

    def __repr__(self) -> str:
        """String representation of OrthogonalSignalGenerator."""
        return (
            f"OrthogonalSignalGenerator(max_signals={self.orthogonal_set_config.max_signals}, "
            f"subcarriers={self.ofdm_config.num_subcarriers}, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'})"
        )
