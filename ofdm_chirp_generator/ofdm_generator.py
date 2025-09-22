"""
Core OFDM signal generation engine.

This module implements the main OFDMGenerator class that orchestrates the complete
signal generation process, integrating chirp modulation with OFDM structure.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .chirp_modulator import ChirpModulator
from .gpu_backend import GPUBackend
from .models import ChirpConfig, OFDMConfig, SignalSet
from .subcarrier_manager import SubcarrierManager

logger = logging.getLogger(__name__)


class OFDMGenerator:
    """Core OFDM signal generation engine.

    This class orchestrates the complete signal generation process by integrating
    chirp modulation with OFDM structure. It provides methods for single signal
    generation and manages the overall signal generation workflow.

    Requirements addressed:
    - 1.1: Generate OFDM signals with configurable parameters
    - 1.2: Create specified number of subcarriers with configured spacing
    - 2.1: Generate chirp signals with predefined length
    - 2.2: Allow independent phase setting for each subcarrier
    """

    def __init__(self, ofdm_config: OFDMConfig, gpu_backend: Optional[GPUBackend] = None):
        """Initialize OFDM generator.

        Args:
            ofdm_config: OFDM configuration parameters
            gpu_backend: GPU backend for acceleration (creates new if None)
        """
        self.ofdm_config = ofdm_config
        self.gpu_backend = gpu_backend or GPUBackend()

        # Initialize component managers
        self.subcarrier_manager = SubcarrierManager(ofdm_config, self.gpu_backend)
        self.chirp_modulator = ChirpModulator(ofdm_config, self.gpu_backend)

        # Calculate signal parameters
        self._signal_length = int(ofdm_config.signal_duration * ofdm_config.sampling_rate)

        # Validate configuration compatibility
        self._validate_generator_configuration()

        logger.info(
            f"OFDMGenerator initialized: {ofdm_config.num_subcarriers} subcarriers, "
            f"{self._signal_length} samples, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}"
        )

    def _validate_generator_configuration(self) -> None:
        """Validate generator configuration for consistency.

        Requirements:
        - 1.1: Validate configuration parameters
        - Ensure all components are compatible

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate OFDM structure
        structure_validation = self.subcarrier_manager.validate_ofdm_structure()

        if not structure_validation["frequency_symmetry"]:
            raise ValueError("OFDM structure validation failed: frequency symmetry")

        if not structure_validation["spacing_consistency"]:
            raise ValueError("OFDM structure validation failed: spacing consistency")

        if not structure_validation["no_overlaps"]:
            raise ValueError("OFDM structure validation failed: subcarrier overlaps detected")

        if not structure_validation["nyquist_satisfied"]:
            raise ValueError("OFDM structure validation failed: Nyquist criterion not satisfied")

        # Validate chirp constraints
        chirp_constraints = self.chirp_modulator.chirp_length_constraints
        if self._signal_length < chirp_constraints[0]:
            raise ValueError(
                f"Signal length {self._signal_length} below minimum chirp length {chirp_constraints[0]}"
            )

        if self._signal_length > chirp_constraints[1]:
            raise ValueError(
                f"Signal length {self._signal_length} above maximum chirp length {chirp_constraints[1]}"
            )

        logger.debug("Generator configuration validation passed")

    def generate_single_signal(self, phase_array: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """Generate a single OFDM signal with specified phase array.

        Requirements:
        - 1.1: Generate OFDM signals with configurable parameters
        - 1.2: Create specified number of subcarriers with configured spacing
        - 2.1: Generate chirp signals with predefined length
        - 2.2: Allow independent phase setting for each subcarrier

        Args:
            phase_array: Phase offsets for each subcarrier [num_subcarriers]

        Returns:
            Generated OFDM signal array

        Raises:
            ValueError: If phase array is invalid
        """
        # Validate phase array
        validated_phases = self.chirp_modulator.validate_phase_array(phase_array)

        # Generate OFDM signal using subcarrier manager
        ofdm_signal = self.subcarrier_manager.assemble_ofdm_signal(
            validated_phases, self._signal_length
        )

        logger.debug(
            f"Generated single OFDM signal: {len(validated_phases)} subcarriers, "
            f"{len(ofdm_signal)} samples"
        )

        return ofdm_signal

    def generate_signal_with_chirp_config(
        self, chirp_config: ChirpConfig
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Generate OFDM signal using ChirpConfig parameters.

        Args:
            chirp_config: Chirp configuration with phase matrix

        Returns:
            Generated OFDM signal array

        Raises:
            ValueError: If chirp config is incompatible
        """
        # Extract phase array for single signal (first row of phase matrix)
        if chirp_config.phase_matrix.ndim != 2:
            raise ValueError("ChirpConfig phase_matrix must be 2D array")

        if chirp_config.phase_matrix.shape[1] != self.ofdm_config.num_subcarriers:
            raise ValueError(
                f"ChirpConfig phase matrix has {chirp_config.phase_matrix.shape[1]} columns, "
                f"expected {self.ofdm_config.num_subcarriers}"
            )

        # Use first row of phase matrix
        phase_array = chirp_config.phase_matrix[0, :]

        # Generate signal with amplitude scaling
        ofdm_signal = self.generate_single_signal(phase_array)

        # Apply amplitude scaling from chirp config
        ofdm_signal *= chirp_config.amplitude

        return ofdm_signal

    def create_signal_set(
        self, phase_matrix: np.ndarray, orthogonality_score: float = 0.0
    ) -> SignalSet:
        """Create a SignalSet from multiple phase configurations.

        Args:
            phase_matrix: Phase configurations [num_signals x num_subcarriers]
            orthogonality_score: Orthogonality score for the signal set

        Returns:
            SignalSet containing generated signals

        Raises:
            ValueError: If phase matrix dimensions are invalid
        """
        if phase_matrix.ndim != 2:
            raise ValueError("Phase matrix must be 2D array")

        if phase_matrix.shape[1] != self.ofdm_config.num_subcarriers:
            raise ValueError(
                f"Phase matrix has {phase_matrix.shape[1]} columns, "
                f"expected {self.ofdm_config.num_subcarriers}"
            )

        # Generate signals for each phase configuration
        signals = []
        for i in range(phase_matrix.shape[0]):
            signal = self.generate_single_signal(phase_matrix[i, :])
            # Convert to CPU for storage in SignalSet
            signal_cpu = self.gpu_backend.to_cpu(signal)
            signals.append(signal_cpu)

        # Create SignalSet
        signal_set = SignalSet(
            signals=signals,
            phases=phase_matrix.copy(),
            orthogonality_score=orthogonality_score,
            generation_timestamp=datetime.now(),
            config=self.ofdm_config,
            metadata={
                "generator_backend": "GPU" if self.gpu_backend.is_gpu_available else "CPU",
                "signal_length": self._signal_length,
                "num_signals": len(signals),
            },
        )

        logger.info(f"Created SignalSet with {len(signals)} signals")

        return signal_set

    def get_signal_parameters(self) -> Dict[str, Union[int, float, List[float]]]:
        """Get comprehensive signal generation parameters.

        Returns:
            Dictionary with all signal parameters
        """
        ofdm_info = self.subcarrier_manager.get_ofdm_structure_info()
        chirp_constraints = self.chirp_modulator.chirp_length_constraints

        return {
            "ofdm_config": {
                "num_subcarriers": self.ofdm_config.num_subcarriers,
                "subcarrier_spacing": self.ofdm_config.subcarrier_spacing,
                "bandwidth_per_subcarrier": self.ofdm_config.bandwidth_per_subcarrier,
                "center_frequency": self.ofdm_config.center_frequency,
                "sampling_rate": self.ofdm_config.sampling_rate,
                "signal_duration": self.ofdm_config.signal_duration,
            },
            "signal_properties": {
                "signal_length_samples": self._signal_length,
                "total_bandwidth": ofdm_info["total_bandwidth"],
                "frequency_range": ofdm_info["frequency_range"],
                "subcarrier_frequencies": ofdm_info["subcarrier_frequencies"],
            },
            "chirp_properties": {
                "min_chirp_length": chirp_constraints[0],
                "max_chirp_length": chirp_constraints[1],
                "actual_chirp_length": self._signal_length,
            },
            "backend_info": self.gpu_backend.device_info,
            "validation_status": self.subcarrier_manager.validate_ofdm_structure(),
        }

    def analyze_generated_signal(
        self, signal: Union[np.ndarray, "cp.ndarray"]
    ) -> Dict[str, Union[float, Dict[int, float]]]:
        """Analyze properties of a generated signal.

        Args:
            signal: Generated OFDM signal to analyze

        Returns:
            Dictionary with signal analysis results
        """
        # Convert to CPU for analysis if needed
        signal_cpu = self.gpu_backend.to_cpu(signal)

        # Basic signal properties
        signal_power = float(np.mean(np.abs(signal_cpu) ** 2))
        peak_amplitude = float(np.max(np.abs(signal_cpu)))
        rms_amplitude = float(np.sqrt(np.mean(np.abs(signal_cpu) ** 2)))

        # Frequency domain analysis
        subcarrier_power = self.subcarrier_manager.analyze_subcarrier_power(signal)

        # Calculate peak-to-average power ratio (PAPR)
        papr = peak_amplitude**2 / signal_power if signal_power > 0 else 0.0

        return {
            "signal_power": signal_power,
            "peak_amplitude": peak_amplitude,
            "rms_amplitude": rms_amplitude,
            "papr_db": 10.0 * np.log10(papr) if papr > 0 else -np.inf,
            "subcarrier_power_distribution": subcarrier_power,
            "signal_length": len(signal_cpu),
            "dynamic_range_db": (
                20.0 * np.log10(peak_amplitude / rms_amplitude) if rms_amplitude > 0 else 0.0
            ),
        }

    def validate_signal_generation(self, phase_array: np.ndarray) -> Dict[str, bool]:
        """Validate that signal generation will work with given parameters.

        Args:
            phase_array: Phase array to validate

        Returns:
            Dictionary with validation results
        """
        results = {}

        try:
            # Validate phase array
            self.chirp_modulator.validate_phase_array(phase_array)
            results["phase_array_valid"] = True
        except Exception:
            results["phase_array_valid"] = False

        # Check memory requirements
        try:
            memory_required = self._signal_length * 16  # Complex128 = 16 bytes
            memory_info = self.gpu_backend.get_memory_info()

            if self.gpu_backend.is_gpu_available:
                memory_available = memory_info.get("free_bytes", 0)
                results["sufficient_memory"] = memory_required < memory_available * 0.8
            else:
                results["sufficient_memory"] = True  # Assume CPU has sufficient memory
        except Exception:
            results["sufficient_memory"] = False

        # Validate OFDM structure
        structure_validation = self.subcarrier_manager.validate_ofdm_structure()
        results.update(structure_validation)

        return results

    def get_example_phase_array(self, signal_index: int = 0) -> np.ndarray:
        """Generate an example phase array for testing.

        Args:
            signal_index: Index for generating different phase patterns

        Returns:
            Example phase array for the configured number of subcarriers
        """
        num_subcarriers = self.ofdm_config.num_subcarriers

        # Generate different phase patterns based on signal index
        if signal_index == 0:
            # Linear phase progression
            phases = np.linspace(0, 2 * np.pi, num_subcarriers, endpoint=False)
        elif signal_index == 1:
            # Quadratic phase progression
            indices = np.arange(num_subcarriers)
            phases = (indices**2 * np.pi / num_subcarriers) % (2 * np.pi)
        else:
            # Random phases with seed for reproducibility
            np.random.seed(signal_index)
            phases = np.random.uniform(0, 2 * np.pi, num_subcarriers)

        return phases

    def cleanup_resources(self) -> None:
        """Clean up GPU resources and memory."""
        self.gpu_backend.cleanup_memory()
        logger.debug("OFDMGenerator resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()

    def __repr__(self) -> str:
        """String representation of OFDMGenerator."""
        return (
            f"OFDMGenerator(subcarriers={self.ofdm_config.num_subcarriers}, "
            f"duration={self.ofdm_config.signal_duration}s, "
            f"samples={self._signal_length}, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'})"
        )
