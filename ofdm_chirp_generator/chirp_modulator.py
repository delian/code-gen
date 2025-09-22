"""
Chirp signal modulation for OFDM subcarriers.

This module implements linear frequency modulated (chirp) signal generation
with GPU acceleration and configurable phase offsets for orthogonal signal creation.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np

from .gpu_backend import GPUBackend
from .models import ChirpConfig, OFDMConfig

logger = logging.getLogger(__name__)


class ChirpModulator:
    """Linear frequency modulated signal generator for OFDM subcarriers.

    This class generates chirp signals with configurable phase offsets and
    linear frequency modulation within specified bandwidth constraints.
    Supports GPU acceleration through CuPy integration.
    """

    def __init__(self, ofdm_config: OFDMConfig, gpu_backend: Optional[GPUBackend] = None):
        """Initialize chirp modulator.

        Args:
            ofdm_config: OFDM configuration parameters
            gpu_backend: GPU backend for acceleration (creates new if None)
        """
        self.ofdm_config = ofdm_config
        self.gpu_backend = gpu_backend or GPUBackend()

        # Validate chirp length constraints
        self._validate_chirp_constraints()

        # Pre-calculate time vector for efficiency
        self._time_vector = None
        self._setup_time_vector()

        logger.info(f"ChirpModulator initialized with {ofdm_config.num_subcarriers} subcarriers")

    def _validate_chirp_constraints(self) -> None:
        """Validate chirp length and parameter constraints.

        Requirement 2.4: Validate and constrain chirp length parameters.
        """
        # Calculate minimum chirp length based on sampling rate and bandwidth
        min_samples = int(
            self.ofdm_config.sampling_rate / self.ofdm_config.bandwidth_per_subcarrier
        )
        max_samples = int(self.ofdm_config.signal_duration * self.ofdm_config.sampling_rate)

        if min_samples > max_samples:
            raise ValueError(
                f"Invalid configuration: minimum chirp length ({min_samples}) "
                f"exceeds maximum signal duration ({max_samples} samples)"
            )

        # Store validated constraints
        self._min_chirp_length = max(min_samples, 32)  # Minimum 32 samples
        self._max_chirp_length = max_samples

        logger.debug(
            f"Chirp length constraints: {self._min_chirp_length} - {self._max_chirp_length} samples"
        )

    def _setup_time_vector(self) -> None:
        """Setup time vector for chirp generation."""
        num_samples = int(self.ofdm_config.signal_duration * self.ofdm_config.sampling_rate)
        dt = 1.0 / self.ofdm_config.sampling_rate

        # Create time vector
        time_vector = np.arange(num_samples) * dt

        # Transfer to GPU if available
        self._time_vector = self.gpu_backend.to_gpu(time_vector)

        logger.debug(f"Time vector setup: {num_samples} samples, dt={dt:.2e}s")

    def validate_chirp_length(self, chirp_length: int) -> int:
        """Validate and constrain chirp length parameter.

        Args:
            chirp_length: Requested chirp length in samples

        Returns:
            Validated and constrained chirp length

        Raises:
            ValueError: If chirp length cannot be constrained to valid range
        """
        if chirp_length < self._min_chirp_length:
            logger.warning(
                f"Chirp length {chirp_length} below minimum, using {self._min_chirp_length}"
            )
            return self._min_chirp_length

        if chirp_length > self._max_chirp_length:
            logger.warning(
                f"Chirp length {chirp_length} above maximum, using {self._max_chirp_length}"
            )
            return self._max_chirp_length

        return chirp_length

    def generate_chirp_signal(
        self, subcarrier_index: int, phase_offset: float = 0.0, chirp_length: Optional[int] = None
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Generate a single chirp signal for a subcarrier.

        Requirements:
        - 2.1: Generate chirp signal with predefined length
        - 2.2: Allow independent phase setting for each subcarrier
        - 2.3: Use linear frequency modulation within subcarrier bandwidth

        Args:
            subcarrier_index: Index of the subcarrier (0-based)
            phase_offset: Phase offset in radians for orthogonality
            chirp_length: Length of chirp in samples (uses signal duration if None)

        Returns:
            Generated chirp signal array

        Raises:
            ValueError: If subcarrier index is invalid
        """
        if subcarrier_index < 0 or subcarrier_index >= self.ofdm_config.num_subcarriers:
            raise ValueError(
                f"Invalid subcarrier index {subcarrier_index}. "
                f"Must be in range [0, {self.ofdm_config.num_subcarriers-1}]"
            )

        # Validate chirp length
        if chirp_length is None:
            chirp_length = len(self._time_vector)
        else:
            chirp_length = self.validate_chirp_length(chirp_length)

        # Calculate subcarrier frequency
        subcarrier_freq = self._calculate_subcarrier_frequency(subcarrier_index)

        # Generate linear frequency modulated chirp
        chirp_signal = self._generate_linear_chirp(subcarrier_freq, phase_offset, chirp_length)

        logger.debug(
            f"Generated chirp for subcarrier {subcarrier_index}: "
            f"freq={subcarrier_freq:.2f}Hz, phase={phase_offset:.3f}rad, "
            f"length={chirp_length} samples"
        )

        return chirp_signal

    def generate_multi_chirp_signal(
        self, phase_array: np.ndarray, chirp_length: Optional[int] = None
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Generate OFDM signal with multiple chirp subcarriers.

        Requirements:
        - 2.1: Generate chirp signals with predefined length
        - 2.2: Independent phase setting for each subcarrier
        - 2.3: Linear frequency modulation within subcarrier bandwidth

        Args:
            phase_array: Phase offsets for each subcarrier [num_subcarriers]
            chirp_length: Length of each chirp in samples

        Returns:
            Combined OFDM signal with all chirp subcarriers

        Raises:
            ValueError: If phase array size doesn't match number of subcarriers
        """
        if len(phase_array) != self.ofdm_config.num_subcarriers:
            raise ValueError(
                f"Phase array length {len(phase_array)} doesn't match "
                f"number of subcarriers {self.ofdm_config.num_subcarriers}"
            )

        # Validate chirp length
        if chirp_length is None:
            chirp_length = len(self._time_vector)
        else:
            chirp_length = self.validate_chirp_length(chirp_length)

        # Initialize combined signal
        combined_signal = self.gpu_backend.allocate_signal_memory(
            (chirp_length,), dtype=np.complex128
        )

        # Generate and sum all subcarrier chirps
        for i, phase in enumerate(phase_array):
            chirp = self.generate_chirp_signal(i, phase, chirp_length)
            combined_signal += chirp

        logger.debug(f"Generated multi-chirp OFDM signal with {len(phase_array)} subcarriers")

        return combined_signal

    def _calculate_subcarrier_frequency(self, subcarrier_index: int) -> float:
        """Calculate frequency for a specific subcarrier.

        Args:
            subcarrier_index: Index of the subcarrier

        Returns:
            Subcarrier center frequency in Hz
        """
        # Calculate frequency offset from center
        num_subcarriers = self.ofdm_config.num_subcarriers

        # For even number of subcarriers, center is between middle two
        # For odd number, center is at the middle subcarrier
        if num_subcarriers % 2 == 0:
            # Even: center between indices (n/2-1) and (n/2)
            center_index = (num_subcarriers / 2.0) - 0.5
        else:
            # Odd: center at index (n-1)/2
            center_index = (num_subcarriers - 1) / 2.0

        frequency_offset = (subcarrier_index - center_index) * self.ofdm_config.subcarrier_spacing

        return self.ofdm_config.center_frequency + frequency_offset

    def _generate_linear_chirp(
        self, center_freq: float, phase_offset: float, chirp_length: int
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Generate linear frequency modulated chirp signal.

        Requirement 2.3: Use linear frequency modulation within subcarrier bandwidth.

        Args:
            center_freq: Center frequency of the chirp
            phase_offset: Phase offset for orthogonality
            chirp_length: Length of the chirp in samples

        Returns:
            Generated chirp signal
        """
        # Use appropriate array library based on backend
        if self.gpu_backend.is_gpu_available:
            import cupy as cp

            array_lib = cp
        else:
            array_lib = np

        # Create time vector for this chirp length
        dt = 1.0 / self.ofdm_config.sampling_rate
        t = array_lib.arange(chirp_length) * dt

        # Transfer to GPU if needed
        if self.gpu_backend.is_gpu_available:
            t = self.gpu_backend.to_gpu(t)

        # Calculate chirp parameters
        bandwidth = self.ofdm_config.bandwidth_per_subcarrier
        chirp_duration = chirp_length * dt

        # Linear frequency sweep rate (Hz/s)
        # Sweep from (center_freq - bandwidth/2) to (center_freq + bandwidth/2)
        sweep_rate = bandwidth / chirp_duration

        # Generate linear chirp with phase offset
        # Start frequency: center_freq - bandwidth/2
        # End frequency: center_freq + bandwidth/2
        # Frequency: f(t) = start_freq + sweep_rate * t

        start_freq = center_freq - bandwidth / 2.0

        # Integrate to get phase: φ(t) = 2π * ∫f(t)dt + phase_offset
        # φ(t) = 2π * (start_freq * t + sweep_rate * t²/2) + phase_offset
        phase = 2.0 * array_lib.pi * (start_freq * t + sweep_rate * (t**2 / 2.0)) + phase_offset

        # Generate complex chirp signal
        chirp_signal = array_lib.exp(1j * phase)

        # Apply amplitude scaling
        amplitude = 1.0 / array_lib.sqrt(self.ofdm_config.num_subcarriers)
        chirp_signal *= amplitude

        return chirp_signal

    def get_chirp_characteristics(self, subcarrier_index: int, phase_offset: float = 0.0) -> dict:
        """Get characteristics of a chirp signal for analysis.

        Args:
            subcarrier_index: Index of the subcarrier
            phase_offset: Phase offset in radians

        Returns:
            Dictionary with chirp characteristics
        """
        center_freq = self._calculate_subcarrier_frequency(subcarrier_index)
        bandwidth = self.ofdm_config.bandwidth_per_subcarrier
        duration = self.ofdm_config.signal_duration

        return {
            "subcarrier_index": subcarrier_index,
            "center_frequency": center_freq,
            "bandwidth": bandwidth,
            "duration": duration,
            "phase_offset": phase_offset,
            "start_frequency": center_freq - bandwidth / 2.0,
            "end_frequency": center_freq + bandwidth / 2.0,
            "sweep_rate": bandwidth / duration,
            "num_samples": int(duration * self.ofdm_config.sampling_rate),
        }

    def validate_phase_array(self, phase_array: np.ndarray) -> np.ndarray:
        """Validate and normalize phase array.

        Args:
            phase_array: Array of phase values

        Returns:
            Validated and normalized phase array [0, 2π]
        """
        if len(phase_array) != self.ofdm_config.num_subcarriers:
            raise ValueError(
                f"Phase array length {len(phase_array)} doesn't match "
                f"number of subcarriers {self.ofdm_config.num_subcarriers}"
            )

        # Normalize phases to [0, 2π] range
        normalized_phases = np.mod(phase_array, 2.0 * np.pi)

        return normalized_phases

    @property
    def chirp_length_constraints(self) -> Tuple[int, int]:
        """Get valid chirp length range.

        Returns:
            Tuple of (min_length, max_length) in samples
        """
        return (self._min_chirp_length, self._max_chirp_length)

    def __repr__(self) -> str:
        """String representation of the chirp modulator."""
        return (
            f"ChirpModulator(subcarriers={self.ofdm_config.num_subcarriers}, "
            f"bandwidth={self.ofdm_config.bandwidth_per_subcarrier}Hz, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'})"
        )
