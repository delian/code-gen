"""
Subcarrier management for OFDM signal structure.

This module implements frequency allocation, spacing management, and OFDM signal
assembly from multiple chirp subcarriers with configurable parameters.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .chirp_modulator import ChirpModulator
from .gpu_backend import GPUBackend
from .models import OFDMConfig

logger = logging.getLogger(__name__)


class SubcarrierManager:
    """Manages frequency allocation and spacing for OFDM subcarriers.

    This class handles the organization of subcarriers within an OFDM signal,
    including frequency allocation, spacing validation, and signal assembly
    from multiple chirp-modulated subcarriers.

    Requirements addressed:
    - 1.1: Configurable number of subcarriers, bandwidth, spacing, and center frequency
    - 1.2: Symmetric positioning around center frequency
    - 1.3: Proper subcarrier spacing and bandwidth allocation
    """

    def __init__(self, ofdm_config: OFDMConfig, gpu_backend: Optional[GPUBackend] = None):
        """Initialize subcarrier manager.

        Args:
            ofdm_config: OFDM configuration parameters
            gpu_backend: GPU backend for acceleration (creates new if None)
        """
        self.ofdm_config = ofdm_config
        self.gpu_backend = gpu_backend or GPUBackend()

        # Validate configuration
        self._validate_subcarrier_configuration()

        # Pre-calculate subcarrier frequencies for efficiency
        self._subcarrier_frequencies = self._calculate_all_subcarrier_frequencies()

        # Initialize chirp modulator for signal generation
        self.chirp_modulator = ChirpModulator(ofdm_config, self.gpu_backend)

        logger.info(
            f"SubcarrierManager initialized: {ofdm_config.num_subcarriers} subcarriers, "
            f"spacing={ofdm_config.subcarrier_spacing}Hz, "
            f"center={ofdm_config.center_frequency}Hz"
        )

    def _validate_subcarrier_configuration(self) -> None:
        """Validate subcarrier configuration parameters.

        Requirements:
        - 1.1: Accept and validate configuration parameters
        - Ensure subcarrier spacing is sufficient for bandwidth
        - Validate total bandwidth fits within sampling constraints

        Raises:
            ValueError: If configuration parameters are invalid
        """
        config = self.ofdm_config

        # Check subcarrier spacing vs bandwidth
        if config.bandwidth_per_subcarrier > config.subcarrier_spacing:
            raise ValueError(
                f"Bandwidth per subcarrier ({config.bandwidth_per_subcarrier}Hz) "
                f"cannot exceed subcarrier spacing ({config.subcarrier_spacing}Hz)"
            )

        # Check total bandwidth vs sampling rate (Nyquist criterion)
        total_bandwidth = (
            config.num_subcarriers - 1
        ) * config.subcarrier_spacing + config.bandwidth_per_subcarrier
        nyquist_frequency = config.sampling_rate / 2.0

        if total_bandwidth > nyquist_frequency:
            raise ValueError(
                f"Total OFDM bandwidth ({total_bandwidth}Hz) exceeds "
                f"Nyquist frequency ({nyquist_frequency}Hz)"
            )

        # Check center frequency constraints
        min_freq = config.center_frequency - total_bandwidth / 2.0
        max_freq = config.center_frequency + total_bandwidth / 2.0

        if min_freq < 0:
            raise ValueError(
                f"Minimum subcarrier frequency ({min_freq}Hz) is negative. "
                f"Increase center frequency or reduce bandwidth."
            )

        logger.debug(
            f"Subcarrier configuration validated: "
            f"total_bandwidth={total_bandwidth}Hz, "
            f"frequency_range=[{min_freq}, {max_freq}]Hz"
        )

    def _calculate_all_subcarrier_frequencies(self) -> np.ndarray:
        """Calculate frequencies for all subcarriers.

        Requirements:
        - 1.2: Position subcarriers symmetrically around center frequency
        - 1.3: Apply configured subcarrier spacing

        Returns:
            Array of subcarrier center frequencies
        """
        num_subcarriers = self.ofdm_config.num_subcarriers
        spacing = self.ofdm_config.subcarrier_spacing
        center_freq = self.ofdm_config.center_frequency

        # Calculate center index for symmetric positioning
        if num_subcarriers % 2 == 0:
            # Even number: center between middle two subcarriers
            center_index = (num_subcarriers / 2.0) - 0.5
        else:
            # Odd number: center at middle subcarrier
            center_index = (num_subcarriers - 1) / 2.0

        # Calculate frequency offsets from center
        subcarrier_indices = np.arange(num_subcarriers)
        frequency_offsets = (subcarrier_indices - center_index) * spacing

        # Calculate absolute frequencies
        frequencies = center_freq + frequency_offsets

        logger.debug(
            f"Subcarrier frequencies calculated: "
            f"range=[{frequencies[0]:.1f}, {frequencies[-1]:.1f}]Hz, "
            f"center_index={center_index}"
        )

        return frequencies

    def get_subcarrier_frequency(self, subcarrier_index: int) -> float:
        """Get frequency for a specific subcarrier.

        Args:
            subcarrier_index: Index of the subcarrier (0-based)

        Returns:
            Subcarrier center frequency in Hz

        Raises:
            IndexError: If subcarrier index is out of range
        """
        if subcarrier_index < 0 or subcarrier_index >= self.ofdm_config.num_subcarriers:
            raise IndexError(
                f"Subcarrier index {subcarrier_index} out of range "
                f"[0, {self.ofdm_config.num_subcarriers-1}]"
            )

        return self._subcarrier_frequencies[subcarrier_index]

    def get_all_subcarrier_frequencies(self) -> np.ndarray:
        """Get frequencies for all subcarriers.

        Returns:
            Array of all subcarrier center frequencies
        """
        return self._subcarrier_frequencies.copy()

    def get_subcarrier_bandwidth_range(self, subcarrier_index: int) -> Tuple[float, float]:
        """Get frequency range (bandwidth) for a specific subcarrier.

        Args:
            subcarrier_index: Index of the subcarrier

        Returns:
            Tuple of (start_frequency, end_frequency) in Hz
        """
        center_freq = self.get_subcarrier_frequency(subcarrier_index)
        bandwidth = self.ofdm_config.bandwidth_per_subcarrier

        start_freq = center_freq - bandwidth / 2.0
        end_freq = center_freq + bandwidth / 2.0

        return (start_freq, end_freq)

    def check_subcarrier_overlap(self) -> List[Tuple[int, int]]:
        """Check for overlapping subcarriers.

        Returns:
            List of tuples (i, j) where subcarriers i and j overlap
        """
        overlaps = []

        for i in range(self.ofdm_config.num_subcarriers):
            for j in range(i + 1, self.ofdm_config.num_subcarriers):
                range_i = self.get_subcarrier_bandwidth_range(i)
                range_j = self.get_subcarrier_bandwidth_range(j)

                # Check if ranges overlap (not just touch)
                # Ranges overlap if one starts before the other ends, with some tolerance
                tolerance = 1e-10  # Small tolerance for floating point comparison
                if (range_i[1] - tolerance) > range_j[0] and (range_j[1] - tolerance) > range_i[0]:
                    overlaps.append((i, j))

        return overlaps

    def assemble_ofdm_signal(
        self, phase_array: np.ndarray, signal_length: Optional[int] = None
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Assemble OFDM signal from multiple chirp subcarriers.

        Requirements:
        - 1.1: Create specified number of subcarriers with configured spacing
        - 1.2: Symmetric positioning around center frequency
        - 1.3: Apply subcarrier spacing and bandwidth

        Args:
            phase_array: Phase offsets for each subcarrier [num_subcarriers]
            signal_length: Length of signal in samples (uses config duration if None)

        Returns:
            Assembled OFDM signal with all subcarriers

        Raises:
            ValueError: If phase array size doesn't match number of subcarriers
        """
        if len(phase_array) != self.ofdm_config.num_subcarriers:
            raise ValueError(
                f"Phase array length {len(phase_array)} doesn't match "
                f"number of subcarriers {self.ofdm_config.num_subcarriers}"
            )

        # Calculate signal length
        if signal_length is None:
            signal_length = int(self.ofdm_config.signal_duration * self.ofdm_config.sampling_rate)

        # Use chirp modulator to generate the multi-chirp signal
        ofdm_signal = self.chirp_modulator.generate_multi_chirp_signal(phase_array, signal_length)

        logger.debug(
            f"OFDM signal assembled: {self.ofdm_config.num_subcarriers} subcarriers, "
            f"{signal_length} samples"
        )

        return ofdm_signal

    def generate_frequency_domain_representation(
        self, time_signal: Union[np.ndarray, "cp.ndarray"]
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Generate frequency domain representation of OFDM signal.

        Args:
            time_signal: Time domain OFDM signal

        Returns:
            Frequency domain representation (FFT)
        """
        return self.gpu_backend.perform_fft(time_signal)

    def analyze_subcarrier_power(
        self, ofdm_signal: Union[np.ndarray, "cp.ndarray"]
    ) -> Dict[int, float]:
        """Analyze power distribution across subcarriers.

        Args:
            ofdm_signal: OFDM signal to analyze

        Returns:
            Dictionary mapping subcarrier index to power level
        """
        # Get frequency domain representation
        freq_signal = self.generate_frequency_domain_representation(ofdm_signal)

        # Convert to CPU for analysis if needed
        if hasattr(freq_signal, "get"):
            freq_signal = freq_signal.get()

        # Calculate frequency bins
        sampling_rate = self.ofdm_config.sampling_rate
        signal_length = len(freq_signal)
        freq_bins = np.fft.fftfreq(signal_length, 1.0 / sampling_rate)

        # Analyze power in each subcarrier band
        power_analysis = {}

        for i in range(self.ofdm_config.num_subcarriers):
            start_freq, end_freq = self.get_subcarrier_bandwidth_range(i)

            # Find frequency bins within this subcarrier's range
            mask = (freq_bins >= start_freq) & (freq_bins <= end_freq)

            # Calculate power in this band
            subcarrier_power = np.sum(np.abs(freq_signal[mask]) ** 2)
            power_analysis[i] = float(subcarrier_power)

        return power_analysis

    def get_ofdm_structure_info(self) -> Dict[str, Union[float, int, List[float]]]:
        """Get comprehensive information about OFDM structure.

        Returns:
            Dictionary with OFDM structure parameters
        """
        total_bandwidth = (
            self.ofdm_config.num_subcarriers - 1
        ) * self.ofdm_config.subcarrier_spacing + self.ofdm_config.bandwidth_per_subcarrier

        return {
            "num_subcarriers": self.ofdm_config.num_subcarriers,
            "subcarrier_spacing": self.ofdm_config.subcarrier_spacing,
            "bandwidth_per_subcarrier": self.ofdm_config.bandwidth_per_subcarrier,
            "center_frequency": self.ofdm_config.center_frequency,
            "total_bandwidth": total_bandwidth,
            "subcarrier_frequencies": self._subcarrier_frequencies.tolist(),
            "frequency_range": [
                float(
                    self._subcarrier_frequencies[0] - self.ofdm_config.bandwidth_per_subcarrier / 2
                ),
                float(
                    self._subcarrier_frequencies[-1] + self.ofdm_config.bandwidth_per_subcarrier / 2
                ),
            ],
            "overlapping_subcarriers": self.check_subcarrier_overlap(),
        }

    def validate_ofdm_structure(self) -> Dict[str, bool]:
        """Validate OFDM structure properties.

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Check frequency symmetry
        frequencies = self._subcarrier_frequencies
        center_freq = self.ofdm_config.center_frequency

        # For symmetric arrangement, average of first and last should equal center
        freq_center = (frequencies[0] + frequencies[-1]) / 2.0
        results["frequency_symmetry"] = bool(abs(freq_center - center_freq) < 1e-6)

        # Check spacing consistency
        if len(frequencies) > 1:
            spacings = np.diff(frequencies)
            expected_spacing = self.ofdm_config.subcarrier_spacing
            results["spacing_consistency"] = bool(
                np.allclose(spacings, expected_spacing, rtol=1e-10)
            )
        else:
            results["spacing_consistency"] = True

        # Check for overlaps
        overlaps = self.check_subcarrier_overlap()
        results["no_overlaps"] = bool(len(overlaps) == 0)

        # Check Nyquist criterion
        total_bandwidth = (
            self.ofdm_config.num_subcarriers - 1
        ) * self.ofdm_config.subcarrier_spacing + self.ofdm_config.bandwidth_per_subcarrier
        nyquist_freq = self.ofdm_config.sampling_rate / 2.0
        results["nyquist_satisfied"] = bool(total_bandwidth <= nyquist_freq)

        return results

    def __repr__(self) -> str:
        """String representation of SubcarrierManager."""
        return (
            f"SubcarrierManager(subcarriers={self.ofdm_config.num_subcarriers}, "
            f"spacing={self.ofdm_config.subcarrier_spacing}Hz, "
            f"center={self.ofdm_config.center_frequency}Hz, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'})"
        )
