"""
Core data models for OFDM chirp signal generation.

This module defines the primary data structures used throughout the system
for configuration management and signal representation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class OFDMConfig:
    """Configuration parameters for OFDM signal generation.

    Attributes:
        num_subcarriers: Number of subcarriers in the OFDM signal
        subcarrier_spacing: Frequency spacing between subcarriers in Hz
        bandwidth_per_subcarrier: Bandwidth allocated to each subcarrier in Hz
        center_frequency: Center frequency of the OFDM signal in Hz
        sampling_rate: Sampling rate for signal generation in Hz
        signal_duration: Duration of the generated signal in seconds
    """

    num_subcarriers: int
    subcarrier_spacing: float
    bandwidth_per_subcarrier: float
    center_frequency: float
    sampling_rate: float
    signal_duration: float

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        from .validation import ConfigValidator

        ConfigValidator.validate_ofdm_config(self)


@dataclass
class ChirpConfig:
    """Configuration parameters for chirp signal modulation.

    Attributes:
        chirp_length: Length of each chirp signal in samples
        phase_matrix: Phase offsets for each signal and subcarrier [signals x subcarriers]
        amplitude: Signal amplitude scaling factor
    """

    chirp_length: int
    phase_matrix: np.ndarray
    amplitude: float = 1.0

    def __post_init__(self):
        """Validate chirp configuration parameters after initialization."""
        from .validation import ConfigValidator

        ConfigValidator.validate_chirp_config(self)


@dataclass
class SignalSet:
    """Container for a set of generated orthogonal OFDM signals.

    Attributes:
        signals: List of generated signal arrays
        phases: Phase matrix used for signal generation
        orthogonality_score: Measure of orthogonality between signals (0-1)
        generation_timestamp: When the signals were generated
        config: OFDM configuration used for generation
        metadata: Additional information about the signal set
    """

    signals: List[np.ndarray]
    phases: np.ndarray
    orthogonality_score: float
    generation_timestamp: datetime
    config: OFDMConfig
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal set parameters after initialization."""
        from .validation import ConfigValidator

        ConfigValidator.validate_signal_set(self)

    @property
    def num_signals(self) -> int:
        """Number of signals in the set."""
        return len(self.signals)

    @property
    def signal_length(self) -> int:
        """Length of each signal in samples."""
        return len(self.signals[0]) if self.signals else 0

    def get_signal(self, index: int) -> np.ndarray:
        """Get a specific signal by index.

        Args:
            index: Signal index (0-based)

        Returns:
            Signal array

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.signals):
            raise IndexError(f"Signal index {index} out of range [0, {len(self.signals)-1}]")
        return self.signals[index]
