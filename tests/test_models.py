"""
Unit tests for core data models.

Tests the OFDMConfig, ChirpConfig, and SignalSet classes including
their validation logic and methods.
"""

from datetime import datetime

import numpy as np
import pytest

from ofdm_chirp_generator.models import ChirpConfig, OFDMConfig, SignalSet
from ofdm_chirp_generator.validation import ValidationError


class TestOFDMConfig:
    """Test cases for OFDMConfig class."""

    def test_valid_ofdm_config(self):
        """Test creation of valid OFDM configuration."""
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        assert config.num_subcarriers == 64
        assert config.subcarrier_spacing == 15000.0
        assert config.bandwidth_per_subcarrier == 12000.0
        assert config.center_frequency == 2.4e9
        assert config.sampling_rate == 30.72e6
        assert config.signal_duration == 0.001

    def test_invalid_num_subcarriers(self):
        """Test validation of invalid num_subcarriers."""
        with pytest.raises(ValidationError, match="num_subcarriers must be an integer"):
            OFDMConfig(
                num_subcarriers=64.5,
                subcarrier_spacing=15000.0,
                bandwidth_per_subcarrier=12000.0,
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )

        with pytest.raises(ValidationError, match="num_subcarriers must be >= 1"):
            OFDMConfig(
                num_subcarriers=0,
                subcarrier_spacing=15000.0,
                bandwidth_per_subcarrier=12000.0,
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )

    def test_invalid_frequencies(self):
        """Test validation of invalid frequency parameters."""
        with pytest.raises(ValidationError, match="subcarrier_spacing must be positive"):
            OFDMConfig(
                num_subcarriers=64,
                subcarrier_spacing=-15000.0,
                bandwidth_per_subcarrier=12000.0,
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )

        with pytest.raises(ValidationError, match="bandwidth_per_subcarrier must be positive"):
            OFDMConfig(
                num_subcarriers=64,
                subcarrier_spacing=15000.0,
                bandwidth_per_subcarrier=0.0,
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )

    def test_nyquist_limit_validation(self):
        """Test validation of Nyquist limit constraint."""
        with pytest.raises(ValidationError, match="Total signal bandwidth.*exceeds Nyquist limit"):
            OFDMConfig(
                num_subcarriers=1000,
                subcarrier_spacing=50000.0,  # Total BW = 50 MHz
                bandwidth_per_subcarrier=40000.0,
                center_frequency=2.4e9,
                sampling_rate=30e6,  # Nyquist = 15 MHz
                signal_duration=0.001,
            )

    def test_subcarrier_overlap_validation(self):
        """Test validation of subcarrier overlap constraint."""
        with pytest.raises(
            ValidationError, match="bandwidth_per_subcarrier cannot exceed subcarrier_spacing"
        ):
            OFDMConfig(
                num_subcarriers=64,
                subcarrier_spacing=10000.0,
                bandwidth_per_subcarrier=15000.0,  # Exceeds spacing
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )


class TestChirpConfig:
    """Test cases for ChirpConfig class."""

    def test_valid_chirp_config(self):
        """Test creation of valid chirp configuration."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 64))
        config = ChirpConfig(chirp_length=1000, phase_matrix=phase_matrix, amplitude=1.5)

        assert config.chirp_length == 1000
        assert np.array_equal(config.phase_matrix, phase_matrix)
        assert config.amplitude == 1.5

    def test_default_amplitude(self):
        """Test default amplitude value."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 64))
        config = ChirpConfig(chirp_length=1000, phase_matrix=phase_matrix)

        assert config.amplitude == 1.0

    def test_invalid_chirp_length(self):
        """Test validation of invalid chirp length."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 64))

        with pytest.raises(ValidationError, match="chirp_length must be an integer"):
            ChirpConfig(chirp_length=1000.5, phase_matrix=phase_matrix)

        with pytest.raises(ValidationError, match="chirp_length must be >= 10"):
            ChirpConfig(chirp_length=5, phase_matrix=phase_matrix)

    def test_invalid_phase_matrix(self):
        """Test validation of invalid phase matrix."""
        with pytest.raises(ValidationError, match="phase_matrix must be a numpy array"):
            ChirpConfig(
                chirp_length=1000,
                phase_matrix=[[0, 1], [2, 3]],  # Python list instead of numpy array
            )

        with pytest.raises(ValidationError, match="phase_matrix must be 2-dimensional"):
            ChirpConfig(chirp_length=1000, phase_matrix=np.array([0, 1, 2, 3]))  # 1D array

        with pytest.raises(ValidationError, match="phase_matrix cannot be empty"):
            ChirpConfig(chirp_length=1000, phase_matrix=np.array([]).reshape(0, 0))

    def test_invalid_phase_values(self):
        """Test validation of phase values outside valid range."""
        with pytest.raises(ValidationError, match="All phase values must be in range"):
            ChirpConfig(
                chirp_length=1000,
                phase_matrix=np.array([[-1.0, 1.0], [2.0, 3.0]]),  # Negative value
            )

        with pytest.raises(ValidationError, match="All phase values must be in range"):
            ChirpConfig(
                chirp_length=1000, phase_matrix=np.array([[1.0, 2.0], [3.0, 7.0]])  # Value > 2π
            )

    def test_invalid_amplitude(self):
        """Test validation of invalid amplitude."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 64))

        with pytest.raises(ValidationError, match="amplitude must be a number"):
            ChirpConfig(
                chirp_length=1000,
                phase_matrix=phase_matrix,
                amplitude="1.0",  # String instead of number
            )

        with pytest.raises(ValidationError, match="amplitude must be >="):
            ChirpConfig(chirp_length=1000, phase_matrix=phase_matrix, amplitude=0.0)  # Too small


class TestSignalSet:
    """Test cases for SignalSet class."""

    def create_valid_signal_set(self):
        """Helper to create a valid signal set for testing."""
        signals = [np.random.randn(1000), np.random.randn(1000), np.random.randn(1000)]
        phases = np.random.uniform(0, 2 * np.pi, (3, 64))
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        return SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=0.95,
            generation_timestamp=datetime.now(),
            config=config,
        )

    def test_valid_signal_set(self):
        """Test creation of valid signal set."""
        signal_set = self.create_valid_signal_set()

        assert len(signal_set.signals) == 3
        assert signal_set.phases.shape == (3, 64)
        assert signal_set.orthogonality_score == 0.95
        assert isinstance(signal_set.generation_timestamp, datetime)
        assert isinstance(signal_set.metadata, dict)

    def test_signal_set_properties(self):
        """Test signal set properties."""
        signal_set = self.create_valid_signal_set()

        assert signal_set.num_signals == 3
        assert signal_set.signal_length == 1000

    def test_get_signal_method(self):
        """Test get_signal method."""
        signal_set = self.create_valid_signal_set()

        # Valid index
        signal = signal_set.get_signal(0)
        assert np.array_equal(signal, signal_set.signals[0])

        # Invalid indices
        with pytest.raises(IndexError, match="Signal index 3 out of range"):
            signal_set.get_signal(3)

        with pytest.raises(IndexError, match="Signal index -1 out of range"):
            signal_set.get_signal(-1)

    def test_invalid_signals_list(self):
        """Test validation of invalid signals list."""
        phases = np.random.uniform(0, 2 * np.pi, (3, 64))
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        with pytest.raises(ValidationError, match="signals must be a list"):
            SignalSet(
                signals=np.array([1, 2, 3]),  # Not a list
                phases=phases,
                orthogonality_score=0.95,
                generation_timestamp=datetime.now(),
                config=config,
            )

        with pytest.raises(ValidationError, match="signals list cannot be empty"):
            SignalSet(
                signals=[],
                phases=phases,
                orthogonality_score=0.95,
                generation_timestamp=datetime.now(),
                config=config,
            )

    def test_mismatched_signal_shapes(self):
        """Test validation of signals with different shapes."""
        signals = [
            np.random.randn(1000),
            np.random.randn(500),  # Different length
            np.random.randn(1000),
        ]
        phases = np.random.uniform(0, 2 * np.pi, (3, 64))
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        with pytest.raises(ValidationError, match="All signals must have the same shape"):
            SignalSet(
                signals=signals,
                phases=phases,
                orthogonality_score=0.95,
                generation_timestamp=datetime.now(),
                config=config,
            )

    def test_invalid_orthogonality_score(self):
        """Test validation of invalid orthogonality score."""
        signals = [np.random.randn(1000)]
        phases = np.random.uniform(0, 2 * np.pi, (1, 64))
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        with pytest.raises(ValidationError, match="orthogonality_score must be in range"):
            SignalSet(
                signals=signals,
                phases=phases,
                orthogonality_score=1.5,  # > 1.0
                generation_timestamp=datetime.now(),
                config=config,
            )

    def test_phase_signal_count_mismatch(self):
        """Test validation when phase matrix and signal count don't match."""
        signals = [np.random.randn(1000), np.random.randn(1000)]  # 2 signals
        phases = np.random.uniform(0, 2 * np.pi, (3, 64))  # 3 rows
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        with pytest.raises(
            ValidationError, match="phases array first dimension.*must match number of signals"
        ):
            SignalSet(
                signals=signals,
                phases=phases,
                orthogonality_score=0.95,
                generation_timestamp=datetime.now(),
                config=config,
            )
