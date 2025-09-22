"""
Unit tests for validation module.

Tests the ConfigValidator class and its validation methods for all
configuration parameters and cross-parameter compatibility.
"""

from datetime import datetime

import numpy as np
import pytest

from ofdm_chirp_generator.models import ChirpConfig, OFDMConfig, SignalSet
from ofdm_chirp_generator.validation import ConfigValidator, ValidationError


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    def test_validate_ofdm_config_valid(self):
        """Test validation of valid OFDM configuration."""
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        # Should not raise any exception
        ConfigValidator.validate_ofdm_config(config)

    def test_validate_ofdm_config_boundary_values(self):
        """Test validation with boundary values."""
        # Minimum values
        config_min = OFDMConfig(
            num_subcarriers=1,
            subcarrier_spacing=1.0,
            bandwidth_per_subcarrier=0.5,
            center_frequency=0.0,
            sampling_rate=1000.0,
            signal_duration=1e-6,
        )
        ConfigValidator.validate_ofdm_config(config_min)

        # Maximum values (respecting Nyquist limit)
        config_max = OFDMConfig(
            num_subcarriers=1000,  # Reduced to respect Nyquist
            subcarrier_spacing=5e5,  # 500 kHz spacing
            bandwidth_per_subcarrier=4e5,  # 400 kHz per subcarrier
            center_frequency=1e12,
            sampling_rate=1e9,  # 1 GHz sampling rate (Nyquist = 500 MHz)
            signal_duration=3600.0,
        )
        ConfigValidator.validate_ofdm_config(config_max)

    def test_validate_chirp_config_valid(self):
        """Test validation of valid chirp configuration."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 64))
        config = ChirpConfig(chirp_length=1000, phase_matrix=phase_matrix, amplitude=1.5)

        # Should not raise any exception
        ConfigValidator.validate_chirp_config(config)

    def test_validate_chirp_config_boundary_phases(self):
        """Test validation with boundary phase values."""
        # All zeros (minimum)
        phase_matrix_min = np.zeros((2, 32))
        config_min = ChirpConfig(chirp_length=100, phase_matrix=phase_matrix_min, amplitude=1e-10)
        ConfigValidator.validate_chirp_config(config_min)

        # All 2π (maximum)
        phase_matrix_max = np.full((2, 32), 2 * np.pi)
        config_max = ChirpConfig(
            chirp_length=1000000, phase_matrix=phase_matrix_max, amplitude=1e10
        )
        ConfigValidator.validate_chirp_config(config_max)

    def test_validate_signal_set_valid(self):
        """Test validation of valid signal set."""
        signals = [np.random.randn(1000) for _ in range(3)]
        phases = np.random.uniform(0, 2 * np.pi, (3, 64))
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        signal_set = SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=0.95,
            generation_timestamp=datetime.now(),
            config=config,
            metadata={"test": "value"},
        )

        # Should not raise any exception
        ConfigValidator.validate_signal_set(signal_set)

    def test_validate_parameter_compatibility_valid(self):
        """Test validation of compatible OFDM and chirp configurations."""
        ofdm_config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 64))  # 64 subcarriers
        chirp_config = ChirpConfig(
            chirp_length=500,  # Less than total samples (30720)
            phase_matrix=phase_matrix,
            amplitude=1.0,
        )

        # Should not raise any exception
        ConfigValidator.validate_parameter_compatibility(ofdm_config, chirp_config)

    def test_validate_parameter_compatibility_dimension_mismatch(self):
        """Test validation with mismatched dimensions."""
        ofdm_config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )

        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 32))  # 32 != 64
        chirp_config = ChirpConfig(chirp_length=500, phase_matrix=phase_matrix, amplitude=1.0)

        with pytest.raises(
            ValidationError, match="Phase matrix has 32 columns.*but OFDM config specifies 64"
        ):
            ConfigValidator.validate_parameter_compatibility(ofdm_config, chirp_config)

    def test_validate_parameter_compatibility_chirp_too_long(self):
        """Test validation when chirp length exceeds signal duration."""
        ofdm_config = OFDMConfig(
            num_subcarriers=4,  # Reduced to respect Nyquist
            subcarrier_spacing=100.0,  # Total BW = 400 Hz
            bandwidth_per_subcarrier=80.0,
            center_frequency=2.4e9,
            sampling_rate=1000.0,  # Nyquist = 500 Hz
            signal_duration=0.001,  # Only 1 sample total
        )

        phase_matrix = np.random.uniform(0, 2 * np.pi, (4, 4))
        chirp_config = ChirpConfig(
            chirp_length=1000, phase_matrix=phase_matrix, amplitude=1.0  # Much longer than 1 sample
        )

        with pytest.raises(ValidationError, match="Chirp length.*exceeds total signal samples"):
            ConfigValidator.validate_parameter_compatibility(ofdm_config, chirp_config)

    def test_validation_error_messages(self):
        """Test that validation errors contain helpful messages."""
        # Test specific error message content
        with pytest.raises(ValidationError) as exc_info:
            OFDMConfig(
                num_subcarriers=0,
                subcarrier_spacing=15000.0,
                bandwidth_per_subcarrier=12000.0,
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )

        assert "num_subcarriers must be >= 1" in str(exc_info.value)

    def test_type_validation(self):
        """Test validation of parameter types."""
        # Test non-numeric types
        with pytest.raises(ValidationError, match="must be a number"):
            OFDMConfig(
                num_subcarriers=64,
                subcarrier_spacing="15000",  # String instead of number
                bandwidth_per_subcarrier=12000.0,
                center_frequency=2.4e9,
                sampling_rate=30.72e6,
                signal_duration=0.001,
            )

        # Test non-integer for integer field
        with pytest.raises(ValidationError, match="must be an integer"):
            ChirpConfig(
                chirp_length=1000.5,  # Float instead of int
                phase_matrix=np.random.uniform(0, 2 * np.pi, (4, 64)),
                amplitude=1.0,
            )

    def test_edge_case_validations(self):
        """Test edge cases and corner conditions."""
        # Test exactly at Nyquist limit
        config = OFDMConfig(
            num_subcarriers=100,
            subcarrier_spacing=150.0,  # Total BW = 15 kHz
            bandwidth_per_subcarrier=100.0,
            center_frequency=1e6,
            sampling_rate=30000.0,  # Nyquist = 15 kHz
            signal_duration=0.001,
        )
        ConfigValidator.validate_ofdm_config(config)

        # Test single subcarrier
        config_single = OFDMConfig(
            num_subcarriers=1,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=500.0,
            center_frequency=1e6,
            sampling_rate=10000.0,
            signal_duration=0.001,
        )
        ConfigValidator.validate_ofdm_config(config_single)

    def test_cross_validation_constraints(self):
        """Test cross-parameter validation constraints."""
        # Test bandwidth equals spacing (boundary case)
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=15000.0,  # Equal to spacing
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=0.001,
        )
        ConfigValidator.validate_ofdm_config(config)

        # Test very small signal duration
        config_short = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,
            bandwidth_per_subcarrier=12000.0,
            center_frequency=2.4e9,
            sampling_rate=30.72e6,
            signal_duration=1e-6,  # Minimum duration
        )
        ConfigValidator.validate_ofdm_config(config_short)
