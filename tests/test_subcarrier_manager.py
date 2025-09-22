"""
Unit tests for SubcarrierManager class.

Tests OFDM signal structure, frequency allocation, and signal assembly.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.subcarrier_manager import SubcarrierManager

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestSubcarrierManager:
    """Test suite for SubcarrierManager functionality."""

    @pytest.fixture
    def ofdm_config(self):
        """Standard OFDM configuration for testing."""
        return OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,  # 1 kHz
            bandwidth_per_subcarrier=500.0,  # 500 Hz
            center_frequency=10000.0,  # 10 kHz
            sampling_rate=50000.0,  # 50 kHz
            signal_duration=0.1,  # 100 ms
        )

    @pytest.fixture
    def gpu_backend(self):
        """GPU backend for testing."""
        return GPUBackend(force_cpu=True)  # Force CPU for consistent testing

    @pytest.fixture
    def subcarrier_manager(self, ofdm_config, gpu_backend):
        """SubcarrierManager instance for testing."""
        return SubcarrierManager(ofdm_config, gpu_backend)

    def test_initialization(self, ofdm_config, gpu_backend):
        """Test SubcarrierManager initialization."""
        manager = SubcarrierManager(ofdm_config, gpu_backend)

        assert manager.ofdm_config == ofdm_config
        assert manager.gpu_backend == gpu_backend
        assert manager._subcarrier_frequencies is not None
        assert len(manager._subcarrier_frequencies) == ofdm_config.num_subcarriers
        assert manager.chirp_modulator is not None

    def test_subcarrier_frequency_calculation(self, subcarrier_manager):
        """Test subcarrier frequency calculation and symmetry.

        Requirements: 1.1, 1.2 - Configurable parameters and symmetric positioning
        """
        config = subcarrier_manager.ofdm_config
        frequencies = subcarrier_manager.get_all_subcarrier_frequencies()

        # Test array properties
        assert len(frequencies) == config.num_subcarriers
        assert isinstance(frequencies, np.ndarray)

        # Test frequency symmetry around center frequency
        # For even number of subcarriers, average of first and last should equal center
        center_freq_calculated = (frequencies[0] + frequencies[-1]) / 2.0
        assert abs(center_freq_calculated - config.center_frequency) < 1e-6

        # Test subcarrier spacing
        if len(frequencies) > 1:
            spacings = np.diff(frequencies)
            expected_spacing = config.subcarrier_spacing
            np.testing.assert_allclose(spacings, expected_spacing, rtol=1e-10)

        # Test individual frequency access
        for i in range(config.num_subcarriers):
            freq = subcarrier_manager.get_subcarrier_frequency(i)
            assert abs(freq - frequencies[i]) < 1e-10

    def test_subcarrier_frequency_symmetry_odd_subcarriers(self):
        """Test frequency symmetry with odd number of subcarriers."""
        config = OFDMConfig(
            num_subcarriers=7,  # Odd number
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=500.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.1,
        )

        manager = SubcarrierManager(config)
        frequencies = manager.get_all_subcarrier_frequencies()

        # For odd number, middle subcarrier should be at center frequency
        middle_index = (config.num_subcarriers - 1) // 2
        assert abs(frequencies[middle_index] - config.center_frequency) < 1e-6

        # Check symmetry
        for i in range(config.num_subcarriers // 2):
            left_freq = frequencies[i]
            right_freq = frequencies[-(i + 1)]
            center_offset_left = abs(left_freq - config.center_frequency)
            center_offset_right = abs(right_freq - config.center_frequency)
            assert abs(center_offset_left - center_offset_right) < 1e-6

    def test_subcarrier_bandwidth_range(self, subcarrier_manager):
        """Test subcarrier bandwidth range calculation."""
        config = subcarrier_manager.ofdm_config

        for i in range(config.num_subcarriers):
            start_freq, end_freq = subcarrier_manager.get_subcarrier_bandwidth_range(i)
            center_freq = subcarrier_manager.get_subcarrier_frequency(i)

            # Check bandwidth calculation
            bandwidth = end_freq - start_freq
            assert abs(bandwidth - config.bandwidth_per_subcarrier) < 1e-10

            # Check center frequency
            calculated_center = (start_freq + end_freq) / 2.0
            assert abs(calculated_center - center_freq) < 1e-10

    def test_subcarrier_overlap_detection(self, subcarrier_manager):
        """Test detection of overlapping subcarriers."""
        # With current config (spacing=1000Hz, bandwidth=500Hz), no overlaps expected
        overlaps = subcarrier_manager.check_subcarrier_overlap()
        assert len(overlaps) == 0

        # Test overlap detection logic by manually creating a manager with overlapping config
        # We'll bypass the OFDMConfig validation by creating the manager directly
        from unittest.mock import patch

        from ofdm_chirp_generator.models import OFDMConfig

        # Create config that would normally fail validation
        with patch("ofdm_chirp_generator.validation.ConfigValidator.validate_ofdm_config"):
            overlap_config = OFDMConfig(
                num_subcarriers=4,
                subcarrier_spacing=400.0,  # 400 Hz spacing
                bandwidth_per_subcarrier=500.0,  # 500 Hz bandwidth - will overlap!
                center_frequency=5000.0,
                sampling_rate=20000.0,
                signal_duration=0.1,
            )

        # Create manager with bypassed validation
        overlap_manager = SubcarrierManager.__new__(SubcarrierManager)
        overlap_manager.ofdm_config = overlap_config
        overlap_manager.gpu_backend = GPUBackend(force_cpu=True)

        # Manually initialize without validation
        overlap_manager._subcarrier_frequencies = (
            overlap_manager._calculate_all_subcarrier_frequencies()
        )

        overlaps = overlap_manager.check_subcarrier_overlap()

        # Should detect overlaps between adjacent subcarriers
        assert len(overlaps) > 0

        # Verify overlap detection logic
        for i, j in overlaps:
            range_i = overlap_manager.get_subcarrier_bandwidth_range(i)
            range_j = overlap_manager.get_subcarrier_bandwidth_range(j)

            # Ranges should actually overlap (not just touch)
            tolerance = 1e-10
            assert (range_i[1] - tolerance) > range_j[0] and (range_j[1] - tolerance) > range_i[0]

    def test_ofdm_signal_assembly(self, subcarrier_manager):
        """Test OFDM signal assembly from chirp subcarriers.

        Requirements: 1.1, 1.2, 1.3 - Signal assembly with proper structure
        """
        config = subcarrier_manager.ofdm_config
        phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)

        # Test signal assembly
        ofdm_signal = subcarrier_manager.assemble_ofdm_signal(phase_array)

        # Check signal properties
        assert ofdm_signal is not None
        assert len(ofdm_signal) > 0
        assert np.iscomplexobj(ofdm_signal)

        # Check signal length
        expected_length = int(config.signal_duration * config.sampling_rate)
        assert len(ofdm_signal) == expected_length

        # Test with custom signal length
        custom_length = 2048
        custom_signal = subcarrier_manager.assemble_ofdm_signal(phase_array, custom_length)
        assert len(custom_signal) == custom_length

    def test_frequency_domain_representation(self, subcarrier_manager):
        """Test frequency domain analysis of OFDM signals."""
        config = subcarrier_manager.ofdm_config
        phase_array = np.zeros(config.num_subcarriers)  # All phases zero for simplicity

        # Generate OFDM signal
        time_signal = subcarrier_manager.assemble_ofdm_signal(phase_array)

        # Get frequency domain representation
        freq_signal = subcarrier_manager.generate_frequency_domain_representation(time_signal)

        # Check properties
        assert freq_signal is not None
        assert len(freq_signal) == len(time_signal)
        assert np.iscomplexobj(freq_signal)

        # Verify it's actually the FFT
        if hasattr(time_signal, "get"):
            time_signal_cpu = time_signal.get()
        else:
            time_signal_cpu = time_signal

        if hasattr(freq_signal, "get"):
            freq_signal_cpu = freq_signal.get()
        else:
            freq_signal_cpu = freq_signal

        expected_fft = np.fft.fft(time_signal_cpu)
        np.testing.assert_allclose(freq_signal_cpu, expected_fft, rtol=1e-10)

    def test_subcarrier_power_analysis(self, subcarrier_manager):
        """Test power analysis across subcarriers."""
        config = subcarrier_manager.ofdm_config

        # Create signal with different power levels per subcarrier
        phase_array = np.zeros(config.num_subcarriers)
        ofdm_signal = subcarrier_manager.assemble_ofdm_signal(phase_array)

        # Analyze power distribution
        power_analysis = subcarrier_manager.analyze_subcarrier_power(ofdm_signal)

        # Check results
        assert isinstance(power_analysis, dict)
        assert len(power_analysis) == config.num_subcarriers

        # All power values should be non-negative
        for i, power in power_analysis.items():
            assert isinstance(i, int)
            assert 0 <= i < config.num_subcarriers
            assert isinstance(power, float)
            assert power >= 0.0

    def test_ofdm_structure_info(self, subcarrier_manager):
        """Test OFDM structure information retrieval."""
        info = subcarrier_manager.get_ofdm_structure_info()
        config = subcarrier_manager.ofdm_config

        # Check required fields
        required_fields = [
            "num_subcarriers",
            "subcarrier_spacing",
            "bandwidth_per_subcarrier",
            "center_frequency",
            "total_bandwidth",
            "subcarrier_frequencies",
            "frequency_range",
            "overlapping_subcarriers",
        ]

        for field in required_fields:
            assert field in info

        # Verify values
        assert info["num_subcarriers"] == config.num_subcarriers
        assert info["subcarrier_spacing"] == config.subcarrier_spacing
        assert info["bandwidth_per_subcarrier"] == config.bandwidth_per_subcarrier
        assert info["center_frequency"] == config.center_frequency

        # Check calculated values
        expected_total_bw = (
            config.num_subcarriers - 1
        ) * config.subcarrier_spacing + config.bandwidth_per_subcarrier
        assert abs(info["total_bandwidth"] - expected_total_bw) < 1e-10

        # Check frequency list
        assert len(info["subcarrier_frequencies"]) == config.num_subcarriers

        # Check frequency range
        freq_range = info["frequency_range"]
        assert len(freq_range) == 2
        assert freq_range[0] < freq_range[1]

    def test_ofdm_structure_validation(self, subcarrier_manager):
        """Test OFDM structure validation."""
        validation = subcarrier_manager.validate_ofdm_structure()

        # Check validation fields
        required_fields = [
            "frequency_symmetry",
            "spacing_consistency",
            "no_overlaps",
            "nyquist_satisfied",
        ]
        for field in required_fields:
            assert field in validation
            assert isinstance(validation[field], bool)

        # With valid configuration, all should pass
        assert validation["frequency_symmetry"] is True
        assert validation["spacing_consistency"] is True
        assert validation["no_overlaps"] is True
        assert validation["nyquist_satisfied"] is True

    def test_invalid_subcarrier_index(self, subcarrier_manager):
        """Test error handling for invalid subcarrier indices."""
        config = subcarrier_manager.ofdm_config

        # Test negative index
        with pytest.raises(IndexError, match="out of range"):
            subcarrier_manager.get_subcarrier_frequency(-1)

        # Test index too large
        with pytest.raises(IndexError, match="out of range"):
            subcarrier_manager.get_subcarrier_frequency(config.num_subcarriers)

    def test_invalid_phase_array_size(self, subcarrier_manager):
        """Test error handling for invalid phase array sizes."""
        wrong_size_array = np.array([0.0, np.pi])  # Wrong size

        with pytest.raises(ValueError, match="Phase array length"):
            subcarrier_manager.assemble_ofdm_signal(wrong_size_array)

    def test_configuration_validation_errors(self):
        """Test configuration validation error cases."""
        # Test bandwidth > spacing - this is caught by OFDMConfig validation
        from ofdm_chirp_generator.validation import ValidationError

        with pytest.raises(
            ValidationError, match="bandwidth_per_subcarrier cannot exceed subcarrier_spacing"
        ):
            invalid_config = OFDMConfig(
                num_subcarriers=4,
                subcarrier_spacing=500.0,
                bandwidth_per_subcarrier=600.0,  # > spacing
                center_frequency=5000.0,
                sampling_rate=20000.0,
                signal_duration=0.1,
            )

        # Test total bandwidth > Nyquist frequency - this is also caught by OFDMConfig validation
        with pytest.raises(ValidationError, match="Total signal bandwidth.*exceeds.*Nyquist"):
            invalid_config = OFDMConfig(
                num_subcarriers=20,
                subcarrier_spacing=2000.0,  # 20 * 2000 = 40kHz total
                bandwidth_per_subcarrier=1000.0,
                center_frequency=25000.0,
                sampling_rate=50000.0,  # Nyquist = 25kHz < 40kHz
                signal_duration=0.1,
            )

        # Test SubcarrierManager specific validation - negative frequency
        from unittest.mock import patch

        with patch("ofdm_chirp_generator.validation.ConfigValidator.validate_ofdm_config"):
            invalid_config = OFDMConfig(
                num_subcarriers=10,
                subcarrier_spacing=1000.0,
                bandwidth_per_subcarrier=500.0,
                center_frequency=2000.0,  # Too low for 10 subcarriers
                sampling_rate=50000.0,
                signal_duration=0.1,
            )

        with pytest.raises(ValueError, match="Minimum subcarrier frequency.*is negative"):
            SubcarrierManager(invalid_config)

    def test_repr_string(self, subcarrier_manager):
        """Test string representation of SubcarrierManager."""
        repr_str = repr(subcarrier_manager)

        assert "SubcarrierManager" in repr_str
        assert str(subcarrier_manager.ofdm_config.num_subcarriers) in repr_str
        assert str(subcarrier_manager.ofdm_config.subcarrier_spacing) in repr_str
        assert str(subcarrier_manager.ofdm_config.center_frequency) in repr_str
        assert "CPU" in repr_str or "GPU" in repr_str


class TestSubcarrierManagerFrequencyDomainProperties:
    """Test frequency domain properties of OFDM signals."""

    @pytest.fixture
    def test_config(self):
        """Configuration optimized for frequency domain testing."""
        return OFDMConfig(
            num_subcarriers=16,
            subcarrier_spacing=1000.0,  # 1 kHz
            bandwidth_per_subcarrier=800.0,  # 800 Hz
            center_frequency=16000.0,  # 16 kHz
            sampling_rate=64000.0,  # 64 kHz
            signal_duration=0.064,  # 64 ms for nice FFT size (4096 samples)
        )

    def test_frequency_domain_subcarrier_separation(self, test_config):
        """Test that subcarriers are properly separated in frequency domain."""
        manager = SubcarrierManager(test_config)

        # Generate signal with single active subcarrier
        for active_subcarrier in [
            0,
            test_config.num_subcarriers // 2,
            test_config.num_subcarriers - 1,
        ]:
            phase_array = np.zeros(test_config.num_subcarriers)
            # Only activate one subcarrier
            phase_array[active_subcarrier] = 0.0

            # Create signal with only one active subcarrier by setting others to zero amplitude
            # We'll generate individual chirps and combine manually
            time_signal = manager.chirp_modulator.generate_chirp_signal(active_subcarrier, 0.0)

            # Analyze frequency content
            freq_signal = manager.generate_frequency_domain_representation(time_signal)
            power_analysis = manager.analyze_subcarrier_power(time_signal)

            # The active subcarrier should have most of the power
            max_power_subcarrier = max(power_analysis.keys(), key=lambda k: power_analysis[k])
            assert max_power_subcarrier == active_subcarrier

    def test_ofdm_signal_frequency_properties(self, test_config):
        """Test frequency properties of complete OFDM signal."""
        manager = SubcarrierManager(test_config)

        # Generate OFDM signal with all subcarriers active
        phase_array = np.random.uniform(0, 2 * np.pi, test_config.num_subcarriers)
        ofdm_signal = manager.assemble_ofdm_signal(phase_array)

        # Analyze frequency domain
        freq_signal = manager.generate_frequency_domain_representation(ofdm_signal)

        # Convert to CPU for analysis
        if hasattr(freq_signal, "get"):
            freq_signal = freq_signal.get()

        # Calculate frequency bins
        signal_length = len(freq_signal)
        freq_bins = np.fft.fftfreq(signal_length, 1.0 / test_config.sampling_rate)

        # Check that energy is concentrated around subcarrier frequencies
        subcarrier_freqs = manager.get_all_subcarrier_frequencies()

        for i, target_freq in enumerate(subcarrier_freqs):
            # Find closest frequency bin
            closest_bin = np.argmin(np.abs(freq_bins - target_freq))

            # Check that there's significant energy near this frequency
            # (This is a basic check - more sophisticated analysis could be done)
            energy_at_freq = np.abs(freq_signal[closest_bin]) ** 2
            assert energy_at_freq > 0  # Should have some energy

    def test_subcarrier_orthogonality_frequency_domain(self, test_config):
        """Test orthogonality properties in frequency domain."""
        manager = SubcarrierManager(test_config)

        # Generate two different OFDM signals
        phase_array1 = np.zeros(test_config.num_subcarriers)
        phase_array2 = np.ones(test_config.num_subcarriers) * np.pi  # π phase shift

        signal1 = manager.assemble_ofdm_signal(phase_array1)
        signal2 = manager.assemble_ofdm_signal(phase_array2)

        # Compute cross-correlation using GPU backend
        correlation = manager.gpu_backend.compute_correlation(signal1, signal2)

        # Signals with π phase difference should have low correlation
        # (This tests the infrastructure for orthogonality analysis)
        assert isinstance(correlation, float)
        assert correlation >= 0.0  # Correlation magnitude should be non-negative


class TestSubcarrierManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_subcarrier(self):
        """Test with single subcarrier configuration."""
        config = OFDMConfig(
            num_subcarriers=1,
            subcarrier_spacing=1000.0,  # Not used with single subcarrier
            bandwidth_per_subcarrier=500.0,
            center_frequency=5000.0,
            sampling_rate=20000.0,
            signal_duration=0.1,
        )

        manager = SubcarrierManager(config)

        # Single subcarrier should be at center frequency
        freq = manager.get_subcarrier_frequency(0)
        assert abs(freq - config.center_frequency) < 1e-10

        # Should be able to generate signal
        phase_array = np.array([0.0])
        signal = manager.assemble_ofdm_signal(phase_array)
        assert signal is not None
        assert len(signal) > 0

    def test_maximum_subcarriers(self):
        """Test with large number of subcarriers."""
        config = OFDMConfig(
            num_subcarriers=128,
            subcarrier_spacing=100.0,  # 100 Hz spacing
            bandwidth_per_subcarrier=80.0,  # 80 Hz bandwidth
            center_frequency=50000.0,  # 50 kHz center
            sampling_rate=200000.0,  # 200 kHz sampling
            signal_duration=0.1,
        )

        manager = SubcarrierManager(config)

        # Should handle large configurations
        phase_array = np.random.uniform(0, 2 * np.pi, 128)
        signal = manager.assemble_ofdm_signal(phase_array)
        assert signal is not None
        assert len(signal) > 0

        # Validation should still pass
        validation = manager.validate_ofdm_structure()
        assert all(validation.values())

    def test_minimal_spacing(self):
        """Test with minimal valid spacing configuration."""
        config = OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=1000.0,  # Equal to spacing (boundary case)
            center_frequency=5000.0,
            sampling_rate=20000.0,
            signal_duration=0.1,
        )

        manager = SubcarrierManager(config)

        # With equal spacing and bandwidth, subcarriers should touch but not overlap
        overlaps = manager.check_subcarrier_overlap()
        assert len(overlaps) == 0  # Should not overlap, just touch

        # Signal generation should still work
        phase_array = np.random.uniform(0, 2 * np.pi, 4)
        signal = manager.assemble_ofdm_signal(phase_array)
        assert signal is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
