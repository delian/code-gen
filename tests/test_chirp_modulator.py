"""
Unit tests for ChirpModulator class.

Tests chirp signal generation, phase accuracy, and GPU/CPU compatibility.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.chirp_modulator import ChirpModulator
from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestChirpModulator:
    """Test suite for ChirpModulator functionality."""

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
    def chirp_modulator(self, ofdm_config, gpu_backend):
        """ChirpModulator instance for testing."""
        return ChirpModulator(ofdm_config, gpu_backend)

    def test_initialization(self, ofdm_config, gpu_backend):
        """Test ChirpModulator initialization."""
        modulator = ChirpModulator(ofdm_config, gpu_backend)

        assert modulator.ofdm_config == ofdm_config
        assert modulator.gpu_backend == gpu_backend
        assert modulator._time_vector is not None
        assert len(modulator._time_vector) == int(
            ofdm_config.signal_duration * ofdm_config.sampling_rate
        )

    def test_chirp_length_validation(self, chirp_modulator):
        """Test chirp length validation and constraints."""
        # Test valid length
        valid_length = 1000
        result = chirp_modulator.validate_chirp_length(valid_length)
        assert result == valid_length

        # Test length below minimum
        min_length, max_length = chirp_modulator.chirp_length_constraints
        too_small = min_length - 10
        result = chirp_modulator.validate_chirp_length(too_small)
        assert result == min_length

        # Test length above maximum
        too_large = max_length + 1000
        result = chirp_modulator.validate_chirp_length(too_large)
        assert result == max_length

    def test_subcarrier_frequency_calculation(self, chirp_modulator):
        """Test subcarrier frequency calculation."""
        config = chirp_modulator.ofdm_config

        # For 8 subcarriers (even), center should be between indices 3 and 4
        # So center_index = 3.5, and frequencies should be symmetric around center_frequency

        # Test symmetry: subcarriers should be symmetric around center frequency
        frequencies = []
        for i in range(config.num_subcarriers):
            freq = chirp_modulator._calculate_subcarrier_frequency(i)
            frequencies.append(freq)

        # Check that frequencies are symmetric around center frequency
        center_freq_avg = (frequencies[0] + frequencies[-1]) / 2.0
        assert abs(center_freq_avg - config.center_frequency) < 1e-6

        # Check spacing between adjacent subcarriers
        for i in range(1, len(frequencies)):
            spacing = frequencies[i] - frequencies[i - 1]
            assert abs(spacing - config.subcarrier_spacing) < 1e-6

    def test_single_chirp_generation(self, chirp_modulator):
        """Test generation of single chirp signal."""
        subcarrier_index = 2
        phase_offset = np.pi / 4

        chirp_signal = chirp_modulator.generate_chirp_signal(subcarrier_index, phase_offset)

        # Check signal properties
        assert chirp_signal is not None
        assert len(chirp_signal) > 0
        assert np.iscomplexobj(chirp_signal)

        # Check signal amplitude (should be normalized)
        expected_amplitude = 1.0 / np.sqrt(chirp_modulator.ofdm_config.num_subcarriers)
        actual_amplitude = np.mean(np.abs(chirp_signal))
        assert abs(actual_amplitude - expected_amplitude) < 0.1

    def test_chirp_signal_characteristics(self, chirp_modulator):
        """Test chirp signal characteristics and linearity."""
        subcarrier_index = 3
        phase_offset = 0.0

        # Generate chirp signal
        chirp_signal = chirp_modulator.generate_chirp_signal(subcarrier_index, phase_offset)

        # Get expected characteristics
        characteristics = chirp_modulator.get_chirp_characteristics(subcarrier_index, phase_offset)

        # Verify characteristics
        assert characteristics["subcarrier_index"] == subcarrier_index
        assert characteristics["phase_offset"] == phase_offset
        assert characteristics["bandwidth"] == chirp_modulator.ofdm_config.bandwidth_per_subcarrier
        assert characteristics["duration"] == chirp_modulator.ofdm_config.signal_duration

        # Test frequency sweep linearity by analyzing instantaneous frequency
        dt = 1.0 / chirp_modulator.ofdm_config.sampling_rate

        # Calculate instantaneous frequency from phase derivative
        phase = np.angle(chirp_signal)
        # Unwrap phase to handle 2π discontinuities
        unwrapped_phase = np.unwrap(phase)
        # Compute frequency as derivative of phase
        inst_freq = np.diff(unwrapped_phase) / (2 * np.pi * dt)

        # Check that frequency sweep is approximately linear
        # (allowing for some numerical noise)
        time_samples = np.arange(len(inst_freq)) * dt
        freq_trend = np.polyfit(time_samples, inst_freq, 1)
        sweep_rate = freq_trend[0]  # Slope of frequency vs time

        expected_sweep_rate = characteristics["bandwidth"] / characteristics["duration"]
        # Allow larger tolerance for numerical approximation due to discrete differentiation
        tolerance = 0.2 * expected_sweep_rate  # 20% tolerance
        assert abs(sweep_rate - expected_sweep_rate) < tolerance

    def test_phase_offset_accuracy(self, chirp_modulator):
        """Test phase offset accuracy in generated chirps."""
        subcarrier_index = 1
        phase_offsets = [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

        chirp_signals = []
        for phase in phase_offsets:
            signal = chirp_modulator.generate_chirp_signal(subcarrier_index, phase)
            chirp_signals.append(signal)

        # Compare phase differences between signals
        reference_signal = chirp_signals[0]  # 0 phase offset

        for i, expected_phase in enumerate(phase_offsets[1:], 1):
            test_signal = chirp_signals[i]

            # Calculate phase difference at the start of the signal
            ref_phase = np.angle(reference_signal[0])
            test_phase = np.angle(test_signal[0])
            phase_diff = test_phase - ref_phase

            # Normalize to [0, 2π]
            phase_diff = np.mod(phase_diff, 2 * np.pi)
            expected_normalized = np.mod(expected_phase, 2 * np.pi)

            # Allow small numerical tolerance
            assert abs(phase_diff - expected_normalized) < 0.01

    def test_multi_chirp_generation(self, chirp_modulator):
        """Test generation of multi-chirp OFDM signal."""
        num_subcarriers = chirp_modulator.ofdm_config.num_subcarriers
        phase_array = np.random.uniform(0, 2 * np.pi, num_subcarriers)

        combined_signal = chirp_modulator.generate_multi_chirp_signal(phase_array)

        # Check signal properties
        assert combined_signal is not None
        assert len(combined_signal) > 0
        assert np.iscomplexobj(combined_signal)

        # Signal should be sum of individual chirps
        # Generate individual chirps and sum them
        expected_signal = np.zeros_like(combined_signal)
        for i, phase in enumerate(phase_array):
            chirp = chirp_modulator.generate_chirp_signal(i, phase, len(combined_signal))
            expected_signal += chirp

        # Compare signals (allow for small numerical differences)
        if hasattr(combined_signal, "get"):  # CuPy array
            combined_signal = combined_signal.get()
        if hasattr(expected_signal, "get"):  # CuPy array
            expected_signal = expected_signal.get()

        correlation = np.corrcoef(np.real(combined_signal), np.real(expected_signal))[0, 1]
        assert correlation > 0.99  # Very high correlation expected

    def test_invalid_subcarrier_index(self, chirp_modulator):
        """Test error handling for invalid subcarrier indices."""
        num_subcarriers = chirp_modulator.ofdm_config.num_subcarriers

        # Test negative index
        with pytest.raises(ValueError, match="Invalid subcarrier index"):
            chirp_modulator.generate_chirp_signal(-1)

        # Test index too large
        with pytest.raises(ValueError, match="Invalid subcarrier index"):
            chirp_modulator.generate_chirp_signal(num_subcarriers)

    def test_invalid_phase_array_size(self, chirp_modulator):
        """Test error handling for invalid phase array sizes."""
        wrong_size_array = np.array([0.0, np.pi])  # Wrong size

        with pytest.raises(ValueError, match="Phase array length"):
            chirp_modulator.generate_multi_chirp_signal(wrong_size_array)

    def test_phase_array_validation(self, chirp_modulator):
        """Test phase array validation and normalization."""
        num_subcarriers = chirp_modulator.ofdm_config.num_subcarriers

        # Test with phases outside [0, 2π] range
        phases = np.array(
            [3 * np.pi, -np.pi / 2, 5 * np.pi, -2 * np.pi] + [0.0] * (num_subcarriers - 4)
        )

        normalized = chirp_modulator.validate_phase_array(phases)

        # All phases should be in [0, 2π] range
        assert np.all(normalized >= 0.0)
        assert np.all(normalized < 2 * np.pi)

        # Check specific normalizations
        assert abs(normalized[0] - np.pi) < 1e-10  # 3π -> π
        assert abs(normalized[1] - (3 * np.pi / 2)) < 1e-10  # -π/2 -> 3π/2

    def test_chirp_length_constraints_property(self, chirp_modulator):
        """Test chirp length constraints property."""
        min_length, max_length = chirp_modulator.chirp_length_constraints

        assert isinstance(min_length, int)
        assert isinstance(max_length, int)
        assert min_length > 0
        assert max_length > min_length
        assert min_length >= 32  # Minimum constraint

    def test_repr_string(self, chirp_modulator):
        """Test string representation of ChirpModulator."""
        repr_str = repr(chirp_modulator)

        assert "ChirpModulator" in repr_str
        assert str(chirp_modulator.ofdm_config.num_subcarriers) in repr_str
        assert str(chirp_modulator.ofdm_config.bandwidth_per_subcarrier) in repr_str
        assert "CPU" in repr_str or "GPU" in repr_str

    def test_gpu_cpu_compatibility(self, ofdm_config):
        """Test that GPU and CPU backends produce identical results."""
        # Create CPU backend
        cpu_backend = GPUBackend(force_cpu=True)
        cpu_modulator = ChirpModulator(ofdm_config, cpu_backend)

        # Generate signal with CPU
        phase_array = np.array(
            [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 0.0, np.pi / 3, 2 * np.pi / 3]
        )
        cpu_signal = cpu_modulator.generate_multi_chirp_signal(phase_array)

        # Convert to numpy for comparison
        if hasattr(cpu_signal, "get"):
            cpu_signal = cpu_signal.get()

        # If GPU is available, test GPU backend too
        try:
            gpu_backend = GPUBackend(force_cpu=False)
            if gpu_backend.is_gpu_available:
                gpu_modulator = ChirpModulator(ofdm_config, gpu_backend)
                gpu_signal = gpu_modulator.generate_multi_chirp_signal(phase_array)

                if hasattr(gpu_signal, "get"):
                    gpu_signal = gpu_signal.get()

                # Signals should be nearly identical
                np.testing.assert_allclose(cpu_signal, gpu_signal, rtol=1e-10, atol=1e-12)
        except ImportError:
            # CuPy not available, skip GPU test
            pass

    def test_memory_management(self, chirp_modulator):
        """Test memory management during signal generation."""
        # Generate multiple signals to test memory handling
        phase_arrays = [
            np.random.uniform(0, 2 * np.pi, chirp_modulator.ofdm_config.num_subcarriers)
            for _ in range(10)
        ]

        signals = []
        for phases in phase_arrays:
            signal = chirp_modulator.generate_multi_chirp_signal(phases)
            signals.append(signal)

        # All signals should be generated successfully
        assert len(signals) == 10
        for signal in signals:
            assert signal is not None
            assert len(signal) > 0

        # Test cleanup
        chirp_modulator.gpu_backend.cleanup_memory()


class TestChirpModulatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_minimal_configuration(self):
        """Test with minimal valid configuration."""
        config = OFDMConfig(
            num_subcarriers=2,
            subcarrier_spacing=100.0,
            bandwidth_per_subcarrier=50.0,
            center_frequency=1000.0,
            sampling_rate=2000.0,  # Higher sampling rate to satisfy Nyquist
            signal_duration=0.1,  # Longer duration to allow sufficient chirp length
        )

        modulator = ChirpModulator(config)

        # Should be able to generate signals
        signal = modulator.generate_chirp_signal(0, 0.0)
        assert signal is not None
        assert len(signal) > 0

    def test_large_configuration(self):
        """Test with large configuration parameters."""
        config = OFDMConfig(
            num_subcarriers=64,
            subcarrier_spacing=15000.0,  # 15 kHz (LTE-like)
            bandwidth_per_subcarrier=15000.0,
            center_frequency=2400000000.0,  # 2.4 GHz
            sampling_rate=30720000.0,  # 30.72 MHz
            signal_duration=0.001,  # 1 ms
        )

        modulator = ChirpModulator(config)

        # Should handle large configurations
        phases = np.random.uniform(0, 2 * np.pi, 64)
        signal = modulator.generate_multi_chirp_signal(phases)
        assert signal is not None
        assert len(signal) > 0

    def test_invalid_configuration_constraints(self):
        """Test invalid configuration that violates constraints."""
        # Create a config that passes OFDMConfig validation but fails ChirpModulator
        # We need: bandwidth_per_subcarrier <= subcarrier_spacing
        # And: total_bandwidth <= sampling_rate/2
        # But: min_chirp_length > max_signal_length

        config = OFDMConfig(
            num_subcarriers=2,
            subcarrier_spacing=1000.0,  # 1 kHz spacing
            bandwidth_per_subcarrier=500.0,  # 500 Hz bandwidth (< spacing)
            center_frequency=5000.0,
            sampling_rate=10000.0,  # Total BW = 2*1000 = 2kHz < 5kHz (Nyquist)
            signal_duration=0.0001,  # Very short: 0.1ms = 1 sample at 10kHz
        )

        # This should fail in ChirpModulator due to chirp length constraints
        # min_chirp_length = sampling_rate/bandwidth = 10000/500 = 20 samples
        # max_signal_length = duration * sampling_rate = 0.0001 * 10000 = 1 sample
        # 20 > 1, so it should fail
        with pytest.raises(ValueError, match="minimum chirp length.*exceeds maximum"):
            ChirpModulator(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
