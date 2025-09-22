"""
Integration tests for OFDM signal generation engine.

This module tests the complete end-to-end signal generation workflow,
including integration between chirp modulation and OFDM structure.
"""

from datetime import datetime

import numpy as np
import pytest

from ofdm_chirp_generator import ChirpConfig, GPUBackend, OFDMConfig, OFDMGenerator, SignalSet


class TestOFDMGenerator:
    """Test suite for OFDMGenerator class."""

    @pytest.fixture
    def basic_ofdm_config(self):
        """Basic OFDM configuration for testing."""
        return OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,  # 1 kHz
            bandwidth_per_subcarrier=800.0,  # 800 Hz
            center_frequency=10000.0,  # 10 kHz
            sampling_rate=50000.0,  # 50 kHz
            signal_duration=0.01,  # 10 ms - longer duration to satisfy chirp constraints
        )

    @pytest.fixture
    def large_ofdm_config(self):
        """Larger OFDM configuration for performance testing."""
        return OFDMConfig(
            num_subcarriers=32,
            subcarrier_spacing=500.0,  # 500 Hz
            bandwidth_per_subcarrier=400.0,  # 400 Hz
            center_frequency=20000.0,  # 20 kHz
            sampling_rate=100000.0,  # 100 kHz
            signal_duration=0.01,  # 10 ms
        )

    @pytest.fixture
    def gpu_backend(self):
        """GPU backend for testing."""
        return GPUBackend()

    def test_generator_initialization(self, basic_ofdm_config):
        """Test OFDMGenerator initialization.

        Requirements: 1.1 - Accept configuration parameters
        """
        generator = OFDMGenerator(basic_ofdm_config)

        assert generator.ofdm_config == basic_ofdm_config
        assert generator.gpu_backend is not None
        assert generator.subcarrier_manager is not None
        assert generator.chirp_modulator is not None
        assert generator._signal_length == int(
            basic_ofdm_config.signal_duration * basic_ofdm_config.sampling_rate
        )

    def test_generator_initialization_with_gpu_backend(self, basic_ofdm_config, gpu_backend):
        """Test OFDMGenerator initialization with provided GPU backend."""
        generator = OFDMGenerator(basic_ofdm_config, gpu_backend)

        assert generator.gpu_backend is gpu_backend
        assert generator.ofdm_config == basic_ofdm_config

    def test_single_signal_generation(self, basic_ofdm_config):
        """Test single OFDM signal generation.

        Requirements: 1.1, 1.2, 2.1, 2.2 - Generate OFDM signals with chirp modulation
        """
        generator = OFDMGenerator(basic_ofdm_config)

        # Create phase array
        phase_array = np.linspace(0, 2 * np.pi, basic_ofdm_config.num_subcarriers, endpoint=False)

        # Generate signal
        signal = generator.generate_single_signal(phase_array)

        # Verify signal properties
        expected_length = int(basic_ofdm_config.signal_duration * basic_ofdm_config.sampling_rate)
        assert len(signal) == expected_length
        assert signal.dtype == np.complex128 or str(signal.dtype) == "complex128"

        # Verify signal is not all zeros
        assert np.any(np.abs(signal) > 0)

        # Verify signal has reasonable amplitude
        rms_amplitude = np.sqrt(np.mean(np.abs(signal) ** 2))
        assert 0.1 < rms_amplitude < 10.0

    def test_single_signal_generation_different_phases(self, basic_ofdm_config):
        """Test that different phase arrays produce different signals.

        Requirements: 2.2 - Independent phase setting for each subcarrier
        """
        generator = OFDMGenerator(basic_ofdm_config)

        # Generate two different phase arrays
        phase_array1 = np.zeros(basic_ofdm_config.num_subcarriers)
        phase_array2 = np.ones(basic_ofdm_config.num_subcarriers) * np.pi

        # Generate signals
        signal1 = generator.generate_single_signal(phase_array1)
        signal2 = generator.generate_single_signal(phase_array2)

        # Convert to CPU for comparison if needed
        if hasattr(signal1, "get"):
            signal1 = signal1.get()
        if hasattr(signal2, "get"):
            signal2 = signal2.get()

        # Signals should be different
        assert not np.allclose(signal1, signal2, rtol=1e-10)

    def test_signal_generation_with_chirp_config(self, basic_ofdm_config):
        """Test signal generation using ChirpConfig."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Create chirp config
        phase_matrix = np.random.uniform(0, 2 * np.pi, (3, basic_ofdm_config.num_subcarriers))
        chirp_config = ChirpConfig(
            chirp_length=int(basic_ofdm_config.signal_duration * basic_ofdm_config.sampling_rate),
            phase_matrix=phase_matrix,
            amplitude=0.5,
        )

        # Generate signal
        signal = generator.generate_signal_with_chirp_config(chirp_config)

        # Verify signal properties
        expected_length = int(basic_ofdm_config.signal_duration * basic_ofdm_config.sampling_rate)
        assert len(signal) == expected_length

        # Verify amplitude scaling was applied
        rms_amplitude = np.sqrt(np.mean(np.abs(signal) ** 2))
        assert rms_amplitude < 1.0  # Should be scaled down by amplitude factor

    def test_create_signal_set(self, basic_ofdm_config):
        """Test creation of SignalSet from multiple phase configurations."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Create phase matrix for multiple signals
        num_signals = 4
        phase_matrix = np.random.uniform(
            0, 2 * np.pi, (num_signals, basic_ofdm_config.num_subcarriers)
        )
        orthogonality_score = 0.95

        # Create signal set
        signal_set = generator.create_signal_set(phase_matrix, orthogonality_score)

        # Verify SignalSet properties
        assert isinstance(signal_set, SignalSet)
        assert signal_set.num_signals == num_signals
        assert len(signal_set.signals) == num_signals
        assert signal_set.orthogonality_score == orthogonality_score
        assert np.array_equal(signal_set.phases, phase_matrix)
        assert signal_set.config == basic_ofdm_config
        assert isinstance(signal_set.generation_timestamp, datetime)

        # Verify all signals have correct length
        expected_length = int(basic_ofdm_config.signal_duration * basic_ofdm_config.sampling_rate)
        for signal in signal_set.signals:
            assert len(signal) == expected_length
            assert isinstance(signal, np.ndarray)  # Should be converted to CPU arrays

    def test_get_signal_parameters(self, basic_ofdm_config):
        """Test retrieval of signal generation parameters."""
        generator = OFDMGenerator(basic_ofdm_config)

        params = generator.get_signal_parameters()

        # Verify parameter structure
        assert "ofdm_config" in params
        assert "signal_properties" in params
        assert "chirp_properties" in params
        assert "backend_info" in params
        assert "validation_status" in params

        # Verify OFDM config parameters
        ofdm_params = params["ofdm_config"]
        assert ofdm_params["num_subcarriers"] == basic_ofdm_config.num_subcarriers
        assert ofdm_params["subcarrier_spacing"] == basic_ofdm_config.subcarrier_spacing
        assert ofdm_params["center_frequency"] == basic_ofdm_config.center_frequency

        # Verify signal properties
        signal_props = params["signal_properties"]
        expected_length = int(basic_ofdm_config.signal_duration * basic_ofdm_config.sampling_rate)
        assert signal_props["signal_length_samples"] == expected_length
        assert "total_bandwidth" in signal_props
        assert "frequency_range" in signal_props
        assert "subcarrier_frequencies" in signal_props

    def test_analyze_generated_signal(self, basic_ofdm_config):
        """Test analysis of generated signal properties."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Generate a signal
        phase_array = np.linspace(0, 2 * np.pi, basic_ofdm_config.num_subcarriers, endpoint=False)
        signal = generator.generate_single_signal(phase_array)

        # Analyze signal
        analysis = generator.analyze_generated_signal(signal)

        # Verify analysis results
        assert "signal_power" in analysis
        assert "peak_amplitude" in analysis
        assert "rms_amplitude" in analysis
        assert "papr_db" in analysis
        assert "subcarrier_power_distribution" in analysis
        assert "signal_length" in analysis
        assert "dynamic_range_db" in analysis

        # Verify reasonable values
        assert analysis["signal_power"] > 0
        assert analysis["peak_amplitude"] > 0
        assert analysis["rms_amplitude"] > 0
        assert analysis["signal_length"] == len(signal)

        # Verify subcarrier power distribution
        power_dist = analysis["subcarrier_power_distribution"]
        assert len(power_dist) == basic_ofdm_config.num_subcarriers
        assert all(power >= 0 for power in power_dist.values())

    def test_validate_signal_generation(self, basic_ofdm_config):
        """Test validation of signal generation parameters."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Test with valid phase array
        valid_phases = np.linspace(0, 2 * np.pi, basic_ofdm_config.num_subcarriers, endpoint=False)
        validation = generator.validate_signal_generation(valid_phases)

        assert "phase_array_valid" in validation
        assert "sufficient_memory" in validation
        assert "frequency_symmetry" in validation
        assert "spacing_consistency" in validation
        assert "no_overlaps" in validation
        assert "nyquist_satisfied" in validation

        assert validation["phase_array_valid"] is True

        # Test with invalid phase array
        invalid_phases = np.array([1, 2])  # Wrong size
        validation_invalid = generator.validate_signal_generation(invalid_phases)
        assert validation_invalid["phase_array_valid"] is False

    def test_get_example_phase_array(self, basic_ofdm_config):
        """Test generation of example phase arrays."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Test different signal indices
        for signal_index in range(3):
            phases = generator.get_example_phase_array(signal_index)

            assert len(phases) == basic_ofdm_config.num_subcarriers
            assert np.all(phases >= 0)
            assert np.all(phases < 2 * np.pi)

        # Different indices should produce different patterns
        phases0 = generator.get_example_phase_array(0)
        phases1 = generator.get_example_phase_array(1)
        assert not np.allclose(phases0, phases1)

    def test_end_to_end_workflow(self, basic_ofdm_config):
        """Test complete end-to-end signal generation workflow.

        Requirements: Integration of all components (1.1, 1.2, 2.1, 2.2)
        """
        # Initialize generator
        generator = OFDMGenerator(basic_ofdm_config)

        # Get system parameters
        params = generator.get_signal_parameters()
        assert params["validation_status"]["frequency_symmetry"] is True

        # Generate multiple signals with different phases
        num_signals = 3
        signals = []
        phase_matrix = np.zeros((num_signals, basic_ofdm_config.num_subcarriers))

        for i in range(num_signals):
            phases = generator.get_example_phase_array(i)
            phase_matrix[i, :] = phases

            # Validate before generation
            validation = generator.validate_signal_generation(phases)
            assert validation["phase_array_valid"] is True

            # Generate signal
            signal = generator.generate_single_signal(phases)
            signals.append(signal)

            # Analyze signal
            analysis = generator.analyze_generated_signal(signal)
            assert analysis["signal_power"] > 0

        # Create signal set
        signal_set = generator.create_signal_set(phase_matrix, orthogonality_score=0.8)

        # Verify signal set
        assert signal_set.num_signals == num_signals
        assert len(signal_set.signals) == num_signals

        # Verify signals are different
        for i in range(num_signals):
            for j in range(i + 1, num_signals):
                signal_i = generator.gpu_backend.to_cpu(signals[i])
                signal_j = generator.gpu_backend.to_cpu(signals[j])
                assert not np.allclose(signal_i, signal_j, rtol=1e-10)

        # Clean up
        generator.cleanup_resources()

    def test_large_signal_generation(self, large_ofdm_config):
        """Test signal generation with larger configuration."""
        generator = OFDMGenerator(large_ofdm_config)

        # Generate signal with larger configuration
        phase_array = np.random.uniform(0, 2 * np.pi, large_ofdm_config.num_subcarriers)
        signal = generator.generate_single_signal(phase_array)

        # Verify signal properties
        expected_length = int(large_ofdm_config.signal_duration * large_ofdm_config.sampling_rate)
        assert len(signal) == expected_length

        # Analyze signal
        analysis = generator.analyze_generated_signal(signal)
        assert analysis["signal_power"] > 0
        assert len(analysis["subcarrier_power_distribution"]) == large_ofdm_config.num_subcarriers

    def test_context_manager(self, basic_ofdm_config):
        """Test OFDMGenerator as context manager."""
        with OFDMGenerator(basic_ofdm_config) as generator:
            phase_array = np.zeros(basic_ofdm_config.num_subcarriers)
            signal = generator.generate_single_signal(phase_array)
            assert len(signal) > 0

        # Context manager should clean up resources automatically

    def test_invalid_configurations(self):
        """Test generator behavior with invalid configurations."""
        # Test configuration that violates Nyquist criterion
        from ofdm_chirp_generator.validation import ValidationError

        with pytest.raises(ValidationError, match="Nyquist"):
            invalid_config = OFDMConfig(
                num_subcarriers=100,
                subcarrier_spacing=1000.0,
                bandwidth_per_subcarrier=800.0,
                center_frequency=10000.0,
                sampling_rate=10000.0,  # Too low sampling rate
                signal_duration=0.001,
            )

    def test_phase_array_validation_errors(self, basic_ofdm_config):
        """Test error handling for invalid phase arrays."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Test wrong size phase array
        wrong_size_phases = np.array([0, 1])
        with pytest.raises(ValueError, match="doesn't match"):
            generator.generate_single_signal(wrong_size_phases)

        # Test empty phase array
        empty_phases = np.array([])
        with pytest.raises(ValueError):
            generator.generate_single_signal(empty_phases)

    def test_chirp_config_validation_errors(self, basic_ofdm_config):
        """Test error handling for invalid ChirpConfig."""
        generator = OFDMGenerator(basic_ofdm_config)
        from ofdm_chirp_generator.validation import ValidationError

        # Test 1D phase matrix - this should fail at ChirpConfig creation
        with pytest.raises(ValidationError, match="2-dimensional"):
            invalid_chirp_config = ChirpConfig(
                chirp_length=100,
                phase_matrix=np.array([1, 2, 3]),  # 1D instead of 2D
                amplitude=1.0,
            )

        # Test wrong number of columns - this should fail at generator level
        invalid_chirp_config2 = ChirpConfig(
            chirp_length=100,
            phase_matrix=np.random.uniform(0, 2 * np.pi, (2, 5)),  # Wrong number of columns
            amplitude=1.0,
        )

        with pytest.raises(ValueError, match="columns"):
            generator.generate_signal_with_chirp_config(invalid_chirp_config2)

    def test_signal_set_creation_errors(self, basic_ofdm_config):
        """Test error handling in signal set creation."""
        generator = OFDMGenerator(basic_ofdm_config)

        # Test 1D phase matrix
        with pytest.raises(ValueError, match="2D array"):
            generator.create_signal_set(np.array([1, 2, 3]))

        # Test wrong number of columns
        wrong_cols_matrix = np.random.uniform(0, 2 * np.pi, (2, 5))
        with pytest.raises(ValueError, match="columns"):
            generator.create_signal_set(wrong_cols_matrix)


class TestOFDMGeneratorIntegration:
    """Integration tests for OFDMGenerator with other components."""

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return OFDMConfig(
            num_subcarriers=16,
            subcarrier_spacing=625.0,  # 625 Hz
            bandwidth_per_subcarrier=500.0,  # 500 Hz
            center_frequency=15000.0,  # 15 kHz
            sampling_rate=80000.0,  # 80 kHz
            signal_duration=0.002,  # 2 ms
        )

    def test_integration_with_subcarrier_manager(self, integration_config):
        """Test integration between OFDMGenerator and SubcarrierManager."""
        generator = OFDMGenerator(integration_config)

        # Test that subcarrier manager is properly integrated
        subcarrier_info = generator.subcarrier_manager.get_ofdm_structure_info()

        assert subcarrier_info["num_subcarriers"] == integration_config.num_subcarriers
        assert subcarrier_info["center_frequency"] == integration_config.center_frequency

        # Generate signal and verify subcarrier structure
        phase_array = np.linspace(0, 2 * np.pi, integration_config.num_subcarriers, endpoint=False)
        signal = generator.generate_single_signal(phase_array)

        # Analyze subcarrier power distribution
        power_analysis = generator.subcarrier_manager.analyze_subcarrier_power(signal)
        assert len(power_analysis) == integration_config.num_subcarriers
        assert all(power > 0 for power in power_analysis.values())

    def test_integration_with_chirp_modulator(self, integration_config):
        """Test integration between OFDMGenerator and ChirpModulator."""
        generator = OFDMGenerator(integration_config)

        # Test that chirp modulator is properly integrated
        chirp_constraints = generator.chirp_modulator.chirp_length_constraints
        signal_length = int(integration_config.signal_duration * integration_config.sampling_rate)

        assert chirp_constraints[0] <= signal_length <= chirp_constraints[1]

        # Generate signals with different chirp characteristics
        for i in range(integration_config.num_subcarriers):
            chirp_info = generator.chirp_modulator.get_chirp_characteristics(i)

            assert chirp_info["subcarrier_index"] == i
            assert chirp_info["bandwidth"] == integration_config.bandwidth_per_subcarrier
            assert chirp_info["duration"] == integration_config.signal_duration

    def test_integration_with_gpu_backend(self, integration_config):
        """Test integration between OFDMGenerator and GPUBackend."""
        # Test with CPU backend
        cpu_backend = GPUBackend(force_cpu=True)
        generator_cpu = OFDMGenerator(integration_config, cpu_backend)

        phase_array = np.zeros(integration_config.num_subcarriers)
        signal_cpu = generator_cpu.generate_single_signal(phase_array)

        # Test with GPU backend (if available)
        gpu_backend = GPUBackend(force_cpu=False)
        generator_gpu = OFDMGenerator(integration_config, gpu_backend)

        signal_gpu = generator_gpu.generate_single_signal(phase_array)

        # Convert GPU signal to CPU for comparison
        signal_gpu_cpu = generator_gpu.gpu_backend.to_cpu(signal_gpu)

        # Results should be very similar (allowing for numerical precision differences)
        assert np.allclose(signal_cpu, signal_gpu_cpu, rtol=1e-10, atol=1e-12)

    def test_memory_management_integration(self, integration_config):
        """Test memory management across all components."""
        generator = OFDMGenerator(integration_config)

        # Generate multiple signals to test memory management
        signals = []
        for i in range(10):
            phase_array = generator.get_example_phase_array(i)
            signal = generator.generate_single_signal(phase_array)
            signals.append(signal)

        # Check memory info
        memory_info = generator.gpu_backend.get_memory_info()
        assert "backend" in memory_info

        # Clean up and verify cleanup
        generator.cleanup_resources()

        # Memory should be cleaned up (if GPU backend)
        if generator.gpu_backend.is_gpu_available:
            memory_info_after = generator.gpu_backend.get_memory_info()
            # Memory usage should not increase significantly
            assert memory_info_after is not None


if __name__ == "__main__":
    pytest.main([__file__])
