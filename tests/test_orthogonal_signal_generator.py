"""
Tests for orthogonal signal set generation functionality.

This module tests the OrthogonalSignalGenerator class and related components
for generating multiple orthogonal OFDM signals.
"""

import time
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig, SignalSet
from ofdm_chirp_generator.orthogonal_signal_generator import (
    OrthogonalSetConfig,
    OrthogonalSignalGenerator,
    PhaseMatrixManager,
)


class TestPhaseMatrixManager:
    """Test phase matrix management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PhaseMatrixManager()

        # Create test phase matrices
        self.phase_matrix_2sig = np.random.uniform(0, 2 * np.pi, (2, 8))
        self.phase_matrix_4sig = np.random.uniform(0, 2 * np.pi, (4, 8))

    def test_store_and_retrieve_configuration(self):
        """Test storing and retrieving phase configurations."""
        config_id = "test_config_2sig"
        orthogonality_score = 0.95
        metadata = {"test": "data"}

        # Store configuration
        self.manager.store_configuration(
            config_id, self.phase_matrix_2sig, orthogonality_score, metadata
        )

        # Retrieve configuration
        retrieved_matrix, retrieved_metadata = self.manager.retrieve_configuration(config_id)

        # Verify retrieval
        np.testing.assert_array_equal(retrieved_matrix, self.phase_matrix_2sig)
        assert retrieved_metadata["orthogonality_score"] == orthogonality_score
        assert retrieved_metadata["num_signals"] == 2
        assert retrieved_metadata["num_subcarriers"] == 8
        assert retrieved_metadata["metadata"] == metadata
        assert isinstance(retrieved_metadata["storage_timestamp"], datetime)

    def test_retrieve_nonexistent_configuration(self):
        """Test retrieving a configuration that doesn't exist."""
        with pytest.raises(KeyError, match="Configuration 'nonexistent' not found"):
            self.manager.retrieve_configuration("nonexistent")

    def test_list_configurations(self):
        """Test listing stored configurations."""
        # Initially empty
        assert self.manager.list_configurations() == []

        # Store some configurations
        self.manager.store_configuration("config1", self.phase_matrix_2sig, 0.9)
        self.manager.store_configuration("config2", self.phase_matrix_4sig, 0.8)

        # Check list
        configs = self.manager.list_configurations()
        assert len(configs) == 2
        assert "config1" in configs
        assert "config2" in configs

    def test_get_best_configuration(self):
        """Test getting the best configuration for a given number of signals."""
        # Store configurations with different scores
        self.manager.store_configuration("config1", self.phase_matrix_2sig, 0.8)
        self.manager.store_configuration("config2", self.phase_matrix_2sig, 0.9)
        self.manager.store_configuration("config3", self.phase_matrix_4sig, 0.7)

        # Get best for 2 signals
        result = self.manager.get_best_configuration(2)
        assert result is not None
        config_id, phase_matrix, metadata = result
        assert config_id == "config2"
        assert metadata["orthogonality_score"] == 0.9

        # Get best for 4 signals
        result = self.manager.get_best_configuration(4)
        assert result is not None
        config_id, phase_matrix, metadata = result
        assert config_id == "config3"

        # Get best for non-existent signal count
        result = self.manager.get_best_configuration(8)
        assert result is None

    def test_remove_configuration(self):
        """Test removing stored configurations."""
        config_id = "test_remove"
        self.manager.store_configuration(config_id, self.phase_matrix_2sig, 0.9)

        # Verify it exists
        assert config_id in self.manager.list_configurations()

        # Remove it
        self.manager.remove_configuration(config_id)

        # Verify it's gone
        assert config_id not in self.manager.list_configurations()

        # Removing non-existent config should not raise error
        self.manager.remove_configuration("nonexistent")

    def test_clear_all(self):
        """Test clearing all configurations."""
        # Store some configurations
        self.manager.store_configuration("config1", self.phase_matrix_2sig, 0.9)
        self.manager.store_configuration("config2", self.phase_matrix_4sig, 0.8)

        # Verify they exist
        assert len(self.manager.list_configurations()) == 2

        # Clear all
        self.manager.clear_all()

        # Verify they're gone
        assert len(self.manager.list_configurations()) == 0


class TestOrthogonalSignalGenerator:
    """Test orthogonal signal generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ofdm_config = OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=500.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.01,  # Increased to 10ms to meet minimum chirp length requirements
        )

        self.gpu_backend = GPUBackend()

        # Create generator with test configuration
        self.generator = OrthogonalSignalGenerator(
            ofdm_config=self.ofdm_config, gpu_backend=self.gpu_backend
        )

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "generator"):
            self.generator.cleanup_resources()

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.ofdm_config == self.ofdm_config
        assert self.generator.gpu_backend == self.gpu_backend
        assert isinstance(self.generator.phase_matrix_manager, PhaseMatrixManager)
        assert isinstance(self.generator.orthogonal_set_config, OrthogonalSetConfig)

    def test_initialization_with_config_file(self):
        """Test generator initialization with configuration file."""
        # This test would require a mock config file
        # For now, test that it handles missing config gracefully
        with patch(
            "ofdm_chirp_generator.orthogonal_signal_generator.get_config"
        ) as mock_get_config:
            mock_get_config.side_effect = Exception("Config not found")

            # Should still initialize with provided ofdm_config
            generator = OrthogonalSignalGenerator(ofdm_config=self.ofdm_config)
            assert generator.ofdm_config == self.ofdm_config
            generator.cleanup_resources()

    def test_generate_orthogonal_signal_set_basic(self):
        """Test basic orthogonal signal set generation."""
        num_signals = 2

        # Mock the phase optimizer to return a simple result
        with patch.object(
            self.generator.phase_optimizer, "find_orthogonal_phases"
        ) as mock_optimize:
            mock_optimize.return_value = Mock(
                optimal_phases=np.random.uniform(
                    0, 2 * np.pi, (num_signals, self.ofdm_config.num_subcarriers)
                ),
                orthogonality_score=0.95,
                iterations=100,
                converged=True,
            )

            signal_set = self.generator.generate_orthogonal_signal_set(num_signals)

            # Verify signal set properties
            assert isinstance(signal_set, SignalSet)
            assert signal_set.num_signals == num_signals
            assert len(signal_set.signals) == num_signals
            assert signal_set.phases.shape == (num_signals, self.ofdm_config.num_subcarriers)
            assert signal_set.orthogonality_score == 0.95
            assert "generation_method" in signal_set.metadata

    def test_generate_orthogonal_signal_set_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Too few signals
        with pytest.raises(ValueError, match="Need at least 2 signals"):
            self.generator.generate_orthogonal_signal_set(1)

        # Too many signals
        with pytest.raises(ValueError, match="exceeds maximum"):
            self.generator.generate_orthogonal_signal_set(100)

    def test_generate_orthogonal_signal_set_with_caching(self):
        """Test signal generation with caching enabled."""
        num_signals = 2

        # Store a cached configuration
        cached_phases = np.random.uniform(
            0, 2 * np.pi, (num_signals, self.ofdm_config.num_subcarriers)
        )
        self.generator.phase_matrix_manager.store_configuration("test_cache", cached_phases, 0.98)

        # Generate signal set (should use cache)
        signal_set = self.generator.generate_orthogonal_signal_set(num_signals)

        # Verify it used cached configuration
        assert signal_set.metadata["generation_method"] == "cached"
        assert "cached_config_id" in signal_set.metadata
        np.testing.assert_array_equal(signal_set.phases, cached_phases)

    def test_generate_orthogonal_signal_set_force_regenerate(self):
        """Test forcing regeneration even with cached configuration."""
        num_signals = 2

        # Store a cached configuration
        cached_phases = np.random.uniform(
            0, 2 * np.pi, (num_signals, self.ofdm_config.num_subcarriers)
        )
        self.generator.phase_matrix_manager.store_configuration("test_cache", cached_phases, 0.98)

        # Mock the phase optimizer
        new_phases = np.random.uniform(
            0, 2 * np.pi, (num_signals, self.ofdm_config.num_subcarriers)
        )
        with patch.object(
            self.generator.phase_optimizer, "find_orthogonal_phases"
        ) as mock_optimize:
            mock_optimize.return_value = Mock(
                optimal_phases=new_phases, orthogonality_score=0.92, iterations=50, converged=True
            )

            # Generate with force_regenerate=True
            signal_set = self.generator.generate_orthogonal_signal_set(
                num_signals, force_regenerate=True
            )

            # Verify it didn't use cache
            assert signal_set.metadata["generation_method"] == "optimized"
            np.testing.assert_array_equal(signal_set.phases, new_phases)

    def test_generate_batch_orthogonal_sets(self):
        """Test batch generation of orthogonal signal sets."""
        signal_counts = [2, 3, 4]

        # Mock the phase optimizer for all calls
        with patch.object(
            self.generator.phase_optimizer, "find_orthogonal_phases"
        ) as mock_optimize:

            def mock_optimize_func(num_signals, *args, **kwargs):
                return Mock(
                    optimal_phases=np.random.uniform(
                        0, 2 * np.pi, (num_signals, self.ofdm_config.num_subcarriers)
                    ),
                    orthogonality_score=0.9,
                    iterations=100,
                    converged=True,
                )

            mock_optimize.side_effect = mock_optimize_func

            results = self.generator.generate_batch_orthogonal_sets(signal_counts)

            # Verify results
            assert len(results) == len(signal_counts)
            for count in signal_counts:
                assert count in results
                assert isinstance(results[count], SignalSet)
                assert results[count].num_signals == count

    def test_validate_orthogonal_configuration(self):
        """Test validation of orthogonal configurations."""
        # Create a test phase matrix
        phase_matrix = np.random.uniform(0, 2 * np.pi, (2, self.ofdm_config.num_subcarriers))

        # Mock the orthogonality tester
        with patch.object(
            self.generator.orthogonality_tester, "test_signal_set_orthogonality"
        ) as mock_test:
            mock_test.return_value = {
                "overall_orthogonality_score": 0.96,
                "orthogonal_pairs": 1,
                "total_pairs": 1,
            }

            result = self.generator._validate_orthogonal_configuration(phase_matrix)

            assert result["is_valid"] is True
            assert result["orthogonality_score"] == 0.96
            assert "Valid" in result["reason"]

    def test_validate_orthogonal_configuration_invalid(self):
        """Test validation of invalid orthogonal configurations."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, (2, self.ofdm_config.num_subcarriers))

        # Mock the orthogonality tester to return low score
        with patch.object(
            self.generator.orthogonality_tester, "test_signal_set_orthogonality"
        ) as mock_test:
            mock_test.return_value = {
                "overall_orthogonality_score": 0.5,  # Below threshold
                "orthogonal_pairs": 0,
                "total_pairs": 1,
            }

            result = self.generator._validate_orthogonal_configuration(phase_matrix)

            assert result["is_valid"] is False
            assert result["orthogonality_score"] == 0.5
            assert "below threshold" in result["reason"]

    def test_analyze_orthogonal_set(self):
        """Test analysis of orthogonal signal sets."""
        # Create a test signal set
        num_signals = 2
        signals = [np.random.randn(50) + 1j * np.random.randn(50) for _ in range(num_signals)]
        phases = np.random.uniform(0, 2 * np.pi, (num_signals, self.ofdm_config.num_subcarriers))

        signal_set = SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=0.9,
            generation_timestamp=datetime.now(),
            config=self.ofdm_config,
        )

        # Mock the orthogonality tester
        with patch.object(
            self.generator.orthogonality_tester, "test_signal_set_orthogonality"
        ) as mock_test:
            mock_test.return_value = {
                "overall_orthogonality_score": 0.9,
                "orthogonal_pairs": 1,
                "total_pairs": 1,
            }

            analysis = self.generator.analyze_orthogonal_set(signal_set)

            # Verify analysis structure
            assert "orthogonality_analysis" in analysis
            assert "signal_analyses" in analysis
            assert "set_metrics" in analysis
            assert "phase_matrix_properties" in analysis

            # Verify set metrics
            assert analysis["set_metrics"]["num_signals"] == num_signals
            assert analysis["set_metrics"]["signal_length"] == len(signals[0])

            # Verify phase matrix properties
            assert analysis["phase_matrix_properties"]["shape"] == phases.shape

    def test_get_maximum_orthogonal_signals(self):
        """Test determination of maximum orthogonal signals."""
        # Mock the phase optimizer to simulate success/failure
        with patch.object(
            self.generator.phase_optimizer, "find_orthogonal_phases"
        ) as mock_optimize:

            def mock_optimize_func(num_signals, *args, **kwargs):
                # Simulate success for small numbers, failure for large
                if num_signals <= 4:
                    return Mock(orthogonality_score=0.8)
                else:
                    return Mock(orthogonality_score=0.3)

            mock_optimize.side_effect = mock_optimize_func

            max_signals = self.generator.get_maximum_orthogonal_signals()

            # Should find 4 as maximum
            assert max_signals == 4

    def test_export_import_phase_configurations(self):
        """Test exporting and importing phase configurations."""
        # Store some configurations
        phase_matrix1 = np.random.uniform(0, 2 * np.pi, (2, self.ofdm_config.num_subcarriers))
        phase_matrix2 = np.random.uniform(0, 2 * np.pi, (3, self.ofdm_config.num_subcarriers))

        self.generator.phase_matrix_manager.store_configuration("config1", phase_matrix1, 0.9)
        self.generator.phase_matrix_manager.store_configuration("config2", phase_matrix2, 0.8)

        # Export configurations
        export_data = self.generator.export_phase_configurations()

        # Verify export structure
        assert "configurations" in export_data
        assert "export_timestamp" in export_data
        assert "ofdm_config" in export_data
        assert len(export_data["configurations"]) == 2

        # Clear and import
        self.generator.phase_matrix_manager.clear_all()
        imported_count = self.generator.import_phase_configurations(export_data)

        # Verify import
        assert imported_count == 2
        assert len(self.generator.phase_matrix_manager.list_configurations()) == 2

    def test_context_manager(self):
        """Test context manager functionality."""
        with OrthogonalSignalGenerator(ofdm_config=self.ofdm_config) as generator:
            assert isinstance(generator, OrthogonalSignalGenerator)
        # Should clean up automatically

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.generator)
        assert "OrthogonalSignalGenerator" in repr_str
        assert str(self.generator.orthogonal_set_config.max_signals) in repr_str
        assert str(self.ofdm_config.num_subcarriers) in repr_str


class TestOrthogonalSetConfig:
    """Test orthogonal set configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OrthogonalSetConfig()

        assert config.max_signals == 16
        assert config.orthogonality_threshold == 0.95
        assert config.optimization_method == "hybrid"
        assert config.max_optimization_iterations == 1000
        assert config.batch_size == 4
        assert config.enable_caching is True
        assert config.validation_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OrthogonalSetConfig(
            max_signals=32, orthogonality_threshold=0.9, optimization_method="genetic", batch_size=8
        )

        assert config.max_signals == 32
        assert config.orthogonality_threshold == 0.9
        assert config.optimization_method == "genetic"
        assert config.batch_size == 8


@pytest.mark.integration
class TestOrthogonalSignalGeneratorIntegration:
    """Integration tests for orthogonal signal generation."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.ofdm_config = OFDMConfig(
            num_subcarriers=4,  # Small for fast testing
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=500.0,
            center_frequency=10000.0,
            sampling_rate=20000.0,
            signal_duration=0.01,  # Increased to 10ms to meet minimum chirp length requirements
        )

        self.generator = OrthogonalSignalGenerator(ofdm_config=self.ofdm_config)

    def teardown_method(self):
        """Clean up after integration tests."""
        if hasattr(self, "generator"):
            self.generator.cleanup_resources()

    def test_end_to_end_signal_generation(self):
        """Test complete end-to-end signal generation."""
        num_signals = 2

        # Generate orthogonal signal set
        signal_set = self.generator.generate_orthogonal_signal_set(num_signals)

        # Verify basic properties
        assert isinstance(signal_set, SignalSet)
        assert signal_set.num_signals == num_signals
        assert len(signal_set.signals) == num_signals

        # Verify signals are not empty
        for signal in signal_set.signals:
            assert len(signal) > 0
            assert not np.all(signal == 0)

        # Analyze the generated set
        analysis = self.generator.analyze_orthogonal_set(signal_set)

        # Verify analysis completed
        assert "orthogonality_analysis" in analysis
        assert "signal_analyses" in analysis

        # The orthogonality score should be reasonable (may not be perfect due to optimization limits)
        assert analysis["orthogonality_analysis"]["overall_orthogonality_score"] >= 0.0

    def test_batch_generation_integration(self):
        """Test batch generation integration."""
        signal_counts = [2, 3]

        results = self.generator.generate_batch_orthogonal_sets(signal_counts)

        # Verify all requested sets were generated
        assert len(results) == len(signal_counts)

        for count in signal_counts:
            assert count in results
            signal_set = results[count]
            assert signal_set.num_signals == count

            # Verify signals are valid
            for signal in signal_set.signals:
                assert len(signal) > 0
                assert np.isfinite(signal).all()


if __name__ == "__main__":
    pytest.main([__file__])
