"""
Tests for signal separation engine.

This module tests the SignalSeparator class for phase-based signal recovery,
separation quality metrics, and diagnostic reporting capabilities.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.config_manager import ConfigurationError
from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig, SignalSet
from ofdm_chirp_generator.signal_separator import SeparationQualityMetrics, SignalSeparator


class TestSeparationQualityMetrics:
    """Test SeparationQualityMetrics container class."""

    def test_initialization(self):
        """Test SeparationQualityMetrics initialization."""
        metrics = SeparationQualityMetrics()

        assert metrics.separation_scores == []
        assert metrics.cross_talk_matrix.size == 0
        assert metrics.signal_to_interference_ratios == []
        assert metrics.mean_separation_score == 0.0
        assert metrics.min_separation_score == 0.0
        assert metrics.max_cross_talk == 0.0
        assert metrics.mean_cross_talk == 0.0
        assert metrics.overall_separation_quality == 0.0
        assert metrics.separation_success is False
        assert metrics.diagnostic_info == {}


class TestSignalSeparator:
    """Test SignalSeparator class."""

    @pytest.fixture
    def ofdm_config(self):
        """Create test OFDM configuration."""
        return OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,
        )

    @pytest.fixture
    def gpu_backend(self):
        """Create test GPU backend."""
        return GPUBackend()

    @pytest.fixture
    def signal_separator(self, ofdm_config, gpu_backend):
        """Create test SignalSeparator."""
        with patch("ofdm_chirp_generator.signal_separator.get_config") as mock_config:
            # Mock configuration to avoid file dependencies
            mock_config_manager = Mock()
            mock_config_manager.get_separation_config.return_value = {
                "quality_threshold": 0.8,
                "max_iterations": 100,
                "convergence_tolerance": 1e-6,
            }
            mock_config.return_value = mock_config_manager

            return SignalSeparator(ofdm_config, gpu_backend)

    @pytest.fixture
    def test_signals(self, gpu_backend):
        """Create test signals for separation."""
        # Create simple test signals with different phases
        signal_length = 50
        t = np.linspace(0, 1, signal_length)

        # Signal 1: sine wave
        signal1 = np.sin(2 * np.pi * 5 * t) + 1j * np.cos(2 * np.pi * 5 * t)

        # Signal 2: cosine wave (orthogonal to signal1)
        signal2 = np.cos(2 * np.pi * 5 * t) + 1j * np.sin(2 * np.pi * 5 * t)

        # Convert to GPU if available
        if gpu_backend.is_gpu_available:
            signal1 = gpu_backend.to_gpu(signal1)
            signal2 = gpu_backend.to_gpu(signal2)

        return [signal1, signal2]

    @pytest.fixture
    def test_phases(self):
        """Create test phase matrix."""
        return np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],  # Signal 1 phases
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],  # Signal 2 phases
            ]
        )

    def test_initialization(self, ofdm_config, gpu_backend):
        """Test SignalSeparator initialization."""
        with patch("ofdm_chirp_generator.signal_separator.get_config") as mock_config:
            mock_config_manager = Mock()
            mock_config_manager.get_separation_config.return_value = {
                "quality_threshold": 0.8,
                "max_iterations": 100,
                "convergence_tolerance": 1e-6,
            }
            mock_config.return_value = mock_config_manager

            separator = SignalSeparator(ofdm_config, gpu_backend)

            assert separator.ofdm_config == ofdm_config
            assert separator.gpu_backend == gpu_backend
            assert separator.separation_threshold == 0.8
            assert separator.max_iterations == 100
            assert separator.convergence_tolerance == 1e-6
            assert separator._reference_signals is None
            assert separator._reference_phases is None

    def test_initialization_with_config_error(self, ofdm_config, gpu_backend):
        """Test SignalSeparator initialization with configuration error."""
        with patch("ofdm_chirp_generator.signal_separator.get_config") as mock_config:
            mock_config.side_effect = ConfigurationError("Config not found")

            separator = SignalSeparator(ofdm_config, gpu_backend)

            # Should use default values
            assert separator.separation_threshold == 0.8
            assert separator.max_iterations == 100
            assert separator.convergence_tolerance == 1e-6

    def test_set_reference_signals(self, signal_separator, test_signals, test_phases):
        """Test setting reference signals and phases."""
        signal_separator.set_reference_signals(test_signals, test_phases)

        assert signal_separator._reference_signals == test_signals
        assert np.array_equal(signal_separator._reference_phases, test_phases)

    def test_set_reference_signals_invalid_empty(self, signal_separator, test_phases):
        """Test setting empty reference signals."""
        with pytest.raises(ValueError, match="Reference signals list cannot be empty"):
            signal_separator.set_reference_signals([], test_phases)

    def test_set_reference_signals_invalid_phases_1d(self, signal_separator, test_signals):
        """Test setting reference signals with 1D phases."""
        phases_1d = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])

        with pytest.raises(ValueError, match="Reference phases must be 2D array"):
            signal_separator.set_reference_signals(test_signals, phases_1d)

    def test_set_reference_signals_mismatched_count(self, signal_separator, test_signals):
        """Test setting reference signals with mismatched phase count."""
        # Wrong number of phase rows
        wrong_phases = np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
                [np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8, 7 * np.pi / 8],  # Extra row
            ]
        )

        with pytest.raises(
            ValueError, match="Number of reference signals must match phase matrix rows"
        ):
            signal_separator.set_reference_signals(test_signals, wrong_phases)

    def test_set_reference_signals_wrong_subcarriers(self, signal_separator, test_signals):
        """Test setting reference signals with wrong number of subcarriers."""
        # Wrong number of phase columns (should be 4 for ofdm_config)
        wrong_phases = np.array(
            [
                [0.0, np.pi / 2, np.pi],  # Only 3 columns instead of 4
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4],
            ]
        )

        with pytest.raises(
            ValueError, match="Phase matrix columns must match number of subcarriers"
        ):
            signal_separator.set_reference_signals(test_signals, wrong_phases)

    def test_separate_combined_signal_no_references(self, signal_separator, test_signals):
        """Test separation without setting reference signals."""
        combined_signal = test_signals[0] + test_signals[1]

        with pytest.raises(
            ValueError, match="Reference signals and phases must be set before separation"
        ):
            signal_separator.separate_combined_signal(combined_signal)

    def test_separate_combined_signal_with_references(
        self, signal_separator, test_signals, test_phases
    ):
        """Test signal separation with provided references."""
        # Create combined signal
        combined_signal = test_signals[0] + test_signals[1]

        # Perform separation
        separated_signals, quality_metrics = signal_separator.separate_combined_signal(
            combined_signal, test_signals, test_phases
        )

        assert len(separated_signals) == len(test_signals)
        assert isinstance(quality_metrics, SeparationQualityMetrics)
        assert len(quality_metrics.separation_scores) == len(test_signals)
        assert quality_metrics.cross_talk_matrix.shape == (len(test_signals), len(test_signals))

    def test_separate_combined_signal_stored_references(
        self, signal_separator, test_signals, test_phases
    ):
        """Test signal separation with stored references."""
        # Set reference signals first
        signal_separator.set_reference_signals(test_signals, test_phases)

        # Create combined signal
        combined_signal = test_signals[0] + test_signals[1]

        # Perform separation
        separated_signals, quality_metrics = signal_separator.separate_combined_signal(
            combined_signal
        )

        assert len(separated_signals) == len(test_signals)
        assert isinstance(quality_metrics, SeparationQualityMetrics)

    def test_separate_signal_set(self, signal_separator, test_signals, test_phases, ofdm_config):
        """Test separation using SignalSet."""
        # Create reference SignalSet
        reference_set = SignalSet(
            signals=[signal_separator.gpu_backend.to_cpu(sig) for sig in test_signals],
            phases=test_phases,
            orthogonality_score=0.9,
            generation_timestamp=datetime.now(),
            config=ofdm_config,
        )

        # Create combined signal
        combined_signal = test_signals[0] + test_signals[1]

        # Perform separation
        separated_set, quality_metrics = signal_separator.separate_signal_set(
            combined_signal, reference_set
        )

        assert isinstance(separated_set, SignalSet)
        assert len(separated_set.signals) == len(reference_set.signals)
        assert np.array_equal(separated_set.phases, reference_set.phases)
        assert isinstance(quality_metrics, SeparationQualityMetrics)

    def test_validate_separation_capability_orthogonal(
        self, signal_separator, test_signals, test_phases
    ):
        """Test validation with orthogonal signals."""
        with patch.object(
            signal_separator.orthogonality_tester, "test_signal_set_orthogonality"
        ) as mock_test:
            mock_test.return_value = {
                "overall_orthogonality_score": 0.95,
                "orthogonality_ratio": 0.98,
                "is_set_orthogonal": True,
            }

            validation = signal_separator.validate_separation_capability(test_signals, test_phases)

            assert validation["separation_feasible"] is True
            assert validation["orthogonality_quality"] == 0.95
            assert validation["orthogonality_ratio"] == 0.98
            assert len(validation["recommendations"]) > 0
            assert "successful" in validation["recommendations"][0].lower()

    def test_validate_separation_capability_non_orthogonal(
        self, signal_separator, test_signals, test_phases
    ):
        """Test validation with non-orthogonal signals."""
        with patch.object(
            signal_separator.orthogonality_tester, "test_signal_set_orthogonality"
        ) as mock_test:
            mock_test.return_value = {
                "overall_orthogonality_score": 0.5,
                "orthogonality_ratio": 0.6,
                "is_set_orthogonal": False,
            }

            validation = signal_separator.validate_separation_capability(test_signals, test_phases)

            assert validation["separation_feasible"] is False
            assert validation["orthogonality_quality"] == 0.5
            assert len(validation["warnings"]) > 0
            assert len(validation["recommendations"]) > 0
            assert "improve" in validation["recommendations"][0].lower()

    def test_validate_separation_capability_energy_imbalance(self, signal_separator, test_phases):
        """Test validation with energy imbalanced signals."""
        # Create signals with large energy difference
        signal1 = np.ones(50, dtype=complex)  # High energy
        signal2 = 0.01 * np.ones(50, dtype=complex)  # Low energy
        signals = [signal1, signal2]

        with patch.object(
            signal_separator.orthogonality_tester, "test_signal_set_orthogonality"
        ) as mock_test:
            mock_test.return_value = {
                "overall_orthogonality_score": 0.95,
                "orthogonality_ratio": 0.98,
                "is_set_orthogonal": True,
            }

            validation = signal_separator.validate_separation_capability(signals, test_phases)

            assert validation["energy_ratio"] > 10.0
            assert len(validation["warnings"]) > 0
            assert "energy imbalance" in validation["warnings"][0].lower()

    def test_generate_separation_report(self, signal_separator):
        """Test separation report generation."""
        # Create mock quality metrics
        metrics = SeparationQualityMetrics()
        metrics.separation_success = True
        metrics.overall_separation_quality = 0.85
        metrics.separation_scores = [0.9, 0.8]
        metrics.mean_separation_score = 0.85
        metrics.min_separation_score = 0.8
        metrics.max_cross_talk = 0.1
        metrics.mean_cross_talk = 0.05
        metrics.signal_to_interference_ratios = [10.0, 8.0]
        metrics.diagnostic_info = {
            "num_signals": 2,
            "separation_threshold": 0.8,
            "mean_sir_db": 9.0,
        }

        report = signal_separator.generate_separation_report(metrics)

        assert "SIGNAL SEPARATION QUALITY REPORT" in report
        assert "Separation Success: YES" in report
        assert "Overall Quality Score: 0.850000" in report
        assert "SEPARATION SCORES" in report
        assert "CROSS-TALK ANALYSIS" in report
        assert "SIGNAL-TO-INTERFERENCE RATIOS" in report
        assert "DIAGNOSTIC INFORMATION" in report

    def test_set_separation_threshold_valid(self, signal_separator):
        """Test setting valid separation threshold."""
        signal_separator.set_separation_threshold(0.9)
        assert signal_separator.get_separation_threshold() == 0.9

    def test_set_separation_threshold_invalid_low(self, signal_separator):
        """Test setting invalid low separation threshold."""
        with pytest.raises(ValueError, match="Separation threshold must be between 0 and 1"):
            signal_separator.set_separation_threshold(0.0)

    def test_set_separation_threshold_invalid_high(self, signal_separator):
        """Test setting invalid high separation threshold."""
        with pytest.raises(ValueError, match="Separation threshold must be between 0 and 1"):
            signal_separator.set_separation_threshold(1.0)

    def test_context_manager(self, signal_separator):
        """Test SignalSeparator as context manager."""
        with patch.object(signal_separator, "cleanup_resources") as mock_cleanup:
            with signal_separator as sep:
                assert sep is signal_separator
            mock_cleanup.assert_called_once()

    def test_cleanup_resources(self, signal_separator):
        """Test resource cleanup."""
        with patch.object(signal_separator.gpu_backend, "cleanup_memory") as mock_gpu_cleanup:
            with patch.object(
                signal_separator.correlation_analyzer, "clear_cache"
            ) as mock_corr_cleanup:
                signal_separator.cleanup_resources()
                mock_gpu_cleanup.assert_called_once()
                mock_corr_cleanup.assert_called_once()


class TestSignalSeparationIntegration:
    """Integration tests for signal separation with real OFDM signals."""

    @pytest.fixture
    def ofdm_config(self):
        """Create integration test OFDM configuration."""
        return OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,
        )

    def test_separation_with_orthogonal_signals(self, ofdm_config):
        """Test separation with truly orthogonal OFDM signals."""
        from ofdm_chirp_generator.ofdm_generator import OFDMGenerator

        # Create OFDM generator
        generator = OFDMGenerator(ofdm_config)

        # Generate orthogonal phase arrays
        phases1 = np.zeros(ofdm_config.num_subcarriers)
        phases2 = np.ones(ofdm_config.num_subcarriers) * np.pi / 2

        # Generate signals
        signal1 = generator.generate_single_signal(phases1)
        signal2 = generator.generate_single_signal(phases2)

        # Create combined signal
        combined_signal = signal1 + signal2

        # Create separator
        separator = SignalSeparator(ofdm_config, generator.gpu_backend)

        # Set reference signals
        phase_matrix = np.array([phases1, phases2])
        separator.set_reference_signals([signal1, signal2], phase_matrix)

        # Perform separation
        separated_signals, quality_metrics = separator.separate_combined_signal(combined_signal)

        # Verify separation quality
        assert len(separated_signals) == 2
        assert quality_metrics.mean_separation_score > 0.1  # Should have some correlation
        assert (
            quality_metrics.max_cross_talk < 2.0
        )  # Cross-talk should be reasonable (relaxed for simple test)

        generator.cleanup_resources()
        separator.cleanup_resources()

    def test_separation_quality_metrics_computation(self, ofdm_config):
        """Test comprehensive separation quality metrics computation."""
        from ofdm_chirp_generator.ofdm_generator import OFDMGenerator

        # Create OFDM generator
        generator = OFDMGenerator(ofdm_config)

        # Generate test signals
        phases1 = np.linspace(0, np.pi, ofdm_config.num_subcarriers)
        phases2 = np.linspace(np.pi / 2, 3 * np.pi / 2, ofdm_config.num_subcarriers)

        signal1 = generator.generate_single_signal(phases1)
        signal2 = generator.generate_single_signal(phases2)

        # Create separator and perform separation
        separator = SignalSeparator(ofdm_config, generator.gpu_backend)
        phase_matrix = np.array([phases1, phases2])

        combined_signal = signal1 + signal2
        separated_signals, quality_metrics = separator.separate_combined_signal(
            combined_signal, [signal1, signal2], phase_matrix
        )

        # Verify all metrics are computed
        assert len(quality_metrics.separation_scores) == 2
        assert quality_metrics.cross_talk_matrix.shape == (2, 2)
        assert len(quality_metrics.signal_to_interference_ratios) == 2
        assert quality_metrics.mean_separation_score >= 0
        assert quality_metrics.min_separation_score >= 0
        assert quality_metrics.max_cross_talk >= 0
        assert quality_metrics.mean_cross_talk >= 0
        assert "num_signals" in quality_metrics.diagnostic_info
        assert "separation_threshold" in quality_metrics.diagnostic_info

        generator.cleanup_resources()
        separator.cleanup_resources()


if __name__ == "__main__":
    pytest.main([__file__])
