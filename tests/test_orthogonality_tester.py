"""
Tests for the OrthogonalityTester class.

This module tests orthogonality evaluation, signal separation quality,
and comprehensive orthogonality analysis functionality.
"""

import numpy as np
import pytest

from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.orthogonality_tester import OrthogonalityTester

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class TestOrthogonalityTester:
    """Test cases for OrthogonalityTester."""

    @pytest.fixture
    def tester(self):
        """Create an OrthogonalityTester instance."""
        return OrthogonalityTester()

    @pytest.fixture
    def orthogonal_signals(self):
        """Create a set of orthogonal test signals."""
        # Create orthogonal Walsh functions (Hadamard matrix)
        walsh_signals = [
            np.array([1, 1, 1, 1], dtype=float),
            np.array([1, -1, 1, -1], dtype=float),
            np.array([1, 1, -1, -1], dtype=float),
            np.array([1, -1, -1, 1], dtype=float),
        ]
        return walsh_signals

    @pytest.fixture
    def non_orthogonal_signals(self):
        """Create a set of non-orthogonal test signals."""
        # Create signals that are definitely not orthogonal
        np.random.seed(42)  # For reproducible results
        base_signal = np.array([1, 2, 3, 4, 5], dtype=float)
        signals = [
            base_signal,  # Original signal
            base_signal + 0.1 * np.random.randn(5),  # Noisy version (correlated)
            2 * base_signal,  # Scaled version (perfectly correlated)
        ]
        return signals

    def test_tester_initialization(self):
        """Test OrthogonalityTester initialization."""
        tester = OrthogonalityTester()
        assert tester.backend is not None
        assert tester.correlation_analyzer is not None
        assert tester.orthogonality_threshold == 0.1

    def test_tester_with_custom_backend(self):
        """Test OrthogonalityTester with custom GPU backend."""
        backend = GPUBackend(force_cpu=True)
        tester = OrthogonalityTester(backend)
        assert tester.backend is backend
        assert not tester.backend.is_gpu_available

    def test_signal_pair_orthogonality_orthogonal(self, tester, orthogonal_signals):
        """Test orthogonality testing with orthogonal signal pairs."""
        sig1, sig2 = orthogonal_signals[0], orthogonal_signals[1]

        result = tester.test_signal_pair_orthogonality(sig1, sig2)

        # Check result structure
        expected_keys = [
            "max_correlation",
            "orthogonality_score",
            "is_orthogonal",
            "threshold",
            "signal1_energy",
            "signal2_energy",
            "energy_ratio",
        ]
        for key in expected_keys:
            assert key in result

        # Orthogonal signals should have low correlation
        assert result["max_correlation"] < 0.1
        assert result["is_orthogonal"] is True
        assert result["orthogonality_score"] > 0.9

    def test_signal_pair_orthogonality_non_orthogonal(self, tester):
        """Test orthogonality testing with non-orthogonal signal pairs."""
        # Create identical signals (maximum correlation)
        sig1 = np.sin(2 * np.pi * np.linspace(0, 1, 100))
        sig2 = sig1.copy()

        result = tester.test_signal_pair_orthogonality(sig1, sig2)

        # Identical signals should have maximum correlation
        assert result["max_correlation"] > 0.9
        assert result["is_orthogonal"] is False
        assert result["orthogonality_score"] < 0.1

    def test_signal_set_orthogonality_orthogonal(self, tester, orthogonal_signals):
        """Test orthogonality testing with orthogonal signal set."""
        result = tester.test_signal_set_orthogonality(orthogonal_signals)

        # Check result structure
        expected_keys = [
            "num_signals",
            "correlation_matrix",
            "cross_correlations",
            "max_cross_correlation",
            "mean_cross_correlation",
            "std_cross_correlation",
            "orthogonal_pairs",
            "total_pairs",
            "orthogonality_ratio",
            "overall_orthogonality_score",
            "is_set_orthogonal",
            "threshold",
        ]
        for key in expected_keys:
            assert key in result

        # Orthogonal set should have high orthogonality ratio
        assert result["orthogonality_ratio"] > 0.8
        assert result["is_set_orthogonal"] is True
        assert result["max_cross_correlation"] < 0.2

    def test_signal_set_orthogonality_non_orthogonal(self, tester, non_orthogonal_signals):
        """Test orthogonality testing with non-orthogonal signal set."""
        result = tester.test_signal_set_orthogonality(non_orthogonal_signals)

        # Non-orthogonal set should have low orthogonality ratio
        assert result["orthogonality_ratio"] <= 0.5
        assert result["is_set_orthogonal"] is False
        assert result["max_cross_correlation"] > 0.3

    def test_signal_set_insufficient_signals(self, tester):
        """Test error handling with insufficient signals."""
        with pytest.raises(ValueError, match="Need at least 2 signals"):
            tester.test_signal_set_orthogonality([np.array([1, 2, 3])])

    def test_separation_quality_evaluation(self, tester, orthogonal_signals):
        """Test signal separation quality evaluation."""
        original_signals = orthogonal_signals[:3]

        # Create "separated" signals (same as original for perfect separation)
        separated_signals = [sig.copy() for sig in original_signals]

        result = tester.evaluate_separation_quality(original_signals, separated_signals)

        # Check result structure
        expected_keys = [
            "separation_scores",
            "correlation_scores",
            "cross_talk_matrix",
            "mean_separation_score",
            "min_separation_score",
            "max_cross_talk",
            "mean_cross_talk",
            "separation_quality",
        ]
        for key in expected_keys:
            assert key in result

        # Perfect separation should have high scores
        assert result["mean_separation_score"] > 0.9
        assert result["min_separation_score"] > 0.9
        assert result["max_cross_talk"] < 0.1

    def test_separation_quality_mismatched_signals(self, tester, orthogonal_signals):
        """Test error handling with mismatched signal counts."""
        original_signals = orthogonal_signals[:3]
        separated_signals = orthogonal_signals[:2]

        with pytest.raises(ValueError, match="Number of original and separated signals must match"):
            tester.evaluate_separation_quality(original_signals, separated_signals)

    def test_optimal_threshold_finding(self, tester, orthogonal_signals):
        """Test optimal threshold finding."""
        result = tester.find_optimal_threshold(
            orthogonal_signals, threshold_range=(0.01, 0.5), num_points=20
        )

        # Check result structure
        expected_keys = [
            "thresholds",
            "orthogonality_ratios",
            "optimal_threshold",
            "optimal_ratio",
            "threshold_range",
        ]
        for key in expected_keys:
            assert key in result

        # Check array lengths
        assert len(result["thresholds"]) == 20
        assert len(result["orthogonality_ratios"]) == 20

        # Optimal threshold should be in range
        assert 0.01 <= result["optimal_threshold"] <= 0.5
        assert 0.0 <= result["optimal_ratio"] <= 1.0

    def test_threshold_setting(self, tester):
        """Test orthogonality threshold setting and getting."""
        # Test setting valid threshold
        tester.set_orthogonality_threshold(0.05)
        assert tester.get_orthogonality_threshold() == 0.05

        # Test invalid thresholds
        with pytest.raises(ValueError, match="Orthogonality threshold must be between 0 and 1"):
            tester.set_orthogonality_threshold(0.0)

        with pytest.raises(ValueError, match="Orthogonality threshold must be between 0 and 1"):
            tester.set_orthogonality_threshold(1.0)

        with pytest.raises(ValueError, match="Orthogonality threshold must be between 0 and 1"):
            tester.set_orthogonality_threshold(-0.1)

    def test_orthogonality_report_generation(self, tester, orthogonal_signals):
        """Test orthogonality report generation."""
        report = tester.generate_orthogonality_report(orthogonal_signals)

        assert isinstance(report, str)
        assert "ORTHOGONALITY ANALYSIS REPORT" in report
        assert "Number of signals:" in report
        assert "CORRELATION STATISTICS:" in report
        assert "ORTHOGONALITY METRICS:" in report
        assert "BACKEND INFORMATION:" in report

        # Report should contain actual values
        assert "Maximum cross-correlation:" in report
        assert "Orthogonality ratio:" in report

    def test_custom_threshold_in_methods(self, tester, orthogonal_signals):
        """Test using custom thresholds in testing methods."""
        # Test with very strict threshold
        result_strict = tester.test_signal_set_orthogonality(orthogonal_signals, threshold=0.01)

        # Test with lenient threshold
        result_lenient = tester.test_signal_set_orthogonality(orthogonal_signals, threshold=0.5)

        # Strict threshold should result in fewer orthogonal pairs
        assert result_strict["orthogonal_pairs"] <= result_lenient["orthogonal_pairs"]
        assert result_strict["threshold"] == 0.01
        assert result_lenient["threshold"] == 0.5

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_cpu_consistency(self, orthogonal_signals):
        """Test that GPU and CPU implementations give consistent results."""
        # Create testers with GPU and CPU backends
        gpu_tester = OrthogonalityTester(GPUBackend(force_cpu=False))
        cpu_tester = OrthogonalityTester(GPUBackend(force_cpu=True))

        # Skip if GPU not available
        if not gpu_tester.backend.is_gpu_available:
            pytest.skip("GPU not available")

        # Compare results
        gpu_result = gpu_tester.test_signal_set_orthogonality(orthogonal_signals)
        cpu_result = cpu_tester.test_signal_set_orthogonality(orthogonal_signals)

        # Key metrics should be very close
        assert abs(gpu_result["max_cross_correlation"] - cpu_result["max_cross_correlation"]) < 1e-6
        assert (
            abs(gpu_result["mean_cross_correlation"] - cpu_result["mean_cross_correlation"]) < 1e-6
        )
        assert gpu_result["orthogonal_pairs"] == cpu_result["orthogonal_pairs"]

    def test_energy_calculations(self, tester):
        """Test signal energy calculations in orthogonality testing."""
        # Create signals with different energies
        sig1 = np.array([1, 0, 1, 0], dtype=float)  # Energy = 2
        sig2 = np.array([2, 0, 2, 0], dtype=float)  # Energy = 8

        result = tester.test_signal_pair_orthogonality(sig1, sig2)

        assert abs(result["signal1_energy"] - 2.0) < 1e-10
        assert abs(result["signal2_energy"] - 8.0) < 1e-10
        assert abs(result["energy_ratio"] - 4.0) < 1e-10

    def test_correlation_matrix_properties(self, tester, orthogonal_signals):
        """Test properties of correlation matrix."""
        result = tester.test_signal_set_orthogonality(orthogonal_signals)
        corr_matrix = result["correlation_matrix"]

        # Matrix should be square
        n_signals = len(orthogonal_signals)
        assert corr_matrix.shape == (n_signals, n_signals)

        # Diagonal should be 1 (self-correlation)
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Matrix should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)

        # All values should be between 0 and 1 for normalized correlation
        assert np.all(corr_matrix >= 0)
        assert np.all(corr_matrix <= 1)

    def test_cross_correlations_extraction(self, tester, orthogonal_signals):
        """Test extraction of cross-correlations from correlation matrix."""
        result = tester.test_signal_set_orthogonality(orthogonal_signals)

        n_signals = len(orthogonal_signals)
        expected_pairs = n_signals * (n_signals - 1) // 2

        assert len(result["cross_correlations"]) == expected_pairs
        assert result["total_pairs"] == expected_pairs

        # Cross-correlations should not include diagonal elements
        corr_matrix = result["correlation_matrix"]
        cross_corrs = result["cross_correlations"]

        # Extract upper triangular elements manually
        expected_cross_corrs = []
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                expected_cross_corrs.append(corr_matrix[i, j])

        assert np.allclose(cross_corrs, expected_cross_corrs)
