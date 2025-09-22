"""
Tests for the CorrelationAnalyzer class.

This module tests cross-correlation computation, GPU acceleration,
and correlation analysis functionality.
"""

import numpy as np
import pytest

from ofdm_chirp_generator.correlation_analyzer import CorrelationAnalyzer
from ofdm_chirp_generator.gpu_backend import GPUBackend

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a CorrelationAnalyzer instance."""
        return CorrelationAnalyzer()

    @pytest.fixture
    def test_signals(self):
        """Create test signals for correlation analysis."""
        # Create two orthogonal sinusoidal signals
        t = np.linspace(0, 1, 1000)
        sig1 = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        sig2 = np.cos(2 * np.pi * 10 * t)  # 10 Hz cosine wave (orthogonal)
        sig3 = np.sin(2 * np.pi * 20 * t)  # 20 Hz sine wave

        return sig1, sig2, sig3

    def test_analyzer_initialization(self):
        """Test CorrelationAnalyzer initialization."""
        analyzer = CorrelationAnalyzer()
        assert analyzer.backend is not None
        assert hasattr(analyzer, "_correlation_cache")

    def test_analyzer_with_custom_backend(self):
        """Test CorrelationAnalyzer with custom GPU backend."""
        backend = GPUBackend(force_cpu=True)
        analyzer = CorrelationAnalyzer(backend)
        assert analyzer.backend is backend
        assert not analyzer.backend.is_gpu_available

    def test_cross_correlation_computation(self, analyzer, test_signals):
        """Test cross-correlation computation between signals."""
        sig1, sig2, sig3 = test_signals

        # Test cross-correlation between orthogonal signals
        correlation = analyzer.compute_cross_correlation(sig1, sig2, normalize=True)
        assert len(correlation) == len(sig1)

        # Test that correlation is computed correctly
        max_corr = analyzer.compute_max_correlation(sig1, sig2, normalize=True)
        assert isinstance(max_corr, float)
        assert 0 <= max_corr <= 1

    def test_autocorrelation(self, analyzer, test_signals):
        """Test autocorrelation computation."""
        sig1, _, _ = test_signals

        autocorr = analyzer.compute_autocorrelation(sig1, normalize=True)
        assert len(autocorr) == len(sig1)

        # Maximum of normalized autocorrelation should be 1
        if analyzer.backend.is_gpu_available and hasattr(autocorr, "get"):
            max_autocorr = float(cp.max(cp.abs(autocorr)).get())
        else:
            max_autocorr = float(np.max(np.abs(autocorr)))

        assert abs(max_autocorr - 1.0) < 1e-10

    def test_correlation_matrix(self, analyzer, test_signals):
        """Test correlation matrix computation."""
        sig1, sig2, sig3 = test_signals
        signals = [sig1, sig2, sig3]

        corr_matrix = analyzer.compute_correlation_matrix(signals, normalize=True)

        # Check matrix properties
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric

        # All values should be between 0 and 1 for normalized correlation
        assert np.all(corr_matrix >= 0)
        assert np.all(corr_matrix <= 1)

    def test_peak_finding(self, analyzer, test_signals):
        """Test correlation peak finding."""
        sig1, _, _ = test_signals

        autocorr = analyzer.compute_autocorrelation(sig1, normalize=True)
        peaks_idx, peaks_val = analyzer.find_correlation_peaks(autocorr, threshold=0.5)

        assert isinstance(peaks_idx, np.ndarray)
        assert isinstance(peaks_val, np.ndarray)
        assert len(peaks_idx) == len(peaks_val)

        # All peak values should be above threshold
        assert np.all(peaks_val >= 0.5)

    def test_orthogonal_signals_correlation(self, analyzer):
        """Test correlation between known orthogonal signals."""
        # Create orthogonal Walsh functions
        n = 8
        walsh1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        walsh2 = np.array([1, -1, 1, -1, 1, -1, 1, -1])

        # These should have zero cross-correlation
        max_corr = analyzer.compute_max_correlation(walsh1, walsh2, normalize=True)
        assert abs(max_corr) < 1e-10

    def test_identical_signals_correlation(self, analyzer, test_signals):
        """Test correlation between identical signals."""
        sig1, _, _ = test_signals

        # Correlation between identical signals should be 1
        max_corr = analyzer.compute_max_correlation(sig1, sig1, normalize=True)
        assert abs(max_corr - 1.0) < 1e-10

    def test_different_length_signals(self, analyzer):
        """Test correlation between signals of different lengths."""
        sig1 = np.sin(2 * np.pi * np.linspace(0, 1, 100))
        sig2 = np.sin(2 * np.pi * np.linspace(0, 1, 150))

        # Should handle different lengths gracefully
        correlation = analyzer.compute_cross_correlation(sig1, sig2, normalize=True)
        assert len(correlation) == max(len(sig1), len(sig2))

        max_corr = analyzer.compute_max_correlation(sig1, sig2, normalize=True)
        assert isinstance(max_corr, float)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_cpu_consistency(self, test_signals):
        """Test that GPU and CPU implementations give consistent results."""
        sig1, sig2, _ = test_signals

        # Create analyzers with GPU and CPU backends
        gpu_analyzer = CorrelationAnalyzer(GPUBackend(force_cpu=False))
        cpu_analyzer = CorrelationAnalyzer(GPUBackend(force_cpu=True))

        # Skip if GPU not available
        if not gpu_analyzer.backend.is_gpu_available:
            pytest.skip("GPU not available")

        # Compare results
        gpu_corr = gpu_analyzer.compute_max_correlation(sig1, sig2, normalize=True)
        cpu_corr = cpu_analyzer.compute_max_correlation(sig1, sig2, normalize=True)

        assert abs(gpu_corr - cpu_corr) < 1e-6

    def test_memory_management(self, analyzer):
        """Test memory management functionality."""
        memory_info = analyzer.get_memory_usage()
        assert isinstance(memory_info, dict)
        assert "backend" in memory_info

        # Test cache clearing
        analyzer.clear_cache()
        assert len(analyzer._correlation_cache) == 0

    def test_normalization_effects(self, analyzer):
        """Test effects of normalization on correlation results."""
        # Use non-orthogonal signals for this test
        sig1 = np.array([1, 2, 3, 4, 5], dtype=float)
        sig2 = np.array([1, 2, 3, 4, 6], dtype=float)  # Similar but not identical

        # Scale one signal
        sig1_scaled = sig1 * 10

        # Normalized correlation should be the same
        corr_norm = analyzer.compute_max_correlation(sig1, sig2, normalize=True)
        corr_scaled_norm = analyzer.compute_max_correlation(sig1_scaled, sig2, normalize=True)

        assert abs(corr_norm - corr_scaled_norm) < 1e-10

        # Unnormalized correlation should be different
        corr_unnorm = analyzer.compute_max_correlation(sig1, sig2, normalize=False)
        corr_scaled_unnorm = analyzer.compute_max_correlation(sig1_scaled, sig2, normalize=False)

        assert abs(corr_unnorm - corr_scaled_unnorm) > 1e-6
