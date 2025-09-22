"""
Cross-correlation analysis for OFDM signal orthogonality testing.

This module provides GPU-accelerated correlation computation between signals
to evaluate orthogonality and signal separation quality.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from .gpu_backend import GPUBackend

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class CorrelationAnalyzer:
    """GPU-accelerated cross-correlation analyzer for signal orthogonality testing."""

    def __init__(self, gpu_backend: Optional[GPUBackend] = None):
        """Initialize correlation analyzer.

        Args:
            gpu_backend: GPU backend instance. If None, creates a new one.
        """
        self.backend = gpu_backend or GPUBackend()
        self._correlation_cache = {}

    def compute_cross_correlation(
        self,
        sig1: Union[np.ndarray, "cp.ndarray"],
        sig2: Union[np.ndarray, "cp.ndarray"],
        normalize: bool = True,
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Compute full cross-correlation between two signals using FFT.

        Args:
            sig1: First signal array
            sig2: Second signal array
            normalize: Whether to normalize the correlation

        Returns:
            Cross-correlation array
        """
        try:
            if self.backend.is_gpu_available:
                return self._compute_cross_correlation_gpu(sig1, sig2, normalize)
            else:
                return self._compute_cross_correlation_cpu(sig1, sig2, normalize)
        except Exception as e:
            logger.warning(f"Cross-correlation computation failed: {e}")
            return self._compute_cross_correlation_cpu(sig1, sig2, normalize)

    def _compute_cross_correlation_gpu(
        self,
        sig1: Union[np.ndarray, "cp.ndarray"],
        sig2: Union[np.ndarray, "cp.ndarray"],
        normalize: bool,
    ) -> "cp.ndarray":
        """GPU implementation of cross-correlation."""
        # Transfer to GPU
        sig1_gpu = self.backend.to_gpu(sig1)
        sig2_gpu = self.backend.to_gpu(sig2)

        # Ensure same length by zero-padding if necessary
        max_len = max(len(sig1_gpu), len(sig2_gpu))
        if len(sig1_gpu) < max_len:
            sig1_gpu = cp.pad(sig1_gpu, (0, max_len - len(sig1_gpu)))
        if len(sig2_gpu) < max_len:
            sig2_gpu = cp.pad(sig2_gpu, (0, max_len - len(sig2_gpu)))

        # Compute cross-correlation using FFT
        sig1_fft = cp.fft.fft(sig1_gpu)
        sig2_fft = cp.fft.fft(sig2_gpu)
        correlation = cp.fft.ifft(sig1_fft * cp.conj(sig2_fft))

        if normalize:
            # Normalize by signal energies for proper correlation coefficient
            energy1 = cp.sqrt(cp.sum(cp.abs(sig1_gpu) ** 2))
            energy2 = cp.sqrt(cp.sum(cp.abs(sig2_gpu) ** 2))
            if energy1 > 0 and energy2 > 0:
                correlation = correlation / (energy1 * energy2)

        return correlation

    def _compute_cross_correlation_cpu(
        self,
        sig1: Union[np.ndarray, "cp.ndarray"],
        sig2: Union[np.ndarray, "cp.ndarray"],
        normalize: bool,
    ) -> np.ndarray:
        """CPU implementation of cross-correlation."""
        # Convert to CPU arrays
        sig1_cpu = self.backend.to_cpu(sig1)
        sig2_cpu = self.backend.to_cpu(sig2)

        # Ensure same length by zero-padding if necessary
        max_len = max(len(sig1_cpu), len(sig2_cpu))
        if len(sig1_cpu) < max_len:
            sig1_cpu = np.pad(sig1_cpu, (0, max_len - len(sig1_cpu)))
        if len(sig2_cpu) < max_len:
            sig2_cpu = np.pad(sig2_cpu, (0, max_len - len(sig2_cpu)))

        # Compute cross-correlation using FFT
        sig1_fft = np.fft.fft(sig1_cpu)
        sig2_fft = np.fft.fft(sig2_cpu)
        correlation = np.fft.ifft(sig1_fft * np.conj(sig2_fft))

        if normalize:
            # Normalize by signal energies for proper correlation coefficient
            energy1 = np.sqrt(np.sum(np.abs(sig1_cpu) ** 2))
            energy2 = np.sqrt(np.sum(np.abs(sig2_cpu) ** 2))
            if energy1 > 0 and energy2 > 0:
                correlation = correlation / (energy1 * energy2)

        return correlation

    def compute_max_correlation(
        self,
        sig1: Union[np.ndarray, "cp.ndarray"],
        sig2: Union[np.ndarray, "cp.ndarray"],
        normalize: bool = True,
    ) -> float:
        """Compute maximum cross-correlation value between two signals.

        For orthogonality testing, this computes the zero-lag correlation
        (normalized dot product) which is the most relevant measure.

        Args:
            sig1: First signal array
            sig2: Second signal array
            normalize: Whether to normalize the correlation

        Returns:
            Maximum correlation value (float)
        """
        if self.backend.is_gpu_available:
            sig1_gpu = self.backend.to_gpu(sig1)
            sig2_gpu = self.backend.to_gpu(sig2)

            # Ensure same length by truncating to minimum length
            min_len = min(len(sig1_gpu), len(sig2_gpu))
            sig1_gpu = sig1_gpu[:min_len]
            sig2_gpu = sig2_gpu[:min_len]

            # Compute zero-lag correlation (dot product)
            dot_product = cp.sum(sig1_gpu * cp.conj(sig2_gpu))

            if normalize:
                energy1 = cp.sqrt(cp.sum(cp.abs(sig1_gpu) ** 2))
                energy2 = cp.sqrt(cp.sum(cp.abs(sig2_gpu) ** 2))
                if energy1 > 0 and energy2 > 0:
                    correlation = dot_product / (energy1 * energy2)
                else:
                    correlation = 0.0
            else:
                correlation = dot_product

            return float(cp.abs(correlation).get())
        else:
            sig1_cpu = self.backend.to_cpu(sig1)
            sig2_cpu = self.backend.to_cpu(sig2)

            # Ensure same length by truncating to minimum length
            min_len = min(len(sig1_cpu), len(sig2_cpu))
            sig1_cpu = sig1_cpu[:min_len]
            sig2_cpu = sig2_cpu[:min_len]

            # Compute zero-lag correlation (dot product)
            dot_product = np.sum(sig1_cpu * np.conj(sig2_cpu))

            if normalize:
                energy1 = np.sqrt(np.sum(np.abs(sig1_cpu) ** 2))
                energy2 = np.sqrt(np.sum(np.abs(sig2_cpu) ** 2))
                if energy1 > 0 and energy2 > 0:
                    correlation = dot_product / (energy1 * energy2)
                else:
                    correlation = 0.0
            else:
                correlation = dot_product

            return float(np.abs(correlation))

    def compute_correlation_matrix(
        self, signals: List[Union[np.ndarray, "cp.ndarray"]], normalize: bool = True
    ) -> np.ndarray:
        """Compute cross-correlation matrix for a set of signals.

        Args:
            signals: List of signal arrays
            normalize: Whether to normalize correlations

        Returns:
            Correlation matrix where element (i,j) is max correlation between signals i and j
        """
        n_signals = len(signals)
        correlation_matrix = np.zeros((n_signals, n_signals))

        for i in range(n_signals):
            for j in range(i, n_signals):
                if i == j:
                    correlation_matrix[i, j] = 1.0  # Self-correlation is always 1
                else:
                    corr = self.compute_max_correlation(signals[i], signals[j], normalize)
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr  # Symmetric matrix

        return correlation_matrix

    def compute_autocorrelation(
        self, signal: Union[np.ndarray, "cp.ndarray"], normalize: bool = True
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Compute autocorrelation of a signal.

        Args:
            signal: Input signal array
            normalize: Whether to normalize the autocorrelation

        Returns:
            Autocorrelation array
        """
        return self.compute_cross_correlation(signal, signal, normalize)

    def find_correlation_peaks(
        self, correlation: Union[np.ndarray, "cp.ndarray"], threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks in correlation function above threshold.

        Args:
            correlation: Correlation array
            threshold: Minimum peak height

        Returns:
            Tuple of (peak_indices, peak_values)
        """
        # Convert to CPU for peak finding
        corr_cpu = self.backend.to_cpu(correlation)
        corr_abs = np.abs(corr_cpu)

        # Find peaks above threshold
        peak_indices = []
        peak_values = []

        for i in range(1, len(corr_abs) - 1):
            if (
                corr_abs[i] > threshold
                and corr_abs[i] > corr_abs[i - 1]
                and corr_abs[i] > corr_abs[i + 1]
            ):
                peak_indices.append(i)
                peak_values.append(corr_abs[i])

        return np.array(peak_indices), np.array(peak_values)

    def clear_cache(self):
        """Clear correlation computation cache."""
        self._correlation_cache.clear()

    def get_memory_usage(self) -> dict:
        """Get memory usage information from the backend."""
        return self.backend.get_memory_info()
