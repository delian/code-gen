"""
Orthogonality testing framework for OFDM signal evaluation.

This module provides comprehensive orthogonality testing capabilities
for evaluating signal pair orthogonality and signal set quality.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config_manager import ConfigurationError, get_config
from .correlation_analyzer import CorrelationAnalyzer
from .gpu_backend import GPUBackend

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class OrthogonalityTester:
    """Comprehensive orthogonality testing for OFDM signal pairs and sets."""

    def __init__(self, gpu_backend: Optional[GPUBackend] = None, config_file: Optional[str] = None):
        """Initialize orthogonality tester.

        Args:
            gpu_backend: GPU backend instance. If None, creates a new one.
            config_file: Path to configuration file (uses default if None)
        """
        self.backend = gpu_backend or GPUBackend()
        self.correlation_analyzer = CorrelationAnalyzer(self.backend)

        # Load orthogonality threshold from configuration
        try:
            config_manager = get_config(config_file)
            orth_config = config_manager.get_orthogonality_config()
            self.orthogonality_threshold = orth_config["default_threshold"]
            logger.info(
                f"Loaded orthogonality threshold from config: {self.orthogonality_threshold}"
            )
        except (ConfigurationError, Exception) as e:
            self.orthogonality_threshold = 0.1  # Default threshold for orthogonality
            logger.warning(
                f"Could not load orthogonality configuration: {e}. Using default threshold: {self.orthogonality_threshold}"
            )

    def test_signal_pair_orthogonality(
        self,
        sig1: Union[np.ndarray, "cp.ndarray"],
        sig2: Union[np.ndarray, "cp.ndarray"],
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Test orthogonality between two signals.

        Args:
            sig1: First signal array
            sig2: Second signal array
            threshold: Orthogonality threshold (uses default if None)

        Returns:
            Dictionary with orthogonality metrics
        """
        if threshold is None:
            threshold = self.orthogonality_threshold

        # Compute cross-correlation
        max_correlation = self.correlation_analyzer.compute_max_correlation(
            sig1, sig2, normalize=True
        )

        # Compute signal energies
        if self.backend.is_gpu_available:
            sig1_gpu = self.backend.to_gpu(sig1)
            sig2_gpu = self.backend.to_gpu(sig2)
            energy1 = float(cp.sum(cp.abs(sig1_gpu) ** 2).get())
            energy2 = float(cp.sum(cp.abs(sig2_gpu) ** 2).get())
        else:
            sig1_cpu = self.backend.to_cpu(sig1)
            sig2_cpu = self.backend.to_cpu(sig2)
            energy1 = float(np.sum(np.abs(sig1_cpu) ** 2))
            energy2 = float(np.sum(np.abs(sig2_cpu) ** 2))

        # Calculate orthogonality score (1 - normalized correlation)
        orthogonality_score = 1.0 - max_correlation

        # Determine if signals are orthogonal
        is_orthogonal = max_correlation < threshold

        return {
            "max_correlation": max_correlation,
            "orthogonality_score": orthogonality_score,
            "is_orthogonal": is_orthogonal,
            "threshold": threshold,
            "signal1_energy": energy1,
            "signal2_energy": energy2,
            "energy_ratio": energy2 / energy1 if energy1 > 0 else float("inf"),
        }

    def test_signal_set_orthogonality(
        self, signals: List[Union[np.ndarray, "cp.ndarray"]], threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """Test orthogonality for a complete set of signals.

        Args:
            signals: List of signal arrays
            threshold: Orthogonality threshold (uses default if None)

        Returns:
            Dictionary with comprehensive orthogonality analysis
        """
        if threshold is None:
            threshold = self.orthogonality_threshold

        n_signals = len(signals)
        if n_signals < 2:
            raise ValueError("Need at least 2 signals for orthogonality testing")

        # Compute correlation matrix
        correlation_matrix = self.correlation_analyzer.compute_correlation_matrix(
            signals, normalize=True
        )

        # Extract off-diagonal elements (cross-correlations)
        cross_correlations = []
        orthogonal_pairs = 0
        total_pairs = 0

        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                corr = correlation_matrix[i, j]
                cross_correlations.append(corr)
                if corr < threshold:
                    orthogonal_pairs += 1
                total_pairs += 1

        cross_correlations = np.array(cross_correlations)

        # Calculate overall metrics
        max_cross_correlation = np.max(cross_correlations)
        mean_cross_correlation = np.mean(cross_correlations)
        std_cross_correlation = np.std(cross_correlations)
        orthogonality_ratio = orthogonal_pairs / total_pairs if total_pairs > 0 else 0.0

        # Overall orthogonality score (higher is better)
        overall_orthogonality_score = 1.0 - mean_cross_correlation

        # Determine if the entire set is orthogonal
        is_set_orthogonal = orthogonality_ratio >= 0.95  # 95% of pairs must be orthogonal

        return {
            "num_signals": n_signals,
            "correlation_matrix": correlation_matrix,
            "cross_correlations": cross_correlations,
            "max_cross_correlation": max_cross_correlation,
            "mean_cross_correlation": mean_cross_correlation,
            "std_cross_correlation": std_cross_correlation,
            "orthogonal_pairs": orthogonal_pairs,
            "total_pairs": total_pairs,
            "orthogonality_ratio": orthogonality_ratio,
            "overall_orthogonality_score": overall_orthogonality_score,
            "is_set_orthogonal": is_set_orthogonal,
            "threshold": threshold,
        }

    def evaluate_separation_quality(
        self,
        original_signals: List[Union[np.ndarray, "cp.ndarray"]],
        separated_signals: List[Union[np.ndarray, "cp.ndarray"]],
    ) -> Dict[str, any]:
        """Evaluate quality of signal separation.

        Args:
            original_signals: List of original signal arrays
            separated_signals: List of separated signal arrays

        Returns:
            Dictionary with separation quality metrics
        """
        if len(original_signals) != len(separated_signals):
            raise ValueError("Number of original and separated signals must match")

        n_signals = len(original_signals)
        separation_scores = []
        correlation_scores = []

        for i in range(n_signals):
            # Compute correlation between original and separated signal
            corr = self.correlation_analyzer.compute_max_correlation(
                original_signals[i], separated_signals[i], normalize=True
            )
            correlation_scores.append(corr)

            # Separation score is the correlation with the intended signal
            separation_scores.append(corr)

        # Calculate cross-talk (correlation with unintended signals)
        cross_talk_matrix = np.zeros((n_signals, n_signals))
        for i in range(n_signals):
            for j in range(n_signals):
                if i != j:
                    cross_talk = self.correlation_analyzer.compute_max_correlation(
                        original_signals[i], separated_signals[j], normalize=True
                    )
                    cross_talk_matrix[i, j] = cross_talk

        # Calculate metrics
        mean_separation_score = np.mean(separation_scores)
        min_separation_score = np.min(separation_scores)
        max_cross_talk = np.max(cross_talk_matrix)
        mean_cross_talk = np.mean(cross_talk_matrix[cross_talk_matrix > 0])

        # Overall separation quality (higher is better)
        separation_quality = mean_separation_score - max_cross_talk

        return {
            "separation_scores": separation_scores,
            "correlation_scores": correlation_scores,
            "cross_talk_matrix": cross_talk_matrix,
            "mean_separation_score": mean_separation_score,
            "min_separation_score": min_separation_score,
            "max_cross_talk": max_cross_talk,
            "mean_cross_talk": mean_cross_talk,
            "separation_quality": separation_quality,
        }

    def find_optimal_threshold(
        self,
        signals: List[Union[np.ndarray, "cp.ndarray"]],
        threshold_range: Tuple[float, float] = (0.01, 0.5),
        num_points: int = 50,
    ) -> Dict[str, any]:
        """Find optimal orthogonality threshold for a signal set.

        Args:
            signals: List of signal arrays
            threshold_range: Range of thresholds to test (min, max)
            num_points: Number of threshold points to test

        Returns:
            Dictionary with optimal threshold analysis
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        orthogonality_ratios = []

        for threshold in thresholds:
            result = self.test_signal_set_orthogonality(signals, threshold)
            orthogonality_ratios.append(result["orthogonality_ratio"])

        orthogonality_ratios = np.array(orthogonality_ratios)

        # Find threshold that maximizes orthogonality ratio
        optimal_idx = np.argmax(orthogonality_ratios)
        optimal_threshold = thresholds[optimal_idx]
        optimal_ratio = orthogonality_ratios[optimal_idx]

        return {
            "thresholds": thresholds,
            "orthogonality_ratios": orthogonality_ratios,
            "optimal_threshold": optimal_threshold,
            "optimal_ratio": optimal_ratio,
            "threshold_range": threshold_range,
        }

    def generate_orthogonality_report(
        self, signals: List[Union[np.ndarray, "cp.ndarray"]], threshold: Optional[float] = None
    ) -> str:
        """Generate a comprehensive orthogonality report.

        Args:
            signals: List of signal arrays
            threshold: Orthogonality threshold (uses default if None)

        Returns:
            Formatted report string
        """
        if threshold is None:
            threshold = self.orthogonality_threshold

        result = self.test_signal_set_orthogonality(signals, threshold)

        report = []
        report.append("=" * 60)
        report.append("ORTHOGONALITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Number of signals: {result['num_signals']}")
        report.append(f"Orthogonality threshold: {threshold:.4f}")
        report.append("")

        report.append("CORRELATION STATISTICS:")
        report.append(f"  Maximum cross-correlation: {result['max_cross_correlation']:.6f}")
        report.append(f"  Mean cross-correlation: {result['mean_cross_correlation']:.6f}")
        report.append(f"  Std cross-correlation: {result['std_cross_correlation']:.6f}")
        report.append("")

        report.append("ORTHOGONALITY METRICS:")
        report.append(f"  Orthogonal pairs: {result['orthogonal_pairs']}/{result['total_pairs']}")
        report.append(f"  Orthogonality ratio: {result['orthogonality_ratio']:.2%}")
        report.append(f"  Overall orthogonality score: {result['overall_orthogonality_score']:.6f}")
        report.append(f"  Set is orthogonal: {'YES' if result['is_set_orthogonal'] else 'NO'}")
        report.append("")

        report.append("BACKEND INFORMATION:")
        device_info = self.backend.device_info
        report.append(f"  Compute backend: {device_info['backend']}")
        if device_info["backend"] == "GPU":
            report.append(f"  Device: {device_info['device_name']}")
            report.append(
                f"  Memory usage: {device_info['memory_total'] - device_info['memory_free']:,} / {device_info['memory_total']:,} bytes"
            )

        report.append("=" * 60)

        return "\n".join(report)

    def set_orthogonality_threshold(self, threshold: float):
        """Set the default orthogonality threshold.

        Args:
            threshold: New orthogonality threshold (0 < threshold < 1)
        """
        if not 0 < threshold < 1:
            raise ValueError("Orthogonality threshold must be between 0 and 1")
        self.orthogonality_threshold = threshold

    def get_orthogonality_threshold(self) -> float:
        """Get the current orthogonality threshold."""
        return self.orthogonality_threshold
