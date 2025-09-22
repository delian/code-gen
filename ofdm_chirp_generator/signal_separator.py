"""
Signal separation engine for phase-based recovery of orthogonal OFDM signals.

This module implements the SignalSeparator class that can separate overlapping
orthogonal OFDM signals using phase-based correlation analysis and provides
comprehensive separation quality metrics and diagnostic reporting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config_manager import ConfigurationError, get_config
from .correlation_analyzer import CorrelationAnalyzer
from .error_handling import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    SeparationError,
    create_error_context,
    with_error_handling,
)
from .gpu_backend import GPUBackend
from .models import OFDMConfig, SignalSet
from .orthogonality_tester import OrthogonalityTester

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class SeparationQualityMetrics:
    """Container for signal separation quality metrics."""

    def __init__(self):
        self.separation_scores: List[float] = []
        self.cross_talk_matrix: np.ndarray = np.array([])
        self.signal_to_interference_ratios: List[float] = []
        self.mean_separation_score: float = 0.0
        self.min_separation_score: float = 0.0
        self.max_cross_talk: float = 0.0
        self.mean_cross_talk: float = 0.0
        self.overall_separation_quality: float = 0.0
        self.separation_success: bool = False
        self.diagnostic_info: Dict[str, any] = {}


class SignalSeparator:
    """Phase-based signal separation engine for orthogonal OFDM signals.

    This class implements signal separation capabilities using phase-based
    correlation analysis to recover individual signals from overlapping
    orthogonal transmissions.

    Requirements addressed:
    - 6.1: Separate two orthogonal signals when combined
    - 6.2: Use phase-based correlation analysis for separation
    - 6.3: Report separation quality metrics
    - 6.4: Provide diagnostic information about separation failures
    """

    def __init__(
        self,
        ofdm_config: OFDMConfig,
        gpu_backend: Optional[GPUBackend] = None,
        config_file: Optional[str] = None,
    ):
        """Initialize signal separator.

        Args:
            ofdm_config: OFDM configuration parameters
            gpu_backend: GPU backend for acceleration (creates new if None)
            config_file: Path to configuration file (uses default if None)
        """
        self.ofdm_config = ofdm_config
        self.gpu_backend = gpu_backend or GPUBackend()
        self._error_handler = ErrorHandler()

        try:
            self.correlation_analyzer = CorrelationAnalyzer(self.gpu_backend)
            self.orthogonality_tester = OrthogonalityTester(self.gpu_backend, config_file)
        except Exception as e:
            context = create_error_context("component_initialization", "SignalSeparator")
            self._error_handler.handle_error(e, context)
            raise

        # Load separation configuration
        try:
            config_manager = get_config(config_file)
            sep_config = config_manager.get_separation_config()
            self.separation_threshold = sep_config.get("quality_threshold", 0.8)
            self.max_iterations = sep_config.get("max_iterations", 100)
            self.convergence_tolerance = sep_config.get("convergence_tolerance", 1e-6)
            logger.info(f"Loaded separation configuration: threshold={self.separation_threshold}")
        except (ConfigurationError, Exception) as e:
            context = create_error_context(
                "config_load", "SignalSeparator", config_file=config_file
            )
            self._error_handler.handle_error(e, context)
            # Use default values
            self.separation_threshold = 0.8
            self.max_iterations = 100
            self.convergence_tolerance = 1e-6
            logger.warning(f"Could not load separation configuration: {e}. Using defaults.")

        # Internal state
        self._reference_signals: Optional[List[Union[np.ndarray, "cp.ndarray"]]] = None
        self._reference_phases: Optional[np.ndarray] = None

        logger.info(f"SignalSeparator initialized with {ofdm_config.num_subcarriers} subcarriers")

    def set_reference_signals(
        self, reference_signals: List[Union[np.ndarray, "cp.ndarray"]], reference_phases: np.ndarray
    ) -> None:
        """Set reference signals and phases for separation.

        Args:
            reference_signals: List of reference signal arrays
            reference_phases: Phase matrix used to generate reference signals

        Raises:
            ValueError: If reference data is invalid
        """
        if len(reference_signals) == 0:
            raise ValueError("Reference signals list cannot be empty")

        if reference_phases.ndim != 2:
            raise ValueError("Reference phases must be 2D array")

        if reference_phases.shape[0] != len(reference_signals):
            raise ValueError("Number of reference signals must match phase matrix rows")

        if reference_phases.shape[1] != self.ofdm_config.num_subcarriers:
            raise ValueError("Phase matrix columns must match number of subcarriers")

        self._reference_signals = reference_signals
        self._reference_phases = reference_phases.copy()

        logger.info(f"Set {len(reference_signals)} reference signals for separation")

    def separate_combined_signal(
        self,
        combined_signal: Union[np.ndarray, "cp.ndarray"],
        reference_signals: Optional[List[Union[np.ndarray, "cp.ndarray"]]] = None,
        reference_phases: Optional[np.ndarray] = None,
    ) -> Tuple[List[Union[np.ndarray, "cp.ndarray"]], SeparationQualityMetrics]:
        """Separate a combined signal into individual orthogonal components.

        Requirements:
        - 6.1: Separate two orthogonal signals when combined
        - 6.2: Use phase-based correlation analysis for separation

        Args:
            combined_signal: Combined signal containing multiple orthogonal signals
            reference_signals: Reference signals for separation (uses stored if None)
            reference_phases: Reference phase matrix (uses stored if None)

        Returns:
            Tuple of (separated_signals, quality_metrics)

        Raises:
            ValueError: If reference signals are not available or invalid
        """
        # Use provided references or stored ones
        if reference_signals is not None and reference_phases is not None:
            self.set_reference_signals(reference_signals, reference_phases)

        if self._reference_signals is None or self._reference_phases is None:
            raise ValueError("Reference signals and phases must be set before separation")

        logger.info(
            f"Starting separation of combined signal with {len(self._reference_signals)} reference signals"
        )

        # Initialize separation results
        separated_signals = []
        quality_metrics = SeparationQualityMetrics()

        try:
            # Validate input signal
            if np.any(np.isnan(self.gpu_backend.to_cpu(combined_signal))) or np.any(
                np.isinf(self.gpu_backend.to_cpu(combined_signal))
            ):
                raise SeparationError(
                    "Combined signal contains NaN or Inf values",
                    0.0,
                    self.separation_threshold,
                    ErrorSeverity.HIGH,
                )

            # Perform phase-based separation for each reference signal
            for i, ref_signal in enumerate(self._reference_signals):
                try:
                    separated_signal = self._separate_single_signal(
                        combined_signal, ref_signal, self._reference_phases[i, :]
                    )
                    separated_signals.append(separated_signal)
                except Exception as sep_error:
                    context = create_error_context(
                        "single_signal_separation", "SignalSeparator", signal_index=i
                    )
                    self._error_handler.handle_error(sep_error, context)
                    # Add zero signal as fallback
                    fallback_signal = self.gpu_backend.allocate_signal_memory(combined_signal.shape)
                    separated_signals.append(fallback_signal)

            # Compute separation quality metrics
            quality_metrics = self._compute_separation_quality(
                self._reference_signals, separated_signals, combined_signal
            )

            # Check if separation meets quality threshold
            if quality_metrics.overall_separation_quality < self.separation_threshold:
                error = SeparationError(
                    f"Separation quality {quality_metrics.overall_separation_quality:.4f} "
                    f"below threshold {self.separation_threshold:.4f}",
                    quality_metrics.overall_separation_quality,
                    self.separation_threshold,
                    ErrorSeverity.MEDIUM,
                )
                context = create_error_context("separation_quality_check", "SignalSeparator")
                self._error_handler.handle_error(error, context)

            logger.info(
                f"Separation completed with quality score: {quality_metrics.overall_separation_quality:.4f}"
            )

        except Exception as e:
            if isinstance(e, SeparationError):
                sep_error = e
            else:
                sep_error = SeparationError(
                    f"Signal separation failed: {e}",
                    0.0,
                    self.separation_threshold,
                    ErrorSeverity.HIGH,
                )

            context = create_error_context(
                "signal_separation",
                "SignalSeparator",
                num_reference_signals=len(self._reference_signals),
            )
            self._error_handler.handle_error(sep_error, context)

            logger.error(f"Signal separation failed: {e}")
            quality_metrics.diagnostic_info["error"] = str(e)
            quality_metrics.separation_success = False

        return separated_signals, quality_metrics

    def _separate_single_signal(
        self,
        combined_signal: Union[np.ndarray, "cp.ndarray"],
        reference_signal: Union[np.ndarray, "cp.ndarray"],
        reference_phases: np.ndarray,
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Separate a single signal from the combined signal using phase correlation.

        Args:
            combined_signal: Combined signal containing multiple components
            reference_signal: Reference signal for the target component
            reference_phases: Phase array for the reference signal

        Returns:
            Separated signal component
        """
        # Convert to GPU if available
        if self.gpu_backend.is_gpu_available:
            combined_gpu = self.gpu_backend.to_gpu(combined_signal)
            reference_gpu = self.gpu_backend.to_gpu(reference_signal)

            # Compute cross-correlation between combined and reference signals
            correlation = self.correlation_analyzer.compute_cross_correlation(
                combined_gpu, reference_gpu, normalize=True
            )

            # Find the peak correlation to determine the signal component
            correlation_abs = cp.abs(correlation)
            peak_idx = cp.argmax(correlation_abs)
            peak_value = correlation[peak_idx]

            # Extract signal component using correlation-based weighting
            # This is a simplified separation - in practice, more sophisticated
            # algorithms like MMSE or ML estimation could be used
            separation_weight = peak_value / cp.abs(peak_value) if cp.abs(peak_value) > 0 else 0
            separated_signal = combined_gpu * separation_weight

            # Apply phase correction based on reference phases
            phase_correction = cp.exp(1j * self.gpu_backend.to_gpu(reference_phases))
            if len(phase_correction) == len(separated_signal):
                separated_signal = separated_signal * phase_correction

        else:
            # CPU implementation
            combined_cpu = self.gpu_backend.to_cpu(combined_signal)
            reference_cpu = self.gpu_backend.to_cpu(reference_signal)

            # Compute cross-correlation
            correlation = self.correlation_analyzer.compute_cross_correlation(
                combined_cpu, reference_cpu, normalize=True
            )

            # Find peak correlation
            correlation_abs = np.abs(correlation)
            peak_idx = np.argmax(correlation_abs)
            peak_value = correlation[peak_idx]

            # Extract signal component
            separation_weight = peak_value / np.abs(peak_value) if np.abs(peak_value) > 0 else 0
            separated_signal = combined_cpu * separation_weight

            # Apply phase correction
            phase_correction = np.exp(1j * reference_phases)
            if len(phase_correction) == len(separated_signal):
                separated_signal = separated_signal * phase_correction

        return separated_signal

    def _compute_separation_quality(
        self,
        original_signals: List[Union[np.ndarray, "cp.ndarray"]],
        separated_signals: List[Union[np.ndarray, "cp.ndarray"]],
        combined_signal: Union[np.ndarray, "cp.ndarray"],
    ) -> SeparationQualityMetrics:
        """Compute comprehensive separation quality metrics.

        Requirements:
        - 6.3: Report separation quality metrics
        - 6.4: Provide diagnostic information

        Args:
            original_signals: Original reference signals
            separated_signals: Separated signal components
            combined_signal: Original combined signal

        Returns:
            SeparationQualityMetrics object with comprehensive metrics
        """
        metrics = SeparationQualityMetrics()

        try:
            n_signals = len(original_signals)

            # Compute separation scores (correlation with original signals)
            separation_scores = []
            for i in range(n_signals):
                corr = self.correlation_analyzer.compute_max_correlation(
                    original_signals[i], separated_signals[i], normalize=True
                )
                separation_scores.append(corr)

            metrics.separation_scores = separation_scores

            # Compute cross-talk matrix (correlation with unintended signals)
            cross_talk_matrix = np.zeros((n_signals, n_signals))
            for i in range(n_signals):
                for j in range(n_signals):
                    if i != j:
                        cross_talk = self.correlation_analyzer.compute_max_correlation(
                            original_signals[i], separated_signals[j], normalize=True
                        )
                        cross_talk_matrix[i, j] = cross_talk

            metrics.cross_talk_matrix = cross_talk_matrix

            # Compute Signal-to-Interference Ratios (SIR)
            sir_values = []
            for i in range(n_signals):
                signal_power = separation_scores[i] ** 2
                interference_power = np.sum(cross_talk_matrix[i, :] ** 2)
                sir = signal_power / interference_power if interference_power > 0 else float("inf")
                sir_values.append(sir)

            metrics.signal_to_interference_ratios = sir_values

            # Calculate summary statistics
            metrics.mean_separation_score = np.mean(separation_scores)
            metrics.min_separation_score = np.min(separation_scores)
            metrics.max_cross_talk = np.max(cross_talk_matrix)
            metrics.mean_cross_talk = (
                np.mean(cross_talk_matrix[cross_talk_matrix > 0])
                if np.any(cross_talk_matrix > 0)
                else 0.0
            )

            # Overall separation quality (higher is better)
            metrics.overall_separation_quality = (
                metrics.mean_separation_score - metrics.max_cross_talk
            )

            # Determine separation success
            metrics.separation_success = (
                metrics.mean_separation_score >= self.separation_threshold
                and metrics.max_cross_talk <= (1.0 - self.separation_threshold)
            )

            # Add diagnostic information
            metrics.diagnostic_info = {
                "num_signals": n_signals,
                "separation_threshold": self.separation_threshold,
                "mean_sir_db": (
                    10 * np.log10(np.mean(sir_values)) if np.mean(sir_values) > 0 else -np.inf
                ),
                "min_sir_db": (
                    10 * np.log10(np.min(sir_values)) if np.min(sir_values) > 0 else -np.inf
                ),
                "combined_signal_power": float(
                    np.mean(np.abs(self.gpu_backend.to_cpu(combined_signal)) ** 2)
                ),
                "separation_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error computing separation quality metrics: {e}")
            metrics.diagnostic_info["computation_error"] = str(e)
            metrics.separation_success = False

        return metrics

    def separate_signal_set(
        self, combined_signal: Union[np.ndarray, "cp.ndarray"], reference_signal_set: SignalSet
    ) -> Tuple[SignalSet, SeparationQualityMetrics]:
        """Separate a combined signal using a reference SignalSet.

        Args:
            combined_signal: Combined signal to separate
            reference_signal_set: Reference SignalSet with original signals and phases

        Returns:
            Tuple of (separated_SignalSet, quality_metrics)
        """
        # Set reference signals from SignalSet
        self.set_reference_signals(reference_signal_set.signals, reference_signal_set.phases)

        # Perform separation
        separated_signals, quality_metrics = self.separate_combined_signal(combined_signal)

        # Create new SignalSet with separated signals
        # Ensure orthogonality score is in valid range [0, 1]
        orthogonality_score = max(0.0, min(1.0, quality_metrics.overall_separation_quality))

        separated_signal_set = SignalSet(
            signals=[self.gpu_backend.to_cpu(sig) for sig in separated_signals],
            phases=reference_signal_set.phases.copy(),
            orthogonality_score=orthogonality_score,
            generation_timestamp=datetime.now(),
            config=reference_signal_set.config,
            metadata={
                "separation_quality": quality_metrics.overall_separation_quality,
                "separation_success": quality_metrics.separation_success,
                "original_orthogonality_score": reference_signal_set.orthogonality_score,
                "separation_method": "phase_based_correlation",
            },
        )

        return separated_signal_set, quality_metrics

    def validate_separation_capability(
        self, reference_signals: List[Union[np.ndarray, "cp.ndarray"]], reference_phases: np.ndarray
    ) -> Dict[str, any]:
        """Validate that separation is feasible with given reference signals.

        Args:
            reference_signals: Reference signals for separation
            reference_phases: Reference phase matrix

        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            "separation_feasible": False,
            "orthogonality_quality": 0.0,
            "recommendations": [],
            "warnings": [],
        }

        try:
            # Test orthogonality of reference signals
            orthogonality_result = self.orthogonality_tester.test_signal_set_orthogonality(
                reference_signals
            )

            validation_results["orthogonality_quality"] = orthogonality_result[
                "overall_orthogonality_score"
            ]
            validation_results["orthogonality_ratio"] = orthogonality_result["orthogonality_ratio"]

            # Check if signals are sufficiently orthogonal for separation
            if orthogonality_result["is_set_orthogonal"]:
                validation_results["separation_feasible"] = True
            else:
                validation_results["warnings"].append(
                    f"Signal set orthogonality ratio {orthogonality_result['orthogonality_ratio']:.2%} "
                    f"is below recommended threshold for reliable separation"
                )

            # Check signal energy balance
            signal_energies = []
            for signal in reference_signals:
                signal_cpu = self.gpu_backend.to_cpu(signal)
                energy = float(np.sum(np.abs(signal_cpu) ** 2))
                signal_energies.append(energy)

            energy_ratio = (
                np.max(signal_energies) / np.min(signal_energies)
                if np.min(signal_energies) > 0
                else float("inf")
            )
            validation_results["energy_ratio"] = energy_ratio

            if energy_ratio > 10.0:  # 10 dB difference
                validation_results["warnings"].append(
                    f"Large energy imbalance detected (ratio: {energy_ratio:.1f}). "
                    f"Consider normalizing signal amplitudes."
                )

            # Provide recommendations
            if validation_results["separation_feasible"]:
                validation_results["recommendations"].append(
                    "Separation should be successful with current signals"
                )
            else:
                validation_results["recommendations"].append(
                    "Improve signal orthogonality before attempting separation"
                )
                validation_results["recommendations"].append(
                    "Consider phase optimization to reduce cross-correlation"
                )

        except Exception as e:
            validation_results["error"] = str(e)
            validation_results["warnings"].append(f"Validation failed: {e}")

        return validation_results

    def generate_separation_report(self, quality_metrics: SeparationQualityMetrics) -> str:
        """Generate a comprehensive separation quality report.

        Requirements:
        - 6.3: Report separation quality metrics
        - 6.4: Provide diagnostic information

        Args:
            quality_metrics: Separation quality metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("SIGNAL SEPARATION QUALITY REPORT")
        report.append("=" * 60)

        # Overall results
        report.append(
            f"Separation Success: {'YES' if quality_metrics.separation_success else 'NO'}"
        )
        report.append(f"Overall Quality Score: {quality_metrics.overall_separation_quality:.6f}")
        report.append("")

        # Separation scores
        report.append("SEPARATION SCORES (correlation with original signals):")
        for i, score in enumerate(quality_metrics.separation_scores):
            report.append(f"  Signal {i+1}: {score:.6f}")
        report.append(f"  Mean: {quality_metrics.mean_separation_score:.6f}")
        report.append(f"  Minimum: {quality_metrics.min_separation_score:.6f}")
        report.append("")

        # Cross-talk analysis
        report.append("CROSS-TALK ANALYSIS:")
        report.append(f"  Maximum cross-talk: {quality_metrics.max_cross_talk:.6f}")
        report.append(f"  Mean cross-talk: {quality_metrics.mean_cross_talk:.6f}")
        report.append("")

        # Signal-to-Interference Ratios
        if quality_metrics.signal_to_interference_ratios:
            report.append("SIGNAL-TO-INTERFERENCE RATIOS:")
            for i, sir in enumerate(quality_metrics.signal_to_interference_ratios):
                sir_db = 10 * np.log10(sir) if sir > 0 else -np.inf
                report.append(f"  Signal {i+1}: {sir_db:.2f} dB")
            report.append("")

        # Diagnostic information
        if quality_metrics.diagnostic_info:
            report.append("DIAGNOSTIC INFORMATION:")
            for key, value in quality_metrics.diagnostic_info.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.6f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Backend information
        device_info = self.gpu_backend.device_info
        report.append("BACKEND INFORMATION:")
        report.append(f"  Compute backend: {device_info['backend']}")
        if device_info["backend"] == "GPU":
            report.append(f"  Device: {device_info['device_name']}")

        report.append("=" * 60)

        return "\n".join(report)

    def set_separation_threshold(self, threshold: float) -> None:
        """Set the separation quality threshold.

        Args:
            threshold: New separation threshold (0 < threshold < 1)

        Raises:
            ValueError: If threshold is out of valid range
        """
        if not 0 < threshold < 1:
            raise ValueError("Separation threshold must be between 0 and 1")
        self.separation_threshold = threshold
        logger.info(f"Separation threshold set to {threshold}")

    def get_separation_threshold(self) -> float:
        """Get the current separation threshold."""
        return self.separation_threshold

    def cleanup_resources(self) -> None:
        """Clean up GPU resources and memory."""
        self.gpu_backend.cleanup_memory()
        self.correlation_analyzer.clear_cache()
        logger.debug("SignalSeparator resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()
