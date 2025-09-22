"""
Real-time performance validation for OFDM signal generation.

This module provides real-time performance validation to ensure the system
can generate signals fast enough for practical real-time applications.

Requirements addressed:
- 4.2: Real-time GPU acceleration validation
- 3.3: Real-time processing capability verification
"""

import logging
import queue
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .gpu_backend import GPUBackend
from .models import OFDMConfig
from .ofdm_generator import OFDMGenerator
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class RealTimeMetrics:
    """Container for real-time performance metrics."""

    timestamp: datetime
    signal_duration: float  # Duration of generated signal in seconds
    generation_time: float  # Time taken to generate signal in seconds
    real_time_factor: float  # signal_duration / generation_time
    meets_real_time: bool  # True if generation_time <= signal_duration
    latency_ms: float  # Generation latency in milliseconds
    throughput_signals_per_sec: float  # Signals generated per second
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class RealTimeRequirements:
    """Real-time performance requirements."""

    max_latency_ms: float = 10.0  # Maximum acceptable latency
    min_real_time_factor: float = 1.0  # Minimum real-time factor (1.0 = exactly real-time)
    target_real_time_factor: float = 2.0  # Target real-time factor (2.0 = 2x faster than real-time)
    min_throughput_signals_per_sec: float = 100.0  # Minimum throughput requirement
    max_jitter_ms: float = 5.0  # Maximum acceptable timing jitter
    sustained_duration_sec: float = 10.0  # Duration for sustained performance test


class RealTimeValidator:
    """Real-time performance validator for OFDM signal generation.

    This class validates that the OFDM signal generation system can meet
    real-time performance requirements for practical applications.
    """

    def __init__(
        self,
        ofdm_config: OFDMConfig,
        gpu_backend: Optional[GPUBackend] = None,
        requirements: Optional[RealTimeRequirements] = None,
    ):
        """Initialize real-time validator.

        Args:
            ofdm_config: OFDM configuration for testing
            gpu_backend: GPU backend instance (creates new if None)
            requirements: Real-time performance requirements
        """
        self.ofdm_config = ofdm_config
        self.gpu_backend = gpu_backend or GPUBackend()
        self.requirements = requirements or RealTimeRequirements()

        # Initialize components
        self.ofdm_generator = OFDMGenerator(ofdm_config, self.gpu_backend)
        self.performance_optimizer = PerformanceOptimizer(self.gpu_backend)

        # Metrics storage
        self.metrics_history: List[RealTimeMetrics] = []
        self._validation_active = False
        self._validation_thread = None

        logger.info(
            f"RealTimeValidator initialized: "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}, "
            f"signal_duration={ofdm_config.signal_duration}s, "
            f"target_rt_factor={self.requirements.target_real_time_factor}"
        )

    def validate_single_signal_generation(
        self, phase_array: Optional[np.ndarray] = None
    ) -> RealTimeMetrics:
        """Validate real-time performance for single signal generation.

        Args:
            phase_array: Phase array for signal generation (random if None)

        Returns:
            Real-time performance metrics
        """
        if phase_array is None:
            phase_array = np.random.uniform(0, 2 * np.pi, self.ofdm_config.num_subcarriers)

        # Measure generation time
        start_time = time.time()
        signal = self.ofdm_generator.generate_single_signal(phase_array)
        generation_time = time.time() - start_time

        # Calculate metrics
        signal_duration = self.ofdm_config.signal_duration
        real_time_factor = (
            signal_duration / generation_time if generation_time > 0 else float("inf")
        )
        meets_real_time = generation_time <= signal_duration
        latency_ms = generation_time * 1000
        throughput = 1.0 / generation_time if generation_time > 0 else float("inf")

        # Create metrics object
        metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            signal_duration=signal_duration,
            generation_time=generation_time,
            real_time_factor=real_time_factor,
            meets_real_time=meets_real_time,
            latency_ms=latency_ms,
            throughput_signals_per_sec=throughput,
        )

        self.metrics_history.append(metrics)

        logger.debug(
            f"Single signal validation: {generation_time:.4f}s, "
            f"RT factor: {real_time_factor:.2f}, "
            f"meets RT: {meets_real_time}"
        )

        return metrics

    def validate_sustained_performance(
        self, duration_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate sustained real-time performance over extended period.

        Args:
            duration_sec: Test duration in seconds (uses requirement default if None)

        Returns:
            Dictionary with sustained performance analysis
        """
        test_duration = duration_sec or self.requirements.sustained_duration_sec

        logger.info(f"Starting sustained performance validation for {test_duration}s")

        start_test = time.time()
        test_metrics = []
        signals_generated = 0

        while (time.time() - start_test) < test_duration:
            # Generate random phase array
            phase_array = np.random.uniform(0, 2 * np.pi, self.ofdm_config.num_subcarriers)

            # Validate single signal
            metrics = self.validate_single_signal_generation(phase_array)
            test_metrics.append(metrics)
            signals_generated += 1

            # Brief pause to simulate realistic usage pattern
            time.sleep(0.001)  # 1ms pause

        actual_test_duration = time.time() - start_test

        # Analyze sustained performance
        generation_times = [m.generation_time for m in test_metrics]
        real_time_factors = [m.real_time_factor for m in test_metrics]
        latencies = [m.latency_ms for m in test_metrics]

        # Calculate statistics
        avg_generation_time = statistics.mean(generation_times)
        max_generation_time = max(generation_times)
        min_generation_time = min(generation_times)
        std_generation_time = statistics.stdev(generation_times) if len(generation_times) > 1 else 0

        avg_real_time_factor = statistics.mean(real_time_factors)
        min_real_time_factor = min(real_time_factors)

        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0

        overall_throughput = signals_generated / actual_test_duration

        # Check requirements compliance
        meets_latency_req = max_latency <= self.requirements.max_latency_ms
        meets_rt_factor_req = min_real_time_factor >= self.requirements.min_real_time_factor
        meets_throughput_req = (
            overall_throughput >= self.requirements.min_throughput_signals_per_sec
        )
        meets_jitter_req = jitter <= self.requirements.max_jitter_ms

        all_requirements_met = all(
            [meets_latency_req, meets_rt_factor_req, meets_throughput_req, meets_jitter_req]
        )

        # Calculate percentage of signals meeting real-time
        real_time_success_rate = sum(1 for m in test_metrics if m.meets_real_time) / len(
            test_metrics
        )

        analysis = {
            "test_duration_sec": actual_test_duration,
            "signals_generated": signals_generated,
            "overall_throughput": overall_throughput,
            "real_time_success_rate": real_time_success_rate,
            "generation_time_stats": {
                "average": avg_generation_time,
                "minimum": min_generation_time,
                "maximum": max_generation_time,
                "std_deviation": std_generation_time,
            },
            "real_time_factor_stats": {
                "average": avg_real_time_factor,
                "minimum": min_real_time_factor,
                "target": self.requirements.target_real_time_factor,
                "meets_target": avg_real_time_factor >= self.requirements.target_real_time_factor,
            },
            "latency_stats": {
                "average_ms": avg_latency,
                "maximum_ms": max_latency,
                "jitter_ms": jitter,
                "requirement_ms": self.requirements.max_latency_ms,
            },
            "requirements_compliance": {
                "meets_latency": meets_latency_req,
                "meets_real_time_factor": meets_rt_factor_req,
                "meets_throughput": meets_throughput_req,
                "meets_jitter": meets_jitter_req,
                "all_requirements_met": all_requirements_met,
            },
            "backend_info": {
                "type": "GPU" if self.gpu_backend.is_gpu_available else "CPU",
                "device_info": self.gpu_backend.device_info,
            },
        }

        logger.info(
            f"Sustained validation complete: {signals_generated} signals in {actual_test_duration:.2f}s, "
            f"throughput: {overall_throughput:.1f} signals/s, "
            f"RT success: {real_time_success_rate:.1%}, "
            f"all reqs met: {all_requirements_met}"
        )

        return analysis

    def validate_batch_processing(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Validate real-time performance for batch processing.

        Args:
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary with batch processing analysis
        """
        logger.info(f"Validating batch processing for sizes: {batch_sizes}")

        batch_results = {}

        for batch_size in batch_sizes:
            logger.debug(f"Testing batch size: {batch_size}")

            # Generate phase arrays for batch
            phase_arrays = [
                np.random.uniform(0, 2 * np.pi, self.ofdm_config.num_subcarriers)
                for _ in range(batch_size)
            ]

            # Use performance optimizer for batch processing
            def generate_signal(phase_array):
                return self.ofdm_generator.generate_single_signal(phase_array)

            start_time = time.time()
            signals = self.performance_optimizer.optimize_signal_generation(
                generate_signal, phase_arrays, f"batch_validation_{batch_size}"
            )
            batch_generation_time = time.time() - start_time

            # Calculate batch metrics
            avg_time_per_signal = batch_generation_time / batch_size
            batch_throughput = batch_size / batch_generation_time

            # Compare to single signal performance
            single_signal_time = self.validate_single_signal_generation().generation_time
            batch_efficiency = (
                single_signal_time / avg_time_per_signal if avg_time_per_signal > 0 else 1.0
            )

            # Real-time analysis for batch
            total_signal_duration = batch_size * self.ofdm_config.signal_duration
            batch_real_time_factor = (
                total_signal_duration / batch_generation_time
                if batch_generation_time > 0
                else float("inf")
            )
            meets_batch_real_time = batch_generation_time <= total_signal_duration

            batch_results[batch_size] = {
                "batch_generation_time": batch_generation_time,
                "avg_time_per_signal": avg_time_per_signal,
                "batch_throughput": batch_throughput,
                "batch_efficiency": batch_efficiency,
                "batch_real_time_factor": batch_real_time_factor,
                "meets_batch_real_time": meets_batch_real_time,
                "signals_generated": len(signals),
            }

        # Find optimal batch size
        optimal_batch_size = max(
            batch_results.keys(), key=lambda bs: batch_results[bs]["batch_efficiency"]
        )

        analysis = {
            "batch_results": batch_results,
            "optimal_batch_size": optimal_batch_size,
            "optimal_efficiency": batch_results[optimal_batch_size]["batch_efficiency"],
            "optimal_throughput": batch_results[optimal_batch_size]["batch_throughput"],
        }

        logger.info(
            f"Batch validation complete. Optimal batch size: {optimal_batch_size}, "
            f"efficiency: {batch_results[optimal_batch_size]['batch_efficiency']:.2f}x"
        )

        return analysis

    def validate_memory_constrained_performance(self) -> Dict[str, Any]:
        """Validate performance under memory constraints.

        Returns:
            Dictionary with memory-constrained performance analysis
        """
        logger.info("Validating performance under memory constraints")

        # Test with increasing memory pressure
        memory_test_results = []

        # Generate increasingly large batches to create memory pressure
        batch_sizes = [1, 5, 10, 20, 50, 100]

        for batch_size in batch_sizes:
            try:
                # Check memory before test
                memory_before = self.performance_optimizer.memory_manager.get_memory_statistics()

                # Generate batch
                phase_arrays = [
                    np.random.uniform(0, 2 * np.pi, self.ofdm_config.num_subcarriers)
                    for _ in range(batch_size)
                ]

                def generate_signal(phase_array):
                    return self.ofdm_generator.generate_single_signal(phase_array)

                start_time = time.time()
                signals = self.performance_optimizer.optimize_signal_generation(
                    generate_signal, phase_arrays, f"memory_test_{batch_size}"
                )
                generation_time = time.time() - start_time

                # Check memory after test
                memory_after = self.performance_optimizer.memory_manager.get_memory_statistics()

                # Calculate performance metrics
                avg_time_per_signal = generation_time / batch_size
                throughput = batch_size / generation_time

                # Check for memory pressure
                pressure = self.performance_optimizer.memory_manager.check_memory_pressure()

                memory_test_results.append(
                    {
                        "batch_size": batch_size,
                        "generation_time": generation_time,
                        "avg_time_per_signal": avg_time_per_signal,
                        "throughput": throughput,
                        "memory_pressure": pressure["under_pressure"],
                        "memory_recommendation": pressure["recommendation"],
                        "memory_before": memory_before,
                        "memory_after": memory_after,
                        "success": True,
                    }
                )

                logger.debug(
                    f"Memory test batch {batch_size}: {generation_time:.4f}s, "
                    f"pressure: {pressure['under_pressure']}"
                )

            except Exception as e:
                memory_test_results.append(
                    {
                        "batch_size": batch_size,
                        "generation_time": 0,
                        "avg_time_per_signal": 0,
                        "throughput": 0,
                        "memory_pressure": True,
                        "memory_recommendation": f"Failed: {str(e)}",
                        "success": False,
                        "error": str(e),
                    }
                )

                logger.warning(f"Memory test failed for batch size {batch_size}: {e}")

        # Analyze results
        successful_tests = [r for r in memory_test_results if r["success"]]

        if successful_tests:
            max_successful_batch = max(r["batch_size"] for r in successful_tests)
            performance_degradation = []

            # Calculate performance degradation with increasing memory pressure
            baseline_throughput = successful_tests[0]["throughput"] if successful_tests else 0

            for result in successful_tests:
                if baseline_throughput > 0:
                    degradation = (baseline_throughput - result["throughput"]) / baseline_throughput
                    performance_degradation.append(degradation)

            avg_degradation = (
                statistics.mean(performance_degradation) if performance_degradation else 0
            )
        else:
            max_successful_batch = 0
            avg_degradation = 1.0  # 100% degradation if all failed

        analysis = {
            "memory_test_results": memory_test_results,
            "max_successful_batch_size": max_successful_batch,
            "average_performance_degradation": avg_degradation,
            "memory_scalability_score": (
                max_successful_batch / max(batch_sizes) if batch_sizes else 0
            ),
            "backend_type": "GPU" if self.gpu_backend.is_gpu_available else "CPU",
        }

        logger.info(
            f"Memory constraint validation complete. Max batch: {max_successful_batch}, "
            f"avg degradation: {avg_degradation:.1%}"
        )

        return analysis

    def generate_real_time_report(self) -> Dict[str, Any]:
        """Generate comprehensive real-time performance report.

        Returns:
            Dictionary with complete real-time performance analysis
        """
        logger.info("Generating comprehensive real-time performance report")

        # Run all validation tests
        single_signal_metrics = self.validate_single_signal_generation()
        sustained_analysis = self.validate_sustained_performance()
        batch_analysis = self.validate_batch_processing([1, 5, 10, 20])
        memory_analysis = self.validate_memory_constrained_performance()

        # Overall assessment
        overall_real_time_capable = all(
            [
                single_signal_metrics.meets_real_time,
                sustained_analysis["requirements_compliance"]["all_requirements_met"],
                batch_analysis["batch_results"][1][
                    "meets_batch_real_time"
                ],  # At least single signal batch
            ]
        )

        # Performance recommendations
        recommendations = []

        if not single_signal_metrics.meets_real_time:
            recommendations.append("Single signal generation does not meet real-time requirements")

        if single_signal_metrics.real_time_factor < self.requirements.target_real_time_factor:
            recommendations.append(
                f"Real-time factor ({single_signal_metrics.real_time_factor:.2f}) below target ({self.requirements.target_real_time_factor})"
            )

        if sustained_analysis["real_time_success_rate"] < 0.95:
            recommendations.append("Sustained performance has inconsistent real-time capability")

        if memory_analysis["average_performance_degradation"] > 0.2:
            recommendations.append("Significant performance degradation under memory pressure")

        if not recommendations:
            recommendations.append("Real-time performance meets all requirements")

        report = {
            "timestamp": datetime.now().isoformat(),
            "ofdm_config": {
                "num_subcarriers": self.ofdm_config.num_subcarriers,
                "signal_duration": self.ofdm_config.signal_duration,
                "sampling_rate": self.ofdm_config.sampling_rate,
            },
            "requirements": {
                "max_latency_ms": self.requirements.max_latency_ms,
                "min_real_time_factor": self.requirements.min_real_time_factor,
                "target_real_time_factor": self.requirements.target_real_time_factor,
                "min_throughput": self.requirements.min_throughput_signals_per_sec,
            },
            "backend_info": {
                "type": "GPU" if self.gpu_backend.is_gpu_available else "CPU",
                "device_info": self.gpu_backend.device_info,
            },
            "single_signal_performance": {
                "generation_time": single_signal_metrics.generation_time,
                "real_time_factor": single_signal_metrics.real_time_factor,
                "meets_real_time": single_signal_metrics.meets_real_time,
                "latency_ms": single_signal_metrics.latency_ms,
                "throughput": single_signal_metrics.throughput_signals_per_sec,
            },
            "sustained_performance": sustained_analysis,
            "batch_performance": batch_analysis,
            "memory_constrained_performance": memory_analysis,
            "overall_assessment": {
                "real_time_capable": overall_real_time_capable,
                "performance_grade": self._calculate_performance_grade(
                    single_signal_metrics, sustained_analysis, batch_analysis, memory_analysis
                ),
                "recommendations": recommendations,
            },
        }

        logger.info(
            f"Real-time report generated. Overall capable: {overall_real_time_capable}, "
            f"grade: {report['overall_assessment']['performance_grade']}"
        )

        return report

    def _calculate_performance_grade(
        self,
        single_metrics: RealTimeMetrics,
        sustained_analysis: Dict,
        batch_analysis: Dict,
        memory_analysis: Dict,
    ) -> str:
        """Calculate overall performance grade.

        Args:
            single_metrics: Single signal performance metrics
            sustained_analysis: Sustained performance analysis
            batch_analysis: Batch performance analysis
            memory_analysis: Memory-constrained performance analysis

        Returns:
            Performance grade (A, B, C, D, F)
        """
        score = 0

        # Single signal performance (25 points)
        if single_metrics.meets_real_time:
            score += 15
            if single_metrics.real_time_factor >= self.requirements.target_real_time_factor:
                score += 10

        # Sustained performance (25 points)
        if sustained_analysis["requirements_compliance"]["all_requirements_met"]:
            score += 15
            if sustained_analysis["real_time_success_rate"] >= 0.99:
                score += 10

        # Batch performance (25 points)
        optimal_efficiency = batch_analysis["optimal_efficiency"]
        if optimal_efficiency >= 1.0:
            score += 15
            if optimal_efficiency >= 1.5:
                score += 10

        # Memory performance (25 points)
        memory_scalability = memory_analysis["memory_scalability_score"]
        if memory_scalability >= 0.5:
            score += 15
            if memory_scalability >= 0.8:
                score += 10

        # Convert score to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def cleanup_resources(self):
        """Clean up validator resources."""
        self.ofdm_generator.cleanup_resources()
        self.performance_optimizer.cleanup_resources()
        logger.debug("RealTimeValidator resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()
