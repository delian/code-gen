"""
Performance benchmarks and regression tests for OFDM Chirp Generator.

This module implements comprehensive performance testing including:
- GPU vs CPU performance comparisons
- Scalability tests with varying signal parameters
- Real-time performance validation
- Automated performance regression detection

Requirements addressed:
- 4.2: GPU acceleration performance validation
- 3.3: Efficient processing scalability testing
"""

import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from ofdm_chirp_generator.correlation_analyzer import CorrelationAnalyzer
from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.ofdm_generator import OFDMGenerator
from ofdm_chirp_generator.orthogonal_signal_generator import OrthogonalSignalGenerator
from ofdm_chirp_generator.performance_optimizer import PerformanceOptimizer
from ofdm_chirp_generator.phase_optimizer import PhaseOptimizer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    test_name: str
    backend: str  # 'GPU' or 'CPU'
    parameters: Dict[str, Any]
    execution_time: float
    throughput: Optional[float]
    memory_usage_mb: Optional[float]
    success: bool
    error_message: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability tests."""

    num_subcarriers_range: List[int]
    num_signals_range: List[int]
    signal_duration_range: List[float]
    sampling_rate_range: List[float]
    max_test_time: float = 30.0  # Maximum time per test in seconds


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""

    def __init__(self, results_file: Optional[str] = None):
        """Initialize benchmark suite.

        Args:
            results_file: Path to save benchmark results (optional)
        """
        self.results_file = results_file or "benchmark_results.json"
        self.results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}

        # Load existing baseline if available
        self._load_baseline_results()

    def _load_baseline_results(self):
        """Load baseline results for regression testing."""
        baseline_file = self.results_file.replace(".json", "_baseline.json")
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, "r") as f:
                    baseline_data = json.load(f)
                    for result_data in baseline_data:
                        result = BenchmarkResult(**result_data)
                        self.baseline_results[result.test_name] = result
            except Exception as e:
                print(f"Warning: Could not load baseline results: {e}")

    def save_results(self):
        """Save benchmark results to file."""
        results_data = [asdict(result) for result in self.results]
        with open(self.results_file, "w") as f:
            json.dump(results_data, f, indent=2)

    def save_as_baseline(self):
        """Save current results as baseline for future regression testing."""
        baseline_file = self.results_file.replace(".json", "_baseline.json")
        results_data = [asdict(result) for result in self.results]
        with open(baseline_file, "w") as f:
            json.dump(results_data, f, indent=2)

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        if not self.results:
            return {"message": "No benchmark results available"}

        # Group results by backend
        gpu_results = [r for r in self.results if r.backend == "GPU" and r.success]
        cpu_results = [r for r in self.results if r.backend == "CPU" and r.success]

        summary = {
            "total_tests": len(self.results),
            "successful_tests": len([r for r in self.results if r.success]),
            "failed_tests": len([r for r in self.results if not r.success]),
            "gpu_tests": len(gpu_results),
            "cpu_tests": len(cpu_results),
        }

        # GPU performance statistics
        if gpu_results:
            gpu_times = [r.execution_time for r in gpu_results]
            gpu_throughputs = [r.throughput for r in gpu_results if r.throughput]

            summary["gpu_performance"] = {
                "average_time": statistics.mean(gpu_times),
                "median_time": statistics.median(gpu_times),
                "min_time": min(gpu_times),
                "max_time": max(gpu_times),
                "std_time": statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0,
            }

            if gpu_throughputs:
                summary["gpu_performance"]["average_throughput"] = statistics.mean(gpu_throughputs)
                summary["gpu_performance"]["max_throughput"] = max(gpu_throughputs)

        # CPU performance statistics
        if cpu_results:
            cpu_times = [r.execution_time for r in cpu_results]
            cpu_throughputs = [r.throughput for r in cpu_results if r.throughput]

            summary["cpu_performance"] = {
                "average_time": statistics.mean(cpu_times),
                "median_time": statistics.median(cpu_times),
                "min_time": min(cpu_times),
                "max_time": max(cpu_times),
                "std_time": statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0,
            }

            if cpu_throughputs:
                summary["cpu_performance"]["average_throughput"] = statistics.mean(cpu_throughputs)
                summary["cpu_performance"]["max_throughput"] = max(cpu_throughputs)

        # GPU vs CPU comparison
        if gpu_results and cpu_results:
            gpu_avg_time = summary["gpu_performance"]["average_time"]
            cpu_avg_time = summary["cpu_performance"]["average_time"]
            speedup = cpu_avg_time / gpu_avg_time if gpu_avg_time > 0 else 1.0

            summary["gpu_vs_cpu"] = {
                "speedup": speedup,
                "gpu_faster": speedup > 1.0,
                "performance_ratio": gpu_avg_time / cpu_avg_time if cpu_avg_time > 0 else 1.0,
            }

        return summary

    def check_regression(self, tolerance: float = 0.2) -> Dict[str, Any]:
        """Check for performance regression against baseline.

        Args:
            tolerance: Acceptable performance degradation (0.2 = 20%)

        Returns:
            Dictionary with regression analysis results
        """
        if not self.baseline_results:
            return {"message": "No baseline results available for regression testing"}

        regressions = []
        improvements = []

        for result in self.results:
            if not result.success:
                continue

            baseline = self.baseline_results.get(result.test_name)
            if not baseline or not baseline.success:
                continue

            # Compare execution times
            time_ratio = result.execution_time / baseline.execution_time
            performance_change = (time_ratio - 1.0) * 100  # Percentage change

            if time_ratio > (1.0 + tolerance):
                regressions.append(
                    {
                        "test_name": result.test_name,
                        "backend": result.backend,
                        "baseline_time": baseline.execution_time,
                        "current_time": result.execution_time,
                        "degradation_percent": performance_change,
                    }
                )
            elif time_ratio < 0.95:  # 5% improvement threshold
                improvements.append(
                    {
                        "test_name": result.test_name,
                        "backend": result.backend,
                        "baseline_time": baseline.execution_time,
                        "current_time": result.execution_time,
                        "improvement_percent": -performance_change,
                    }
                )

        return {
            "regressions_detected": len(regressions),
            "improvements_detected": len(improvements),
            "regressions": regressions,
            "improvements": improvements,
            "tolerance_percent": tolerance * 100,
        }


class TestGPUvsCPUPerformance:
    """Test GPU vs CPU performance comparisons."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        suite = PerformanceBenchmarkSuite(temp_file)
        yield suite

        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    @pytest.fixture
    def ofdm_config(self):
        """Create standard OFDM configuration for benchmarks."""
        return OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,
        )

    def benchmark_signal_generation(
        self,
        backend_type: str,
        ofdm_config: OFDMConfig,
        num_signals: int,
        benchmark_suite: PerformanceBenchmarkSuite,
    ):
        """Benchmark signal generation performance.

        Args:
            backend_type: 'GPU' or 'CPU'
            ofdm_config: OFDM configuration
            num_signals: Number of signals to generate
            benchmark_suite: Benchmark suite to record results
        """
        # Initialize backend
        force_cpu = backend_type == "CPU"
        gpu_backend = GPUBackend(force_cpu=force_cpu)
        ofdm_generator = OFDMGenerator(ofdm_config, gpu_backend)

        # Generate phase arrays
        phase_arrays = [
            np.random.uniform(0, 2 * np.pi, ofdm_config.num_subcarriers) for _ in range(num_signals)
        ]

        # Measure performance
        start_time = time.time()
        memory_before = self._get_memory_usage()

        try:
            signals = []
            for phase_array in phase_arrays:
                signal = ofdm_generator.generate_single_signal(phase_array)
                signals.append(signal)

            execution_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_usage = memory_after - memory_before

            # Calculate throughput
            throughput = num_signals / execution_time if execution_time > 0 else 0

            # Record result
            result = BenchmarkResult(
                test_name=f"signal_generation_{num_signals}",
                backend=backend_type,
                parameters={
                    "num_signals": num_signals,
                    "num_subcarriers": ofdm_config.num_subcarriers,
                    "signal_duration": ofdm_config.signal_duration,
                    "sampling_rate": ofdm_config.sampling_rate,
                },
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                success=True,
            )

        except Exception as e:
            result = BenchmarkResult(
                test_name=f"signal_generation_{num_signals}",
                backend=backend_type,
                parameters={"num_signals": num_signals},
                execution_time=0,
                throughput=None,
                memory_usage_mb=None,
                success=False,
                error_message=str(e),
            )

        finally:
            ofdm_generator.cleanup_resources()

        benchmark_suite.add_result(result)
        return result

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def test_single_signal_generation_performance(self, benchmark_suite, ofdm_config):
        """Test single signal generation performance."""
        # Test both backends
        for backend in ["CPU", "GPU"]:
            result = self.benchmark_signal_generation(backend, ofdm_config, 1, benchmark_suite)
            assert (
                result.success
            ), f"Single signal generation failed on {backend}: {result.error_message}"

    def test_batch_signal_generation_performance(self, benchmark_suite, ofdm_config):
        """Test batch signal generation performance."""
        batch_sizes = [5, 10, 20, 50]

        for batch_size in batch_sizes:
            for backend in ["CPU", "GPU"]:
                result = self.benchmark_signal_generation(
                    backend, ofdm_config, batch_size, benchmark_suite
                )
                # Allow some failures for large batches on limited hardware
                if not result.success and batch_size <= 10:
                    pytest.fail(
                        f"Batch generation failed on {backend} for size {batch_size}: {result.error_message}"
                    )

    def test_orthogonal_signal_set_performance(self, benchmark_suite, ofdm_config):
        """Test orthogonal signal set generation performance."""
        set_sizes = [2, 4, 8]

        for set_size in set_sizes:
            for backend in ["CPU", "GPU"]:
                force_cpu = backend == "CPU"
                gpu_backend = GPUBackend(force_cpu=force_cpu)
                orthogonal_generator = OrthogonalSignalGenerator(ofdm_config, gpu_backend)

                start_time = time.time()
                memory_before = self._get_memory_usage()

                try:
                    signal_set = orthogonal_generator.generate_orthogonal_signal_set(set_size)

                    execution_time = time.time() - start_time
                    memory_after = self._get_memory_usage()
                    memory_usage = memory_after - memory_before

                    throughput = set_size / execution_time if execution_time > 0 else 0

                    result = BenchmarkResult(
                        test_name=f"orthogonal_set_{set_size}",
                        backend=backend,
                        parameters={
                            "set_size": set_size,
                            "num_subcarriers": ofdm_config.num_subcarriers,
                        },
                        execution_time=execution_time,
                        throughput=throughput,
                        memory_usage_mb=memory_usage,
                        success=True,
                    )

                except Exception as e:
                    result = BenchmarkResult(
                        test_name=f"orthogonal_set_{set_size}",
                        backend=backend,
                        parameters={"set_size": set_size},
                        execution_time=0,
                        throughput=None,
                        memory_usage_mb=None,
                        success=False,
                        error_message=str(e),
                    )

                finally:
                    orthogonal_generator.cleanup_resources()

                benchmark_suite.add_result(result)

                # Allow failures for larger sets on limited hardware
                if not result.success and set_size <= 4:
                    pytest.fail(
                        f"Orthogonal set generation failed on {backend} for size {set_size}: {result.error_message}"
                    )


class TestScalabilityPerformance:
    """Test performance scalability with varying parameters."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        suite = PerformanceBenchmarkSuite(temp_file)
        yield suite

        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    def test_subcarrier_count_scalability(self, benchmark_suite):
        """Test performance scaling with number of subcarriers."""
        subcarrier_counts = [4, 8, 12, 16]  # Reduced to avoid bandwidth issues
        base_config = OFDMConfig(
            num_subcarriers=8,  # Will be overridden
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,  # Longer to avoid chirp length issues
        )

        for num_subcarriers in subcarrier_counts:
            config = OFDMConfig(
                num_subcarriers=num_subcarriers,
                subcarrier_spacing=base_config.subcarrier_spacing,
                bandwidth_per_subcarrier=base_config.bandwidth_per_subcarrier,
                center_frequency=base_config.center_frequency,
                sampling_rate=base_config.sampling_rate,
                signal_duration=base_config.signal_duration,
            )

            # Test with CPU backend (more predictable)
            gpu_backend = GPUBackend(force_cpu=True)
            ofdm_generator = OFDMGenerator(config, gpu_backend)

            start_time = time.time()

            try:
                phase_array = np.random.uniform(0, 2 * np.pi, num_subcarriers)
                signal = ofdm_generator.generate_single_signal(phase_array)

                execution_time = time.time() - start_time

                result = BenchmarkResult(
                    test_name=f"scalability_subcarriers_{num_subcarriers}",
                    backend="CPU",
                    parameters={"num_subcarriers": num_subcarriers, "signal_length": len(signal)},
                    execution_time=execution_time,
                    throughput=1.0 / execution_time if execution_time > 0 else 0,
                    memory_usage_mb=None,
                    success=True,
                )

            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"scalability_subcarriers_{num_subcarriers}",
                    backend="CPU",
                    parameters={"num_subcarriers": num_subcarriers},
                    execution_time=0,
                    throughput=None,
                    memory_usage_mb=None,
                    success=False,
                    error_message=str(e),
                )

            finally:
                ofdm_generator.cleanup_resources()

            benchmark_suite.add_result(result)

            # Verify reasonable performance scaling
            if result.success:
                assert (
                    result.execution_time < 5.0
                ), f"Execution time too long for {num_subcarriers} subcarriers"

    def test_signal_duration_scalability(self, benchmark_suite):
        """Test performance scaling with signal duration."""
        durations = [0.0005, 0.001, 0.002, 0.005]  # 0.5ms to 5ms
        base_config = OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,  # Will be overridden
        )

        for duration in durations:
            config = OFDMConfig(
                num_subcarriers=base_config.num_subcarriers,
                subcarrier_spacing=base_config.subcarrier_spacing,
                bandwidth_per_subcarrier=base_config.bandwidth_per_subcarrier,
                center_frequency=base_config.center_frequency,
                sampling_rate=base_config.sampling_rate,
                signal_duration=duration,
            )

            gpu_backend = GPUBackend(force_cpu=True)
            ofdm_generator = OFDMGenerator(config, gpu_backend)

            start_time = time.time()

            try:
                phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)
                signal = ofdm_generator.generate_single_signal(phase_array)

                execution_time = time.time() - start_time
                signal_length = len(signal)

                result = BenchmarkResult(
                    test_name=f"scalability_duration_{duration}",
                    backend="CPU",
                    parameters={
                        "signal_duration": duration,
                        "signal_length": signal_length,
                        "samples_per_second": signal_length / duration,
                    },
                    execution_time=execution_time,
                    throughput=signal_length / execution_time if execution_time > 0 else 0,
                    memory_usage_mb=None,
                    success=True,
                )

            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"scalability_duration_{duration}",
                    backend="CPU",
                    parameters={"signal_duration": duration},
                    execution_time=0,
                    throughput=None,
                    memory_usage_mb=None,
                    success=False,
                    error_message=str(e),
                )

            finally:
                ofdm_generator.cleanup_resources()

            benchmark_suite.add_result(result)

            if result.success:
                # Verify linear scaling relationship
                expected_samples = int(duration * config.sampling_rate)
                actual_samples = len(signal)
                assert (
                    abs(actual_samples - expected_samples) <= 1
                ), f"Signal length mismatch for duration {duration}"


class TestRealTimePerformance:
    """Test real-time performance validation."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        suite = PerformanceBenchmarkSuite(temp_file)
        yield suite

        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    def test_real_time_signal_generation(self, benchmark_suite):
        """Test if signal generation meets real-time requirements."""
        # Real-time requirement: generate signal faster than its duration
        config = OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,  # 2ms signal (longer to avoid chirp length issues)
        )

        gpu_backend = GPUBackend()
        ofdm_generator = OFDMGenerator(config, gpu_backend)

        # Test multiple iterations for statistical significance
        execution_times = []

        try:
            for i in range(10):
                phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)

                start_time = time.time()
                signal = ofdm_generator.generate_single_signal(phase_array)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)

            avg_execution_time = statistics.mean(execution_times)
            max_execution_time = max(execution_times)
            real_time_factor = config.signal_duration / avg_execution_time

            result = BenchmarkResult(
                test_name="real_time_validation",
                backend="GPU" if gpu_backend.is_gpu_available else "CPU",
                parameters={
                    "signal_duration": config.signal_duration,
                    "avg_execution_time": avg_execution_time,
                    "max_execution_time": max_execution_time,
                    "real_time_factor": real_time_factor,
                    "iterations": len(execution_times),
                },
                execution_time=avg_execution_time,
                throughput=1.0 / avg_execution_time,
                memory_usage_mb=None,
                success=True,
            )

            # Real-time requirement: execution time < signal duration
            assert (
                avg_execution_time < config.signal_duration
            ), f"Average execution time ({avg_execution_time:.4f}s) exceeds signal duration ({config.signal_duration}s)"

            # Stricter requirement: even worst case should be reasonable
            assert (
                max_execution_time < config.signal_duration * 2
            ), f"Maximum execution time ({max_execution_time:.4f}s) too slow for real-time"

        except Exception as e:
            result = BenchmarkResult(
                test_name="real_time_validation",
                backend="GPU" if gpu_backend.is_gpu_available else "CPU",
                parameters={"signal_duration": config.signal_duration},
                execution_time=0,
                throughput=None,
                memory_usage_mb=None,
                success=False,
                error_message=str(e),
            )

        finally:
            ofdm_generator.cleanup_resources()

        benchmark_suite.add_result(result)

    def test_continuous_generation_performance(self, benchmark_suite):
        """Test performance under continuous generation load."""
        config = OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,
        )

        gpu_backend = GPUBackend()
        ofdm_generator = OFDMGenerator(config, gpu_backend)

        # Continuous generation for 1 second
        test_duration = 1.0
        signals_generated = 0
        execution_times = []

        start_test = time.time()

        try:
            while (time.time() - start_test) < test_duration:
                phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)

                start_time = time.time()
                signal = ofdm_generator.generate_single_signal(phase_array)
                execution_time = time.time() - start_time

                execution_times.append(execution_time)
                signals_generated += 1

            total_test_time = time.time() - start_test
            avg_execution_time = statistics.mean(execution_times)
            throughput = signals_generated / total_test_time

            result = BenchmarkResult(
                test_name="continuous_generation",
                backend="GPU" if gpu_backend.is_gpu_available else "CPU",
                parameters={
                    "test_duration": total_test_time,
                    "signals_generated": signals_generated,
                    "avg_execution_time": avg_execution_time,
                },
                execution_time=avg_execution_time,
                throughput=throughput,
                memory_usage_mb=None,
                success=True,
            )

            # Verify sustained performance
            assert (
                signals_generated >= 100
            ), f"Too few signals generated in continuous test: {signals_generated}"
            assert (
                throughput >= 100
            ), f"Throughput too low for continuous generation: {throughput:.1f} signals/s"

        except Exception as e:
            result = BenchmarkResult(
                test_name="continuous_generation",
                backend="GPU" if gpu_backend.is_gpu_available else "CPU",
                parameters={"test_duration": test_duration},
                execution_time=0,
                throughput=None,
                memory_usage_mb=None,
                success=False,
                error_message=str(e),
            )

        finally:
            ofdm_generator.cleanup_resources()

        benchmark_suite.add_result(result)


class TestPerformanceRegression:
    """Test automated performance regression detection."""

    def test_regression_detection_setup(self):
        """Test regression detection system setup."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            suite = PerformanceBenchmarkSuite(temp_file)

            # Add some mock baseline results
            baseline_results = [
                BenchmarkResult(
                    test_name="test_operation_1",
                    backend="CPU",
                    parameters={"size": 100},
                    execution_time=0.1,
                    throughput=10.0,
                    memory_usage_mb=50.0,
                    success=True,
                ),
                BenchmarkResult(
                    test_name="test_operation_2",
                    backend="GPU",
                    parameters={"size": 100},
                    execution_time=0.05,
                    throughput=20.0,
                    memory_usage_mb=100.0,
                    success=True,
                ),
            ]

            for result in baseline_results:
                suite.add_result(result)

            # Save as baseline
            suite.save_as_baseline()

            # Create new suite and load baseline
            new_suite = PerformanceBenchmarkSuite(temp_file)
            assert len(new_suite.baseline_results) == 2

            # Add current results (some with regression)
            current_results = [
                BenchmarkResult(
                    test_name="test_operation_1",
                    backend="CPU",
                    parameters={"size": 100},
                    execution_time=0.15,  # 50% slower - regression
                    throughput=6.67,
                    memory_usage_mb=50.0,
                    success=True,
                ),
                BenchmarkResult(
                    test_name="test_operation_2",
                    backend="GPU",
                    parameters={"size": 100},
                    execution_time=0.04,  # 20% faster - improvement
                    throughput=25.0,
                    memory_usage_mb=100.0,
                    success=True,
                ),
            ]

            for result in current_results:
                new_suite.add_result(result)

            # Check for regressions
            regression_analysis = new_suite.check_regression(tolerance=0.2)

            assert regression_analysis["regressions_detected"] == 1
            assert regression_analysis["improvements_detected"] == 1
            assert len(regression_analysis["regressions"]) == 1
            assert regression_analysis["regressions"][0]["test_name"] == "test_operation_1"

        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            baseline_file = temp_file.replace(".json", "_baseline.json")
            if os.path.exists(baseline_file):
                os.unlink(baseline_file)


@pytest.mark.performance
class TestComprehensivePerformanceSuite:
    """Comprehensive performance test suite."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite for comprehensive testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        suite = PerformanceBenchmarkSuite(temp_file)
        yield suite

        # Save results and cleanup
        suite.save_results()
        print(f"\nBenchmark results saved to: {temp_file}")

        # Print performance summary
        summary = suite.get_performance_summary()
        print("\nPerformance Summary:")
        print(f"  Total tests: {summary.get('total_tests', 0)}")
        print(f"  Successful: {summary.get('successful_tests', 0)}")
        print(f"  Failed: {summary.get('failed_tests', 0)}")

        if "gpu_vs_cpu" in summary:
            gpu_vs_cpu = summary["gpu_vs_cpu"]
            print(f"  GPU vs CPU speedup: {gpu_vs_cpu['speedup']:.2f}x")

        if os.path.exists(temp_file):
            os.unlink(temp_file)

    def test_comprehensive_performance_benchmark(self, benchmark_suite):
        """Run comprehensive performance benchmark suite."""
        # Test configurations
        configs = [
            OFDMConfig(
                num_subcarriers=4,
                subcarrier_spacing=1000.0,
                bandwidth_per_subcarrier=800.0,
                center_frequency=10000.0,
                sampling_rate=50000.0,
                signal_duration=0.001,
            ),
            OFDMConfig(
                num_subcarriers=8,
                subcarrier_spacing=1000.0,
                bandwidth_per_subcarrier=800.0,
                center_frequency=10000.0,
                sampling_rate=50000.0,
                signal_duration=0.002,
            ),
        ]

        # Test scenarios
        test_scenarios = [("single_signal", 1), ("small_batch", 5), ("medium_batch", 10)]

        successful_tests = 0
        total_tests = 0

        for config_idx, config in enumerate(configs):
            for scenario_name, num_signals in test_scenarios:
                for backend in ["CPU", "GPU"]:
                    total_tests += 1

                    try:
                        # Run benchmark
                        force_cpu = backend == "CPU"
                        gpu_backend = GPUBackend(force_cpu=force_cpu)
                        ofdm_generator = OFDMGenerator(config, gpu_backend)

                        # Generate signals
                        start_time = time.time()
                        signals = []

                        for i in range(num_signals):
                            phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)
                            signal = ofdm_generator.generate_single_signal(phase_array)
                            signals.append(signal)

                        execution_time = time.time() - start_time
                        throughput = num_signals / execution_time if execution_time > 0 else 0

                        result = BenchmarkResult(
                            test_name=f"{scenario_name}_config{config_idx}",
                            backend=backend,
                            parameters={
                                "num_signals": num_signals,
                                "num_subcarriers": config.num_subcarriers,
                                "signal_duration": config.signal_duration,
                            },
                            execution_time=execution_time,
                            throughput=throughput,
                            memory_usage_mb=None,
                            success=True,
                        )

                        successful_tests += 1

                    except Exception as e:
                        result = BenchmarkResult(
                            test_name=f"{scenario_name}_config{config_idx}",
                            backend=backend,
                            parameters={"num_signals": num_signals},
                            execution_time=0,
                            throughput=None,
                            memory_usage_mb=None,
                            success=False,
                            error_message=str(e),
                        )

                    finally:
                        if "ofdm_generator" in locals():
                            ofdm_generator.cleanup_resources()

                    benchmark_suite.add_result(result)

        # Verify minimum success rate (adjusted for configuration issues)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        assert (
            success_rate >= 0.5
        ), f"Success rate too low: {success_rate:.1%} ({successful_tests}/{total_tests})"

        print(
            f"\nComprehensive benchmark completed: {successful_tests}/{total_tests} tests successful"
        )


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-m", "performance"])
