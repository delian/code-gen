#!/usr/bin/env python3
"""
Performance Benchmarks Demo

This script demonstrates the comprehensive performance benchmarking capabilities
of the OFDM Chirp Generator, including GPU vs CPU comparisons, scalability tests,
real-time validation, and regression detection.
"""

import json
import logging
import os
import time
from typing import Any, Dict

import numpy as np

from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.ofdm_generator import OFDMGenerator
from ofdm_chirp_generator.orthogonal_signal_generator import OrthogonalSignalGenerator
from ofdm_chirp_generator.performance_optimizer import PerformanceOptimizer
from ofdm_chirp_generator.real_time_validator import RealTimeRequirements, RealTimeValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_configs() -> Dict[str, OFDMConfig]:
    """Create various OFDM configurations for testing."""
    configs = {
        "small": OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,  # 2ms (longer to avoid chirp length issues)
        ),
        "medium": OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,  # 2ms
        ),
        "large": OFDMConfig(
            num_subcarriers=12,  # Reduced from 16 to avoid bandwidth issues
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.005,  # 5ms
        ),
    }
    return configs


def demo_gpu_vs_cpu_performance():
    """Demonstrate GPU vs CPU performance comparison."""
    print("\n" + "=" * 60)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("=" * 60)

    configs = create_test_configs()

    for config_name, config in configs.items():
        print(f"\nTesting configuration: {config_name}")
        print(f"  Subcarriers: {config.num_subcarriers}")
        print(f"  Signal duration: {config.signal_duration}s")

        results = {}

        for backend_name in ["CPU", "GPU"]:
            print(f"\n  {backend_name} Backend:")

            try:
                # Initialize backend
                force_cpu = backend_name == "CPU"
                gpu_backend = GPUBackend(force_cpu=force_cpu)
                ofdm_generator = OFDMGenerator(config, gpu_backend)

                # Test single signal generation
                phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)

                # Warm-up run
                _ = ofdm_generator.generate_single_signal(phase_array)

                # Timed runs
                times = []
                for i in range(10):
                    start_time = time.time()
                    signal = ofdm_generator.generate_single_signal(phase_array)
                    execution_time = time.time() - start_time
                    times.append(execution_time)

                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)

                results[backend_name] = {
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "throughput": 1.0 / avg_time,
                    "signal_length": len(signal),
                }

                print(f"    Average time: {avg_time:.4f}s ± {std_time:.4f}s")
                print(f"    Range: {min_time:.4f}s - {max_time:.4f}s")
                print(f"    Throughput: {1.0/avg_time:.1f} signals/s")
                print(f"    Signal length: {len(signal):,} samples")

            except Exception as e:
                print(f"    ERROR: {e}")
                results[backend_name] = {"error": str(e)}

            finally:
                if "ofdm_generator" in locals():
                    ofdm_generator.cleanup_resources()

        # Compare results
        if (
            "CPU" in results
            and "GPU" in results
            and "error" not in results["CPU"]
            and "error" not in results["GPU"]
        ):
            cpu_time = results["CPU"]["avg_time"]
            gpu_time = results["GPU"]["avg_time"]
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

            print(f"\n  Performance Comparison:")
            print(f"    GPU Speedup: {speedup:.2f}x")
            print(f"    GPU Efficiency: {(speedup - 1) * 100:.1f}% improvement")

            if speedup > 1.0:
                print(f"    ✓ GPU is faster than CPU")
            else:
                print(f"    ⚠ CPU is faster than GPU (overhead may dominate)")


def demo_scalability_testing():
    """Demonstrate scalability testing with varying parameters."""
    print("\n" + "=" * 60)
    print("SCALABILITY TESTING")
    print("=" * 60)

    # Test scalability with number of subcarriers
    print("\n1. Subcarrier Count Scalability")
    subcarrier_counts = [4, 8, 12, 16]  # Reduced max to avoid bandwidth issues
    base_config = OFDMConfig(
        num_subcarriers=8,  # Will be overridden
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002,  # Longer to avoid chirp length issues
    )

    scalability_results = {}

    for num_subcarriers in subcarrier_counts:
        config = OFDMConfig(
            num_subcarriers=num_subcarriers,
            subcarrier_spacing=base_config.subcarrier_spacing,
            bandwidth_per_subcarrier=base_config.bandwidth_per_subcarrier,
            center_frequency=base_config.center_frequency,
            sampling_rate=base_config.sampling_rate,
            signal_duration=base_config.signal_duration,
        )

        try:
            # Use CPU for consistent results
            gpu_backend = GPUBackend(force_cpu=True)
            ofdm_generator = OFDMGenerator(config, gpu_backend)

            # Time signal generation
            phase_array = np.random.uniform(0, 2 * np.pi, num_subcarriers)

            times = []
            for _ in range(5):
                start_time = time.time()
                signal = ofdm_generator.generate_single_signal(phase_array)
                execution_time = time.time() - start_time
                times.append(execution_time)

            avg_time = np.mean(times)
            complexity_factor = num_subcarriers / 8.0  # Relative to base case

            scalability_results[num_subcarriers] = {
                "avg_time": avg_time,
                "complexity_factor": complexity_factor,
                "efficiency": (
                    (
                        complexity_factor
                        / (avg_time / scalability_results.get(8, {}).get("avg_time", avg_time))
                    )
                    if 8 in scalability_results
                    else 1.0
                ),
                "signal_length": len(signal),
            }

            print(
                f"  {num_subcarriers:2d} subcarriers: {avg_time:.4f}s (complexity: {complexity_factor:.1f}x)"
            )

        except Exception as e:
            print(f"  {num_subcarriers:2d} subcarriers: ERROR - {e}")

        finally:
            if "ofdm_generator" in locals():
                ofdm_generator.cleanup_resources()

    # Analyze scalability
    if len(scalability_results) >= 2:
        print(f"\n  Scalability Analysis:")
        base_time = scalability_results[min(scalability_results.keys())]["avg_time"]

        for num_subcarriers, result in scalability_results.items():
            scaling_factor = result["avg_time"] / base_time
            theoretical_scaling = result["complexity_factor"]
            efficiency = theoretical_scaling / scaling_factor if scaling_factor > 0 else 0

            print(
                f"    {num_subcarriers:2d} subcarriers: {scaling_factor:.2f}x time, {efficiency:.1%} efficiency"
            )

    # Test scalability with signal duration
    print("\n2. Signal Duration Scalability")
    durations = [0.0005, 0.001, 0.002, 0.005]  # 0.5ms to 5ms

    for duration in durations:
        config = OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=duration,
        )

        try:
            gpu_backend = GPUBackend(force_cpu=True)
            ofdm_generator = OFDMGenerator(config, gpu_backend)

            phase_array = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)

            start_time = time.time()
            signal = ofdm_generator.generate_single_signal(phase_array)
            execution_time = time.time() - start_time

            samples_per_ms = len(signal) / (duration * 1000)

            print(
                f"  {duration*1000:4.1f}ms signal: {execution_time:.4f}s, {len(signal):,} samples ({samples_per_ms:.0f}/ms)"
            )

        except Exception as e:
            print(f"  {duration*1000:4.1f}ms signal: ERROR - {e}")

        finally:
            if "ofdm_generator" in locals():
                ofdm_generator.cleanup_resources()


def demo_real_time_validation():
    """Demonstrate real-time performance validation."""
    print("\n" + "=" * 60)
    print("REAL-TIME PERFORMANCE VALIDATION")
    print("=" * 60)

    # Create real-time requirements
    requirements = RealTimeRequirements(
        max_latency_ms=5.0,
        min_real_time_factor=1.0,
        target_real_time_factor=2.0,
        min_throughput_signals_per_sec=200.0,
        max_jitter_ms=2.0,
        sustained_duration_sec=5.0,  # Shorter for demo
    )

    config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002,  # 2ms signal
    )

    print(f"Configuration:")
    print(f"  Signal duration: {config.signal_duration*1000:.1f}ms")
    print(f"  Requirements:")
    print(f"    Max latency: {requirements.max_latency_ms:.1f}ms")
    print(f"    Target real-time factor: {requirements.target_real_time_factor:.1f}x")
    print(f"    Min throughput: {requirements.min_throughput_signals_per_sec:.0f} signals/s")

    try:
        gpu_backend = GPUBackend()
        validator = RealTimeValidator(config, gpu_backend, requirements)

        print(f"\nBackend: {'GPU' if gpu_backend.is_gpu_available else 'CPU'}")

        # Single signal validation
        print(f"\n1. Single Signal Validation")
        single_metrics = validator.validate_single_signal_generation()

        print(f"  Generation time: {single_metrics.generation_time*1000:.2f}ms")
        print(f"  Real-time factor: {single_metrics.real_time_factor:.2f}x")
        print(f"  Meets real-time: {'✓' if single_metrics.meets_real_time else '✗'}")
        print(f"  Latency: {single_metrics.latency_ms:.2f}ms")
        print(f"  Throughput: {single_metrics.throughput_signals_per_sec:.1f} signals/s")

        # Sustained performance validation
        print(f"\n2. Sustained Performance Validation")
        sustained_analysis = validator.validate_sustained_performance(3.0)  # 3 seconds for demo

        print(f"  Test duration: {sustained_analysis['test_duration_sec']:.1f}s")
        print(f"  Signals generated: {sustained_analysis['signals_generated']}")
        print(f"  Overall throughput: {sustained_analysis['overall_throughput']:.1f} signals/s")
        print(f"  Real-time success rate: {sustained_analysis['real_time_success_rate']:.1%}")

        rt_stats = sustained_analysis["real_time_factor_stats"]
        print(f"  Real-time factor: avg={rt_stats['average']:.2f}, min={rt_stats['minimum']:.2f}")

        latency_stats = sustained_analysis["latency_stats"]
        print(
            f"  Latency: avg={latency_stats['average_ms']:.2f}ms, max={latency_stats['maximum_ms']:.2f}ms"
        )
        print(f"  Jitter: {latency_stats['jitter_ms']:.2f}ms")

        compliance = sustained_analysis["requirements_compliance"]
        print(f"  Requirements compliance: {'✓' if compliance['all_requirements_met'] else '✗'}")

        # Batch performance validation
        print(f"\n3. Batch Performance Validation")
        batch_analysis = validator.validate_batch_processing([1, 5, 10])

        print(f"  Optimal batch size: {batch_analysis['optimal_batch_size']}")
        print(f"  Optimal efficiency: {batch_analysis['optimal_efficiency']:.2f}x")
        print(f"  Optimal throughput: {batch_analysis['optimal_throughput']:.1f} signals/s")

        for batch_size, result in batch_analysis["batch_results"].items():
            print(
                f"    Batch {batch_size}: {result['avg_time_per_signal']*1000:.2f}ms/signal, "
                f"{result['batch_efficiency']:.2f}x efficiency"
            )

        # Memory-constrained performance
        print(f"\n4. Memory-Constrained Performance")
        memory_analysis = validator.validate_memory_constrained_performance()

        print(f"  Max successful batch: {memory_analysis['max_successful_batch_size']}")
        print(
            f"  Performance degradation: {memory_analysis['average_performance_degradation']:.1%}"
        )
        print(f"  Memory scalability score: {memory_analysis['memory_scalability_score']:.2f}")

        # Generate comprehensive report
        print(f"\n5. Comprehensive Real-Time Report")
        report = validator.generate_real_time_report()

        assessment = report["overall_assessment"]
        print(f"  Real-time capable: {'✓' if assessment['real_time_capable'] else '✗'}")
        print(f"  Performance grade: {assessment['performance_grade']}")
        print(f"  Recommendations:")
        for rec in assessment["recommendations"]:
            print(f"    - {rec}")

        # Save report
        report_file = "real_time_performance_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Detailed report saved to: {report_file}")

    except Exception as e:
        print(f"Real-time validation failed: {e}")

    finally:
        if "validator" in locals():
            validator.cleanup_resources()


def demo_batch_processing_optimization():
    """Demonstrate batch processing optimization."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING OPTIMIZATION")
    print("=" * 60)

    config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002,
    )

    batch_sizes = [1, 2, 5, 10, 20]

    for backend_name in ["CPU", "GPU"]:
        print(f"\n{backend_name} Backend Batch Processing:")

        try:
            force_cpu = backend_name == "CPU"
            gpu_backend = GPUBackend(force_cpu=force_cpu)
            optimizer = PerformanceOptimizer(gpu_backend)
            ofdm_generator = OFDMGenerator(config, gpu_backend)

            def generate_signal(phase_array):
                return ofdm_generator.generate_single_signal(phase_array)

            batch_results = {}

            for batch_size in batch_sizes:
                # Generate phase arrays
                phase_arrays = [
                    np.random.uniform(0, 2 * np.pi, config.num_subcarriers)
                    for _ in range(batch_size)
                ]

                # Test optimized batch processing
                start_time = time.time()
                optimized_signals = optimizer.optimize_signal_generation(
                    generate_signal, phase_arrays, f"batch_demo_{batch_size}"
                )
                optimized_time = time.time() - start_time

                # Test direct processing
                start_time = time.time()
                direct_signals = [generate_signal(pa) for pa in phase_arrays]
                direct_time = time.time() - start_time

                # Calculate metrics
                optimized_throughput = batch_size / optimized_time if optimized_time > 0 else 0
                direct_throughput = batch_size / direct_time if direct_time > 0 else 0
                speedup = direct_time / optimized_time if optimized_time > 0 else 1.0

                batch_results[batch_size] = {
                    "optimized_time": optimized_time,
                    "direct_time": direct_time,
                    "speedup": speedup,
                    "optimized_throughput": optimized_throughput,
                    "direct_throughput": direct_throughput,
                }

                print(
                    f"  Batch {batch_size:2d}: "
                    f"optimized={optimized_time:.4f}s, "
                    f"direct={direct_time:.4f}s, "
                    f"speedup={speedup:.2f}x, "
                    f"throughput={optimized_throughput:.1f} signals/s"
                )

            # Find optimal batch size
            optimal_batch = max(
                batch_results.keys(), key=lambda bs: batch_results[bs]["optimized_throughput"]
            )

            print(f"\n  Optimal batch size: {optimal_batch}")
            print(
                f"  Peak throughput: {batch_results[optimal_batch]['optimized_throughput']:.1f} signals/s"
            )

        except Exception as e:
            print(f"  ERROR: {e}")

        finally:
            if "optimizer" in locals():
                optimizer.cleanup_resources()
            if "ofdm_generator" in locals():
                ofdm_generator.cleanup_resources()


def demo_performance_regression_detection():
    """Demonstrate performance regression detection."""
    print("\n" + "=" * 60)
    print("PERFORMANCE REGRESSION DETECTION")
    print("=" * 60)

    # This would typically be run as part of CI/CD pipeline
    print("This demo shows how regression detection would work in practice.")
    print("In a real scenario, baseline results would be stored from previous runs.")

    # Simulate baseline results
    baseline_results = {
        "signal_generation": {"execution_time": 0.001, "backend": "GPU"},
        "batch_processing_10": {"execution_time": 0.008, "backend": "GPU"},
        "orthogonal_set_4": {"execution_time": 0.015, "backend": "GPU"},
    }

    # Simulate current results (with some regression)
    current_results = {
        "signal_generation": {"execution_time": 0.0012, "backend": "GPU"},  # 20% slower
        "batch_processing_10": {"execution_time": 0.007, "backend": "GPU"},  # 12.5% faster
        "orthogonal_set_4": {"execution_time": 0.019, "backend": "GPU"},  # 26.7% slower
    }

    print(f"\nBaseline vs Current Performance:")
    print(f"{'Test Name':<20} {'Baseline':<10} {'Current':<10} {'Change':<10} {'Status'}")
    print("-" * 60)

    tolerance = 0.15  # 15% tolerance
    regressions = []
    improvements = []

    for test_name in baseline_results:
        if test_name in current_results:
            baseline_time = baseline_results[test_name]["execution_time"]
            current_time = current_results[test_name]["execution_time"]

            change_percent = ((current_time - baseline_time) / baseline_time) * 100

            if change_percent > tolerance * 100:
                status = "REGRESSION"
                regressions.append(test_name)
            elif change_percent < -5:  # 5% improvement threshold
                status = "IMPROVEMENT"
                improvements.append(test_name)
            else:
                status = "OK"

            print(
                f"{test_name:<20} {baseline_time:<10.4f} {current_time:<10.4f} "
                f"{change_percent:>+7.1f}% {status}"
            )

    print(f"\nRegression Analysis:")
    print(f"  Tolerance: {tolerance*100:.0f}%")
    print(f"  Regressions detected: {len(regressions)}")
    print(f"  Improvements detected: {len(improvements)}")

    if regressions:
        print(f"  ⚠ Performance regressions in: {', '.join(regressions)}")
    else:
        print(f"  ✓ No performance regressions detected")

    if improvements:
        print(f"  ✓ Performance improvements in: {', '.join(improvements)}")


def main():
    """Run all performance benchmark demos."""
    print("OFDM Chirp Generator - Performance Benchmarks Demo")
    print("=" * 60)

    try:
        # Run individual demos
        demo_gpu_vs_cpu_performance()
        demo_scalability_testing()
        demo_batch_processing_optimization()
        demo_real_time_validation()
        demo_performance_regression_detection()

        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARKS DEMO COMPLETED")
        print("=" * 60)
        print("\nKey Performance Insights:")
        print("1. GPU acceleration provides significant speedup for complex operations")
        print("2. Performance scales predictably with problem complexity")
        print("3. Batch processing optimization improves throughput")
        print("4. Real-time requirements can be validated systematically")
        print("5. Regression detection helps maintain performance over time")
        print(
            "\nFor automated benchmarking, use: uv run python scripts/run_performance_benchmarks.py"
        )

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
