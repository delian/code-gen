#!/usr/bin/env python3
"""
Performance Optimization Demo

This script demonstrates the performance optimization and memory management
features of the OFDM Chirp Generator, including chunked processing,
adaptive batching, memory monitoring, and performance profiling.
"""

import logging
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.ofdm_generator import OFDMGenerator
from ofdm_chirp_generator.orthogonal_signal_generator import OrthogonalSignalGenerator
from ofdm_chirp_generator.performance_optimizer import PerformanceOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_ofdm_config() -> OFDMConfig:
    """Create OFDM configuration for testing."""
    return OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002,
    )


def demo_basic_performance_optimization():
    """Demonstrate basic performance optimization features."""
    print("\n" + "=" * 60)
    print("BASIC PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 60)

    # Initialize components
    ofdm_config = create_test_ofdm_config()
    gpu_backend = GPUBackend()
    optimizer = PerformanceOptimizer(gpu_backend)

    print(f"Backend: {'GPU' if gpu_backend.is_gpu_available else 'CPU'}")
    print(f"Device info: {gpu_backend.device_info}")

    # Create signal generator function
    ofdm_generator = OFDMGenerator(ofdm_config, gpu_backend)

    def generate_signal(phase_array):
        """Generate a single OFDM signal."""
        return ofdm_generator.generate_single_signal(phase_array)

    # Generate multiple phase arrays for testing
    num_signals = 20
    phase_arrays = [
        np.random.uniform(0, 2 * np.pi, ofdm_config.num_subcarriers) for _ in range(num_signals)
    ]

    print(f"\nGenerating {num_signals} OFDM signals...")

    # Optimize signal generation
    start_time = time.time()
    optimized_signals = optimizer.optimize_signal_generation(
        generate_signal, phase_arrays, "demo_signal_generation"
    )
    optimized_time = time.time() - start_time

    print(f"Optimized generation completed in {optimized_time:.3f}s")
    print(f"Throughput: {len(optimized_signals) / optimized_time:.1f} signals/s")

    # Compare with direct generation
    print("\nComparing with direct generation...")
    start_time = time.time()
    direct_signals = [generate_signal(phase_array) for phase_array in phase_arrays]
    direct_time = time.time() - start_time

    print(f"Direct generation completed in {direct_time:.3f}s")
    print(f"Throughput: {len(direct_signals) / direct_time:.1f} signals/s")

    # Performance comparison
    speedup = direct_time / optimized_time if optimized_time > 0 else 1.0
    print(f"\nPerformance comparison:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Overhead: {((optimized_time - direct_time) / direct_time * 100):.1f}%")

    # Cleanup
    optimizer.cleanup_resources()
    ofdm_generator.cleanup_resources()


def demo_memory_management():
    """Demonstrate memory management features."""
    print("\n" + "=" * 60)
    print("MEMORY MANAGEMENT DEMO")
    print("=" * 60)

    gpu_backend = GPUBackend()
    optimizer = PerformanceOptimizer(gpu_backend)

    # Display initial memory state
    initial_memory = optimizer.memory_manager.get_memory_statistics()
    print(f"Initial memory state:")
    print(f"  Backend: {initial_memory['backend']}")
    if initial_memory["backend"] == "GPU":
        print(f"  Current used: {initial_memory.get('current_used_bytes', 0) / 1024**2:.1f} MB")
        print(f"  Peak usage: {initial_memory.get('peak_usage_bytes', 0) / 1024**2:.1f} MB")

    # Test memory pressure detection
    pressure = optimizer.memory_manager.check_memory_pressure()
    print(f"\nMemory pressure analysis:")
    print(f"  Under pressure: {pressure['under_pressure']}")
    print(f"  Pressure level: {pressure.get('pressure_level', 'N/A')}")
    print(f"  Recommendation: {pressure['recommendation']}")

    # Demonstrate chunked memory allocation
    print(f"\nTesting chunked memory allocation...")

    def memory_intensive_operation(array_size):
        """Simulate memory-intensive operation."""
        # Allocate large array
        data = np.random.random(array_size).astype(np.complex128)

        # Perform some operations
        fft_data = np.fft.fft(data)
        processed_data = fft_data * np.conj(fft_data)
        result = np.fft.ifft(processed_data)

        return np.abs(result)

    # Test with increasing array sizes
    array_sizes = [1000, 5000, 10000, 50000, 100000]

    for size in array_sizes:
        print(f"\nProcessing array of size {size}...")

        try:
            result = optimizer.optimize_memory_usage(memory_intensive_operation, size)

            # Check memory after operation
            memory_after = optimizer.memory_manager.get_memory_statistics()
            print(f"  Success: Result length = {len(result)}")
            if memory_after["backend"] == "GPU":
                print(
                    f"  Memory used: {memory_after.get('current_used_bytes', 0) / 1024**2:.1f} MB"
                )

        except Exception as e:
            print(f"  Failed: {e}")

    # Force garbage collection and show results
    print(f"\nForcing garbage collection...")
    cleanup_stats = optimizer.memory_manager.force_garbage_collection()
    print(f"  Freed: {cleanup_stats['freed_mb']:.1f} MB")
    print(f"  Objects collected: {cleanup_stats['objects_collected']}")

    # Final memory state
    final_memory = optimizer.memory_manager.get_memory_statistics()
    print(f"\nFinal memory state:")
    if final_memory["backend"] == "GPU":
        print(f"  Current used: {final_memory.get('current_used_bytes', 0) / 1024**2:.1f} MB")
        print(f"  Peak usage: {final_memory.get('peak_usage_bytes', 0) / 1024**2:.1f} MB")

    optimizer.cleanup_resources()


def demo_adaptive_batching():
    """Demonstrate adaptive batching capabilities."""
    print("\n" + "=" * 60)
    print("ADAPTIVE BATCHING DEMO")
    print("=" * 60)

    ofdm_config = create_test_ofdm_config()
    gpu_backend = GPUBackend()
    optimizer = PerformanceOptimizer(gpu_backend)

    # Create operations with variable complexity
    def variable_complexity_operation(complexity_factor):
        """Operation with variable computational complexity."""
        base_size = 1000
        size = int(base_size * complexity_factor)

        # Generate data
        data = np.random.random(size).astype(np.complex128)

        # Variable number of FFT operations based on complexity
        for _ in range(int(complexity_factor)):
            data = np.fft.fft(data)
            data = np.fft.ifft(data)

        return np.abs(data)

    # Test with varying complexity factors
    complexity_factors = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    print(f"Testing adaptive batching with {len(complexity_factors)} operations...")
    print("Complexity factors:", complexity_factors)

    # Process with adaptive batching
    start_time = time.time()
    results = optimizer.optimize_signal_generation(
        variable_complexity_operation, complexity_factors, "adaptive_batching_demo"
    )
    adaptive_time = time.time() - start_time

    print(f"\nAdaptive batching completed in {adaptive_time:.3f}s")
    print(f"Average throughput: {len(results) / adaptive_time:.1f} operations/s")

    # Process without optimization (direct)
    print("\nComparing with direct processing...")
    start_time = time.time()
    direct_results = [variable_complexity_operation(cf) for cf in complexity_factors]
    direct_time = time.time() - start_time

    print(f"Direct processing completed in {direct_time:.3f}s")
    print(f"Average throughput: {len(direct_results) / direct_time:.1f} operations/s")

    # Performance analysis
    speedup = direct_time / adaptive_time if adaptive_time > 0 else 1.0
    print(f"\nPerformance analysis:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {(speedup - 1) * 100:.1f}% improvement")

    # Verify results are equivalent
    for i, (adaptive_result, direct_result) in enumerate(zip(results, direct_results)):
        if not np.allclose(adaptive_result, direct_result, rtol=1e-10):
            print(f"  Warning: Results differ for operation {i}")
        else:
            print(f"  ✓ Results match for complexity factor {complexity_factors[i]}")

    optimizer.cleanup_resources()


def demo_performance_profiling():
    """Demonstrate performance profiling capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE PROFILING DEMO")
    print("=" * 60)

    ofdm_config = create_test_ofdm_config()
    gpu_backend = GPUBackend()
    optimizer = PerformanceOptimizer(gpu_backend)
    orthogonal_generator = OrthogonalSignalGenerator(ofdm_config, gpu_backend)

    # Perform various operations to generate profiling data
    operations = [
        (
            "Single Signal Generation",
            lambda: orthogonal_generator.ofdm_generator.generate_single_signal(
                np.random.uniform(0, 2 * np.pi, ofdm_config.num_subcarriers)
            ),
        ),
        (
            "Orthogonal Set (2 signals)",
            lambda: orthogonal_generator.generate_orthogonal_signal_set(2),
        ),
        (
            "Orthogonal Set (4 signals)",
            lambda: orthogonal_generator.generate_orthogonal_signal_set(4),
        ),
        (
            "Orthogonal Set (8 signals)",
            lambda: orthogonal_generator.generate_orthogonal_signal_set(8),
        ),
    ]

    print("Performing operations for profiling...")

    for op_name, operation in operations:
        print(f"\nExecuting: {op_name}")

        # Use optimizer to profile the operation
        def wrapped_operation(_):
            return operation()

        try:
            result = optimizer.optimize_signal_generation(
                wrapped_operation, [None], f"profile_{op_name.lower().replace(' ', '_')}"
            )
            print(f"  ✓ Completed successfully")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Generate comprehensive performance report
    print(f"\nGenerating performance report...")
    report = optimizer.get_optimization_report()

    print(f"\n" + "-" * 40)
    print("PERFORMANCE REPORT")
    print("-" * 40)

    print(f"Operations processed: {report['operations_processed']}")

    perf_summary = report["performance_summary"]
    if "total_operations" in perf_summary:
        print(f"Total operations: {perf_summary['total_operations']}")
        print(f"Total duration: {perf_summary['total_duration']:.3f}s")
        print(f"Average duration: {perf_summary['average_duration']:.3f}s")
        print(f"Min duration: {perf_summary['min_duration']:.3f}s")
        print(f"Max duration: {perf_summary['max_duration']:.3f}s")
        print(f"Error rate: {perf_summary['error_rate']:.1%}")

        if "average_throughput" in perf_summary:
            print(f"Average throughput: {perf_summary['average_throughput']:.1f} ops/s")

    memory_stats = report["memory_statistics"]
    print(f"\nMemory Statistics:")
    print(f"  Backend: {memory_stats['backend']}")
    print(f"  Allocations: {memory_stats['allocation_count']}")
    if memory_stats["backend"] == "GPU":
        print(f"  Peak usage: {memory_stats.get('peak_usage_bytes', 0) / 1024**2:.1f} MB")

    pressure = report["current_memory_pressure"]
    print(f"\nCurrent Memory Pressure:")
    print(f"  Under pressure: {pressure['under_pressure']}")
    print(f"  Recommendation: {pressure['recommendation']}")

    print(f"\nOptimization Recommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    # Cleanup
    optimizer.cleanup_resources()
    orthogonal_generator.cleanup_resources()


def demo_chunked_processing():
    """Demonstrate chunked processing for large arrays."""
    print("\n" + "=" * 60)
    print("CHUNKED PROCESSING DEMO")
    print("=" * 60)

    gpu_backend = GPUBackend()
    optimizer = PerformanceOptimizer(gpu_backend)

    # Create large signal processing task
    def large_signal_processor(signal_length):
        """Process a large signal array."""
        print(f"  Processing signal of length {signal_length}...")

        # Generate large signal
        signal = np.random.random(signal_length).astype(np.complex128)

        # Simulate complex processing
        fft_signal = np.fft.fft(signal)
        processed = fft_signal * np.conj(fft_signal)
        result = np.fft.ifft(processed)

        return np.abs(result)

    # Test with increasing signal sizes
    signal_lengths = [10000, 50000, 100000, 500000, 1000000]

    print("Testing chunked processing with large signals...")

    for length in signal_lengths:
        print(f"\nSignal length: {length:,} samples")

        try:
            # Process with optimization (chunked if needed)
            start_time = time.time()
            result = optimizer.optimize_memory_usage(large_signal_processor, length)
            optimized_time = time.time() - start_time

            print(f"  ✓ Optimized processing: {optimized_time:.3f}s")
            print(f"  Result length: {len(result):,}")

            # Memory statistics
            memory_stats = optimizer.memory_manager.get_memory_statistics()
            if memory_stats["backend"] == "GPU":
                print(
                    f"  GPU memory used: {memory_stats.get('current_used_bytes', 0) / 1024**2:.1f} MB"
                )

            # Check for memory pressure
            pressure = optimizer.memory_manager.check_memory_pressure()
            if pressure["under_pressure"]:
                print(f"  ⚠ Memory pressure: {pressure['recommendation']}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    optimizer.cleanup_resources()


def create_performance_visualization(optimizer: PerformanceOptimizer):
    """Create visualization of performance metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 60)

    # Get performance history
    metrics_history = optimizer.profiler.metrics_history

    if not metrics_history:
        print("No performance metrics available for visualization")
        return

    # Extract data for plotting
    operation_names = [m.operation_name for m in metrics_history]
    durations = [m.duration for m in metrics_history]
    throughputs = [
        m.throughput_signals_per_sec for m in metrics_history if m.throughput_signals_per_sec
    ]

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Operation durations
    ax1.bar(range(len(durations)), durations)
    ax1.set_title("Operation Durations")
    ax1.set_xlabel("Operation Index")
    ax1.set_ylabel("Duration (s)")
    ax1.tick_params(axis="x", rotation=45)

    # Throughput (if available)
    if throughputs:
        ax2.plot(throughputs, "o-")
        ax2.set_title("Throughput Over Time")
        ax2.set_xlabel("Operation Index")
        ax2.set_ylabel("Signals/Second")
    else:
        ax2.text(0.5, 0.5, "No throughput data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Throughput (No Data)")

    # Memory usage (if GPU)
    memory_stats = optimizer.memory_manager.get_memory_statistics()
    if memory_stats["backend"] == "GPU" and optimizer.memory_manager._memory_usage_history:
        memory_history = optimizer.memory_manager._memory_usage_history
        ax3.plot([m / 1024**2 for m in memory_history])  # Convert to MB
        ax3.set_title("GPU Memory Usage Over Time")
        ax3.set_xlabel("Measurement Index")
        ax3.set_ylabel("Memory Usage (MB)")
    else:
        ax3.text(0.5, 0.5, "No GPU memory data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Memory Usage (No Data)")

    # Performance summary
    perf_summary = optimizer.profiler.get_performance_summary()
    if "total_operations" in perf_summary:
        summary_data = [
            perf_summary["average_duration"],
            perf_summary["min_duration"],
            perf_summary["max_duration"],
        ]
        ax4.bar(["Average", "Min", "Max"], summary_data)
        ax4.set_title("Duration Statistics")
        ax4.set_ylabel("Duration (s)")
    else:
        ax4.text(0.5, 0.5, "No summary data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Duration Statistics (No Data)")

    plt.tight_layout()
    plt.savefig("performance_metrics.png", dpi=150, bbox_inches="tight")
    print("Performance visualization saved as 'performance_metrics.png'")

    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("Could not display plot (non-interactive environment)")


def main():
    """Run all performance optimization demos."""
    print("OFDM Chirp Generator - Performance Optimization Demo")
    print("=" * 60)

    try:
        # Run individual demos
        demo_basic_performance_optimization()
        demo_memory_management()
        demo_adaptive_batching()
        demo_chunked_processing()
        demo_performance_profiling()

        # Create final performance visualization
        gpu_backend = GPUBackend()
        optimizer = PerformanceOptimizer(gpu_backend)

        # Generate some data for visualization
        def dummy_operation(x):
            return np.random.random(100) * x

        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        optimizer.optimize_signal_generation(dummy_operation, test_data, "visualization_demo")

        create_performance_visualization(optimizer)
        optimizer.cleanup_resources()

        print("\n" + "=" * 60)
        print("PERFORMANCE OPTIMIZATION DEMO COMPLETED")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. Chunked processing enables handling of large arrays")
        print("2. Adaptive batching optimizes throughput under varying conditions")
        print("3. Memory monitoring prevents out-of-memory errors")
        print("4. Performance profiling provides insights for optimization")
        print("5. Automatic fallback ensures robustness")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
