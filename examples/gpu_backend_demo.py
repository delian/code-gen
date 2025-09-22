#!/usr/bin/env python3
"""
GPU Backend Demonstration

This example demonstrates the GPU backend functionality with CuPy integration,
including automatic CPU fallback when GPU is not available.
"""

import time

import numpy as np

from ofdm_chirp_generator import GPUBackend, MemoryManager


def demonstrate_gpu_backend():
    """Demonstrate GPU backend capabilities."""
    print("=== OFDM Chirp Generator GPU Backend Demo ===\n")

    # Initialize GPU backend
    print("1. Initializing GPU Backend...")
    backend = GPUBackend()

    device_info = backend.device_info
    print(f"   Backend: {device_info['backend']}")
    print(f"   Device: {device_info['device_name']}")

    if backend.is_gpu_available:
        print(
            f"   GPU Memory: {device_info['memory_free'] / 1e9:.2f} GB free / {device_info['memory_total'] / 1e9:.2f} GB total"
        )
        print(f"   Compute Capability: {device_info['compute_capability']}")

    print()

    # Memory allocation demonstration
    print("2. Memory Allocation Test...")
    shapes = [(1000,), (100, 10), (50, 20, 2)]

    for shape in shapes:
        array = backend.allocate_signal_memory(shape, np.complex128)
        print(f"   Allocated array shape {shape}: {type(array).__name__}")

    print()

    # Array transfer demonstration
    print("3. Array Transfer Test...")
    cpu_array = np.random.random(1000) + 1j * np.random.random(1000)
    print(f"   Original array type: {type(cpu_array).__name__}")

    gpu_array = backend.to_gpu(cpu_array)
    print(f"   GPU array type: {type(gpu_array).__name__}")

    cpu_back = backend.to_cpu(gpu_array)
    print(f"   Transferred back type: {type(cpu_back).__name__}")
    print(f"   Arrays equal: {np.allclose(cpu_array, cpu_back)}")

    print()

    # FFT performance comparison
    print("4. FFT Performance Test...")
    test_sizes = [1024, 4096, 16384]

    for size in test_sizes:
        # Create test signal
        t = np.linspace(0, 1, size)
        signal = np.sin(2 * np.pi * 10 * t) + 1j * np.cos(2 * np.pi * 20 * t)

        # Time FFT operation
        start_time = time.time()
        fft_result = backend.perform_fft(signal)
        end_time = time.time()

        print(f"   FFT size {size}: {(end_time - start_time) * 1000:.2f} ms")

    print()

    # Cross-correlation demonstration
    print("5. Cross-Correlation Test...")

    # Create two test signals
    signal1 = np.random.random(1000) + 1j * np.random.random(1000)
    signal2 = np.random.random(1000) + 1j * np.random.random(1000)

    # Compute correlations
    auto_corr = backend.compute_correlation(signal1, signal1)
    cross_corr = backend.compute_correlation(signal1, signal2)

    print(f"   Auto-correlation: {auto_corr:.4f}")
    print(f"   Cross-correlation: {cross_corr:.4f}")
    print(f"   Auto > Cross: {auto_corr > cross_corr}")

    print()

    # Memory management demonstration
    print("6. Memory Management Test...")
    memory_manager = MemoryManager(backend)

    # Get optimal chunk size
    chunk_size = memory_manager.get_optimal_chunk_size()
    print(f"   Optimal chunk size: {chunk_size:,} elements")

    # Allocate chunked memory
    total_shape = (10000,)
    chunks = memory_manager.allocate_chunked_memory(total_shape, 1000)
    print(f"   Allocated {len(chunks)} chunks for shape {total_shape}")

    # Memory info
    memory_info = backend.get_memory_info()
    print(f"   Memory backend: {memory_info['backend']}")

    if memory_info["used_bytes"] is not None:
        print(f"   Used memory: {memory_info['used_bytes'] / 1e6:.2f} MB")

    print()

    # Cleanup
    print("7. Cleanup...")
    memory_manager.cleanup_all()
    backend.cleanup_memory()
    print("   Memory cleanup completed")

    print("\n=== Demo Complete ===")


def demonstrate_cpu_fallback():
    """Demonstrate CPU fallback functionality."""
    print("\n=== CPU Fallback Demonstration ===\n")

    # Force CPU backend
    print("1. Forcing CPU Backend...")
    cpu_backend = GPUBackend(force_cpu=True)

    device_info = cpu_backend.device_info
    print(f"   Backend: {device_info['backend']}")
    print(f"   Device: {device_info['device_name']}")

    print()

    # Test operations on CPU
    print("2. Testing CPU Operations...")

    # Create test signal
    signal = np.random.random(1000) + 1j * np.random.random(1000)

    # FFT
    fft_result = cpu_backend.perform_fft(signal)
    ifft_result = cpu_backend.perform_ifft(fft_result)

    print(f"   FFT roundtrip error: {np.max(np.abs(signal - ifft_result)):.2e}")

    # Correlation
    correlation = cpu_backend.compute_correlation(signal, signal)
    print(f"   Auto-correlation: {correlation:.4f}")

    print("\n=== CPU Fallback Demo Complete ===")


def demonstrate_error_handling():
    """Demonstrate error handling and robustness."""
    print("\n=== Error Handling Demonstration ===\n")

    backend = GPUBackend()

    print("1. Testing with Various Array Sizes...")

    # Test with different sizes
    sizes = [10, 100, 1000, 10000]

    for size in sizes:
        try:
            array = backend.allocate_signal_memory((size,))
            fft_result = backend.perform_fft(array)
            print(f"   Size {size}: OK")
        except Exception as e:
            print(f"   Size {size}: Error - {e}")

    print()

    print("2. Testing Mixed Data Types...")

    # Test with different data types
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]

    for dtype in dtypes:
        try:
            array = backend.allocate_signal_memory((100,), dtype)
            print(f"   {dtype.__name__}: OK")
        except Exception as e:
            print(f"   {dtype.__name__}: Error - {e}")

    print("\n=== Error Handling Demo Complete ===")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_gpu_backend()
    demonstrate_cpu_fallback()
    demonstrate_error_handling()
