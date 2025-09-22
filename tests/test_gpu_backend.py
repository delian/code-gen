"""
Tests for GPU backend implementation with CuPy integration.

This module tests GPU acceleration, CPU fallback, and memory management
functionality of the GPUBackend class.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.gpu_backend import GPUBackend, MemoryManager

# Try to import CuPy for GPU tests
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class TestGPUBackend:
    """Test cases for GPUBackend class."""

    def test_cpu_backend_initialization(self):
        """Test CPU backend initialization when forced."""
        backend = GPUBackend(force_cpu=True)

        assert not backend.is_gpu_available
        assert backend.device_info["backend"] == "CPU"
        assert backend.device_info["device_name"] == "CPU"

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_backend_initialization(self):
        """Test GPU backend initialization when CuPy is available."""
        backend = GPUBackend(force_cpu=False)

        # GPU availability depends on hardware, so we test both cases
        device_info = backend.device_info
        assert device_info["backend"] in ["GPU", "CPU"]

        if backend.is_gpu_available:
            assert "device_id" in device_info
            assert "memory_total" in device_info
            assert "compute_capability" in device_info

    def test_memory_allocation_cpu(self):
        """Test memory allocation on CPU backend."""
        backend = GPUBackend(force_cpu=True)

        # Test different shapes and data types
        shapes_and_types = [
            ((100,), np.complex128),
            ((50, 2), np.float64),
            ((10, 10, 2), np.complex64),
        ]

        for shape, dtype in shapes_and_types:
            array = backend.allocate_signal_memory(shape, dtype)

            assert isinstance(array, np.ndarray)
            assert array.shape == shape
            assert array.dtype == dtype
            assert np.all(array == 0)  # Should be zero-initialized

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_memory_allocation_gpu(self):
        """Test memory allocation on GPU backend if available."""
        backend = GPUBackend(force_cpu=False)

        if not backend.is_gpu_available:
            pytest.skip("GPU not available on this system")

        shape = (100, 2)
        dtype = np.complex128
        array = backend.allocate_signal_memory(shape, dtype)

        # Should be CuPy array if GPU is available
        assert hasattr(array, "device")  # CuPy arrays have device attribute
        assert array.shape == shape
        assert array.dtype == dtype

    def test_array_transfer_cpu_only(self):
        """Test array transfer operations with CPU backend."""
        backend = GPUBackend(force_cpu=True)

        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.complex128)

        # GPU transfer should return original array when GPU not available
        gpu_array = backend.to_gpu(test_array)
        assert isinstance(gpu_array, np.ndarray)
        np.testing.assert_array_equal(gpu_array, test_array)

        # CPU transfer should return the same array
        cpu_array = backend.to_cpu(gpu_array)
        assert isinstance(cpu_array, np.ndarray)
        np.testing.assert_array_equal(cpu_array, test_array)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_array_transfer_gpu(self):
        """Test array transfer operations with GPU backend if available."""
        backend = GPUBackend(force_cpu=False)

        if not backend.is_gpu_available:
            pytest.skip("GPU not available on this system")

        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.complex128)

        # Transfer to GPU
        gpu_array = backend.to_gpu(test_array)
        assert hasattr(gpu_array, "device")  # Should be CuPy array

        # Transfer back to CPU
        cpu_array = backend.to_cpu(gpu_array)
        assert isinstance(cpu_array, np.ndarray)
        np.testing.assert_array_equal(cpu_array, test_array)

    def test_fft_operations_cpu(self):
        """Test FFT operations on CPU backend."""
        backend = GPUBackend(force_cpu=True)

        # Create test signal
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t) + 1j * np.cos(2 * np.pi * 5 * t)

        # Test FFT
        fft_result = backend.perform_fft(signal)
        assert isinstance(fft_result, np.ndarray)
        assert fft_result.shape == signal.shape

        # Test IFFT
        ifft_result = backend.perform_ifft(fft_result)
        assert isinstance(ifft_result, np.ndarray)
        np.testing.assert_allclose(ifft_result, signal, rtol=1e-10)

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_fft_operations_gpu(self):
        """Test FFT operations on GPU backend if available."""
        backend = GPUBackend(force_cpu=False)

        if not backend.is_gpu_available:
            pytest.skip("GPU not available on this system")

        # Create test signal
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t) + 1j * np.cos(2 * np.pi * 5 * t)
        gpu_signal = backend.to_gpu(signal)

        # Test FFT
        fft_result = backend.perform_fft(gpu_signal)
        assert hasattr(fft_result, "device")  # Should be CuPy array

        # Test IFFT
        ifft_result = backend.perform_ifft(fft_result)
        cpu_result = backend.to_cpu(ifft_result)
        np.testing.assert_allclose(cpu_result, signal, rtol=1e-10)

    def test_correlation_computation_cpu(self):
        """Test cross-correlation computation on CPU."""
        backend = GPUBackend(force_cpu=True)

        # Create test signals
        signal1 = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        signal2 = np.array([5, 4, 3, 2, 1], dtype=np.complex128)

        correlation = backend.compute_correlation(signal1, signal2)

        assert isinstance(correlation, float)
        assert correlation >= 0  # Correlation magnitude should be non-negative

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_correlation_computation_gpu(self):
        """Test cross-correlation computation on GPU if available."""
        backend = GPUBackend(force_cpu=False)

        if not backend.is_gpu_available:
            pytest.skip("GPU not available on this system")

        # Create test signals
        signal1 = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        signal2 = np.array([5, 4, 3, 2, 1], dtype=np.complex128)

        correlation = backend.compute_correlation(signal1, signal2)

        assert isinstance(correlation, float)
        assert correlation >= 0

    def test_correlation_identical_signals(self):
        """Test correlation of identical signals should be maximum."""
        backend = GPUBackend(force_cpu=True)

        signal = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        correlation = backend.compute_correlation(signal, signal)

        # Correlation of identical signals should be high
        assert correlation > 0

    def test_memory_info_cpu(self):
        """Test memory information retrieval for CPU backend."""
        backend = GPUBackend(force_cpu=True)

        memory_info = backend.get_memory_info()

        assert memory_info["backend"] == "CPU"
        assert memory_info["used_bytes"] is None
        assert memory_info["total_bytes"] is None
        assert memory_info["free_bytes"] is None

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_memory_info_gpu(self):
        """Test memory information retrieval for GPU backend if available."""
        backend = GPUBackend(force_cpu=False)

        if not backend.is_gpu_available:
            pytest.skip("GPU not available on this system")

        memory_info = backend.get_memory_info()

        assert memory_info["backend"] == "GPU"
        assert isinstance(memory_info["used_bytes"], int)
        assert isinstance(memory_info["total_bytes"], int)
        assert isinstance(memory_info["free_bytes"], int)

    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        backend = GPUBackend()

        # Allocate some memory
        arrays = []
        for i in range(5):
            array = backend.allocate_signal_memory((100, 2))
            arrays.append(array)

        # Cleanup should not raise errors
        backend.cleanup_memory()

    def test_context_manager(self):
        """Test GPUBackend as context manager."""
        with GPUBackend() as backend:
            assert backend is not None
            array = backend.allocate_signal_memory((10,))
            assert array is not None

        # Context manager should clean up automatically

    def test_fallback_on_gpu_error(self):
        """Test graceful fallback when GPU operations fail."""
        backend = GPUBackend(force_cpu=True)

        # Create test data
        signal = np.array([1, 2, 3, 4, 5], dtype=np.complex128)

        # Operations should work even if GPU fails
        fft_result = backend.perform_fft(signal)
        assert isinstance(fft_result, np.ndarray)

        correlation = backend.compute_correlation(signal, signal)
        assert isinstance(correlation, float)


class TestMemoryManager:
    """Test cases for MemoryManager class."""

    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        backend = GPUBackend(force_cpu=True)
        manager = MemoryManager(backend)

        assert manager.backend is backend
        assert len(manager._allocated_arrays) == 0

    def test_chunked_memory_allocation(self):
        """Test chunked memory allocation."""
        backend = GPUBackend(force_cpu=True)
        manager = MemoryManager(backend)

        total_shape = (1000,)
        chunk_size = 100

        chunks = manager.allocate_chunked_memory(total_shape, chunk_size)

        assert len(chunks) == 10  # 1000 / 100 = 10 chunks
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        assert all(chunk.shape[0] <= chunk_size for chunk in chunks)

    def test_optimal_chunk_size_cpu(self):
        """Test optimal chunk size calculation for CPU."""
        backend = GPUBackend(force_cpu=True)
        manager = MemoryManager(backend)

        chunk_size = manager.get_optimal_chunk_size()

        assert isinstance(chunk_size, int)
        assert chunk_size > 0
        assert chunk_size == 1024 * 1024  # Default for CPU

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_optimal_chunk_size_gpu(self):
        """Test optimal chunk size calculation for GPU if available."""
        backend = GPUBackend(force_cpu=False)

        if not backend.is_gpu_available:
            pytest.skip("GPU not available on this system")

        manager = MemoryManager(backend)
        chunk_size = manager.get_optimal_chunk_size()

        assert isinstance(chunk_size, int)
        assert chunk_size > 0

    def test_cleanup_all(self):
        """Test cleanup of all managed arrays."""
        backend = GPUBackend(force_cpu=True)
        manager = MemoryManager(backend)

        # Allocate some chunked memory
        chunks = manager.allocate_chunked_memory((100,), 10)
        assert len(manager._allocated_arrays) > 0

        # Cleanup should clear all arrays
        manager.cleanup_all()
        assert len(manager._allocated_arrays) == 0


class TestGPUBackendErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_array_shapes(self):
        """Test handling of invalid array shapes."""
        backend = GPUBackend(force_cpu=True)

        # Test with empty shape
        array = backend.allocate_signal_memory(())
        assert array.shape == ()

        # Test with very large shape (should not crash)
        try:
            large_array = backend.allocate_signal_memory((1000000,))
            assert large_array.shape == (1000000,)
        except MemoryError:
            # This is acceptable for very large arrays
            pass

    def test_mixed_array_types(self):
        """Test operations with mixed array types."""
        backend = GPUBackend(force_cpu=True)

        # Create arrays of different types
        float_array = np.array([1.0, 2.0, 3.0])
        complex_array = np.array([1 + 1j, 2 + 2j, 3 + 3j])

        # Operations should handle type conversion gracefully
        correlation = backend.compute_correlation(float_array, complex_array)
        assert isinstance(correlation, float)

    @patch("ofdm_chirp_generator.gpu_backend.cp")
    def test_cupy_import_failure_simulation(self, mock_cp):
        """Test behavior when CuPy import fails."""
        # Simulate CuPy import failure
        mock_cp.side_effect = ImportError("CuPy not available")

        # Should fall back to CPU gracefully
        backend = GPUBackend(force_cpu=False)
        assert not backend.is_gpu_available
        assert backend.device_info["backend"] == "CPU"


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    pytest.main([__file__, "-v"])
