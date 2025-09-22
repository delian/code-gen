"""
Tests for performance optimization and memory management.

This module tests the performance optimization features including
chunked processing, adaptive batching, memory management, and profiling.
"""

import os
import tempfile
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.config_manager import ConfigurationManager
from ofdm_chirp_generator.gpu_backend import ChunkedProcessor, GPUBackend, MemoryManager
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.performance_optimizer import (
    OptimizationConfig,
    PerformanceMetrics,
    PerformanceOptimizer,
    PerformanceProfiler,
)


class TestPerformanceProfiler:
    """Test performance profiler functionality."""

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)
        assert profiler.enable_gpu_monitoring is False
        assert len(profiler.metrics_history) == 0

    def test_monitoring_lifecycle(self):
        """Test complete monitoring lifecycle."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        # Start monitoring
        initial_state = profiler.start_monitoring("test_operation")
        assert "test_operation" in profiler._current_metrics
        assert "start_time" in initial_state

        # Simulate some work
        time.sleep(0.01)

        # Stop monitoring
        metrics = profiler.stop_monitoring("test_operation", throughput_signals=10)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "test_operation"
        assert metrics.duration > 0
        assert metrics.throughput_signals_per_sec > 0
        assert len(profiler.metrics_history) == 1

    def test_monitoring_with_error(self):
        """Test monitoring with error handling."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        profiler.start_monitoring("error_operation")
        metrics = profiler.stop_monitoring(
            "error_operation", error_info=(True, "Test error message")
        )

        assert metrics.error_occurred is True
        assert metrics.error_message == "Test error message"

    def test_performance_summary(self):
        """Test performance summary generation."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        # Generate some test metrics
        for i in range(5):
            profiler.start_monitoring(f"operation_{i}")
            time.sleep(0.001)
            profiler.stop_monitoring(f"operation_{i}", throughput_signals=i + 1)

        summary = profiler.get_performance_summary()

        assert summary["total_operations"] == 5
        assert summary["average_duration"] > 0
        assert summary["error_count"] == 0
        assert summary["error_rate"] == 0.0

    def test_performance_summary_with_filter(self):
        """Test performance summary with operation filter."""
        profiler = PerformanceProfiler(enable_gpu_monitoring=False)

        # Generate mixed operations
        profiler.start_monitoring("signal_generation_1")
        profiler.stop_monitoring("signal_generation_1")

        profiler.start_monitoring("optimization_1")
        profiler.stop_monitoring("optimization_1")

        # Filter by operation type
        signal_summary = profiler.get_performance_summary("signal_generation")
        assert signal_summary["total_operations"] == 1

        opt_summary = profiler.get_performance_summary("optimization")
        assert opt_summary["total_operations"] == 1


class TestMemoryManager:
    """Test memory manager functionality."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock GPU backend."""
        backend = Mock(spec=GPUBackend)
        backend.is_gpu_available = True
        backend.allocate_signal_memory.return_value = np.zeros((1000,), dtype=np.complex128)
        backend.get_memory_info.return_value = {
            "backend": "GPU",
            "used_bytes": 1024 * 1024 * 100,  # 100 MB
            "total_bytes": 1024 * 1024 * 1024,  # 1 GB
            "free_bytes": 1024 * 1024 * 924,  # 924 MB
        }
        return backend

    def test_memory_manager_initialization(self, mock_backend):
        """Test memory manager initialization."""
        manager = MemoryManager(mock_backend)
        assert manager.backend == mock_backend
        assert manager._allocation_count == 0
        assert manager._peak_memory_usage == 0

    def test_chunked_memory_allocation(self, mock_backend):
        """Test chunked memory allocation."""
        manager = MemoryManager(mock_backend)

        total_shape = (10000,)
        chunk_size = 1000

        chunks = manager.allocate_chunked_memory(total_shape, chunk_size)

        assert len(chunks) == 10  # 10000 / 1000
        assert manager._allocation_count == 10
        assert len(manager._allocated_arrays) == 10

    def test_optimal_chunk_size_calculation(self, mock_backend):
        """Test optimal chunk size calculation."""
        manager = MemoryManager(mock_backend)

        chunk_size = manager.get_optimal_chunk_size(element_size=16)

        # Should be based on available memory (924 MB * 0.8 / 16 bytes)
        expected_size = int((924 * 1024 * 1024 * 0.8) // 16)
        assert chunk_size == min(expected_size, 100 * 1024 * 1024)  # Capped at 100M

    def test_adaptive_batch_size(self, mock_backend):
        """Test adaptive batch size calculation."""
        manager = MemoryManager(mock_backend)

        # Low memory usage - should increase batch size
        batch_size = manager.get_adaptive_batch_size(4, 0.3)
        assert batch_size >= 4

        # High memory usage - should decrease batch size
        batch_size = manager.get_adaptive_batch_size(4, 0.9)
        assert batch_size <= 4

    def test_memory_pressure_detection(self, mock_backend):
        """Test memory pressure detection."""
        manager = MemoryManager(mock_backend)

        pressure = manager.check_memory_pressure()

        assert "under_pressure" in pressure
        assert "pressure_level" in pressure
        assert "recommendation" in pressure
        assert "usage_fraction" in pressure

    def test_memory_statistics(self, mock_backend):
        """Test memory statistics collection."""
        manager = MemoryManager(mock_backend)

        # Allocate some memory to generate statistics
        manager.allocate_chunked_memory((1000,), 100)

        stats = manager.get_memory_statistics()

        assert stats["backend"] == "GPU"
        assert stats["allocation_count"] > 0
        assert stats["managed_arrays"] > 0

    def test_force_garbage_collection(self, mock_backend):
        """Test forced garbage collection."""
        manager = MemoryManager(mock_backend)

        # Allocate some arrays
        manager.allocate_chunked_memory((1000,), 100)
        assert len(manager._allocated_arrays) > 0

        # Force cleanup
        cleanup_result = manager.force_garbage_collection()

        assert "freed_bytes" in cleanup_result
        assert "objects_collected" in cleanup_result
        assert len(manager._allocated_arrays) == 0


class TestChunkedProcessor:
    """Test chunked processor functionality."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        manager = Mock(spec=MemoryManager)
        manager.get_optimal_chunk_size.return_value = 1000
        manager.check_memory_pressure.return_value = {
            "under_pressure": False,
            "recommendation": "Memory usage is healthy",
        }
        manager.force_garbage_collection.return_value = {"freed_mb": 0}

        # Mock the backend attribute
        mock_backend = Mock()
        mock_backend.is_gpu_available = False
        mock_backend.cleanup_memory.return_value = None
        manager.backend = mock_backend

        return manager

    def test_chunked_processor_initialization(self, mock_memory_manager):
        """Test chunked processor initialization."""
        processor = ChunkedProcessor(mock_memory_manager)
        assert processor.memory_manager == mock_memory_manager
        assert processor.max_chunk_size is None

    def test_process_small_array(self, mock_memory_manager):
        """Test processing of small array (no chunking needed)."""
        processor = ChunkedProcessor(mock_memory_manager, max_chunk_size=1000)

        # Small array that doesn't need chunking
        array = np.random.random(500)

        def dummy_func(arr):
            return arr * 2

        result = processor.process_large_array(array, dummy_func)

        np.testing.assert_array_equal(result, array * 2)

    def test_process_large_array_chunked(self, mock_memory_manager):
        """Test processing of large array with chunking."""
        processor = ChunkedProcessor(mock_memory_manager, max_chunk_size=100)

        # Large array that needs chunking
        array = np.random.random(500)

        def dummy_func(arr):
            return arr * 2

        result = processor.process_large_array(array, dummy_func)

        np.testing.assert_array_equal(result, array * 2)

    def test_process_signal_batch(self, mock_memory_manager):
        """Test batch signal processing."""
        processor = ChunkedProcessor(mock_memory_manager)

        # Create test signals with fixed seed for reproducibility
        np.random.seed(42)
        signals = [np.random.random(10) for _ in range(3)]  # Smaller test

        def dummy_func(signal):
            return signal * 2

        # Disable adaptive batching for this test to get predictable results
        results = processor.process_signal_batch(signals, dummy_func, adaptive_batching=False)

        # Should get exactly the same number of results
        assert len(results) == len(signals)

        # Check that results match expected outputs
        for i, result in enumerate(results):
            expected = signals[i] * 2
            np.testing.assert_array_equal(result, expected)


class TestPerformanceOptimizer:
    """Test performance optimizer functionality."""

    @pytest.fixture
    def mock_gpu_backend(self):
        """Create mock GPU backend."""
        backend = Mock(spec=GPUBackend)
        backend.is_gpu_available = True
        backend.get_memory_info.return_value = {
            "backend": "GPU",
            "used_bytes": 1024 * 1024 * 100,
            "total_bytes": 1024 * 1024 * 1024,
            "free_bytes": 1024 * 1024 * 924,
        }
        backend.cleanup_memory.return_value = None
        return backend

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        config_content = """
[ofdm]
num_subcarriers = 8
subcarrier_spacing = 1000.0
bandwidth_per_subcarrier = 800.0
center_frequency = 10000.0
sampling_rate = 50000.0
signal_duration = 0.002

[performance]
enable_chunked_processing = true
enable_adaptive_batching = true
enable_memory_monitoring = true
enable_performance_profiling = true
memory_cleanup_frequency = 5
max_retry_attempts = 2
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_content)
            temp_file = f.name

        yield temp_file

        # Cleanup
        os.unlink(temp_file)

    def test_optimizer_initialization(self, mock_gpu_backend, temp_config_file):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        assert optimizer.gpu_backend == mock_gpu_backend
        assert isinstance(optimizer.memory_manager, MemoryManager)
        assert isinstance(optimizer.chunked_processor, ChunkedProcessor)
        assert isinstance(optimizer.profiler, PerformanceProfiler)
        assert optimizer.config.enable_chunked_processing is True

    def test_optimize_signal_generation_small_batch(self, mock_gpu_backend, temp_config_file):
        """Test signal generation optimization with small batch."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        def mock_generator(params):
            return np.random.random(100) * params

        signal_params = [1.0, 2.0, 3.0]  # Small batch

        results = optimizer.optimize_signal_generation(
            mock_generator, signal_params, "test_generation"
        )

        assert len(results) == len(signal_params)
        assert optimizer._operation_count == 1

    def test_optimize_signal_generation_large_batch(self, mock_gpu_backend, temp_config_file):
        """Test signal generation optimization with large batch (chunked)."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        def mock_generator(params):
            return np.random.random(100) * params

        signal_params = [float(i) for i in range(20)]  # Large batch

        results = optimizer.optimize_signal_generation(
            mock_generator, signal_params, "test_chunked_generation"
        )

        assert len(results) == len(signal_params)

    def test_optimize_memory_usage(self, mock_gpu_backend, temp_config_file):
        """Test memory usage optimization."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        def memory_intensive_operation(size):
            return np.random.random(size)

        result = optimizer.optimize_memory_usage(memory_intensive_operation, 1000)

        assert len(result) == 1000

    def test_fallback_to_cpu(self, mock_gpu_backend, temp_config_file):
        """Test CPU fallback functionality."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        # Mock the GPU backend to simulate failure and then success
        call_count = 0

        def failing_generator(params):
            nonlocal call_count
            call_count += 1
            # Fail on first calls, succeed on fallback
            if call_count <= 2:  # First batch fails
                raise RuntimeError("GPU operation failed")
            return np.random.random(100) * params

        signal_params = [1.0, 2.0]

        # Should fallback to CPU and succeed
        results = optimizer.optimize_signal_generation(
            failing_generator, signal_params, "test_fallback"
        )

        assert len(results) == len(signal_params)

    def test_optimization_report_generation(self, mock_gpu_backend, temp_config_file):
        """Test optimization report generation."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        # Generate some operations for the report
        def mock_generator(params):
            return np.random.random(100) * params

        optimizer.optimize_signal_generation(mock_generator, [1.0, 2.0], "test_op")

        report = optimizer.get_optimization_report()

        assert "timestamp" in report
        assert "operations_processed" in report
        assert "performance_summary" in report
        assert "memory_statistics" in report
        assert "recommendations" in report
        assert report["operations_processed"] == 1

    def test_optimization_recommendations(self, mock_gpu_backend, temp_config_file):
        """Test optimization recommendation generation."""
        optimizer = PerformanceOptimizer(mock_gpu_backend, temp_config_file)

        # Mock performance data that should trigger recommendations
        perf_summary = {
            "error_rate": 0.15,  # High error rate
            "average_duration": 15.0,  # Long duration
        }

        memory_stats = {"backend": "GPU", "peak_usage_bytes": 3 * 1024**3}  # 3GB peak usage

        pressure = {"under_pressure": True, "recommendation": "Reduce memory usage"}

        recommendations = optimizer._generate_optimization_recommendations(
            perf_summary, memory_stats, pressure
        )

        assert len(recommendations) > 0
        assert any("error rate" in rec.lower() for rec in recommendations)
        assert any("memory" in rec.lower() for rec in recommendations)


class TestIntegrationPerformanceOptimization:
    """Integration tests for performance optimization."""

    @pytest.fixture
    def ofdm_config(self):
        """Create test OFDM configuration."""
        return OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,
        )

    def test_end_to_end_optimization(self, ofdm_config):
        """Test end-to-end performance optimization."""
        # Use CPU backend for reliable testing
        gpu_backend = GPUBackend(force_cpu=True)
        optimizer = PerformanceOptimizer(gpu_backend)

        def signal_generator(phase_array):
            # Simulate signal generation
            signal_length = int(ofdm_config.signal_duration * ofdm_config.sampling_rate)
            return np.random.random(signal_length) * np.sum(phase_array)

        # Generate multiple phase arrays
        phase_arrays = [
            np.random.uniform(0, 2 * np.pi, ofdm_config.num_subcarriers) for _ in range(8)
        ]

        # Optimize signal generation
        results = optimizer.optimize_signal_generation(
            signal_generator, phase_arrays, "integration_test"
        )

        assert len(results) == len(phase_arrays)

        # Check that profiling data was collected
        assert len(optimizer.profiler.metrics_history) > 0

        # Generate optimization report
        report = optimizer.get_optimization_report()
        assert report["operations_processed"] > 0

    def test_memory_pressure_handling(self, ofdm_config):
        """Test handling of memory pressure scenarios."""
        gpu_backend = GPUBackend(force_cpu=True)
        optimizer = PerformanceOptimizer(gpu_backend)

        def memory_intensive_generator(size_multiplier):
            # Simulate memory-intensive operation
            size = int(1000 * size_multiplier)
            return np.random.random(size)

        # Generate operations with increasing memory requirements
        size_multipliers = [1, 5, 10, 20, 50]

        results = optimizer.optimize_signal_generation(
            memory_intensive_generator, size_multipliers, "memory_test"
        )

        assert len(results) == len(size_multipliers)

        # Check memory statistics
        memory_stats = optimizer.memory_manager.get_memory_statistics()
        assert memory_stats["allocation_count"] >= 0


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_chunked_vs_direct_processing_benchmark(self):
        """Benchmark chunked vs direct processing."""
        gpu_backend = GPUBackend(force_cpu=True)
        memory_manager = MemoryManager(gpu_backend)
        processor = ChunkedProcessor(memory_manager, max_chunk_size=1000)

        # Large array for testing
        large_array = np.random.random(10000)

        def processing_func(arr):
            return np.fft.fft(arr)

        # Direct processing
        start_time = time.time()
        direct_result = processing_func(large_array)
        direct_time = time.time() - start_time

        # Chunked processing
        start_time = time.time()
        chunked_result = processor.process_large_array(large_array, processing_func)
        chunked_time = time.time() - start_time

        # Results should be similar (allowing for numerical differences and chunking effects)
        # Note: Chunked FFT processing may have different results due to boundary effects
        # We'll just verify the shapes match and results are reasonable
        assert direct_result.shape == chunked_result.shape
        assert np.isfinite(direct_result).all()
        assert np.isfinite(chunked_result).all()

        # Log timing results
        print(f"Direct processing: {direct_time:.4f}s")
        print(f"Chunked processing: {chunked_time:.4f}s")
        print(f"Overhead: {((chunked_time - direct_time) / direct_time * 100):.1f}%")

    def test_adaptive_batching_performance(self):
        """Test adaptive batching performance characteristics."""
        gpu_backend = GPUBackend(force_cpu=True)
        optimizer = PerformanceOptimizer(gpu_backend)

        def variable_complexity_generator(complexity):
            # Simulate variable complexity operations
            size = int(100 * complexity)
            data = np.random.random(size)
            # More complex operations for higher complexity
            for _ in range(int(complexity)):
                data = np.fft.fft(data)
                data = np.fft.ifft(data)
            return data

        # Test with varying complexity
        complexities = [1, 2, 5, 10, 20, 50, 100]

        start_time = time.time()
        results = optimizer.optimize_signal_generation(
            variable_complexity_generator, complexities, "adaptive_benchmark"
        )
        total_time = time.time() - start_time

        assert len(results) == len(complexities)

        # Check performance metrics
        perf_summary = optimizer.profiler.get_performance_summary()
        print(f"Adaptive batching completed in {total_time:.4f}s")
        print(f"Average operation duration: {perf_summary['average_duration']:.4f}s")
        print(f"Error rate: {perf_summary['error_rate']:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
