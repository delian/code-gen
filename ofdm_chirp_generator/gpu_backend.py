"""
GPU backend implementation with CuPy integration and CPU fallback.

This module provides GPU acceleration for OFDM signal generation using CuPy,
with graceful fallback to NumPy CPU computation when GPU is unavailable.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config_manager import ConfigurationError, get_config
from .error_handling import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    GPUError,
    MemoryError,
    create_error_context,
    with_error_handling,
)

# Try to import CuPy, fall back gracefully if not available
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUBackend:
    """GPU computation backend with automatic CPU fallback.

    This class provides a unified interface for GPU-accelerated computations
    using CuPy, with automatic fallback to NumPy when GPU is unavailable.
    """

    def __init__(self, force_cpu: Optional[bool] = None, config_file: Optional[str] = None):
        """Initialize GPU backend.

        Args:
            force_cpu: If True, force CPU computation even if GPU is available (loads from config if None)
            config_file: Path to configuration file (uses default if None)
        """
        self._gpu_available = False
        self._memory_pool = None
        self._device_id = None
        self._error_handler = ErrorHandler()

        # Load GPU configuration
        try:
            config_manager = get_config(config_file)
            gpu_config = config_manager.get_gpu_config()

            if force_cpu is None:
                self._force_cpu = not gpu_config["enable_gpu"]
            else:
                self._force_cpu = force_cpu

            self._memory_limit = config_manager.get_memory_limit_bytes()
            self._cleanup_after_operations = gpu_config["cleanup_memory_after_operations"]

            logger.info(
                f"Loaded GPU configuration: enable_gpu={gpu_config['enable_gpu']}, memory_limit={self._memory_limit/1024**3:.1f}GB"
            )
        except (ConfigurationError, Exception) as e:
            context = create_error_context("gpu_config_load", "GPUBackend", config_file=config_file)
            self._error_handler.handle_error(e, context)
            logger.warning(f"Could not load GPU configuration: {e}. Using defaults.")
            self._force_cpu = force_cpu or False
            self._memory_limit = 4 * 1024 * 1024 * 1024  # 4GB default
            self._cleanup_after_operations = True

        if not self._force_cpu:
            self._gpu_available = self.initialize_gpu()

        if not self._gpu_available:
            logger.info("Using CPU backend (NumPy)")
        else:
            logger.info(f"Using GPU backend (CuPy) on device {self._device_id}")

    def initialize_gpu(self) -> bool:
        """Initialize GPU and check availability.

        Returns:
            True if GPU is available and initialized successfully, False otherwise
        """
        if not CUPY_AVAILABLE:
            error = GPUError(
                "CuPy not available. Install cupy-cuda11x for GPU acceleration.",
                "cupy_import",
                ErrorSeverity.HIGH,
            )
            context = create_error_context("gpu_initialization", "GPUBackend")
            self._error_handler.handle_error(error, context)
            return False

        try:
            # Check if CUDA is available
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                error = GPUError("No CUDA devices found", "cuda_device_check", ErrorSeverity.HIGH)
                context = create_error_context("gpu_initialization", "GPUBackend")
                self._error_handler.handle_error(error, context)
                return False

            # Get current device
            self._device_id = cp.cuda.Device().id

            # Test basic GPU operation
            test_array = cp.array([1, 2, 3])
            result = cp.sum(test_array)

            # Verify result is correct
            if result.get() != 6:
                raise GPUError("GPU computation test failed", "gpu_test", ErrorSeverity.HIGH)

            # Initialize memory pool for efficient memory management
            self._memory_pool = cp.get_default_memory_pool()

            # Check available memory
            mem_info = cp.cuda.Device().mem_info
            if mem_info[0] < 100 * 1024 * 1024:  # Less than 100MB free
                logger.warning(f"Low GPU memory available: {mem_info[0] / 1024**2:.1f} MB")

            logger.info(f"GPU initialized successfully on device {self._device_id}")
            return True

        except Exception as e:
            if isinstance(e, GPUError):
                gpu_error = e
            else:
                gpu_error = GPUError(
                    f"GPU initialization failed: {e}", "gpu_init", ErrorSeverity.HIGH
                )

            context = create_error_context(
                "gpu_initialization",
                "GPUBackend",
                device_count=device_count if "device_count" in locals() else 0,
            )
            self._error_handler.handle_error(gpu_error, context)
            return False

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for computation."""
        return self._gpu_available and not self._force_cpu

    @property
    def device_info(self) -> dict:
        """Get information about the current compute device."""
        if self.is_gpu_available:
            device = cp.cuda.Device(self._device_id)
            return {
                "backend": "GPU",
                "device_id": self._device_id,
                "device_name": device.attributes["Name"].decode(),
                "memory_total": device.mem_info[1],
                "memory_free": device.mem_info[0],
                "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            }
        else:
            return {
                "backend": "CPU",
                "device_name": "CPU",
                "memory_total": None,
                "memory_free": None,
            }

    def allocate_signal_memory(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.complex128
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Allocate memory for signal arrays.

        Args:
            shape: Shape of the array to allocate
            dtype: Data type for the array

        Returns:
            Allocated array (CuPy array if GPU available, NumPy array otherwise)
        """
        # Calculate memory requirements
        element_size = np.dtype(dtype).itemsize
        total_elements = np.prod(shape)
        required_bytes = total_elements * element_size

        try:
            if self.is_gpu_available:
                # Check available memory before allocation
                mem_info = cp.cuda.Device().mem_info
                available_bytes = mem_info[0]

                if required_bytes > available_bytes:
                    error = MemoryError(
                        f"Insufficient GPU memory: required {required_bytes/1024**2:.1f} MB, "
                        f"available {available_bytes/1024**2:.1f} MB",
                        required_bytes,
                        available_bytes,
                        ErrorSeverity.HIGH,
                    )
                    context = create_error_context(
                        "memory_allocation",
                        "GPUBackend",
                        shape=shape,
                        dtype=str(dtype),
                        required_mb=required_bytes / 1024**2,
                    )
                    self._error_handler.handle_error(error, context)

                    # Force CPU fallback
                    logger.warning("GPU memory insufficient, falling back to CPU")
                    self._gpu_available = False
                    return np.zeros(shape, dtype=dtype)

                return cp.zeros(shape, dtype=dtype)
            else:
                return np.zeros(shape, dtype=dtype)

        except Exception as e:
            if isinstance(e, (MemoryError, GPUError)):
                gpu_error = e
            else:
                gpu_error = MemoryError(
                    f"GPU memory allocation failed: {e}", required_bytes, 0, ErrorSeverity.HIGH
                )

            context = create_error_context(
                "memory_allocation",
                "GPUBackend",
                shape=shape,
                dtype=str(dtype),
                required_mb=required_bytes / 1024**2,
            )
            self._error_handler.handle_error(gpu_error, context)

            # Fallback to CPU
            logger.warning(f"GPU memory allocation failed: {e}. Falling back to CPU.")
            self._gpu_available = False
            return np.zeros(shape, dtype=dtype)

    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """Transfer array to GPU if available.

        Args:
            array: NumPy array to transfer

        Returns:
            CuPy array if GPU available, original NumPy array otherwise
        """
        if self.is_gpu_available:
            try:
                return cp.asarray(array)
            except Exception as e:
                logger.warning(f"GPU transfer failed: {e}. Using CPU array.")
                return array
        return array

    def to_cpu(self, array: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
        """Transfer array to CPU.

        Args:
            array: Array to transfer (CuPy or NumPy)

        Returns:
            NumPy array
        """
        if self.is_gpu_available and hasattr(array, "get"):
            return array.get()
        return np.asarray(array)

    @with_error_handling("fft_computation", "GPUBackend")
    def perform_fft(
        self, signal: Union[np.ndarray, "cp.ndarray"]
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Perform FFT on signal array.

        Args:
            signal: Input signal array

        Returns:
            FFT of the input signal
        """
        try:
            if self.is_gpu_available and hasattr(signal, "device"):
                # Check for NaN or Inf values before FFT
                if cp.any(cp.isnan(signal)) or cp.any(cp.isinf(signal)):
                    raise GPUError(
                        "Input signal contains NaN or Inf values", "fft_input_validation"
                    )

                result = cp.fft.fft(signal)

                # Validate result
                if cp.any(cp.isnan(result)) or cp.any(cp.isinf(result)):
                    raise GPUError(
                        "FFT computation produced NaN or Inf values", "fft_output_validation"
                    )

                return result
            else:
                signal_cpu = self.to_cpu(signal)

                # Check for NaN or Inf values
                if np.any(np.isnan(signal_cpu)) or np.any(np.isinf(signal_cpu)):
                    raise GPUError(
                        "Input signal contains NaN or Inf values", "fft_input_validation"
                    )

                result = np.fft.fft(signal_cpu)

                # Validate result
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    raise GPUError(
                        "FFT computation produced NaN or Inf values", "fft_output_validation"
                    )

                return result

        except Exception as e:
            if not isinstance(e, GPUError):
                gpu_error = GPUError(
                    f"GPU FFT failed: {e}", "fft_computation", ErrorSeverity.MEDIUM
                )
            else:
                gpu_error = e

            context = create_error_context(
                "fft_computation",
                "GPUBackend",
                signal_shape=signal.shape,
                signal_dtype=str(signal.dtype),
            )
            self._error_handler.handle_error(gpu_error, context)

            # Fallback to CPU
            logger.warning(f"GPU FFT failed: {e}. Falling back to CPU.")
            signal_cpu = self.to_cpu(signal)
            return np.fft.fft(signal_cpu)

    def perform_ifft(
        self, signal: Union[np.ndarray, "cp.ndarray"]
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Perform inverse FFT on signal array.

        Args:
            signal: Input signal array

        Returns:
            Inverse FFT of the input signal
        """
        try:
            if self.is_gpu_available and hasattr(signal, "device"):
                return cp.fft.ifft(signal)
            else:
                return np.fft.ifft(self.to_cpu(signal))
        except Exception as e:
            logger.warning(f"GPU IFFT failed: {e}. Falling back to CPU.")
            return np.fft.ifft(self.to_cpu(signal))

    def compute_correlation(
        self, sig1: Union[np.ndarray, "cp.ndarray"], sig2: Union[np.ndarray, "cp.ndarray"]
    ) -> float:
        """Compute cross-correlation between two signals.

        Args:
            sig1: First signal array
            sig2: Second signal array

        Returns:
            Maximum cross-correlation value
        """
        try:
            if self.is_gpu_available:
                # Ensure both signals are on GPU
                sig1_gpu = self.to_gpu(sig1)
                sig2_gpu = self.to_gpu(sig2)

                # Compute cross-correlation using FFT
                sig1_fft = cp.fft.fft(sig1_gpu)
                sig2_fft = cp.fft.fft(sig2_gpu)
                correlation = cp.fft.ifft(sig1_fft * cp.conj(sig2_fft))

                # Return maximum correlation value
                return float(cp.max(cp.abs(correlation)).get())
            else:
                # CPU computation
                sig1_cpu = self.to_cpu(sig1)
                sig2_cpu = self.to_cpu(sig2)

                sig1_fft = np.fft.fft(sig1_cpu)
                sig2_fft = np.fft.fft(sig2_cpu)
                correlation = np.fft.ifft(sig1_fft * np.conj(sig2_fft))

                return float(np.max(np.abs(correlation)))

        except Exception as e:
            logger.warning(f"GPU correlation failed: {e}. Falling back to CPU.")
            # Fallback to CPU
            sig1_cpu = self.to_cpu(sig1)
            sig2_cpu = self.to_cpu(sig2)

            sig1_fft = np.fft.fft(sig1_cpu)
            sig2_fft = np.fft.fft(sig2_cpu)
            correlation = np.fft.ifft(sig1_fft * np.conj(sig2_fft))

            return float(np.max(np.abs(correlation)))

    def get_memory_info(self) -> dict:
        """Get current memory usage information.

        Returns:
            Dictionary with memory usage statistics
        """
        if self.is_gpu_available and self._memory_pool:
            return {
                "backend": "GPU",
                "used_bytes": self._memory_pool.used_bytes(),
                "total_bytes": self._memory_pool.total_bytes(),
                "free_bytes": cp.cuda.Device().mem_info[0],
            }
        else:
            return {"backend": "CPU", "used_bytes": None, "total_bytes": None, "free_bytes": None}

    def cleanup_memory(self) -> None:
        """Clean up GPU memory and free unused allocations."""
        if self.is_gpu_available:
            try:
                # Free unused memory blocks
                if self._memory_pool:
                    self._memory_pool.free_all_blocks()

                # Force garbage collection
                cp.cuda.Stream.null.synchronize()

                logger.debug("GPU memory cleanup completed")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_memory()


class MemoryManager:
    """Advanced memory management utilities for GPU arrays."""

    def __init__(self, backend: GPUBackend):
        """Initialize memory manager.

        Args:
            backend: GPU backend instance
        """
        self.backend = backend
        self._allocated_arrays = []
        self._memory_usage_history = []
        self._peak_memory_usage = 0
        self._allocation_count = 0

    def allocate_chunked_memory(
        self, total_shape: Tuple[int, ...], chunk_size: int, dtype: np.dtype = np.complex128
    ) -> list:
        """Allocate memory in chunks to avoid GPU memory issues.

        Args:
            total_shape: Total shape needed
            chunk_size: Size of each chunk
            dtype: Data type

        Returns:
            List of allocated array chunks
        """
        chunks = []
        total_elements = np.prod(total_shape)

        for i in range(0, total_elements, chunk_size):
            chunk_elements = min(chunk_size, total_elements - i)
            chunk_shape = (
                (chunk_elements,) if len(total_shape) == 1 else (chunk_elements, *total_shape[1:])
            )

            chunk = self.backend.allocate_signal_memory(chunk_shape, dtype)
            chunks.append(chunk)
            self._allocated_arrays.append(chunk)
            self._allocation_count += 1

            # Track memory usage
            self._update_memory_tracking()

        return chunks

    def get_optimal_chunk_size(self, element_size: int = 16, safety_factor: float = 0.8) -> int:
        """Calculate optimal chunk size based on available memory.

        Args:
            element_size: Size of each element in bytes
            safety_factor: Fraction of available memory to use (0.0-1.0)

        Returns:
            Optimal chunk size in elements
        """
        if self.backend.is_gpu_available:
            memory_info = self.backend.get_memory_info()
            free_bytes = memory_info.get("free_bytes", 0)

            # Use safety factor of available memory
            usable_bytes = int(free_bytes * safety_factor)
            chunk_size = max(1024, usable_bytes // element_size)

            # Ensure chunk size doesn't exceed reasonable limits
            max_chunk_size = 100 * 1024 * 1024  # 100M elements max
            chunk_size = min(chunk_size, max_chunk_size)

            logger.debug(
                f"Calculated optimal chunk size: {chunk_size} elements ({chunk_size * element_size / 1024**2:.1f} MB)"
            )
            return chunk_size
        else:
            # For CPU, use a reasonable default based on system memory
            return 1024 * 1024  # 1M elements

    def get_adaptive_batch_size(self, base_batch_size: int, current_memory_usage: float) -> int:
        """Calculate adaptive batch size based on current memory usage.

        Args:
            base_batch_size: Base batch size to start from
            current_memory_usage: Current memory usage as fraction (0.0-1.0)

        Returns:
            Adjusted batch size
        """
        if current_memory_usage < 0.5:
            # Low memory usage, can increase batch size
            return min(base_batch_size * 2, base_batch_size * 4)
        elif current_memory_usage < 0.7:
            # Moderate usage, keep current size
            return base_batch_size
        elif current_memory_usage < 0.85:
            # High usage, reduce batch size
            return max(1, base_batch_size // 2)
        else:
            # Very high usage, minimize batch size
            return 1

    def _update_memory_tracking(self) -> None:
        """Update memory usage tracking."""
        memory_info = self.backend.get_memory_info()
        if memory_info["backend"] == "GPU" and memory_info["used_bytes"] is not None:
            current_usage = memory_info["used_bytes"]
            self._peak_memory_usage = max(self._peak_memory_usage, current_usage)

            # Keep history of last 100 measurements
            self._memory_usage_history.append(current_usage)
            if len(self._memory_usage_history) > 100:
                self._memory_usage_history.pop(0)

    def get_memory_statistics(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        memory_info = self.backend.get_memory_info()

        stats = {
            "backend": memory_info["backend"],
            "current_used_bytes": memory_info.get("used_bytes", 0) or 0,
            "current_free_bytes": memory_info.get("free_bytes", 0) or 0,
            "peak_usage_bytes": self._peak_memory_usage,
            "allocation_count": self._allocation_count,
            "managed_arrays": len(self._allocated_arrays),
        }

        if self._memory_usage_history:
            stats.update(
                {
                    "average_usage_bytes": np.mean(self._memory_usage_history),
                    "usage_std_bytes": np.std(self._memory_usage_history),
                    "min_usage_bytes": np.min(self._memory_usage_history),
                    "max_usage_bytes": np.max(self._memory_usage_history),
                }
            )

        return stats

    def check_memory_pressure(self) -> Dict[str, Union[bool, float, str]]:
        """Check if system is under memory pressure.

        Returns:
            Dictionary with memory pressure analysis
        """
        memory_info = self.backend.get_memory_info()

        if memory_info["backend"] == "CPU":
            return {
                "under_pressure": False,
                "pressure_level": "low",
                "usage_fraction": 0.0,
                "recommendation": "CPU backend - no GPU memory pressure",
            }

        total_bytes = memory_info.get("total_bytes", 0) or 0
        free_bytes = memory_info.get("free_bytes", 0) or 0

        if total_bytes == 0:
            return {
                "under_pressure": True,
                "pressure_level": "critical",
                "usage_fraction": 1.0,
                "recommendation": "Cannot determine GPU memory status",
            }

        usage_fraction = 1.0 - (free_bytes / total_bytes)

        if usage_fraction > 0.9:
            pressure_level = "critical"
            under_pressure = True
            recommendation = "Reduce batch size immediately, consider CPU fallback"
        elif usage_fraction > 0.8:
            pressure_level = "high"
            under_pressure = True
            recommendation = "Reduce batch size, increase cleanup frequency"
        elif usage_fraction > 0.7:
            pressure_level = "moderate"
            under_pressure = True
            recommendation = "Monitor closely, consider smaller chunks"
        else:
            pressure_level = "low"
            under_pressure = False
            recommendation = "Memory usage is healthy"

        return {
            "under_pressure": under_pressure,
            "pressure_level": pressure_level,
            "usage_fraction": usage_fraction,
            "recommendation": recommendation,
            "free_mb": free_bytes / (1024**2),
            "total_mb": total_bytes / (1024**2),
        }

    def force_garbage_collection(self) -> Dict[str, Union[int, float]]:
        """Force garbage collection and return memory freed.

        Returns:
            Dictionary with cleanup results
        """
        # Get memory before cleanup
        memory_before = self.backend.get_memory_info()
        before_used = memory_before.get("used_bytes", 0) or 0

        # Clear managed arrays
        self._allocated_arrays.clear()

        # Force GPU cleanup
        self.backend.cleanup_memory()

        # Force Python garbage collection
        import gc

        collected = gc.collect()

        # Get memory after cleanup
        memory_after = self.backend.get_memory_info()
        after_used = memory_after.get("used_bytes", 0) or 0

        freed_bytes = max(0, before_used - after_used)

        logger.info(
            f"Garbage collection freed {freed_bytes / 1024**2:.1f} MB, collected {collected} objects"
        )

        return {
            "freed_bytes": freed_bytes,
            "freed_mb": freed_bytes / (1024**2),
            "objects_collected": collected,
            "before_used_mb": before_used / (1024**2),
            "after_used_mb": after_used / (1024**2),
        }

    def cleanup_all(self) -> None:
        """Clean up all managed arrays."""
        self._allocated_arrays.clear()
        self.backend.cleanup_memory()


class ChunkedProcessor:
    """Processor for handling large arrays in chunks to manage memory."""

    def __init__(self, memory_manager: MemoryManager, max_chunk_size: Optional[int] = None):
        """Initialize chunked processor.

        Args:
            memory_manager: Memory manager instance
            max_chunk_size: Maximum chunk size in elements (auto-calculated if None)
        """
        self.memory_manager = memory_manager
        self.max_chunk_size = max_chunk_size

    def process_large_array(
        self,
        array: Union[np.ndarray, "cp.ndarray"],
        processing_func: callable,
        chunk_overlap: int = 0,
        **func_kwargs,
    ) -> Union[np.ndarray, "cp.ndarray"]:
        """Process a large array in chunks.

        Args:
            array: Large array to process
            processing_func: Function to apply to each chunk
            chunk_overlap: Number of elements to overlap between chunks
            **func_kwargs: Additional arguments for processing function

        Returns:
            Processed array result
        """
        if self.max_chunk_size is None:
            element_size = array.dtype.itemsize
            self.max_chunk_size = self.memory_manager.get_optimal_chunk_size(element_size)

        array_length = len(array)

        # If array is small enough, process directly
        if array_length <= self.max_chunk_size:
            return processing_func(array, **func_kwargs)

        # Process in chunks
        results = []
        effective_chunk_size = self.max_chunk_size - chunk_overlap

        for start_idx in range(0, array_length, effective_chunk_size):
            end_idx = min(start_idx + self.max_chunk_size, array_length)

            # Extract chunk with overlap
            chunk = array[start_idx:end_idx]

            # Process chunk
            chunk_result = processing_func(chunk, **func_kwargs)

            # Handle overlap removal if needed
            if chunk_overlap > 0 and start_idx > 0:
                chunk_result = chunk_result[chunk_overlap:]

            results.append(chunk_result)

            # Check memory pressure and cleanup if needed
            pressure = self.memory_manager.check_memory_pressure()
            if pressure["under_pressure"]:
                logger.warning(f"Memory pressure detected: {pressure['recommendation']}")
                self.memory_manager.force_garbage_collection()

        # Combine results
        if self.memory_manager.backend.is_gpu_available and hasattr(results[0], "device"):
            return cp.concatenate(results)
        else:
            return np.concatenate(results)

    def process_signal_batch(
        self,
        signals: List[Union[np.ndarray, "cp.ndarray"]],
        processing_func: callable,
        adaptive_batching: bool = True,
        **func_kwargs,
    ) -> List[Union[np.ndarray, "cp.ndarray"]]:
        """Process a batch of signals with adaptive batch sizing.

        Args:
            signals: List of signals to process
            processing_func: Function to apply to each signal
            adaptive_batching: Enable adaptive batch size adjustment
            **func_kwargs: Additional arguments for processing function

        Returns:
            List of processed signals
        """
        results = []
        batch_size = 4  # Initial batch size

        for i in range(0, len(signals), batch_size):
            batch_end = min(i + batch_size, len(signals))
            batch = signals[i:batch_end]

            try:
                # Process batch
                batch_results = []
                for signal in batch:
                    result = processing_func(signal, **func_kwargs)
                    batch_results.append(result)

                results.extend(batch_results)

                # Adaptive batch size adjustment
                if adaptive_batching:
                    memory_info = self.memory_manager.backend.get_memory_info()
                    if memory_info["backend"] == "GPU" and memory_info["total_bytes"]:
                        usage_fraction = 1.0 - (
                            memory_info["free_bytes"] / memory_info["total_bytes"]
                        )
                        batch_size = self.memory_manager.get_adaptive_batch_size(
                            batch_size, usage_fraction
                        )

            except Exception as e:
                logger.warning(f"Batch processing failed: {e}. Reducing batch size.")
                batch_size = max(1, batch_size // 2)

                # Retry with smaller batch
                for signal in batch:
                    try:
                        result = processing_func(signal, **func_kwargs)
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"Individual signal processing failed: {e2}")
                        raise

            # Cleanup between batches
            if i % 10 == 0:  # Every 10 batches
                self.memory_manager.backend.cleanup_memory()

        return results
