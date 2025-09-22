"""
Performance optimization and memory management for OFDM signal generation.

This module provides advanced performance optimization features including
chunked processing, adaptive batch sizing, memory profiling, and performance monitoring.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from .config_manager import ConfigurationError, get_config
from .gpu_backend import ChunkedProcessor, GPUBackend, MemoryManager
from .models import OFDMConfig, SignalSet

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: Dict[str, Any]
    memory_after: Dict[str, Any]
    memory_peak: int
    cpu_usage_percent: float
    gpu_utilization: Optional[float] = None
    throughput_signals_per_sec: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    enable_chunked_processing: bool = True
    enable_adaptive_batching: bool = True
    enable_memory_monitoring: bool = True
    enable_performance_profiling: bool = True
    chunk_overlap_samples: int = 0
    memory_cleanup_frequency: int = 10  # Every N operations
    performance_log_frequency: int = 50  # Every N operations
    memory_pressure_threshold: float = 0.8
    auto_fallback_to_cpu: bool = True
    max_retry_attempts: int = 3


class PerformanceProfiler:
    """Performance profiler for monitoring system resource usage."""

    def __init__(self, enable_gpu_monitoring: bool = True):
        """Initialize performance profiler.

        Args:
            enable_gpu_monitoring: Enable GPU utilization monitoring
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring and CUPY_AVAILABLE
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring_active = False
        self._monitor_thread = None
        self._current_metrics = {}

    def start_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """Start monitoring an operation.

        Args:
            operation_name: Name of the operation being monitored

        Returns:
            Initial system state
        """
        initial_state = {
            "operation_name": operation_name,
            "start_time": time.time(),
            "memory_before": self._get_memory_info(),
            "cpu_before": psutil.cpu_percent(),
            "gpu_before": self._get_gpu_utilization() if self.enable_gpu_monitoring else None,
        }

        self._current_metrics[operation_name] = initial_state
        return initial_state

    def stop_monitoring(
        self,
        operation_name: str,
        throughput_signals: Optional[int] = None,
        error_info: Optional[Tuple[bool, str]] = None,
    ) -> PerformanceMetrics:
        """Stop monitoring an operation and record metrics.

        Args:
            operation_name: Name of the operation
            throughput_signals: Number of signals processed (for throughput calculation)
            error_info: Tuple of (error_occurred, error_message)

        Returns:
            Performance metrics for the operation
        """
        if operation_name not in self._current_metrics:
            raise ValueError(f"No monitoring started for operation '{operation_name}'")

        initial_state = self._current_metrics.pop(operation_name)
        end_time = time.time()
        duration = end_time - initial_state["start_time"]

        # Calculate throughput
        throughput = None
        if throughput_signals and duration > 0:
            throughput = throughput_signals / duration

        # Handle error information
        error_occurred = False
        error_message = None
        if error_info:
            error_occurred, error_message = error_info

        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=initial_state["start_time"],
            end_time=end_time,
            duration=duration,
            memory_before=initial_state["memory_before"],
            memory_after=self._get_memory_info(),
            memory_peak=self._get_peak_memory_usage(),
            cpu_usage_percent=psutil.cpu_percent() - initial_state["cpu_before"],
            gpu_utilization=self._get_gpu_utilization() if self.enable_gpu_monitoring else None,
            throughput_signals_per_sec=throughput,
            error_occurred=error_occurred,
            error_message=error_message,
        )

        self.metrics_history.append(metrics)

        # Log performance summary
        logger.info(
            f"Performance: {operation_name} completed in {duration:.3f}s"
            + (f", throughput: {throughput:.1f} signals/s" if throughput else "")
            + (f", ERROR: {error_message}" if error_occurred else "")
        )

        return metrics

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        # System memory
        memory = psutil.virtual_memory()

        info = {
            "system_total_gb": memory.total / (1024**3),
            "system_available_gb": memory.available / (1024**3),
            "system_used_percent": memory.percent,
            "process_memory_mb": psutil.Process().memory_info().rss / (1024**2),
        }

        # GPU memory if available
        if self.enable_gpu_monitoring:
            try:
                gpu_memory = cp.cuda.Device().mem_info
                info.update(
                    {
                        "gpu_free_mb": gpu_memory[0] / (1024**2),
                        "gpu_total_mb": gpu_memory[1] / (1024**2),
                        "gpu_used_mb": (gpu_memory[1] - gpu_memory[0]) / (1024**2),
                    }
                )
            except Exception:
                pass

        return info

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        if not self.enable_gpu_monitoring:
            return None

        try:
            # This is a simplified approach - in practice you might use nvidia-ml-py
            # For now, we'll estimate based on memory usage
            gpu_memory = cp.cuda.Device().mem_info
            utilization = ((gpu_memory[1] - gpu_memory[0]) / gpu_memory[1]) * 100
            return utilization
        except Exception:
            return None

    def _get_peak_memory_usage(self) -> int:
        """Get peak memory usage during operation."""
        try:
            return psutil.Process().memory_info().peak_wset
        except AttributeError:
            # peak_wset not available on all platforms
            return psutil.Process().memory_info().rss

    def get_performance_summary(self, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary statistics.

        Args:
            operation_filter: Filter metrics by operation name

        Returns:
            Dictionary with performance statistics
        """
        filtered_metrics = self.metrics_history
        if operation_filter:
            filtered_metrics = [
                m for m in self.metrics_history if operation_filter in m.operation_name
            ]

        if not filtered_metrics:
            return {"message": "No metrics available"}

        durations = [m.duration for m in filtered_metrics]
        throughputs = [
            m.throughput_signals_per_sec for m in filtered_metrics if m.throughput_signals_per_sec
        ]
        errors = [m for m in filtered_metrics if m.error_occurred]

        summary = {
            "total_operations": len(filtered_metrics),
            "total_duration": sum(durations),
            "average_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "std_duration": np.std(durations),
            "error_count": len(errors),
            "error_rate": len(errors) / len(filtered_metrics) if filtered_metrics else 0,
        }

        if throughputs:
            summary.update(
                {
                    "average_throughput": np.mean(throughputs),
                    "max_throughput": np.max(throughputs),
                    "min_throughput": np.min(throughputs),
                }
            )

        return summary


class PerformanceOptimizer:
    """Main performance optimization coordinator.

    This class provides comprehensive performance optimization including
    chunked processing, adaptive batching, memory management, and profiling.

    Requirements addressed:
    - 4.4: GPU memory monitoring and management
    - 3.3: Efficient processing of large signal arrays
    """

    def __init__(self, gpu_backend: Optional[GPUBackend] = None, config_file: Optional[str] = None):
        """Initialize performance optimizer.

        Args:
            gpu_backend: GPU backend instance (creates new if None)
            config_file: Path to configuration file
        """
        self.gpu_backend = gpu_backend or GPUBackend()
        self.memory_manager = MemoryManager(self.gpu_backend)
        self.chunked_processor = ChunkedProcessor(self.memory_manager)
        self.profiler = PerformanceProfiler()

        # Load optimization configuration
        try:
            config_manager = get_config(config_file)
            opt_config = config_manager.get_optimization_config()

            self.config = OptimizationConfig(
                enable_chunked_processing=opt_config.get("enable_chunked_processing", True),
                enable_adaptive_batching=opt_config.get("enable_adaptive_batching", True),
                enable_memory_monitoring=opt_config.get("enable_memory_monitoring", True),
                enable_performance_profiling=opt_config.get("enable_performance_profiling", True),
                chunk_overlap_samples=opt_config.get("chunk_overlap_samples", 0),
                memory_cleanup_frequency=opt_config.get("memory_cleanup_frequency", 10),
                performance_log_frequency=opt_config.get("performance_log_frequency", 50),
                memory_pressure_threshold=opt_config.get("memory_pressure_threshold", 0.8),
                auto_fallback_to_cpu=opt_config.get("auto_fallback_to_cpu", True),
                max_retry_attempts=opt_config.get("max_retry_attempts", 3),
            )
            logger.info("Loaded performance optimization configuration")
        except (ConfigurationError, Exception) as e:
            self.config = OptimizationConfig()
            logger.warning(f"Could not load optimization configuration: {e}. Using defaults.")

        self._operation_count = 0

        logger.info(
            f"PerformanceOptimizer initialized: "
            f"chunked={self.config.enable_chunked_processing}, "
            f"adaptive={self.config.enable_adaptive_batching}, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}"
        )

    def optimize_signal_generation(
        self,
        generator_func: Callable,
        signal_params: List[Any],
        operation_name: str = "signal_generation",
    ) -> List[Any]:
        """Optimize signal generation with chunked processing and adaptive batching.

        Args:
            generator_func: Function that generates signals
            signal_params: List of parameters for each signal to generate
            operation_name: Name for performance tracking

        Returns:
            List of generated signals
        """
        self._operation_count += 1

        # Start performance monitoring
        if self.config.enable_performance_profiling:
            self.profiler.start_monitoring(f"{operation_name}_{self._operation_count}")

        try:
            # Check memory pressure before starting
            if self.config.enable_memory_monitoring:
                pressure = self.memory_manager.check_memory_pressure()
                if pressure["under_pressure"]:
                    logger.warning(
                        f"Starting operation under memory pressure: {pressure['recommendation']}"
                    )

            # Use chunked processing for large batches
            if self.config.enable_chunked_processing and len(signal_params) > 10:
                results = self._process_signals_chunked(generator_func, signal_params)
            else:
                results = self._process_signals_batch(generator_func, signal_params)

            # Periodic cleanup
            if self._operation_count % self.config.memory_cleanup_frequency == 0:
                cleanup_stats = self.memory_manager.force_garbage_collection()
                logger.debug(f"Periodic cleanup freed {cleanup_stats['freed_mb']:.1f} MB")

            # Stop performance monitoring
            if self.config.enable_performance_profiling:
                self.profiler.stop_monitoring(
                    f"{operation_name}_{self._operation_count}",
                    throughput_signals=len(signal_params),
                )

            return results

        except Exception as e:
            logger.error(f"Signal generation optimization failed: {e}")

            # Record error in profiler
            if self.config.enable_performance_profiling:
                self.profiler.stop_monitoring(
                    f"{operation_name}_{self._operation_count}", error_info=(True, str(e))
                )

            # Attempt fallback if configured
            if self.config.auto_fallback_to_cpu and self.gpu_backend.is_gpu_available:
                logger.info("Attempting CPU fallback for failed operation")
                return self._fallback_to_cpu(generator_func, signal_params)

            raise

    def _process_signals_chunked(
        self, generator_func: Callable, signal_params: List[Any]
    ) -> List[Any]:
        """Process signals using chunked approach for memory efficiency.

        Args:
            generator_func: Signal generation function
            signal_params: Parameters for signal generation

        Returns:
            List of generated signals
        """
        # Calculate optimal chunk size based on memory
        base_chunk_size = 4
        memory_info = self.memory_manager.backend.get_memory_info()

        if memory_info["backend"] == "GPU" and memory_info["total_bytes"]:
            usage_fraction = 1.0 - (memory_info["free_bytes"] / memory_info["total_bytes"])
            chunk_size = self.memory_manager.get_adaptive_batch_size(
                base_chunk_size, usage_fraction
            )
        else:
            chunk_size = base_chunk_size

        logger.debug(f"Processing {len(signal_params)} signals in chunks of {chunk_size}")

        results = []
        for i in range(0, len(signal_params), chunk_size):
            chunk_params = signal_params[i : i + chunk_size]

            # Process chunk with retry logic
            chunk_results = self._process_chunk_with_retry(generator_func, chunk_params)
            results.extend(chunk_results)

            # Monitor memory pressure between chunks
            if self.config.enable_memory_monitoring:
                pressure = self.memory_manager.check_memory_pressure()
                if pressure["under_pressure"]:
                    logger.warning(
                        f"Memory pressure during chunked processing: {pressure['recommendation']}"
                    )
                    self.memory_manager.force_garbage_collection()

                    # Reduce chunk size for next iteration
                    chunk_size = max(1, chunk_size // 2)

        return results

    def _process_signals_batch(
        self, generator_func: Callable, signal_params: List[Any]
    ) -> List[Any]:
        """Process signals using adaptive batching.

        Args:
            generator_func: Signal generation function
            signal_params: Parameters for signal generation

        Returns:
            List of generated signals
        """
        if self.config.enable_adaptive_batching:
            return self.chunked_processor.process_signal_batch(
                signal_params, generator_func, adaptive_batching=True
            )
        else:
            # Simple sequential processing
            return [generator_func(params) for params in signal_params]

    def _process_chunk_with_retry(
        self, generator_func: Callable, chunk_params: List[Any]
    ) -> List[Any]:
        """Process a chunk with retry logic for robustness.

        Args:
            generator_func: Signal generation function
            chunk_params: Parameters for this chunk

        Returns:
            List of generated signals for this chunk
        """
        for attempt in range(self.config.max_retry_attempts):
            try:
                return [generator_func(params) for params in chunk_params]

            except Exception as e:
                logger.warning(f"Chunk processing attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retry_attempts - 1:
                    # Clean up and retry
                    self.memory_manager.force_garbage_collection()
                    time.sleep(0.1)  # Brief pause
                else:
                    # Final attempt failed
                    raise

    def _fallback_to_cpu(self, generator_func: Callable, signal_params: List[Any]) -> List[Any]:
        """Fallback to CPU processing when GPU fails.

        Args:
            generator_func: Signal generation function
            signal_params: Parameters for signal generation

        Returns:
            List of generated signals using CPU
        """
        logger.info("Falling back to CPU processing")

        # Temporarily disable GPU if available
        original_gpu_state = getattr(self.gpu_backend, "_gpu_available", False)
        if hasattr(self.gpu_backend, "_gpu_available"):
            self.gpu_backend._gpu_available = False

        try:
            # Process with CPU
            results = [generator_func(params) for params in signal_params]
            logger.info(f"CPU fallback successful for {len(signal_params)} signals")
            return results

        finally:
            # Restore GPU state if it was available
            if hasattr(self.gpu_backend, "_gpu_available"):
                self.gpu_backend._gpu_available = original_gpu_state

    def optimize_memory_usage(self, target_operation: Callable, *args, **kwargs) -> Any:
        """Optimize memory usage for a specific operation.

        Args:
            target_operation: Operation to optimize
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the optimized operation
        """
        # Pre-operation cleanup
        initial_cleanup = self.memory_manager.force_garbage_collection()
        logger.debug(f"Pre-operation cleanup freed {initial_cleanup['freed_mb']:.1f} MB")

        # Monitor memory during operation
        memory_before = self.memory_manager.backend.get_memory_info()

        try:
            result = target_operation(*args, **kwargs)

            # Post-operation analysis
            memory_after = self.memory_manager.backend.get_memory_info()

            if memory_before["backend"] == "GPU" and memory_after["backend"] == "GPU":
                memory_used = memory_after["used_bytes"] - memory_before["used_bytes"]
                logger.debug(f"Operation used {memory_used / 1024**2:.1f} MB GPU memory")

            return result

        except Exception as e:
            # Emergency cleanup on error
            emergency_cleanup = self.memory_manager.force_garbage_collection()
            logger.warning(
                f"Emergency cleanup after error freed {emergency_cleanup['freed_mb']:.1f} MB"
            )
            raise

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization performance report.

        Returns:
            Dictionary with optimization statistics and recommendations
        """
        # Performance statistics
        perf_summary = self.profiler.get_performance_summary()

        # Memory statistics
        memory_stats = self.memory_manager.get_memory_statistics()

        # Current system state
        current_pressure = self.memory_manager.check_memory_pressure()

        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            perf_summary, memory_stats, current_pressure
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "operations_processed": self._operation_count,
            "performance_summary": perf_summary,
            "memory_statistics": memory_stats,
            "current_memory_pressure": current_pressure,
            "optimization_config": {
                "chunked_processing": self.config.enable_chunked_processing,
                "adaptive_batching": self.config.enable_adaptive_batching,
                "memory_monitoring": self.config.enable_memory_monitoring,
                "auto_fallback": self.config.auto_fallback_to_cpu,
            },
            "recommendations": recommendations,
        }

    def _generate_optimization_recommendations(
        self, perf_summary: Dict, memory_stats: Dict, pressure: Dict
    ) -> List[str]:
        """Generate optimization recommendations based on current performance.

        Args:
            perf_summary: Performance summary statistics
            memory_stats: Memory usage statistics
            pressure: Current memory pressure information

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Performance-based recommendations
        if perf_summary.get("error_rate", 0) > 0.1:
            recommendations.append(
                "High error rate detected - consider reducing batch sizes or enabling CPU fallback"
            )

        if perf_summary.get("average_duration", 0) > 10.0:
            recommendations.append(
                "Long operation durations - consider enabling chunked processing"
            )

        # Memory-based recommendations
        if pressure.get("under_pressure", False):
            recommendations.append(
                f"Memory pressure detected: {pressure.get('recommendation', 'Reduce memory usage')}"
            )

        if memory_stats.get("backend") == "GPU":
            peak_mb = memory_stats.get("peak_usage_bytes", 0) / (1024**2)
            if peak_mb > 2000:  # > 2GB
                recommendations.append(
                    "High GPU memory usage - consider smaller chunk sizes or more frequent cleanup"
                )

        # Configuration recommendations
        if not self.config.enable_chunked_processing and self._operation_count > 100:
            recommendations.append(
                "Consider enabling chunked processing for better memory management"
            )

        if not self.config.enable_adaptive_batching and perf_summary.get("error_rate", 0) > 0.05:
            recommendations.append("Consider enabling adaptive batching to handle memory pressure")

        if not recommendations:
            recommendations.append(
                "Performance optimization is working well - no changes recommended"
            )

        return recommendations

    def cleanup_resources(self) -> None:
        """Clean up all optimization resources."""
        self.memory_manager.cleanup_all()
        logger.debug("PerformanceOptimizer resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()
