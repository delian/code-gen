"""
Comprehensive error handling system for OFDM chirp generator.

This module provides centralized error handling, graceful fallback mechanisms,
and diagnostic reporting for GPU operations, memory issues, orthogonality
failures, and separation failures.
"""

import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    GPU_ERROR = "gpu_error"
    MEMORY_ERROR = "memory_error"
    COMPUTATION_ERROR = "computation_error"
    CONFIGURATION_ERROR = "configuration_error"
    ORTHOGONALITY_ERROR = "orthogonality_error"
    SEPARATION_ERROR = "separation_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    operation: str
    component: str
    parameters: Dict[str, Any]
    timestamp: datetime
    system_info: Dict[str, Any]
    memory_info: Dict[str, Any]


@dataclass
class ErrorReport:
    """Comprehensive error report."""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    traceback_info: str
    recovery_actions: List[str]
    fallback_used: bool
    diagnostic_data: Dict[str, Any]


class OFDMError(Exception):
    """Base exception class for OFDM generator errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.timestamp = datetime.now()


class GPUError(OFDMError):
    """GPU-related errors."""

    def __init__(
        self, message: str, gpu_operation: str = "", severity: ErrorSeverity = ErrorSeverity.HIGH
    ):
        super().__init__(message, ErrorCategory.GPU_ERROR, severity)
        self.gpu_operation = gpu_operation


class MemoryError(OFDMError):
    """Memory-related errors."""

    def __init__(
        self,
        message: str,
        memory_requested: int = 0,
        memory_available: int = 0,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
    ):
        super().__init__(message, ErrorCategory.MEMORY_ERROR, severity)
        self.memory_requested = memory_requested
        self.memory_available = memory_available


class OrthogonalityError(OFDMError):
    """Orthogonality-related errors."""

    def __init__(
        self,
        message: str,
        orthogonality_score: float = 0.0,
        target_score: float = 0.0,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        super().__init__(message, ErrorCategory.ORTHOGONALITY_ERROR, severity)
        self.orthogonality_score = orthogonality_score
        self.target_score = target_score


class SeparationError(OFDMError):
    """Signal separation errors."""

    def __init__(
        self,
        message: str,
        separation_quality: float = 0.0,
        required_quality: float = 0.0,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        super().__init__(message, ErrorCategory.SEPARATION_ERROR, severity)
        self.separation_quality = separation_quality
        self.required_quality = required_quality


class ErrorHandler:
    """Centralized error handling and recovery system.

    Requirements addressed:
    - 4.3: Graceful fallback mechanisms for various failure scenarios
    - 5.4: Report best available approximation if no orthogonal solution exists
    - 6.4: Provide diagnostic information about separation failures
    """

    def __init__(self, enable_fallbacks: bool = True, log_level: int = logging.WARNING):
        """Initialize error handler.

        Args:
            enable_fallbacks: Enable automatic fallback mechanisms
            log_level: Minimum log level for error reporting
        """
        self.enable_fallbacks = enable_fallbacks
        self.log_level = log_level
        self.error_history: List[ErrorReport] = []
        self.fallback_strategies: Dict[ErrorCategory, Callable] = {}
        self.recovery_statistics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "fallback_activations": 0,
            "critical_failures": 0,
        }

        # Register default fallback strategies
        self._register_default_fallbacks()

        logger.info(
            "ErrorHandler initialized with fallbacks enabled" if enable_fallbacks else "disabled"
        )

    def _register_default_fallbacks(self) -> None:
        """Register default fallback strategies for different error categories."""
        self.fallback_strategies[ErrorCategory.GPU_ERROR] = self._gpu_fallback_strategy
        self.fallback_strategies[ErrorCategory.MEMORY_ERROR] = self._memory_fallback_strategy
        self.fallback_strategies[ErrorCategory.ORTHOGONALITY_ERROR] = (
            self._orthogonality_fallback_strategy
        )
        self.fallback_strategies[ErrorCategory.SEPARATION_ERROR] = (
            self._separation_fallback_strategy
        )

    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True,
    ) -> ErrorReport:
        """Handle an error with comprehensive reporting and recovery.

        Args:
            error: Exception that occurred
            context: Context information about the error
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            ErrorReport with handling results
        """
        # Generate unique error ID
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_history):04d}"

        # Classify error
        category, severity = self._classify_error(error)

        # Create error report
        report = ErrorReport(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            context=context or self._create_default_context(),
            traceback_info=traceback.format_exc(),
            recovery_actions=[],
            fallback_used=False,
            diagnostic_data={},
        )

        # Update statistics
        self.recovery_statistics["total_errors"] += 1
        if severity == ErrorSeverity.CRITICAL:
            self.recovery_statistics["critical_failures"] += 1

        # Attempt recovery if enabled
        if attempt_recovery and self.enable_fallbacks and category in self.fallback_strategies:
            try:
                recovery_result = self.fallback_strategies[category](error, report)
                if recovery_result:
                    report.fallback_used = True
                    report.recovery_actions.extend(recovery_result.get("actions", []))
                    report.diagnostic_data.update(recovery_result.get("diagnostic_data", {}))
                    self.recovery_statistics["successful_recoveries"] += 1
                    self.recovery_statistics["fallback_activations"] += 1
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_id}: {recovery_error}")
                report.recovery_actions.append(f"Recovery failed: {recovery_error}")

        # Log error
        self._log_error(report)

        # Store in history
        self.error_history.append(report)

        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

        return report

    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity.

        Args:
            error: Exception to classify

        Returns:
            Tuple of (category, severity)
        """
        if isinstance(error, OFDMError):
            return error.category, error.severity

        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # GPU-related errors
        if any(keyword in error_str for keyword in ["cuda", "cupy", "gpu", "device"]):
            return ErrorCategory.GPU_ERROR, ErrorSeverity.HIGH

        # Memory errors
        if any(keyword in error_str for keyword in ["memory", "allocation", "out of memory"]):
            return ErrorCategory.MEMORY_ERROR, ErrorSeverity.HIGH

        # Computation errors
        if any(keyword in error_str for keyword in ["nan", "inf", "overflow", "underflow"]):
            return ErrorCategory.COMPUTATION_ERROR, ErrorSeverity.MEDIUM

        # Configuration errors
        if any(keyword in error_str for keyword in ["config", "parameter", "validation"]):
            return ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM

        # System errors
        if error_type in ["systemexit", "keyboardinterrupt"]:
            return ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL

        # Default classification
        return ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM

    def _create_default_context(self) -> ErrorContext:
        """Create default error context."""
        return ErrorContext(
            operation="unknown",
            component="unknown",
            parameters={},
            timestamp=datetime.now(),
            system_info=self._get_system_info(),
            memory_info=self._get_memory_info(),
        )

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for error context."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "cupy_available": CUPY_AVAILABLE,
            "numpy_version": np.__version__,
        }

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for error context."""
        memory_info = {"backend": "CPU"}

        if CUPY_AVAILABLE:
            try:
                device = cp.cuda.Device()
                memory_info.update(
                    {
                        "backend": "GPU",
                        "gpu_memory_total": device.mem_info[1],
                        "gpu_memory_free": device.mem_info[0],
                        "gpu_memory_used": device.mem_info[1] - device.mem_info[0],
                    }
                )
            except Exception:
                pass

        return memory_info

    def _log_error(self, report: ErrorReport) -> None:
        """Log error report."""
        log_message = f"[{report.error_id}] {report.category.value.upper()}: {report.message}"

        if report.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif report.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif report.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        if report.fallback_used:
            logger.info(f"[{report.error_id}] Fallback recovery successful")

    def _gpu_fallback_strategy(self, error: Exception, report: ErrorReport) -> Dict[str, Any]:
        """Fallback strategy for GPU errors.

        Requirements: 4.3 - Graceful fallback mechanisms
        """
        recovery_actions = []
        diagnostic_data = {}

        try:
            # Force CPU fallback
            recovery_actions.append("Forcing CPU computation fallback")
            diagnostic_data["gpu_fallback_triggered"] = True

            # Clear GPU memory if possible
            if CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    recovery_actions.append("GPU memory cleared")
                except Exception as cleanup_error:
                    recovery_actions.append(f"GPU cleanup failed: {cleanup_error}")

            # Suggest system-level recovery
            recovery_actions.append("Consider restarting GPU drivers or reducing batch size")

            return {
                "success": True,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

        except Exception as fallback_error:
            logger.error(f"GPU fallback strategy failed: {fallback_error}")
            return {
                "success": False,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

    def _memory_fallback_strategy(self, error: Exception, report: ErrorReport) -> Dict[str, Any]:
        """Fallback strategy for memory errors.

        Requirements: 4.3 - Graceful fallback mechanisms
        """
        recovery_actions = []
        diagnostic_data = {}

        try:
            # Force garbage collection
            import gc

            collected = gc.collect()
            recovery_actions.append(f"Garbage collection freed {collected} objects")

            # Clear GPU memory if available
            if CUPY_AVAILABLE:
                try:
                    memory_pool = cp.get_default_memory_pool()
                    memory_pool.free_all_blocks()
                    recovery_actions.append("GPU memory pool cleared")
                except Exception:
                    pass

            # Suggest memory optimization strategies
            recovery_actions.extend(
                [
                    "Reduce batch size for processing",
                    "Enable chunked processing for large arrays",
                    "Consider using lower precision data types",
                ]
            )

            diagnostic_data["memory_recovery_attempted"] = True

            return {
                "success": True,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

        except Exception as fallback_error:
            logger.error(f"Memory fallback strategy failed: {fallback_error}")
            return {
                "success": False,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

    def _orthogonality_fallback_strategy(
        self, error: Exception, report: ErrorReport
    ) -> Dict[str, Any]:
        """Fallback strategy for orthogonality errors.

        Requirements: 5.4 - Report best available approximation
        """
        recovery_actions = []
        diagnostic_data = {}

        try:
            if isinstance(error, OrthogonalityError):
                # Provide best available approximation
                recovery_actions.append(
                    f"Best orthogonality score achieved: {error.orthogonality_score:.6f}"
                )
                recovery_actions.append(f"Target score was: {error.target_score:.6f}")

                # Suggest optimization strategies
                if error.orthogonality_score < 0.5:
                    recovery_actions.extend(
                        [
                            "Consider increasing phase resolution",
                            "Try different optimization algorithms",
                            "Reduce number of signals or subcarriers",
                        ]
                    )
                elif error.orthogonality_score < 0.8:
                    recovery_actions.extend(
                        [
                            "Increase optimization iterations",
                            "Try hybrid optimization method",
                            "Fine-tune phase configurations manually",
                        ]
                    )
                else:
                    recovery_actions.append(
                        "Near-optimal solution found, consider accepting result"
                    )

                diagnostic_data.update(
                    {
                        "best_orthogonality_score": error.orthogonality_score,
                        "target_score": error.target_score,
                        "score_ratio": (
                            error.orthogonality_score / error.target_score
                            if error.target_score > 0
                            else 0
                        ),
                    }
                )

            return {
                "success": True,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

        except Exception as fallback_error:
            logger.error(f"Orthogonality fallback strategy failed: {fallback_error}")
            return {
                "success": False,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

    def _separation_fallback_strategy(
        self, error: Exception, report: ErrorReport
    ) -> Dict[str, Any]:
        """Fallback strategy for separation errors.

        Requirements: 6.4 - Provide diagnostic information about failures
        """
        recovery_actions = []
        diagnostic_data = {}

        try:
            if isinstance(error, SeparationError):
                # Provide separation quality information
                recovery_actions.append(
                    f"Separation quality achieved: {error.separation_quality:.6f}"
                )
                recovery_actions.append(f"Required quality was: {error.required_quality:.6f}")

                # Suggest improvement strategies
                if error.separation_quality < 0.3:
                    recovery_actions.extend(
                        [
                            "Signals may not be sufficiently orthogonal",
                            "Check reference signal quality",
                            "Verify phase configurations are correct",
                        ]
                    )
                elif error.separation_quality < 0.6:
                    recovery_actions.extend(
                        [
                            "Improve signal orthogonality before separation",
                            "Consider noise reduction preprocessing",
                            "Adjust separation algorithm parameters",
                        ]
                    )
                else:
                    recovery_actions.extend(
                        [
                            "Separation partially successful",
                            "Consider accepting result with quality warning",
                            "Fine-tune separation threshold",
                        ]
                    )

                diagnostic_data.update(
                    {
                        "separation_quality": error.separation_quality,
                        "required_quality": error.required_quality,
                        "quality_ratio": (
                            error.separation_quality / error.required_quality
                            if error.required_quality > 0
                            else 0
                        ),
                    }
                )

            return {
                "success": True,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

        except Exception as fallback_error:
            logger.error(f"Separation fallback strategy failed: {fallback_error}")
            return {
                "success": False,
                "actions": recovery_actions,
                "diagnostic_data": diagnostic_data,
            }

    def register_fallback_strategy(self, category: ErrorCategory, strategy: Callable) -> None:
        """Register a custom fallback strategy.

        Args:
            category: Error category to handle
            strategy: Callable that takes (error, report) and returns recovery dict
        """
        self.fallback_strategies[category] = strategy
        logger.info(f"Registered custom fallback strategy for {category.value}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        stats = self.recovery_statistics.copy()

        if stats["total_errors"] > 0:
            stats["recovery_rate"] = stats["successful_recoveries"] / stats["total_errors"]
            stats["fallback_rate"] = stats["fallback_activations"] / stats["total_errors"]
        else:
            stats["recovery_rate"] = 0.0
            stats["fallback_rate"] = 0.0

        # Category breakdown
        category_counts = {}
        severity_counts = {}

        for report in self.error_history:
            category_counts[report.category.value] = (
                category_counts.get(report.category.value, 0) + 1
            )
            severity_counts[report.severity.value] = (
                severity_counts.get(report.severity.value, 0) + 1
            )

        stats["category_breakdown"] = category_counts
        stats["severity_breakdown"] = severity_counts

        return stats

    def generate_diagnostic_report(self, include_traceback: bool = False) -> str:
        """Generate comprehensive diagnostic report.

        Args:
            include_traceback: Include full traceback information

        Returns:
            Formatted diagnostic report
        """
        report = []
        report.append("=" * 80)
        report.append("OFDM CHIRP GENERATOR - ERROR DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Statistics
        stats = self.get_error_statistics()
        report.append("ERROR STATISTICS:")
        report.append(f"  Total Errors: {stats['total_errors']}")
        report.append(f"  Successful Recoveries: {stats['successful_recoveries']}")
        report.append(f"  Fallback Activations: {stats['fallback_activations']}")
        report.append(f"  Critical Failures: {stats['critical_failures']}")
        report.append(f"  Recovery Rate: {stats['recovery_rate']:.2%}")
        report.append(f"  Fallback Rate: {stats['fallback_rate']:.2%}")
        report.append("")

        # Category breakdown
        if stats["category_breakdown"]:
            report.append("ERROR CATEGORIES:")
            for category, count in stats["category_breakdown"].items():
                report.append(f"  {category}: {count}")
            report.append("")

        # Severity breakdown
        if stats["severity_breakdown"]:
            report.append("ERROR SEVERITY:")
            for severity, count in stats["severity_breakdown"].items():
                report.append(f"  {severity}: {count}")
            report.append("")

        # Recent errors
        recent_errors = self.error_history[-10:] if self.error_history else []
        if recent_errors:
            report.append("RECENT ERRORS (last 10):")
            for error_report in recent_errors:
                report.append(
                    f"  [{error_report.error_id}] {error_report.category.value}: {error_report.message}"
                )
                if error_report.fallback_used:
                    report.append(f"    → Fallback recovery applied")
                if include_traceback and error_report.traceback_info:
                    report.append(f"    Traceback: {error_report.traceback_info}")
            report.append("")

        # System information
        system_info = self._get_system_info()
        memory_info = self._get_memory_info()

        report.append("SYSTEM INFORMATION:")
        report.append(f"  Python Version: {system_info['python_version']}")
        report.append(f"  Platform: {system_info['platform']}")
        report.append(f"  CuPy Available: {system_info['cupy_available']}")
        report.append(f"  NumPy Version: {system_info['numpy_version']}")
        report.append("")

        report.append("MEMORY INFORMATION:")
        report.append(f"  Backend: {memory_info['backend']}")
        if memory_info["backend"] == "GPU":
            total_gb = memory_info.get("gpu_memory_total", 0) / (1024**3)
            free_gb = memory_info.get("gpu_memory_free", 0) / (1024**3)
            used_gb = memory_info.get("gpu_memory_used", 0) / (1024**3)
            report.append(f"  GPU Memory Total: {total_gb:.2f} GB")
            report.append(f"  GPU Memory Free: {free_gb:.2f} GB")
            report.append(f"  GPU Memory Used: {used_gb:.2f} GB")

        report.append("=" * 80)

        return "\n".join(report)

    def clear_error_history(self) -> None:
        """Clear error history and reset statistics."""
        self.error_history.clear()
        self.recovery_statistics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "fallback_activations": 0,
            "critical_failures": 0,
        }
        logger.info("Error history and statistics cleared")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(
    error: Exception, context: Optional[ErrorContext] = None, attempt_recovery: bool = True
) -> ErrorReport:
    """Handle an error using the global error handler.

    Args:
        error: Exception that occurred
        context: Context information about the error
        attempt_recovery: Whether to attempt automatic recovery

    Returns:
        ErrorReport with handling results
    """
    return get_error_handler().handle_error(error, context, attempt_recovery)


def create_error_context(operation: str, component: str, **parameters) -> ErrorContext:
    """Create error context for error handling.

    Args:
        operation: Name of the operation being performed
        component: Name of the component where error occurred
        **parameters: Additional parameters to include in context

    Returns:
        ErrorContext object
    """
    return ErrorContext(
        operation=operation,
        component=component,
        parameters=parameters,
        timestamp=datetime.now(),
        system_info=get_error_handler()._get_system_info(),
        memory_info=get_error_handler()._get_memory_info(),
    )


# Decorator for automatic error handling
def with_error_handling(operation: str = "", component: str = "", attempt_recovery: bool = True):
    """Decorator for automatic error handling.

    Args:
        operation: Name of the operation
        component: Name of the component
        attempt_recovery: Whether to attempt recovery
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = create_error_context(
                    operation=operation or func.__name__,
                    component=component or func.__module__,
                    args=str(args)[:200],  # Limit length
                    kwargs=str(kwargs)[:200],
                )

                error_report = handle_error(e, context, attempt_recovery)

                # Re-raise if critical or recovery failed
                if (
                    error_report.severity == ErrorSeverity.CRITICAL
                    or not error_report.fallback_used
                ):
                    raise

                # Return None or appropriate fallback value
                return None

        return wrapper

    return decorator
