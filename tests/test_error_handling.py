"""
Tests for comprehensive error handling system.

This module tests error handling for GPU operations, memory issues,
orthogonality failures, and separation failures with recovery mechanisms.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.error_handling import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorReport,
    ErrorSeverity,
    GPUError,
    MemoryError,
    OFDMError,
    OrthogonalityError,
    SeparationError,
    create_error_context,
    get_error_handler,
    with_error_handling,
)
from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig


class TestErrorClasses:
    """Test custom error classes."""

    def test_ofdm_error_creation(self):
        """Test OFDMError creation and attributes."""
        error = OFDMError("Test error", ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH)

        assert str(error) == "Test error"
        assert error.category == ErrorCategory.SYSTEM_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert isinstance(error.timestamp, datetime)

    def test_gpu_error_creation(self):
        """Test GPUError creation and attributes."""
        error = GPUError("GPU operation failed", "fft_computation", ErrorSeverity.HIGH)

        assert str(error) == "GPU operation failed"
        assert error.category == ErrorCategory.GPU_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.gpu_operation == "fft_computation"

    def test_memory_error_creation(self):
        """Test MemoryError creation and attributes."""
        error = MemoryError("Out of memory", 1000, 500, ErrorSeverity.CRITICAL)

        assert str(error) == "Out of memory"
        assert error.category == ErrorCategory.MEMORY_ERROR
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.memory_requested == 1000
        assert error.memory_available == 500

    def test_orthogonality_error_creation(self):
        """Test OrthogonalityError creation and attributes."""
        error = OrthogonalityError("Orthogonality failed", 0.5, 0.8, ErrorSeverity.MEDIUM)

        assert str(error) == "Orthogonality failed"
        assert error.category == ErrorCategory.ORTHOGONALITY_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.orthogonality_score == 0.5
        assert error.target_score == 0.8

    def test_separation_error_creation(self):
        """Test SeparationError creation and attributes."""
        error = SeparationError("Separation failed", 0.3, 0.7, ErrorSeverity.MEDIUM)

        assert str(error) == "Separation failed"
        assert error.category == ErrorCategory.SEPARATION_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.separation_quality == 0.3
        assert error.required_quality == 0.7


class TestErrorContext:
    """Test error context creation and usage."""

    def test_create_error_context(self):
        """Test error context creation."""
        context = create_error_context("test_operation", "TestComponent", param1="value1")

        assert context.operation == "test_operation"
        assert context.component == "TestComponent"
        assert context.parameters["param1"] == "value1"
        assert isinstance(context.timestamp, datetime)
        assert isinstance(context.system_info, dict)
        assert isinstance(context.memory_info, dict)


class TestErrorHandler:
    """Test ErrorHandler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler(enable_fallbacks=True)

    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler(enable_fallbacks=False)

        assert handler.enable_fallbacks is False
        assert len(handler.error_history) == 0
        assert len(handler.fallback_strategies) > 0  # Default strategies registered
        assert handler.recovery_statistics["total_errors"] == 0

    def test_error_classification(self):
        """Test error classification by category and severity."""
        # Test GPU error classification
        gpu_error = Exception("CUDA out of memory")
        category, severity = self.error_handler._classify_error(gpu_error)
        assert category == ErrorCategory.GPU_ERROR
        assert severity == ErrorSeverity.HIGH

        # Test memory error classification
        memory_error = Exception("allocation failed")
        category, severity = self.error_handler._classify_error(memory_error)
        assert category == ErrorCategory.MEMORY_ERROR
        assert severity == ErrorSeverity.HIGH

        # Test custom OFDM error
        ofdm_error = GPUError("Test GPU error")
        category, severity = self.error_handler._classify_error(ofdm_error)
        assert category == ErrorCategory.GPU_ERROR
        assert severity == ErrorSeverity.HIGH

    def test_handle_error_basic(self):
        """Test basic error handling without recovery."""
        error = ValueError("Test error")
        context = create_error_context("test_op", "TestComponent")

        report = self.error_handler.handle_error(error, context, attempt_recovery=False)

        assert isinstance(report, ErrorReport)
        assert report.message == "Test error"
        assert report.context == context
        assert report.fallback_used is False
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.recovery_statistics["total_errors"] == 1

    def test_handle_error_with_recovery(self):
        """Test error handling with recovery attempt."""
        # Mock a fallback strategy
        mock_strategy = Mock(
            return_value={
                "success": True,
                "actions": ["Recovery action performed"],
                "diagnostic_data": {"recovered": True},
            }
        )

        self.error_handler.fallback_strategies[ErrorCategory.GPU_ERROR] = mock_strategy

        gpu_error = GPUError("Test GPU error")
        context = create_error_context("gpu_op", "GPUBackend")

        report = self.error_handler.handle_error(gpu_error, context, attempt_recovery=True)

        assert report.fallback_used is True
        assert "Recovery action performed" in report.recovery_actions
        assert report.diagnostic_data["recovered"] is True
        assert self.error_handler.recovery_statistics["successful_recoveries"] == 1
        mock_strategy.assert_called_once()

    def test_gpu_fallback_strategy(self):
        """Test GPU fallback strategy."""
        gpu_error = GPUError("CUDA error")
        report = ErrorReport(
            error_id="test_id",
            category=ErrorCategory.GPU_ERROR,
            severity=ErrorSeverity.HIGH,
            message="Test error",
            context=create_error_context("test", "test"),
            traceback_info="",
            recovery_actions=[],
            fallback_used=False,
            diagnostic_data={},
        )

        result = self.error_handler._gpu_fallback_strategy(gpu_error, report)

        assert result["success"] is True
        assert len(result["actions"]) > 0
        assert "gpu_fallback_triggered" in result["diagnostic_data"]

    def test_memory_fallback_strategy(self):
        """Test memory fallback strategy."""
        memory_error = MemoryError("Out of memory", 1000, 500)
        report = ErrorReport(
            error_id="test_id",
            category=ErrorCategory.MEMORY_ERROR,
            severity=ErrorSeverity.HIGH,
            message="Test error",
            context=create_error_context("test", "test"),
            traceback_info="",
            recovery_actions=[],
            fallback_used=False,
            diagnostic_data={},
        )

        result = self.error_handler._memory_fallback_strategy(memory_error, report)

        assert result["success"] is True
        assert len(result["actions"]) > 0
        assert "memory_recovery_attempted" in result["diagnostic_data"]

    def test_orthogonality_fallback_strategy(self):
        """Test orthogonality fallback strategy."""
        orth_error = OrthogonalityError("Low orthogonality", 0.6, 0.9)
        report = ErrorReport(
            error_id="test_id",
            category=ErrorCategory.ORTHOGONALITY_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            context=create_error_context("test", "test"),
            traceback_info="",
            recovery_actions=[],
            fallback_used=False,
            diagnostic_data={},
        )

        result = self.error_handler._orthogonality_fallback_strategy(orth_error, report)

        assert result["success"] is True
        assert len(result["actions"]) > 0
        assert result["diagnostic_data"]["best_orthogonality_score"] == 0.6
        assert result["diagnostic_data"]["target_score"] == 0.9

    def test_separation_fallback_strategy(self):
        """Test separation fallback strategy."""
        sep_error = SeparationError("Poor separation", 0.4, 0.8)
        report = ErrorReport(
            error_id="test_id",
            category=ErrorCategory.SEPARATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            context=create_error_context("test", "test"),
            traceback_info="",
            recovery_actions=[],
            fallback_used=False,
            diagnostic_data={},
        )

        result = self.error_handler._separation_fallback_strategy(sep_error, report)

        assert result["success"] is True
        assert len(result["actions"]) > 0
        assert result["diagnostic_data"]["separation_quality"] == 0.4
        assert result["diagnostic_data"]["required_quality"] == 0.8

    def test_register_custom_fallback_strategy(self):
        """Test registering custom fallback strategy."""

        def custom_strategy(error, report):
            return {"success": True, "actions": ["Custom action"], "diagnostic_data": {}}

        self.error_handler.register_fallback_strategy(
            ErrorCategory.COMPUTATION_ERROR, custom_strategy
        )

        assert ErrorCategory.COMPUTATION_ERROR in self.error_handler.fallback_strategies
        assert (
            self.error_handler.fallback_strategies[ErrorCategory.COMPUTATION_ERROR]
            == custom_strategy
        )

    def test_get_error_statistics(self):
        """Test error statistics generation."""
        # Generate some test errors
        for i in range(5):
            error = ValueError(f"Test error {i}")
            self.error_handler.handle_error(error, attempt_recovery=False)

        stats = self.error_handler.get_error_statistics()

        assert stats["total_errors"] == 5
        assert stats["recovery_rate"] == 0.0  # No recovery attempted
        assert "category_breakdown" in stats
        assert "severity_breakdown" in stats

    def test_generate_diagnostic_report(self):
        """Test diagnostic report generation."""
        # Add some test errors
        error1 = GPUError("GPU error")
        error2 = MemoryError("Memory error", 1000, 500)

        self.error_handler.handle_error(error1)
        self.error_handler.handle_error(error2)

        report = self.error_handler.generate_diagnostic_report(include_traceback=False)

        assert "ERROR DIAGNOSTIC REPORT" in report
        assert "Total Errors: 2" in report
        assert "ERROR CATEGORIES:" in report
        assert "SYSTEM INFORMATION:" in report

    def test_clear_error_history(self):
        """Test clearing error history."""
        # Add some errors
        for i in range(3):
            error = ValueError(f"Test error {i}")
            self.error_handler.handle_error(error)

        assert len(self.error_handler.error_history) == 3
        assert self.error_handler.recovery_statistics["total_errors"] == 3

        self.error_handler.clear_error_history()

        assert len(self.error_handler.error_history) == 0
        assert self.error_handler.recovery_statistics["total_errors"] == 0


class TestErrorHandlingDecorator:
    """Test error handling decorator."""

    def test_with_error_handling_decorator_success(self):
        """Test decorator with successful function execution."""

        @with_error_handling("test_operation", "TestComponent")
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)
        assert result == 5

    def test_with_error_handling_decorator_error(self):
        """Test decorator with function that raises error."""

        @with_error_handling("test_operation", "TestComponent", attempt_recovery=False)
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_function()

    def test_with_error_handling_decorator_recovery(self):
        """Test decorator with recovery enabled."""
        # Mock the global error handler to return successful recovery
        with patch("ofdm_chirp_generator.error_handling.get_error_handler") as mock_get_handler:
            mock_handler = Mock()
            mock_report = Mock()
            mock_report.severity = ErrorSeverity.MEDIUM
            mock_report.fallback_used = True
            mock_handler.handle_error.return_value = mock_report
            mock_get_handler.return_value = mock_handler

            @with_error_handling("test_operation", "TestComponent", attempt_recovery=True)
            def test_function():
                raise ValueError("Test error")

            result = test_function()
            assert result is None  # Fallback return value


class TestGPUBackendErrorHandling:
    """Test error handling in GPU backend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gpu_backend = GPUBackend(force_cpu=True)  # Force CPU to avoid GPU dependencies

    def test_gpu_initialization_error_handling(self):
        """Test error handling during GPU initialization."""
        # This test runs with force_cpu=True, so GPU should not be available
        assert not self.gpu_backend.is_gpu_available

    def test_memory_allocation_error_handling(self):
        """Test error handling during memory allocation."""
        # Test with extremely large array that should fail
        with patch("numpy.zeros", side_effect=MemoryError("Simulated memory error")):
            with pytest.raises(MemoryError):
                self.gpu_backend.allocate_signal_memory((10**10,), np.complex128)

    def test_fft_error_handling_with_nan_input(self):
        """Test FFT error handling with NaN input."""
        # Create signal with NaN values
        signal = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.complex128)

        # This should handle the error gracefully
        result = self.gpu_backend.perform_fft(signal)

        # The error handler should have been called, but we still get a result
        # (the actual behavior depends on the error handling implementation)
        assert result is not None

    def test_fft_error_handling_with_inf_input(self):
        """Test FFT error handling with Inf input."""
        # Create signal with Inf values
        signal = np.array([1.0, 2.0, np.inf, 4.0], dtype=np.complex128)

        # This should handle the error gracefully
        result = self.gpu_backend.perform_fft(signal)

        # The error handler should have been called
        assert result is not None


class TestIntegrationErrorHandling:
    """Test error handling integration across components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ofdm_config = OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=500.0,
            center_frequency=10000.0,
            sampling_rate=8000.0,
            signal_duration=0.001,
        )

    def test_error_propagation_through_components(self):
        """Test that errors propagate correctly through component stack."""
        # This test would require mocking various components to simulate errors
        # and verify that they are handled appropriately at each level
        pass

    def test_recovery_mechanism_effectiveness(self):
        """Test that recovery mechanisms actually improve system robustness."""
        # This test would measure system behavior with and without error handling
        # to verify that recovery mechanisms provide value
        pass

    def test_diagnostic_information_quality(self):
        """Test that diagnostic information is useful for debugging."""
        error_handler = ErrorHandler()

        # Simulate various error conditions
        gpu_error = GPUError("Test GPU error", "test_operation")
        context = create_error_context("test_op", "TestComponent", param1="value1")

        report = error_handler.handle_error(gpu_error, context)

        # Verify diagnostic information is comprehensive
        assert report.error_id is not None
        assert report.category == ErrorCategory.GPU_ERROR
        assert report.context.operation == "test_op"
        assert report.context.component == "TestComponent"
        assert "param1" in report.context.parameters

    def test_error_handling_performance_impact(self):
        """Test that error handling doesn't significantly impact performance."""
        # This test would measure performance with and without error handling
        # to ensure the overhead is acceptable
        pass


class TestGlobalErrorHandler:
    """Test global error handler functionality."""

    def test_get_global_error_handler(self):
        """Test getting global error handler instance."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        # Should return the same instance
        assert handler1 is handler2
        assert isinstance(handler1, ErrorHandler)

    def test_global_handle_error_function(self):
        """Test global handle_error function."""
        from ofdm_chirp_generator.error_handling import handle_error

        error = ValueError("Test error")
        context = create_error_context("test_op", "TestComponent")

        report = handle_error(error, context, attempt_recovery=False)

        assert isinstance(report, ErrorReport)
        assert report.message == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])
