#!/usr/bin/env python3
"""
Error Handling Demonstration for OFDM Chirp Generator

This script demonstrates the comprehensive error handling capabilities
of the OFDM chirp generator system, including GPU errors, memory issues,
orthogonality failures, and separation failures with recovery mechanisms.
"""

import sys
from pathlib import Path

import numpy as np

# Add the parent directory to the path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ofdm_chirp_generator.error_handling import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    GPUError,
    MemoryError,
    OrthogonalityError,
    SeparationError,
    create_error_context,
    get_error_handler,
)
from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.phase_optimizer import PhaseOptimizer
from ofdm_chirp_generator.signal_separator import SignalSeparator


def demonstrate_error_classification():
    """Demonstrate error classification and handling."""
    print("=" * 60)
    print("ERROR CLASSIFICATION DEMONSTRATION")
    print("=" * 60)

    error_handler = ErrorHandler(enable_fallbacks=True)

    # Test different types of errors
    test_errors = [
        ValueError("Invalid parameter value"),
        RuntimeError("CUDA out of memory"),
        Exception("Memory allocation failed"),
        GPUError("GPU computation failed", "fft_operation"),
        MemoryError("Insufficient memory", 1000000, 500000),
        OrthogonalityError("Poor orthogonality", 0.6, 0.9),
        SeparationError("Separation quality too low", 0.4, 0.8),
    ]

    for i, error in enumerate(test_errors):
        print(f"\nTest {i+1}: {type(error).__name__}")
        print(f"Message: {error}")

        context = create_error_context(f"test_operation_{i}", "DemoComponent", test_id=i)
        report = error_handler.handle_error(error, context, attempt_recovery=True)

        print(f"Category: {report.category.value}")
        print(f"Severity: {report.severity.value}")
        print(f"Fallback used: {report.fallback_used}")
        if report.recovery_actions:
            print(f"Recovery actions: {len(report.recovery_actions)}")
            for action in report.recovery_actions[:2]:  # Show first 2 actions
                print(f"  - {action}")

    # Show statistics
    print(f"\nError handling statistics:")
    stats = error_handler.get_error_statistics()
    print(f"Total errors: {stats['total_errors']}")
    print(f"Successful recoveries: {stats['successful_recoveries']}")
    print(f"Recovery rate: {stats['recovery_rate']:.2%}")


def demonstrate_gpu_error_handling():
    """Demonstrate GPU error handling and fallback mechanisms."""
    print("\n" + "=" * 60)
    print("GPU ERROR HANDLING DEMONSTRATION")
    print("=" * 60)

    # Initialize GPU backend
    gpu_backend = GPUBackend()

    print(f"GPU available: {gpu_backend.is_gpu_available}")
    print(f"Backend: {gpu_backend.device_info['backend']}")

    # Test memory allocation with error handling
    print("\nTesting memory allocation error handling...")

    try:
        # Try to allocate a very large array that might fail
        large_shape = (10**6, 100)  # 100M complex numbers ≈ 1.6GB
        print(f"Attempting to allocate array of shape {large_shape}")

        array = gpu_backend.allocate_signal_memory(large_shape, np.complex128)
        print(f"Allocation successful: {array.shape}, backend: {type(array).__module__}")

        # Clean up
        if hasattr(array, "device"):
            gpu_backend.cleanup_memory()

    except Exception as e:
        print(f"Allocation failed as expected: {e}")

    # Test FFT error handling with problematic input
    print("\nTesting FFT error handling...")

    # Create signal with potential numerical issues
    test_signals = [
        np.array([1, 2, 3, 4], dtype=np.complex128),  # Normal signal
        np.array([1, 2, np.nan, 4], dtype=np.complex128),  # Signal with NaN
        np.array([1, 2, np.inf, 4], dtype=np.complex128),  # Signal with Inf
    ]

    for i, signal in enumerate(test_signals):
        print(f"\nTest signal {i+1}: {signal}")
        try:
            result = gpu_backend.perform_fft(signal)
            print(f"FFT result shape: {result.shape}")
            print(f"Contains NaN: {np.any(np.isnan(gpu_backend.to_cpu(result)))}")
            print(f"Contains Inf: {np.any(np.isinf(gpu_backend.to_cpu(result)))}")
        except Exception as e:
            print(f"FFT failed: {e}")


def demonstrate_orthogonality_error_handling():
    """Demonstrate orthogonality optimization error handling."""
    print("\n" + "=" * 60)
    print("ORTHOGONALITY ERROR HANDLING DEMONSTRATION")
    print("=" * 60)

    # Create a challenging OFDM configuration
    ofdm_config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=500.0,
        center_frequency=10000.0,
        sampling_rate=16000.0,
        signal_duration=0.001,
    )

    try:
        # Initialize phase optimizer
        phase_optimizer = PhaseOptimizer(ofdm_config)

        print(f"Initialized PhaseOptimizer with {ofdm_config.num_subcarriers} subcarriers")

        # Try to find orthogonal phases for many signals (likely to fail)
        num_signals = 16  # Very challenging for 8 subcarriers
        print(f"\nAttempting to find orthogonal phases for {num_signals} signals...")

        # Use a very high orthogonality target that's unlikely to be achieved
        from ofdm_chirp_generator.phase_optimizer import OptimizationConfig

        config = OptimizationConfig(
            max_iterations=50,  # Limited iterations
            orthogonality_target=0.99,  # Very high target
            early_stopping_patience=10,
        )

        result = phase_optimizer.find_orthogonal_phases(num_signals, config, method="brute_force")

        print(f"Optimization completed:")
        print(f"  Converged: {result.converged}")
        print(f"  Orthogonality score: {result.orthogonality_score:.6f}")
        print(f"  Target score: {config.orthogonality_target}")
        print(f"  Iterations: {result.iterations}")

        if not result.converged:
            print(f"  Best available approximation provided")

    except Exception as e:
        print(f"Orthogonality optimization failed: {e}")

        # Show error handler statistics
        error_handler = get_error_handler()
        stats = error_handler.get_error_statistics()
        if stats["total_errors"] > 0:
            print(f"\nError handling statistics:")
            print(f"  Total errors: {stats['total_errors']}")
            print(f"  Recovery rate: {stats['recovery_rate']:.2%}")


def demonstrate_separation_error_handling():
    """Demonstrate signal separation error handling."""
    print("\n" + "=" * 60)
    print("SEPARATION ERROR HANDLING DEMONSTRATION")
    print("=" * 60)

    # Create OFDM configuration
    ofdm_config = OFDMConfig(
        num_subcarriers=4,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=500.0,
        center_frequency=10000.0,
        sampling_rate=8000.0,
        signal_duration=0.001,
    )

    try:
        # Initialize signal separator
        signal_separator = SignalSeparator(ofdm_config)

        print(f"Initialized SignalSeparator with {ofdm_config.num_subcarriers} subcarriers")

        # Create test signals with poor orthogonality (will cause separation issues)
        signal_length = int(ofdm_config.sampling_rate * ofdm_config.signal_duration)

        # Create poorly orthogonal reference signals
        ref_signal1 = np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
        ref_signal2 = ref_signal1 + 0.1 * (
            np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
        )

        reference_signals = [ref_signal1, ref_signal2]
        reference_phases = np.array(
            [
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                [np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8, 7 * np.pi / 8],
            ]
        )

        # Create combined signal
        combined_signal = ref_signal1 + ref_signal2

        print(f"\nAttempting signal separation...")
        print(f"Reference signals: {len(reference_signals)}")
        print(f"Combined signal length: {len(combined_signal)}")

        # Attempt separation (likely to have quality issues)
        separated_signals, quality_metrics = signal_separator.separate_combined_signal(
            combined_signal, reference_signals, reference_phases
        )

        print(f"\nSeparation results:")
        print(f"  Success: {quality_metrics.separation_success}")
        print(f"  Overall quality: {quality_metrics.overall_separation_quality:.6f}")
        print(f"  Mean separation score: {quality_metrics.mean_separation_score:.6f}")
        print(f"  Max cross-talk: {quality_metrics.max_cross_talk:.6f}")

        if not quality_metrics.separation_success:
            print(f"  Separation quality below threshold")
            if quality_metrics.diagnostic_info:
                print(f"  Diagnostic info available: {len(quality_metrics.diagnostic_info)} items")

        # Generate separation report
        report = signal_separator.generate_separation_report(quality_metrics)
        print(f"\nSeparation report generated ({len(report)} characters)")

    except Exception as e:
        print(f"Signal separation failed: {e}")


def demonstrate_diagnostic_reporting():
    """Demonstrate comprehensive diagnostic reporting."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORTING DEMONSTRATION")
    print("=" * 60)

    error_handler = get_error_handler()

    # Generate diagnostic report
    report = error_handler.generate_diagnostic_report(include_traceback=False)

    print("Comprehensive diagnostic report:")
    print(report)

    # Show recent error details
    if error_handler.error_history:
        print(f"\nRecent error details:")
        for i, error_report in enumerate(error_handler.error_history[-3:]):
            print(f"\nError {i+1}:")
            print(f"  ID: {error_report.error_id}")
            print(f"  Category: {error_report.category.value}")
            print(f"  Severity: {error_report.severity.value}")
            print(f"  Message: {error_report.message}")
            print(f"  Fallback used: {error_report.fallback_used}")
            if error_report.recovery_actions:
                print(f"  Recovery actions: {len(error_report.recovery_actions)}")


def main():
    """Run all error handling demonstrations."""
    print("OFDM Chirp Generator - Error Handling Demonstration")
    print("This script demonstrates comprehensive error handling capabilities")
    print("including GPU errors, memory issues, orthogonality failures, and separation failures.")

    try:
        # Run demonstrations
        demonstrate_error_classification()
        demonstrate_gpu_error_handling()
        demonstrate_orthogonality_error_handling()
        demonstrate_separation_error_handling()
        demonstrate_diagnostic_reporting()

        print("\n" + "=" * 60)
        print("ERROR HANDLING DEMONSTRATION COMPLETED")
        print("=" * 60)
        print("All error handling mechanisms have been demonstrated.")
        print("The system provides comprehensive error recovery and diagnostic capabilities.")

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")

        # Even if the demo fails, show error handling in action
        error_handler = get_error_handler()
        context = create_error_context("demonstration", "ErrorHandlingDemo")
        report = error_handler.handle_error(e, context)

        print(f"Error handled by system:")
        print(f"  Category: {report.category.value}")
        print(f"  Severity: {report.severity.value}")
        print(f"  Recovery attempted: {report.fallback_used}")


if __name__ == "__main__":
    main()
