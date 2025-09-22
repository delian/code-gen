#!/usr/bin/env python3
"""
Orthogonality Testing Framework Demo

This script demonstrates the orthogonality testing capabilities of the OFDM
chirp generator, including correlation analysis and signal separation evaluation.
"""

import numpy as np

from ofdm_chirp_generator import (
    ChirpConfig,
    CorrelationAnalyzer,
    OFDMConfig,
    OFDMGenerator,
    OrthogonalityTester,
)


def create_test_signals():
    """Create various test signals for orthogonality demonstration."""
    print("Creating test signals...")

    # Create orthogonal Walsh functions
    walsh_signals = [
        np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float),
        np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float),
        np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float),
        np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=float),
    ]

    # Create sinusoidal signals with different frequencies
    t = np.linspace(0, 1, 1000)
    sine_signals = [
        np.sin(2 * np.pi * 10 * t),  # 10 Hz
        np.sin(2 * np.pi * 15 * t),  # 15 Hz
        np.sin(2 * np.pi * 20 * t),  # 20 Hz
        np.cos(2 * np.pi * 10 * t),  # 10 Hz cosine (orthogonal to first)
    ]

    # Create non-orthogonal signals (close frequencies)
    non_orthogonal_signals = [
        np.sin(2 * np.pi * 10.0 * t),  # 10.0 Hz
        np.sin(2 * np.pi * 10.2 * t),  # 10.2 Hz (close frequency)
        np.sin(2 * np.pi * 10.4 * t),  # 10.4 Hz (close frequency)
    ]

    return walsh_signals, sine_signals, non_orthogonal_signals


def demonstrate_correlation_analysis():
    """Demonstrate correlation analysis capabilities."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Create correlation analyzer
    analyzer = CorrelationAnalyzer()
    print(f"Using backend: {analyzer.backend.device_info['backend']}")

    # Create test signals
    walsh_signals, sine_signals, _ = create_test_signals()

    # Test cross-correlation between orthogonal Walsh functions
    print("\n1. Cross-correlation between orthogonal Walsh functions:")
    for i in range(len(walsh_signals)):
        for j in range(i + 1, len(walsh_signals)):
            corr = analyzer.compute_max_correlation(walsh_signals[i], walsh_signals[j])
            print(f"   Walsh {i+1} vs Walsh {j+1}: {corr:.6f}")

    # Test correlation matrix for sine signals
    print("\n2. Correlation matrix for sinusoidal signals:")
    corr_matrix = analyzer.compute_correlation_matrix(sine_signals[:3])
    print("   Correlation Matrix:")
    for i, row in enumerate(corr_matrix):
        print(f"   Signal {i+1}: {' '.join(f'{val:6.3f}' for val in row)}")

    # Test autocorrelation
    print("\n3. Autocorrelation analysis:")
    autocorr = analyzer.compute_autocorrelation(sine_signals[0])
    peaks_idx, peaks_val = analyzer.find_correlation_peaks(autocorr, threshold=0.5)
    print(f"   Found {len(peaks_idx)} correlation peaks above 0.5 threshold")
    if len(peaks_idx) > 0:
        print(f"   Peak values: {peaks_val[:5]}")  # Show first 5 peaks


def demonstrate_orthogonality_testing():
    """Demonstrate orthogonality testing capabilities."""
    print("\n" + "=" * 60)
    print("ORTHOGONALITY TESTING DEMONSTRATION")
    print("=" * 60)

    # Create orthogonality tester
    tester = OrthogonalityTester()

    # Create test signals
    walsh_signals, sine_signals, non_orthogonal_signals = create_test_signals()

    # Test orthogonal signal set (Walsh functions)
    print("\n1. Testing orthogonal signal set (Walsh functions):")
    result_orthogonal = tester.test_signal_set_orthogonality(walsh_signals)
    print(f"   Orthogonality ratio: {result_orthogonal['orthogonality_ratio']:.2%}")
    print(f"   Max cross-correlation: {result_orthogonal['max_cross_correlation']:.6f}")
    print(f"   Set is orthogonal: {'YES' if result_orthogonal['is_set_orthogonal'] else 'NO'}")

    # Test non-orthogonal signal set
    print("\n2. Testing non-orthogonal signal set (close frequencies):")
    result_non_orthogonal = tester.test_signal_set_orthogonality(non_orthogonal_signals)
    print(f"   Orthogonality ratio: {result_non_orthogonal['orthogonality_ratio']:.2%}")
    print(f"   Max cross-correlation: {result_non_orthogonal['max_cross_correlation']:.6f}")
    print(f"   Set is orthogonal: {'YES' if result_non_orthogonal['is_set_orthogonal'] else 'NO'}")

    # Test signal pair orthogonality
    print("\n3. Testing individual signal pairs:")
    pair_result = tester.test_signal_pair_orthogonality(walsh_signals[0], walsh_signals[1])
    print(f"   Walsh 1 vs Walsh 2:")
    print(f"     Max correlation: {pair_result['max_correlation']:.6f}")
    print(f"     Orthogonality score: {pair_result['orthogonality_score']:.6f}")
    print(f"     Is orthogonal: {'YES' if pair_result['is_orthogonal'] else 'NO'}")


def demonstrate_separation_quality():
    """Demonstrate signal separation quality evaluation."""
    print("\n" + "=" * 60)
    print("SIGNAL SEPARATION QUALITY DEMONSTRATION")
    print("=" * 60)

    tester = OrthogonalityTester()
    walsh_signals, _, _ = create_test_signals()

    # Simulate perfect separation (separated = original)
    print("\n1. Perfect separation scenario:")
    original_signals = walsh_signals[:3]
    perfect_separated = [sig.copy() for sig in original_signals]

    perfect_result = tester.evaluate_separation_quality(original_signals, perfect_separated)
    print(f"   Mean separation score: {perfect_result['mean_separation_score']:.6f}")
    print(f"   Max cross-talk: {perfect_result['max_cross_talk']:.6f}")
    print(f"   Separation quality: {perfect_result['separation_quality']:.6f}")

    # Simulate imperfect separation (add noise)
    print("\n2. Imperfect separation scenario (with noise):")
    noisy_separated = [sig + 0.1 * np.random.randn(len(sig)) for sig in original_signals]

    noisy_result = tester.evaluate_separation_quality(original_signals, noisy_separated)
    print(f"   Mean separation score: {noisy_result['mean_separation_score']:.6f}")
    print(f"   Max cross-talk: {noisy_result['max_cross_talk']:.6f}")
    print(f"   Separation quality: {noisy_result['separation_quality']:.6f}")


def demonstrate_threshold_optimization():
    """Demonstrate optimal threshold finding."""
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    tester = OrthogonalityTester()
    walsh_signals, _, _ = create_test_signals()

    # Find optimal threshold
    print("\nFinding optimal orthogonality threshold...")
    threshold_result = tester.find_optimal_threshold(
        walsh_signals, threshold_range=(0.01, 0.3), num_points=30
    )

    print(f"Optimal threshold: {threshold_result['optimal_threshold']:.4f}")
    print(f"Optimal orthogonality ratio: {threshold_result['optimal_ratio']:.2%}")

    # Show effect of different thresholds
    print("\nThreshold sensitivity analysis:")
    test_thresholds = [0.05, 0.1, 0.15, 0.2]
    for threshold in test_thresholds:
        result = tester.test_signal_set_orthogonality(walsh_signals, threshold=threshold)
        print(f"   Threshold {threshold:.2f}: {result['orthogonality_ratio']:.2%} orthogonal pairs")


def generate_comprehensive_report():
    """Generate and display a comprehensive orthogonality report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ORTHOGONALITY REPORT")
    print("=" * 60)

    tester = OrthogonalityTester()
    walsh_signals, _, _ = create_test_signals()

    # Generate detailed report
    report = tester.generate_orthogonality_report(walsh_signals)
    print(report)


def main():
    """Main demonstration function."""
    print("OFDM Chirp Generator - Orthogonality Testing Framework Demo")
    print("This demo showcases correlation analysis and orthogonality testing capabilities.")

    try:
        # Run all demonstrations
        demonstrate_correlation_analysis()
        demonstrate_orthogonality_testing()
        demonstrate_separation_quality()
        demonstrate_threshold_optimization()
        generate_comprehensive_report()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey takeaways:")
        print("- Walsh functions demonstrate perfect orthogonality")
        print("- Sinusoidal signals show frequency-dependent correlation")
        print("- GPU acceleration provides significant performance benefits")
        print("- Threshold optimization helps fine-tune orthogonality detection")
        print("- Comprehensive reporting enables detailed analysis")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
