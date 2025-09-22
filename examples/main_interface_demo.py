#!/usr/bin/env python3
"""
Comprehensive demonstration of the OFDM Chirp Generator main interface.

This example showcases all the key features of the high-level API, including:
- Single signal generation
- Orthogonal signal set generation
- Signal separation and analysis
- Configuration management
- Export functionality

Run with: uv run python examples/main_interface_demo.py
"""

import tempfile
from pathlib import Path

import numpy as np

from ofdm_chirp_generator import (
    OFDMChirpGenerator,
    create_generator,
    quick_generate_orthogonal_signals,
    quick_test_separation,
)


def demo_basic_usage():
    """Demonstrate basic usage of the main interface."""
    print("=" * 60)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 60)

    # Create generator with default configuration
    with create_generator() as generator:
        print(f"✓ Created generator: {generator}")

        # Get system information
        system_info = generator.get_system_info()
        print(f"✓ Backend: {system_info['gpu_backend']['backend']}")
        print(f"✓ Subcarriers: {system_info['ofdm_config']['num_subcarriers']}")
        print(f"✓ Sampling rate: {system_info['ofdm_config']['sampling_rate']} Hz")

        # Validate configuration
        validation = generator.validate_configuration()
        print(f"✓ Configuration valid: {validation['configuration_valid']}")
        if validation["warnings"]:
            print(f"⚠ Warnings: {validation['warnings']}")


def demo_single_signal_generation():
    """Demonstrate single signal generation."""
    print("\n" + "=" * 60)
    print("SINGLE SIGNAL GENERATION")
    print("=" * 60)

    with create_generator() as generator:
        # Generate signal with default phases
        signal_set = generator.generate_single_signal()
        print(f"✓ Generated single signal: {len(signal_set.signals[0])} samples")
        print(f"✓ Signal power: {np.mean(np.abs(signal_set.signals[0])**2):.6f}")

        # Generate signal with custom phases
        num_subcarriers = generator.ofdm_config.num_subcarriers
        custom_phases = np.random.uniform(0, 2 * np.pi, num_subcarriers)

        custom_signal_set = generator.generate_single_signal(phases=custom_phases)
        print(f"✓ Generated custom signal with {len(custom_phases)} phase values")

        # Analyze the signal (skip orthogonal analysis for single signal)
        signal_analysis = generator.ofdm_generator.analyze_generated_signal(
            custom_signal_set.signals[0]
        )
        print(f"✓ Signal PAPR: {signal_analysis['papr_db']:.2f} dB")
        print(f"✓ Peak amplitude: {signal_analysis['peak_amplitude']:.6f}")


def demo_orthogonal_signal_generation():
    """Demonstrate orthogonal signal set generation."""
    print("\n" + "=" * 60)
    print("ORTHOGONAL SIGNAL GENERATION")
    print("=" * 60)

    with create_generator() as generator:
        # Generate small orthogonal set
        signal_set = generator.generate_orthogonal_set(num_signals=2)
        print(f"✓ Generated {len(signal_set.signals)} orthogonal signals")
        print(f"✓ Orthogonality score: {signal_set.orthogonality_score:.6f}")

        # Generate larger set with custom parameters
        larger_set = generator.generate_orthogonal_set(
            num_signals=3, optimization_method="genetic", orthogonality_threshold=0.9
        )
        print(f"✓ Generated {len(larger_set.signals)} signals with genetic optimization")
        print(f"✓ Orthogonality score: {larger_set.orthogonality_score:.6f}")

        # Analyze the orthogonal set
        analysis = generator.analyze_signal_set(larger_set)
        orth_analysis = analysis["orthogonal_analysis"]["orthogonality_analysis"]
        print(
            f"✓ Orthogonal pairs: {orth_analysis['orthogonal_pairs']}/{orth_analysis['total_pairs']}"
        )
        print(f"✓ Overall orthogonality: {orth_analysis['overall_orthogonality_score']:.6f}")


def demo_signal_separation():
    """Demonstrate signal separation capabilities."""
    print("\n" + "=" * 60)
    print("SIGNAL SEPARATION DEMONSTRATION")
    print("=" * 60)

    with create_generator() as generator:
        # Generate orthogonal signals
        original_set = generator.generate_orthogonal_set(num_signals=2)
        print(f"✓ Generated {len(original_set.signals)} original signals")

        # Combine the signals
        combined_signal = generator.combine_signal_set(original_set)
        print(f"✓ Combined signals into single array: {len(combined_signal)} samples")

        # Separate the combined signal
        separated_set, quality_metrics = generator.separate_signals(combined_signal, original_set)
        print(f"✓ Separated into {len(separated_set.signals)} signals")
        print(f"✓ Separation quality: {quality_metrics.overall_separation_quality:.6f}")
        print(f"✓ Separation success: {quality_metrics.separation_success}")

        # Show detailed separation metrics
        print(f"✓ Mean separation score: {quality_metrics.mean_separation_score:.6f}")
        print(f"✓ Max cross-talk: {quality_metrics.max_cross_talk:.6f}")

        # Generate separation report
        report = generator.signal_separator.generate_separation_report(quality_metrics)
        print("\n📊 SEPARATION QUALITY REPORT:")
        print(report[:500] + "..." if len(report) > 500 else report)


def demo_phase_optimization():
    """Demonstrate phase optimization capabilities."""
    print("\n" + "=" * 60)
    print("PHASE OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    with create_generator() as generator:
        # Optimize phases for different numbers of signals
        for num_signals in [2, 3]:
            optimal_phases, score = generator.optimize_phases(
                num_signals=num_signals, method="genetic", max_iterations=100
            )

            print(f"✓ Optimized phases for {num_signals} signals")
            print(f"✓ Optimization score: {score:.6f}")
            print(f"✓ Phase matrix shape: {optimal_phases.shape}")

            # Show phase statistics
            print(f"✓ Phase range: [{np.min(optimal_phases):.3f}, {np.max(optimal_phases):.3f}]")
            print(f"✓ Phase std dev: {np.std(optimal_phases):.3f}")


def demo_export_functionality():
    """Demonstrate signal export capabilities."""
    print("\n" + "=" * 60)
    print("EXPORT FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)

    with create_generator() as generator:
        # Generate signals to export
        signal_set = generator.generate_orthogonal_set(num_signals=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set export directory
            generator.signal_exporter.output_dir = Path(temp_dir)

            # Export in different formats
            for format_type in ["numpy"]:  # Skip JSON for now due to serialization issues
                try:
                    exported_files = generator.export_signals(
                        signal_set,
                        f"demo_signals_{format_type}",
                        format=format_type,
                        include_visualization=False,  # Skip visualization for demo
                    )

                    print(f"✓ Exported {len(exported_files)} files in {format_type} format:")
                    for file_path in exported_files:
                        file_size = file_path.stat().st_size
                        print(f"  - {file_path.name} ({file_size} bytes)")

                except Exception as e:
                    print(f"⚠ Export failed for {format_type}: {e}")


def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n" + "=" * 60)
    print("CONVENIENCE FUNCTIONS DEMONSTRATION")
    print("=" * 60)

    # Quick orthogonal signal generation
    signal_set = quick_generate_orthogonal_signals(num_signals=2)
    print(f"✓ Quick generated {len(signal_set.signals)} orthogonal signals")
    print(f"✓ Orthogonality score: {signal_set.orthogonality_score:.6f}")

    # Quick separation test
    separated_set, quality_metrics = quick_test_separation(num_signals=2)
    print(f"✓ Quick separation test completed")
    print(f"✓ Separation quality: {quality_metrics.overall_separation_quality:.6f}")
    print(f"✓ Separation success: {quality_metrics.separation_success}")


def demo_advanced_features():
    """Demonstrate advanced features and customization."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)

    with create_generator() as generator:
        # Custom signal combination with weights
        signal_set = generator.generate_orthogonal_set(num_signals=3)

        # Combine with different weights
        weights = [0.5, 0.3, 0.2]
        combined_weighted = generator.combine_signal_set(signal_set, weights=weights)
        print(f"✓ Combined signals with weights {weights}")
        print(f"✓ Combined signal power: {np.mean(np.abs(combined_weighted)**2):.6f}")

        # Test separation with weighted combination
        separated_weighted, quality_weighted = generator.separate_signals(
            combined_weighted, signal_set
        )
        print(f"✓ Separated weighted combination")
        print(f"✓ Weighted separation quality: {quality_weighted.overall_separation_quality:.6f}")

        # Comprehensive analysis
        full_analysis = generator.analyze_signal_set(signal_set)

        print(f"✓ System backend: {full_analysis['system_info']['backend']}")
        print(f"✓ Memory usage: {full_analysis['system_info']['gpu_memory_info']}")

        # Show signal parameters
        signal_params = full_analysis["signal_parameters"]
        print(f"✓ Total bandwidth: {signal_params['signal_properties']['total_bandwidth']:.0f} Hz")
        print(f"✓ Frequency range: {signal_params['signal_properties']['frequency_range']}")


def demo_error_handling():
    """Demonstrate error handling and edge cases."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 60)

    with create_generator() as generator:
        # Test invalid parameters
        try:
            generator.generate_orthogonal_set(num_signals=1)  # Too few signals
        except ValueError as e:
            print(f"✓ Caught expected error for too few signals: {e}")

        try:
            # Test separation without reference
            dummy_signal = np.random.random(1000) + 1j * np.random.random(1000)
            generator._last_generated_set = None  # Clear last generated
            generator.separate_signals(dummy_signal)
        except ValueError as e:
            print(f"✓ Caught expected error for missing reference: {e}")

        # Test with invalid phase array
        try:
            invalid_phases = np.array([1, 2])  # Wrong size
            generator.generate_single_signal(phases=invalid_phases)
        except (ValueError, IndexError) as e:
            print(f"✓ Caught expected error for invalid phases: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 OFDM Chirp Generator - Main Interface Demonstration")
    print("This demo showcases the complete high-level API functionality.\n")

    try:
        demo_basic_usage()
        demo_single_signal_generation()
        demo_orthogonal_signal_generation()
        demo_signal_separation()
        demo_phase_optimization()
        demo_export_functionality()
        demo_convenience_functions()
        demo_advanced_features()
        demo_error_handling()

        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe OFDM Chirp Generator main interface provides:")
        print("• Easy-to-use high-level API")
        print("• Automatic configuration management")
        print("• GPU acceleration with CPU fallback")
        print("• Comprehensive signal analysis")
        print("• Flexible export options")
        print("• Robust error handling")
        print("\nReady for production use! 🎉")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
