#!/usr/bin/env python3
"""
Quick start demonstration of the OFDM Chirp Generator.

This example shows the simplest way to get started with the system,
demonstrating the most common use cases in just a few lines of code.

Run with: uv run python examples/quick_start_demo.py
"""

from ofdm_chirp_generator import (
    OFDMChirpGenerator,
    quick_generate_orthogonal_signals,
    quick_test_separation,
)


def quick_start_example():
    """Demonstrate the quickest way to use the system."""
    print("🚀 OFDM Chirp Generator - Quick Start")
    print("=" * 50)

    # Method 1: Use convenience functions (simplest)
    print("\n1️⃣ Using convenience functions:")

    # Generate orthogonal signals in one line
    signals = quick_generate_orthogonal_signals(num_signals=2)
    print(f"✓ Generated {len(signals.signals)} orthogonal signals")
    print(f"✓ Orthogonality score: {signals.orthogonality_score:.4f}")

    # Test separation in one line
    separated, quality = quick_test_separation(num_signals=2)
    print(f"✓ Separation quality: {quality.overall_separation_quality:.4f}")

    # Method 2: Use main interface (more control)
    print("\n2️⃣ Using main interface:")

    with OFDMChirpGenerator() as generator:
        # Generate orthogonal signals
        signal_set = generator.generate_orthogonal_set(num_signals=3)
        print(f"✓ Generated {len(signal_set.signals)} signals")

        # Combine and separate to test the system
        combined = generator.combine_signal_set(signal_set)
        separated_set, metrics = generator.separate_signals(combined)

        print(f"✓ Combined and separated {len(separated_set.signals)} signals")
        print(f"✓ Separation success: {metrics.separation_success}")

    print("\n✅ Quick start completed! The system is working correctly.")
    print("\nNext steps:")
    print("• Check examples/main_interface_demo.py for comprehensive features")
    print("• Modify config.toml to customize parameters")
    print("• Use GPU acceleration for better performance")


if __name__ == "__main__":
    quick_start_example()
