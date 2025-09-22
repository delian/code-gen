#!/usr/bin/env python3
"""
Signal Separation Demo

This script demonstrates the signal separation capabilities of the OFDM chirp generator.
It shows how to:
1. Generate orthogonal OFDM signals
2. Combine them into a single transmission
3. Separate the individual signals using phase-based correlation analysis
4. Evaluate separation quality and generate diagnostic reports
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ofdm_chirp_generator import (
    OFDMConfig,
    OFDMGenerator,
    OrthogonalSignalGenerator,
    SignalSeparator,
    SignalSet,
    get_config,
)


def create_demo_config():
    """Create demonstration OFDM configuration."""
    return OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.004,  # 4ms signal
    )


def generate_orthogonal_signals(config, num_signals=3):
    """Generate a set of orthogonal OFDM signals."""
    print(f"Generating {num_signals} orthogonal OFDM signals...")

    # Create orthogonal signal generator
    orthogonal_generator = OrthogonalSignalGenerator(config)

    # Generate orthogonal signal set
    signal_set = orthogonal_generator.generate_orthogonal_signal_set(num_signals)

    print(f"Generated signal set with orthogonality score: {signal_set.orthogonality_score:.4f}")
    print(f"Signal length: {signal_set.signal_length} samples")

    return signal_set


def combine_signals(signal_set, weights=None):
    """Combine multiple signals into a single transmission."""
    if weights is None:
        weights = np.ones(signal_set.num_signals)

    print(f"Combining {signal_set.num_signals} signals with weights: {weights}")

    # Combine signals with specified weights
    combined_signal = np.zeros_like(signal_set.signals[0], dtype=complex)
    for i, signal in enumerate(signal_set.signals):
        combined_signal += weights[i] * signal

    # Calculate combined signal properties
    combined_power = np.mean(np.abs(combined_signal) ** 2)
    peak_amplitude = np.max(np.abs(combined_signal))
    papr_db = 10 * np.log10(peak_amplitude**2 / combined_power) if combined_power > 0 else 0

    print(f"Combined signal power: {combined_power:.6f}")
    print(f"Peak-to-Average Power Ratio: {papr_db:.2f} dB")

    return combined_signal


def perform_signal_separation(combined_signal, reference_signal_set, config):
    """Perform signal separation using phase-based correlation analysis."""
    print("\nPerforming signal separation...")

    # Create signal separator
    separator = SignalSeparator(config)

    # Validate separation capability
    validation = separator.validate_separation_capability(
        reference_signal_set.signals, reference_signal_set.phases
    )

    print(f"Separation feasibility: {'YES' if validation['separation_feasible'] else 'NO'}")
    print(f"Reference orthogonality quality: {validation['orthogonality_quality']:.4f}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    # Perform separation
    separated_set, quality_metrics = separator.separate_signal_set(
        combined_signal, reference_signal_set
    )

    print(f"\nSeparation completed!")
    print(f"Separation success: {'YES' if quality_metrics.separation_success else 'NO'}")
    print(f"Overall separation quality: {quality_metrics.overall_separation_quality:.4f}")

    return separated_set, quality_metrics, separator


def analyze_separation_results(original_set, separated_set, quality_metrics):
    """Analyze and display separation results."""
    print("\n" + "=" * 60)
    print("SEPARATION ANALYSIS RESULTS")
    print("=" * 60)

    # Individual signal analysis
    print("\nINDIVIDUAL SIGNAL SEPARATION SCORES:")
    for i, score in enumerate(quality_metrics.separation_scores):
        print(f"  Signal {i+1}: {score:.6f}")

    print(f"\nMean separation score: {quality_metrics.mean_separation_score:.6f}")
    print(f"Minimum separation score: {quality_metrics.min_separation_score:.6f}")

    # Cross-talk analysis
    print(f"\nCROSS-TALK ANALYSIS:")
    print(f"  Maximum cross-talk: {quality_metrics.max_cross_talk:.6f}")
    print(f"  Mean cross-talk: {quality_metrics.mean_cross_talk:.6f}")

    # Signal-to-Interference Ratios
    if quality_metrics.signal_to_interference_ratios:
        print(f"\nSIGNAL-TO-INTERFERENCE RATIOS:")
        for i, sir in enumerate(quality_metrics.signal_to_interference_ratios):
            sir_db = 10 * np.log10(sir) if sir > 0 else -np.inf
            print(f"  Signal {i+1}: {sir_db:.2f} dB")

    # Signal energy comparison
    print(f"\nSIGNAL ENERGY COMPARISON:")
    for i in range(original_set.num_signals):
        original_energy = np.sum(np.abs(original_set.signals[i]) ** 2)
        separated_energy = np.sum(np.abs(separated_set.signals[i]) ** 2)
        energy_ratio = separated_energy / original_energy if original_energy > 0 else 0

        print(
            f"  Signal {i+1}: Original={original_energy:.6f}, "
            f"Separated={separated_energy:.6f}, Ratio={energy_ratio:.4f}"
        )


def plot_separation_results(original_set, combined_signal, separated_set, quality_metrics):
    """Plot separation results for visualization."""
    try:
        num_signals = original_set.num_signals
        time_axis = np.arange(len(combined_signal)) / 50000.0  # Convert to time in seconds

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Signal Separation Results", fontsize=16)

        # Plot 1: Original signals
        ax1 = axes[0, 0]
        for i, signal in enumerate(original_set.signals):
            ax1.plot(time_axis, np.real(signal), label=f"Signal {i+1} (Real)", alpha=0.7)
        ax1.set_title("Original Signals (Real Part)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Combined signal
        ax2 = axes[0, 1]
        ax2.plot(time_axis, np.real(combined_signal), "r-", label="Real Part")
        ax2.plot(time_axis, np.imag(combined_signal), "b-", label="Imaginary Part", alpha=0.7)
        ax2.set_title("Combined Signal")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Separated signals
        ax3 = axes[1, 0]
        for i, signal in enumerate(separated_set.signals):
            ax3.plot(time_axis, np.real(signal), label=f"Separated {i+1} (Real)", alpha=0.7)
        ax3.set_title("Separated Signals (Real Part)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Separation quality metrics
        ax4 = axes[1, 1]

        # Bar plot of separation scores
        signal_indices = range(1, num_signals + 1)
        bars = ax4.bar(
            signal_indices,
            quality_metrics.separation_scores,
            alpha=0.7,
            color="green",
            label="Separation Scores",
        )

        # Add threshold line
        ax4.axhline(y=0.8, color="red", linestyle="--", alpha=0.7, label="Threshold (0.8)")

        ax4.set_title("Separation Quality Scores")
        ax4.set_xlabel("Signal Index")
        ax4.set_ylabel("Separation Score")
        ax4.set_ylim(0, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

        # Plot cross-talk matrix
        if quality_metrics.cross_talk_matrix.size > 0:
            fig2, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(quality_metrics.cross_talk_matrix, cmap="hot", interpolation="nearest")
            ax.set_title("Cross-Talk Matrix")
            ax.set_xlabel("Separated Signal Index")
            ax.set_ylabel("Original Signal Index")

            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label("Cross-Talk Level")

            # Add text annotations
            for i in range(quality_metrics.cross_talk_matrix.shape[0]):
                for j in range(quality_metrics.cross_talk_matrix.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        f"{quality_metrics.cross_talk_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if quality_metrics.cross_talk_matrix[i, j] > 0.5 else "black",
                    )

            plt.tight_layout()
            plt.show()

    except ImportError:
        print("Matplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"Error creating plots: {e}")


def demonstrate_separation_with_noise(original_set, config, noise_level=0.1):
    """Demonstrate separation performance with added noise."""
    print(f"\n" + "=" * 60)
    print(f"SEPARATION WITH NOISE (SNR ≈ {-20*np.log10(noise_level):.1f} dB)")
    print("=" * 60)

    # Add noise to combined signal
    combined_signal = np.sum(original_set.signals, axis=0)
    signal_power = np.mean(np.abs(combined_signal) ** 2)
    noise = (
        noise_level
        * np.sqrt(signal_power)
        * (np.random.randn(len(combined_signal)) + 1j * np.random.randn(len(combined_signal)))
    )
    noisy_combined = combined_signal + noise

    # Perform separation with noise
    separator = SignalSeparator(config)
    separated_set, quality_metrics = separator.separate_signal_set(noisy_combined, original_set)

    print(f"Separation with noise completed!")
    print(f"Separation success: {'YES' if quality_metrics.separation_success else 'NO'}")
    print(f"Overall separation quality: {quality_metrics.overall_separation_quality:.4f}")
    print(f"Mean separation score: {quality_metrics.mean_separation_score:.4f}")
    print(f"Maximum cross-talk: {quality_metrics.max_cross_talk:.4f}")

    return separated_set, quality_metrics


def main():
    """Main demonstration function."""
    print("OFDM Signal Separation Demonstration")
    print("=" * 50)

    # Create configuration
    config = create_demo_config()
    print(f"OFDM Configuration:")
    print(f"  Subcarriers: {config.num_subcarriers}")
    print(f"  Signal duration: {config.signal_duration*1000:.1f} ms")
    print(f"  Sampling rate: {config.sampling_rate/1000:.0f} kHz")

    try:
        # Generate orthogonal signals
        original_signal_set = generate_orthogonal_signals(config, num_signals=3)

        # Combine signals with different weights
        weights = [1.0, 0.8, 1.2]  # Different signal strengths
        combined_signal = combine_signals(original_signal_set, weights)

        # Perform signal separation
        separated_set, quality_metrics, separator = perform_signal_separation(
            combined_signal, original_signal_set, config
        )

        # Generate comprehensive report
        print(separator.generate_separation_report(quality_metrics))

        # Analyze results
        analyze_separation_results(original_signal_set, separated_set, quality_metrics)

        # Plot results
        plot_separation_results(
            original_signal_set, combined_signal, separated_set, quality_metrics
        )

        # Demonstrate separation with noise
        demonstrate_separation_with_noise(original_signal_set, config, noise_level=0.05)
        demonstrate_separation_with_noise(original_signal_set, config, noise_level=0.2)

        # Cleanup resources
        separator.cleanup_resources()

        print(f"\nDemo completed successfully!")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
