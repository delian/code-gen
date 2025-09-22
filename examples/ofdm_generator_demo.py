#!/usr/bin/env python3
"""
OFDM Generator Demo

This example demonstrates the core OFDM signal generation engine,
showing how to generate single signals and signal sets with different
phase configurations.

Run with UV:
    uv run examples/ofdm_generator_demo.py

Or use the installed script:
    uv run ofdm-demo
"""

import numpy as np

from ofdm_chirp_generator import ChirpConfig, OFDMConfig, OFDMGenerator

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    """Demonstrate OFDM signal generation capabilities."""
    print("OFDM Generator Demo")
    print("=" * 50)

    # Create OFDM configuration
    config = OFDMConfig(
        num_subcarriers=16,
        subcarrier_spacing=500.0,  # 500 Hz spacing
        bandwidth_per_subcarrier=400.0,  # 400 Hz bandwidth per subcarrier
        center_frequency=15000.0,  # 15 kHz center frequency
        sampling_rate=80000.0,  # 80 kHz sampling rate
        signal_duration=0.005,  # 5 ms signal duration
    )

    print(f"Configuration:")
    print(f"  Subcarriers: {config.num_subcarriers}")
    print(f"  Spacing: {config.subcarrier_spacing} Hz")
    print(f"  Bandwidth per subcarrier: {config.bandwidth_per_subcarrier} Hz")
    print(f"  Center frequency: {config.center_frequency} Hz")
    print(f"  Sampling rate: {config.sampling_rate} Hz")
    print(f"  Signal duration: {config.signal_duration} s")
    print()

    # Initialize OFDM generator
    with OFDMGenerator(config) as generator:
        print("Generator initialized successfully!")

        # Get system parameters
        params = generator.get_signal_parameters()
        print(f"Backend: {params['backend_info']['backend']}")
        print(f"Signal length: {params['signal_properties']['signal_length_samples']} samples")
        print(f"Total bandwidth: {params['signal_properties']['total_bandwidth']:.1f} Hz")
        print()

        # Demonstrate single signal generation
        print("1. Single Signal Generation")
        print("-" * 30)

        # Generate example phase arrays
        phase_array1 = generator.get_example_phase_array(0)  # Linear phases
        phase_array2 = generator.get_example_phase_array(1)  # Quadratic phases

        print(f"Phase array 1 (linear): {phase_array1[:4]} (first 4 values)")
        print(f"Phase array 2 (quadratic): {phase_array2[:4]} (first 4 values)")

        # Generate signals
        signal1 = generator.generate_single_signal(phase_array1)
        signal2 = generator.generate_single_signal(phase_array2)

        print(f"Generated signal 1: {len(signal1)} samples")
        print(f"Generated signal 2: {len(signal2)} samples")

        # Analyze signals
        analysis1 = generator.analyze_generated_signal(signal1)
        analysis2 = generator.analyze_generated_signal(signal2)

        print(
            f"Signal 1 - RMS: {analysis1['rms_amplitude']:.4f}, PAPR: {analysis1['papr_db']:.2f} dB"
        )
        print(
            f"Signal 2 - RMS: {analysis2['rms_amplitude']:.4f}, PAPR: {analysis2['papr_db']:.2f} dB"
        )
        print()

        # Demonstrate signal set generation
        print("2. Signal Set Generation")
        print("-" * 30)

        # Create phase matrix for multiple signals
        num_signals = 4
        phase_matrix = np.zeros((num_signals, config.num_subcarriers))

        for i in range(num_signals):
            phase_matrix[i, :] = generator.get_example_phase_array(i)

        # Create signal set
        signal_set = generator.create_signal_set(phase_matrix, orthogonality_score=0.85)

        print(f"Created signal set with {signal_set.num_signals} signals")
        print(f"Signal length: {signal_set.signal_length} samples")
        print(f"Orthogonality score: {signal_set.orthogonality_score}")
        print(f"Generation time: {signal_set.generation_timestamp}")
        print()

        # Demonstrate ChirpConfig usage
        print("3. ChirpConfig Usage")
        print("-" * 30)

        # Create chirp configuration
        chirp_config = ChirpConfig(
            chirp_length=int(config.signal_duration * config.sampling_rate),
            phase_matrix=phase_matrix[:2, :],  # Use first 2 signals
            amplitude=0.7,
        )

        # Generate signal using chirp config
        signal_from_config = generator.generate_signal_with_chirp_config(chirp_config)
        analysis_config = generator.analyze_generated_signal(signal_from_config)

        print(f"Signal from ChirpConfig: {len(signal_from_config)} samples")
        print(f"RMS amplitude: {analysis_config['rms_amplitude']:.4f} (scaled by amplitude factor)")
        print()

        # Demonstrate validation
        print("4. Validation and Analysis")
        print("-" * 30)

        # Validate signal generation
        validation = generator.validate_signal_generation(phase_array1)
        print("Validation results:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        print()

        # Show subcarrier power distribution
        power_dist = analysis1["subcarrier_power_distribution"]
        print("Subcarrier power distribution (first 8):")
        for i in range(min(8, len(power_dist))):
            print(f"  Subcarrier {i}: {power_dist[i]:.6f}")
        print()

        # Demonstrate frequency analysis
        print("5. Frequency Analysis")
        print("-" * 30)

        # Get OFDM structure info
        ofdm_info = generator.subcarrier_manager.get_ofdm_structure_info()
        frequencies = ofdm_info["subcarrier_frequencies"]

        print(f"Subcarrier frequencies (Hz):")
        for i, freq in enumerate(frequencies[:8]):  # Show first 8
            print(f"  Subcarrier {i}: {freq:.1f} Hz")
        if len(frequencies) > 8:
            print(f"  ... and {len(frequencies) - 8} more")
        print()

        print(
            f"Frequency range: {ofdm_info['frequency_range'][0]:.1f} - {ofdm_info['frequency_range'][1]:.1f} Hz"
        )
        print(f"Total bandwidth: {ofdm_info['total_bandwidth']:.1f} Hz")

        # Optional: Create plots if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            create_plots(generator, signal1, signal2, config)
        else:
            print("\nNote: Install matplotlib for signal visualization")

        print("\nDemo completed successfully!")


def create_plots(generator, signal1, signal2, config):
    """Create visualization plots of the generated signals."""
    print("\n6. Signal Visualization")
    print("-" * 30)

    # Convert signals to CPU if needed
    if hasattr(signal1, "get"):
        signal1 = signal1.get()
    if hasattr(signal2, "get"):
        signal2 = signal2.get()

    # Create time vector
    t = np.arange(len(signal1)) / config.sampling_rate

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("OFDM Signal Generation Demo", fontsize=14)

    # Time domain plots
    axes[0, 0].plot(t * 1000, np.real(signal1), "b-", linewidth=0.8)
    axes[0, 0].set_title("Signal 1 - Real Part")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t * 1000, np.real(signal2), "r-", linewidth=0.8)
    axes[0, 1].set_title("Signal 2 - Real Part")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Frequency domain plots
    freq1 = np.fft.fftfreq(len(signal1), 1 / config.sampling_rate)
    fft1 = np.fft.fft(signal1)
    freq2 = np.fft.fftfreq(len(signal2), 1 / config.sampling_rate)
    fft2 = np.fft.fft(signal2)

    # Plot positive frequencies only
    pos_mask1 = freq1 >= 0
    pos_mask2 = freq2 >= 0

    axes[1, 0].plot(
        freq1[pos_mask1] / 1000, 20 * np.log10(np.abs(fft1[pos_mask1]) + 1e-10), "b-", linewidth=0.8
    )
    axes[1, 0].set_title("Signal 1 - Frequency Spectrum")
    axes[1, 0].set_xlabel("Frequency (kHz)")
    axes[1, 0].set_ylabel("Magnitude (dB)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, config.sampling_rate / 2000)

    axes[1, 1].plot(
        freq2[pos_mask2] / 1000, 20 * np.log10(np.abs(fft2[pos_mask2]) + 1e-10), "r-", linewidth=0.8
    )
    axes[1, 1].set_title("Signal 2 - Frequency Spectrum")
    axes[1, 1].set_xlabel("Frequency (kHz)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, config.sampling_rate / 2000)

    plt.tight_layout()
    plt.savefig("ofdm_generator_demo.png", dpi=150, bbox_inches="tight")
    print("Plots saved as 'ofdm_generator_demo.png'")

    # Show subcarrier structure
    ofdm_info = generator.subcarrier_manager.get_ofdm_structure_info()
    frequencies = np.array(ofdm_info["subcarrier_frequencies"])

    plt.figure(figsize=(10, 6))
    plt.stem(frequencies / 1000, np.ones(len(frequencies)), basefmt=" ")
    plt.title("OFDM Subcarrier Frequency Allocation")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Subcarrier")
    plt.grid(True, alpha=0.3)

    # Add bandwidth indicators
    for i, freq in enumerate(frequencies):
        bandwidth = config.bandwidth_per_subcarrier
        plt.axvspan(
            (freq - bandwidth / 2) / 1000, (freq + bandwidth / 2) / 1000, alpha=0.2, color="blue"
        )

    plt.tight_layout()
    plt.savefig("ofdm_subcarrier_structure.png", dpi=150, bbox_inches="tight")
    print("Subcarrier structure saved as 'ofdm_subcarrier_structure.png'")


if __name__ == "__main__":
    main()
