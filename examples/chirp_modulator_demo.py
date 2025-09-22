#!/usr/bin/env python3
"""
Demonstration of ChirpModulator functionality.

This example shows how to use the ChirpModulator class to generate
linear frequency modulated (chirp) signals for OFDM subcarriers.
"""

import numpy as np

from ofdm_chirp_generator import ChirpModulator, GPUBackend, OFDMConfig

# Try to import matplotlib, handle gracefully if not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    """Demonstrate ChirpModulator functionality."""
    print("OFDM Chirp Modulator Demonstration")
    print("=" * 40)

    # Create OFDM configuration
    config = OFDMConfig(
        num_subcarriers=4,
        subcarrier_spacing=1000.0,  # 1 kHz
        bandwidth_per_subcarrier=500.0,  # 500 Hz
        center_frequency=10000.0,  # 10 kHz
        sampling_rate=50000.0,  # 50 kHz
        signal_duration=0.01,  # 10 ms
    )

    print(f"Configuration:")
    print(f"  Subcarriers: {config.num_subcarriers}")
    print(f"  Spacing: {config.subcarrier_spacing} Hz")
    print(f"  Bandwidth per subcarrier: {config.bandwidth_per_subcarrier} Hz")
    print(f"  Center frequency: {config.center_frequency} Hz")
    print(f"  Sampling rate: {config.sampling_rate} Hz")
    print(f"  Signal duration: {config.signal_duration} s")
    print()

    # Create GPU backend (will fall back to CPU if GPU unavailable)
    gpu_backend = GPUBackend()
    print(f"Backend: {gpu_backend.device_info['backend']}")
    print()

    # Create chirp modulator
    modulator = ChirpModulator(config, gpu_backend)
    print(f"Chirp length constraints: {modulator.chirp_length_constraints}")
    print()

    # Generate individual chirp signals with different phase offsets
    print("Generating individual chirp signals...")
    phase_offsets = [0.0, np.pi / 4, np.pi / 2, np.pi]
    chirp_signals = []

    for i, phase in enumerate(phase_offsets):
        chirp = modulator.generate_chirp_signal(i, phase)
        chirp_signals.append(chirp)

        # Get characteristics
        characteristics = modulator.get_chirp_characteristics(i, phase)
        print(
            f"  Subcarrier {i}: freq={characteristics['center_frequency']:.1f}Hz, "
            f"phase={phase:.3f}rad, sweep={characteristics['sweep_rate']:.1f}Hz/s"
        )

    print()

    # Generate multi-chirp OFDM signal
    print("Generating multi-chirp OFDM signal...")
    phase_array = np.array([0.0, np.pi / 4, np.pi / 2, np.pi])
    ofdm_signal = modulator.generate_multi_chirp_signal(phase_array)

    # Convert to CPU for analysis if needed
    if hasattr(ofdm_signal, "get"):
        ofdm_signal = ofdm_signal.get()

    print(f"OFDM signal generated: {len(ofdm_signal)} samples")
    print(f"Signal power: {np.mean(np.abs(ofdm_signal)**2):.6f}")
    print()

    # Analyze frequency content
    print("Frequency analysis...")
    fft_signal = np.fft.fft(ofdm_signal)
    frequencies = np.fft.fftfreq(len(ofdm_signal), 1 / config.sampling_rate)

    # Find peak frequencies
    magnitude = np.abs(fft_signal)
    peak_indices = np.argsort(magnitude)[-config.num_subcarriers * 2 :]  # Top peaks
    peak_frequencies = frequencies[peak_indices]
    peak_frequencies = peak_frequencies[peak_frequencies > 0]  # Positive frequencies only
    peak_frequencies = np.sort(peak_frequencies)[: config.num_subcarriers]

    print("Expected vs Actual subcarrier frequencies:")
    for i in range(config.num_subcarriers):
        expected_freq = modulator._calculate_subcarrier_frequency(i)
        if i < len(peak_frequencies):
            actual_freq = peak_frequencies[i]
            error = abs(actual_freq - expected_freq)
            print(
                f"  Subcarrier {i}: expected={expected_freq:.1f}Hz, "
                f"actual={actual_freq:.1f}Hz, error={error:.1f}Hz"
            )
        else:
            print(f"  Subcarrier {i}: expected={expected_freq:.1f}Hz, actual=N/A")

    print()

    # Test phase validation
    print("Testing phase validation...")
    test_phases = np.array([3 * np.pi, -np.pi / 2, 5 * np.pi, -2 * np.pi])
    normalized_phases = modulator.validate_phase_array(test_phases)

    print("Phase normalization:")
    for i, (original, normalized) in enumerate(zip(test_phases, normalized_phases)):
        print(f"  Phase {i}: {original:.3f} -> {normalized:.3f} rad")

    print()

    # Memory cleanup
    gpu_backend.cleanup_memory()
    print("Memory cleanup completed.")

    # Optional: Create plots if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        create_plots(config, chirp_signals, ofdm_signal, phase_offsets)
    else:
        print("Matplotlib not available - skipping plots")


def create_plots(config, chirp_signals, ofdm_signal, phase_offsets):
    """Create visualization plots."""
    print("Creating visualization plots...")

    # Time vector
    t = np.arange(len(ofdm_signal)) / config.sampling_rate

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("OFDM Chirp Modulator Demonstration")

    # Plot 1: Individual chirp signals (real part)
    ax1 = axes[0, 0]
    for i, (chirp, phase) in enumerate(zip(chirp_signals, phase_offsets)):
        if hasattr(chirp, "get"):
            chirp = chirp.get()
        ax1.plot(t[:500], np.real(chirp[:500]), label=f"Subcarrier {i} (φ={phase:.2f})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Individual Chirp Signals (Real Part)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Combined OFDM signal
    ax2 = axes[0, 1]
    ax2.plot(t[:500], np.real(ofdm_signal[:500]), "b-", label="Real")
    ax2.plot(t[:500], np.imag(ofdm_signal[:500]), "r-", label="Imaginary")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Combined OFDM Signal")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Frequency spectrum
    ax3 = axes[1, 0]
    fft_signal = np.fft.fft(ofdm_signal)
    frequencies = np.fft.fftfreq(len(ofdm_signal), 1 / config.sampling_rate)
    magnitude_db = 20 * np.log10(np.abs(fft_signal) + 1e-10)

    # Plot positive frequencies only
    pos_mask = frequencies > 0
    ax3.plot(frequencies[pos_mask], magnitude_db[pos_mask])
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Magnitude (dB)")
    ax3.set_title("Frequency Spectrum")
    ax3.grid(True)

    # Mark expected subcarrier frequencies
    for i in range(config.num_subcarriers):
        freq = (
            config.center_frequency
            + (i - (config.num_subcarriers - 1) / 2) * config.subcarrier_spacing
        )
        ax3.axvline(freq, color="red", linestyle="--", alpha=0.7, label=f"SC{i}" if i == 0 else "")

    # Plot 4: Instantaneous frequency of first chirp
    ax4 = axes[1, 1]
    chirp0 = chirp_signals[0]
    if hasattr(chirp0, "get"):
        chirp0 = chirp0.get()

    # Calculate instantaneous frequency
    phase = np.unwrap(np.angle(chirp0))
    dt = 1.0 / config.sampling_rate
    inst_freq = np.diff(phase) / (2 * np.pi * dt)

    ax4.plot(t[1:1000], inst_freq[:999])
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_title("Instantaneous Frequency (Subcarrier 0)")
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig("chirp_modulator_demo.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Plots saved as 'chirp_modulator_demo.png'")


if __name__ == "__main__":
    main()
