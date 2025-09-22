#!/usr/bin/env python3
"""
OFDM Signal Structure Demonstration

This example demonstrates the OFDM signal structure implementation,
including subcarrier management, frequency allocation, and signal assembly.
"""

import numpy as np

from ofdm_chirp_generator import GPUBackend, OFDMConfig, SubcarrierManager

# Try to import matplotlib, handle gracefully if not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    """Demonstrate OFDM signal structure functionality."""
    print("OFDM Signal Structure Demo")
    print("=" * 50)

    # Create OFDM configuration
    config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,  # 1 kHz spacing
        bandwidth_per_subcarrier=800.0,  # 800 Hz bandwidth per subcarrier
        center_frequency=10000.0,  # 10 kHz center frequency
        sampling_rate=50000.0,  # 50 kHz sampling rate
        signal_duration=0.1,  # 100 ms signal duration
    )

    print(f"Configuration:")
    print(f"  Subcarriers: {config.num_subcarriers}")
    print(f"  Spacing: {config.subcarrier_spacing} Hz")
    print(f"  Bandwidth per subcarrier: {config.bandwidth_per_subcarrier} Hz")
    print(f"  Center frequency: {config.center_frequency} Hz")
    print(f"  Sampling rate: {config.sampling_rate} Hz")
    print(f"  Signal duration: {config.signal_duration} s")
    print()

    # Create GPU backend (will fall back to CPU if GPU not available)
    gpu_backend = GPUBackend()
    print(f"Backend: {gpu_backend.device_info['backend']}")
    print()

    # Create subcarrier manager
    manager = SubcarrierManager(config, gpu_backend)

    # Display subcarrier frequency allocation
    print("Subcarrier Frequency Allocation:")
    frequencies = manager.get_all_subcarrier_frequencies()
    for i, freq in enumerate(frequencies):
        start_freq, end_freq = manager.get_subcarrier_bandwidth_range(i)
        print(f"  Subcarrier {i}: {freq:.1f} Hz (range: {start_freq:.1f} - {end_freq:.1f} Hz)")
    print()

    # Validate OFDM structure
    validation = manager.validate_ofdm_structure()
    print("OFDM Structure Validation:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {key}: {status}")
    print()

    # Get structure information
    info = manager.get_ofdm_structure_info()
    print("OFDM Structure Information:")
    print(f"  Total bandwidth: {info['total_bandwidth']:.1f} Hz")
    print(
        f"  Frequency range: {info['frequency_range'][0]:.1f} - {info['frequency_range'][1]:.1f} Hz"
    )
    print(f"  Overlapping subcarriers: {len(info['overlapping_subcarriers'])}")
    print()

    # Generate OFDM signal with different phase patterns
    print("Generating OFDM signals with different phase patterns...")

    # Pattern 1: All phases zero
    phases_zero = np.zeros(config.num_subcarriers)
    signal_zero = manager.assemble_ofdm_signal(phases_zero)

    # Pattern 2: Random phases
    phases_random = np.random.uniform(0, 2 * np.pi, config.num_subcarriers)
    signal_random = manager.assemble_ofdm_signal(phases_random)

    # Pattern 3: Linear phase progression
    phases_linear = np.linspace(0, 2 * np.pi, config.num_subcarriers, endpoint=False)
    signal_linear = manager.assemble_ofdm_signal(phases_linear)

    print(f"Generated signals:")
    print(f"  Zero phases: {len(signal_zero)} samples")
    print(f"  Random phases: {len(signal_random)} samples")
    print(f"  Linear phases: {len(signal_linear)} samples")
    print()

    # Analyze frequency domain properties
    print("Frequency Domain Analysis:")

    # Analyze power distribution for zero-phase signal
    power_analysis = manager.analyze_subcarrier_power(signal_zero)
    print("Power distribution (zero phases):")
    for i, power in power_analysis.items():
        print(f"  Subcarrier {i}: {power:.2e}")
    print()

    # Create visualization if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        create_visualization(
            manager,
            signal_zero,
            signal_random,
            signal_linear,
            phases_zero,
            phases_random,
            phases_linear,
        )
        print("Visualization saved as 'ofdm_structure_demo.png'")
    else:
        print("Matplotlib not available - skipping visualization")

    print("Demo completed successfully!")


def create_visualization(
    manager, signal_zero, signal_random, signal_linear, phases_zero, phases_random, phases_linear
):
    """Create visualization of OFDM signal structure."""
    config = manager.ofdm_config

    # Convert signals to CPU if needed
    if hasattr(signal_zero, "get"):
        signal_zero = signal_zero.get()
    if hasattr(signal_random, "get"):
        signal_random = signal_random.get()
    if hasattr(signal_linear, "get"):
        signal_linear = signal_linear.get()

    # Create time vector
    time_vector = np.arange(len(signal_zero)) / config.sampling_rate

    # Create frequency vector for FFT
    freq_vector = np.fft.fftfreq(len(signal_zero), 1 / config.sampling_rate)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("OFDM Signal Structure Demonstration", fontsize=16)

    # Plot time domain signals
    signals = [signal_zero, signal_random, signal_linear]
    phase_labels = ["Zero Phases", "Random Phases", "Linear Phases"]

    for i, (signal, label) in enumerate(zip(signals, phase_labels)):
        # Time domain (real part)
        axes[i, 0].plot(time_vector[:1000], np.real(signal[:1000]))
        axes[i, 0].set_title(f"{label} - Time Domain (Real)")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True)

        # Frequency domain (magnitude)
        fft_signal = np.fft.fft(signal)
        axes[i, 1].plot(
            freq_vector[: len(freq_vector) // 2], np.abs(fft_signal[: len(fft_signal) // 2])
        )
        axes[i, 1].set_title(f"{label} - Frequency Domain")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Magnitude")
        axes[i, 1].grid(True)

        # Mark subcarrier frequencies
        frequencies = manager.get_all_subcarrier_frequencies()
        for freq in frequencies:
            if freq >= 0 and freq <= config.sampling_rate / 2:
                axes[i, 1].axvline(x=freq, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("ofdm_structure_demo.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
