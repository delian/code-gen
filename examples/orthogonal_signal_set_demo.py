#!/usr/bin/env python3
"""
Orthogonal Signal Set Generation Demo

This example demonstrates the generation of multiple orthogonal OFDM signals
using the OrthogonalSignalGenerator class.
"""

from datetime import datetime

import numpy as np

from ofdm_chirp_generator import OFDMConfig, OrthogonalSetConfig, OrthogonalSignalGenerator

# Try to import matplotlib, but continue without it if not available
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    """Demonstrate orthogonal signal set generation."""
    print("=" * 60)
    print("ORTHOGONAL SIGNAL SET GENERATION DEMO")
    print("=" * 60)

    # Create OFDM configuration
    ofdm_config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,  # 1 kHz spacing
        bandwidth_per_subcarrier=500.0,  # 500 Hz per subcarrier
        center_frequency=10000.0,  # 10 kHz center
        sampling_rate=50000.0,  # 50 kHz sampling rate
        signal_duration=0.002,  # 2 ms duration
    )

    print(f"OFDM Configuration:")
    print(f"  Subcarriers: {ofdm_config.num_subcarriers}")
    print(f"  Subcarrier spacing: {ofdm_config.subcarrier_spacing} Hz")
    print(f"  Center frequency: {ofdm_config.center_frequency} Hz")
    print(f"  Signal duration: {ofdm_config.signal_duration * 1000} ms")
    print(f"  Samples per signal: {int(ofdm_config.signal_duration * ofdm_config.sampling_rate)}")
    print()

    # Create orthogonal signal generator
    print("Initializing orthogonal signal generator...")
    with OrthogonalSignalGenerator(ofdm_config=ofdm_config) as generator:

        # Configure orthogonal set parameters
        generator.orthogonal_set_config.max_signals = 8
        generator.orthogonal_set_config.orthogonality_threshold = 0.9
        generator.orthogonal_set_config.optimization_method = "hybrid"

        print(f"Generator backend: {'GPU' if generator.gpu_backend.is_gpu_available else 'CPU'}")
        print()

        # Demonstrate single orthogonal set generation
        print("1. SINGLE ORTHOGONAL SET GENERATION")
        print("-" * 40)

        num_signals = 3
        print(f"Generating {num_signals} orthogonal signals...")

        start_time = datetime.now()
        signal_set = generator.generate_orthogonal_signal_set(num_signals)
        generation_time = (datetime.now() - start_time).total_seconds()

        print(f"Generation completed in {generation_time:.3f} seconds")
        print(f"Orthogonality score: {signal_set.orthogonality_score:.6f}")
        print(f"Signal length: {signal_set.signal_length} samples")
        print()

        # Analyze the generated set
        print("Analyzing orthogonal signal set...")
        analysis = generator.analyze_orthogonal_set(signal_set)

        orth_analysis = analysis["orthogonality_analysis"]
        print(f"Orthogonality Analysis:")
        print(f"  Overall score: {orth_analysis['overall_orthogonality_score']:.6f}")
        print(
            f"  Orthogonal pairs: {orth_analysis['orthogonal_pairs']}/{orth_analysis['total_pairs']}"
        )
        print(f"  Max cross-correlation: {orth_analysis['max_cross_correlation']:.6f}")
        print(f"  Mean cross-correlation: {orth_analysis['mean_cross_correlation']:.6f}")
        print()

        # Display signal properties
        set_metrics = analysis["set_metrics"]
        print(f"Signal Set Properties:")
        print(f"  Mean signal power: {set_metrics['mean_signal_power']:.6f}")
        print(f"  Power balance coefficient: {set_metrics['power_balance_coefficient']:.6f}")
        print()

        # Demonstrate batch generation
        print("2. BATCH ORTHOGONAL SET GENERATION")
        print("-" * 40)

        signal_counts = [2, 3, 4]
        print(f"Generating batch sets for {signal_counts} signals...")

        start_time = datetime.now()
        batch_results = generator.generate_batch_orthogonal_sets(signal_counts)
        batch_time = (datetime.now() - start_time).total_seconds()

        print(f"Batch generation completed in {batch_time:.3f} seconds")
        print(f"Generated {len(batch_results)} signal sets")
        print()

        for count, signal_set in batch_results.items():
            print(f"  {count} signals: score={signal_set.orthogonality_score:.6f}")
        print()

        # Demonstrate phase matrix management
        print("3. PHASE MATRIX MANAGEMENT")
        print("-" * 40)

        # Store a custom phase configuration
        custom_phases = np.random.uniform(0, 2 * np.pi, (2, ofdm_config.num_subcarriers))
        generator.phase_matrix_manager.store_configuration(
            "custom_2sig", custom_phases, 0.95, {"source": "manual"}
        )

        # List stored configurations
        configs = generator.phase_matrix_manager.list_configurations()
        print(f"Stored configurations: {len(configs)}")
        for config_id in configs:
            _, metadata = generator.phase_matrix_manager.retrieve_configuration(config_id)
            print(
                f"  {config_id}: score={metadata['orthogonality_score']:.6f}, "
                f"signals={metadata['num_signals']}"
            )
        print()

        # Test cached generation
        print("Testing cached generation...")
        cached_signal_set = generator.generate_orthogonal_signal_set(2)
        if cached_signal_set.metadata.get("generation_method") == "cached":
            print("Successfully used cached configuration!")
            print(f"Cached config ID: {cached_signal_set.metadata.get('cached_config_id')}")
        else:
            print("Generated new configuration (no suitable cache found)")
        print()

        # Demonstrate maximum signal determination
        print("4. MAXIMUM ORTHOGONAL SIGNALS")
        print("-" * 40)

        print("Determining maximum achievable orthogonal signals...")
        max_signals = generator.get_maximum_orthogonal_signals()
        print(f"Maximum orthogonal signals: {max_signals}")
        print()

        # Export/import demonstration
        print("5. CONFIGURATION EXPORT/IMPORT")
        print("-" * 40)

        # Export configurations
        export_data = generator.export_phase_configurations()
        print(f"Exported {len(export_data['configurations'])} configurations")

        # Clear and reimport
        original_count = len(generator.phase_matrix_manager.list_configurations())
        generator.phase_matrix_manager.clear_all()
        print("Cleared all configurations")

        imported_count = generator.import_phase_configurations(export_data)
        print(f"Imported {imported_count} configurations")
        print()

        # Visualization (if matplotlib is available)
        if MATPLOTLIB_AVAILABLE:
            try:
                print("6. SIGNAL VISUALIZATION")
                print("-" * 40)

                # Plot the first few signals from the orthogonal set
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle("Orthogonal OFDM Signal Set Visualization")

                # Time domain plots
                time_axis = (
                    np.arange(signal_set.signal_length) / ofdm_config.sampling_rate * 1000
                )  # ms

                for i in range(min(2, signal_set.num_signals)):
                    signal = signal_set.signals[i]

                    # Real part
                    axes[0, i].plot(time_axis, np.real(signal))
                    axes[0, i].set_title(f"Signal {i+1} - Real Part")
                    axes[0, i].set_xlabel("Time (ms)")
                    axes[0, i].set_ylabel("Amplitude")
                    axes[0, i].grid(True)

                    # Magnitude spectrum
                    freq_axis = (
                        np.fft.fftfreq(len(signal), 1 / ofdm_config.sampling_rate) / 1000
                    )  # kHz
                    spectrum = np.abs(np.fft.fft(signal))

                    axes[1, i].plot(
                        freq_axis[: len(freq_axis) // 2],
                        20 * np.log10(spectrum[: len(spectrum) // 2] + 1e-10),
                    )
                    axes[1, i].set_title(f"Signal {i+1} - Magnitude Spectrum")
                    axes[1, i].set_xlabel("Frequency (kHz)")
                    axes[1, i].set_ylabel("Magnitude (dB)")
                    axes[1, i].grid(True)

                plt.tight_layout()
                plt.show()

                # Plot correlation matrix
                if signal_set.num_signals > 1:
                    correlation_matrix = orth_analysis["correlation_matrix"]

                    plt.figure(figsize=(8, 6))
                    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                    plt.colorbar(label="Correlation Coefficient")
                    plt.title("Signal Cross-Correlation Matrix")
                    plt.xlabel("Signal Index")
                    plt.ylabel("Signal Index")

                    # Add correlation values as text
                    for i in range(correlation_matrix.shape[0]):
                        for j in range(correlation_matrix.shape[1]):
                            plt.text(
                                j,
                                i,
                                f"{correlation_matrix[i, j]:.3f}",
                                ha="center",
                                va="center",
                                color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                            )

                    plt.tight_layout()
                    plt.show()

                print("Visualization completed!")

            except Exception as e:
                print(f"Visualization error: {e}")
        else:
            print("6. SIGNAL VISUALIZATION")
            print("-" * 40)
            print("Matplotlib not available - skipping visualization")

        print()
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Summary
        print("\nSUMMARY:")
        print(
            f"✓ Generated {num_signals} orthogonal signals with score {signal_set.orthogonality_score:.6f}"
        )
        print(f"✓ Batch generated {len(batch_results)} signal sets")
        print(f"✓ Demonstrated phase matrix management and caching")
        print(f"✓ Maximum orthogonal signals: {max_signals}")
        print(f"✓ Configuration export/import functionality")

        if signal_set.orthogonality_score >= 0.8:
            print("\n🎉 Excellent orthogonality achieved!")
        elif signal_set.orthogonality_score >= 0.6:
            print("\n✅ Good orthogonality achieved!")
        else:
            print("\n⚠️  Orthogonality could be improved - try adjusting parameters")


if __name__ == "__main__":
    main()
