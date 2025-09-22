#!/usr/bin/env python3
"""
Signal Export and Visualization Demo

This example demonstrates the signal export and visualization capabilities
of the OFDM chirp generator, including:
- Exporting signals in multiple formats
- Creating visualizations for signal analysis
- Saving and loading phase configurations
- Generating comprehensive signal reports
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from ofdm_chirp_generator import (
    ChirpConfig,
    ChirpModulator,
    OFDMConfig,
    OFDMGenerator,
    PhaseConfigurationManager,
    SignalExporter,
    SignalLoader,
    SignalSet,
    SignalVisualizer,
    SubcarrierManager,
)


def create_sample_signals():
    """Create sample OFDM signals for demonstration."""
    print("Creating sample OFDM signals...")

    # Configure OFDM parameters
    ofdm_config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002,
    )

    # Create phase matrix for orthogonal signals
    num_signals = 3
    phases = np.array(
        [
            [
                0.0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
                np.pi,
                5 * np.pi / 4,
                3 * np.pi / 2,
                7 * np.pi / 4,
            ],
            [
                np.pi / 8,
                3 * np.pi / 8,
                5 * np.pi / 8,
                7 * np.pi / 8,
                9 * np.pi / 8,
                11 * np.pi / 8,
                13 * np.pi / 8,
                15 * np.pi / 8,
            ],
            [
                np.pi / 6,
                np.pi / 3,
                np.pi / 2,
                2 * np.pi / 3,
                5 * np.pi / 6,
                np.pi,
                7 * np.pi / 6,
                4 * np.pi / 3,
            ],
        ]
    )

    # Generate signals using the OFDM generator
    ofdm_generator = OFDMGenerator(ofdm_config)
    signals = []

    for i in range(num_signals):
        signal = ofdm_generator.generate_single_signal(phases[i])
        signals.append(signal)
        print(f"  Generated signal {i+1}: {len(signal)} samples")

    # Create SignalSet
    signal_set = SignalSet(
        signals=signals,
        phases=phases,
        orthogonality_score=0.95,  # Simulated score
        generation_timestamp=datetime.now(),
        config=ofdm_config,
        metadata={
            "demo_type": "export_visualization",
            "generator_version": "1.0",
            "notes": "Sample signals for export demonstration",
        },
    )

    print(f"Created signal set with {signal_set.num_signals} signals")
    return signal_set


def demonstrate_signal_export(signal_set, output_dir):
    """Demonstrate signal export functionality."""
    print("\n" + "=" * 50)
    print("SIGNAL EXPORT DEMONSTRATION")
    print("=" * 50)

    exporter = SignalExporter(output_dir / "exports")

    # Export in different formats
    formats = ["numpy", "csv", "json", "pickle"]
    exported_files = {}

    for format_name in formats:
        print(f"\nExporting signals in {format_name.upper()} format...")
        try:
            filepath = exporter.export_signal_set(
                signal_set, f"demo_signals_{format_name}", format=format_name, include_metadata=True
            )
            exported_files[format_name] = filepath
            print(f"  ✓ Exported to: {filepath}")

            # Show file size
            if filepath.is_file():
                size_kb = filepath.stat().st_size / 1024
                print(f"    File size: {size_kb:.1f} KB")
            elif filepath.is_dir():
                total_size = sum(f.stat().st_size for f in filepath.rglob("*") if f.is_file())
                size_kb = total_size / 1024
                print(f"    Directory size: {size_kb:.1f} KB")

        except Exception as e:
            print(f"  ✗ Export failed: {e}")

    return exported_files


def demonstrate_phase_configuration_management(signal_set, output_dir):
    """Demonstrate phase configuration management."""
    print("\n" + "=" * 50)
    print("PHASE CONFIGURATION MANAGEMENT")
    print("=" * 50)

    manager = PhaseConfigurationManager(output_dir / "phase_configs")

    # Save phase configuration
    print("\nSaving phase configuration...")
    config_path = manager.save_phase_configuration(
        phases=signal_set.phases,
        config_name="demo_orthogonal_phases",
        ofdm_config=signal_set.config,
        orthogonality_score=signal_set.orthogonality_score,
        metadata={
            "description": "Demonstration phase configuration",
            "optimization_method": "manual",
            "target_orthogonality": 0.95,
        },
    )
    print(f"  ✓ Saved configuration to: {config_path}")

    # Load phase configuration
    print("\nLoading phase configuration...")
    loaded_phases, config_data = manager.load_phase_configuration(config_path)
    print(f"  ✓ Loaded phases shape: {loaded_phases.shape}")
    print(f"  ✓ Configuration name: {config_data['config_name']}")
    print(f"  ✓ Orthogonality score: {config_data['orthogonality_score']}")

    # List all configurations
    print("\nListing available configurations...")
    configs = manager.list_configurations()
    for i, config in enumerate(configs):
        print(f"  {i+1}. {config['config_name']}")
        print(f"     Signals: {config['num_signals']}, Subcarriers: {config['num_subcarriers']}")
        print(f"     Orthogonality: {config['orthogonality_score']:.3f}")
        print(f"     Created: {config['generation_timestamp']}")

    return config_path


def demonstrate_signal_loading(exported_files, output_dir):
    """Demonstrate signal loading functionality."""
    print("\n" + "=" * 50)
    print("SIGNAL LOADING DEMONSTRATION")
    print("=" * 50)

    loader = SignalLoader()

    # Test loading from different formats
    loadable_formats = ["numpy", "pickle", "json"]

    for format_name in loadable_formats:
        if format_name in exported_files:
            print(f"\nLoading signals from {format_name.upper()} format...")
            try:
                filepath = exported_files[format_name]
                loaded_signal_set = loader.load_signal_set(filepath)

                print(f"  ✓ Loaded {loaded_signal_set.num_signals} signals")
                print(f"  ✓ Signal length: {loaded_signal_set.signal_length} samples")
                print(f"  ✓ Orthogonality score: {loaded_signal_set.orthogonality_score}")
                print(f"  ✓ Generation timestamp: {loaded_signal_set.generation_timestamp}")

                # Verify signal integrity
                if hasattr(loaded_signal_set, "metadata") and loaded_signal_set.metadata:
                    print(f"  ✓ Metadata preserved: {len(loaded_signal_set.metadata)} items")

            except Exception as e:
                print(f"  ✗ Loading failed: {e}")


def demonstrate_visualization(signal_set, output_dir):
    """Demonstrate signal visualization functionality."""
    print("\n" + "=" * 50)
    print("SIGNAL VISUALIZATION DEMONSTRATION")
    print("=" * 50)

    try:
        visualizer = SignalVisualizer()
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        print("\nCreating individual plots...")

        # Time domain plot
        print("  Creating time domain plot...")
        fig_time = visualizer.plot_signal_time_domain(
            signal_set.signals,
            signal_set.config.sampling_rate,
            title="Demo OFDM Signals - Time Domain",
            save_path=viz_dir / "time_domain.png",
        )
        print("    ✓ Time domain plot saved")

        # Frequency domain plot
        print("  Creating frequency domain plot...")
        fig_freq = visualizer.plot_signal_frequency_domain(
            signal_set.signals,
            signal_set.config.sampling_rate,
            title="Demo OFDM Signals - Frequency Domain",
            save_path=viz_dir / "frequency_domain.png",
        )
        print("    ✓ Frequency domain plot saved")

        # Phase matrix plot
        print("  Creating phase matrix plot...")
        fig_phase = visualizer.plot_phase_matrix(
            signal_set.phases,
            title="Orthogonal Phase Configuration",
            save_path=viz_dir / "phase_matrix.png",
        )
        print("    ✓ Phase matrix plot saved")

        # Orthogonality matrix plot
        print("  Creating orthogonality matrix plot...")
        fig_ortho = visualizer.plot_orthogonality_matrix(
            signal_set.signals,
            title="Signal Orthogonality Analysis",
            save_path=viz_dir / "orthogonality_matrix.png",
        )
        print("    ✓ Orthogonality matrix plot saved")

        # Subcarrier analysis plot
        print("  Creating subcarrier analysis plot...")
        fig_subcarrier = visualizer.plot_subcarrier_analysis(
            signal_set.signals[0],
            signal_set.config,
            title="Detailed Subcarrier Analysis - Signal 0",
            save_path=viz_dir / "subcarrier_analysis.png",
        )
        print("    ✓ Subcarrier analysis plot saved")

        # Comprehensive signal report
        print("\nCreating comprehensive signal report...")
        report_dir = visualizer.create_signal_report(signal_set, output_dir / "signal_report")
        print(f"  ✓ Complete signal report created in: {report_dir}")

        # List generated files
        report_files = list(report_dir.glob("*"))
        print(f"  ✓ Generated {len(report_files)} report files:")
        for file in sorted(report_files):
            print(f"    - {file.name}")

        return viz_dir, report_dir

    except ImportError as e:
        print(f"  ✗ Visualization not available: {e}")
        print("    Install matplotlib to enable visualization: uv add matplotlib")
        return None, None
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
        return None, None


def demonstrate_data_integrity_verification(signal_set, exported_files):
    """Demonstrate data integrity verification."""
    print("\n" + "=" * 50)
    print("DATA INTEGRITY VERIFICATION")
    print("=" * 50)

    loader = SignalLoader()

    print("\nVerifying data integrity across export formats...")

    # Load signals from different formats and compare
    reference_signals = signal_set.signals
    reference_phases = signal_set.phases

    for format_name, filepath in exported_files.items():
        if format_name in ["numpy", "pickle", "json"]:
            try:
                loaded_signal_set = loader.load_signal_set(filepath)

                print(f"\n{format_name.upper()} format verification:")

                # Check number of signals
                if len(loaded_signal_set.signals) == len(reference_signals):
                    print("  ✓ Signal count matches")
                else:
                    print("  ✗ Signal count mismatch")

                # Check phase matrix
                if np.allclose(loaded_signal_set.phases, reference_phases, rtol=1e-10):
                    print("  ✓ Phase matrix matches")
                else:
                    print("  ✗ Phase matrix mismatch")

                # Check orthogonality score
                if (
                    abs(loaded_signal_set.orthogonality_score - signal_set.orthogonality_score)
                    < 1e-10
                ):
                    print("  ✓ Orthogonality score matches")
                else:
                    print("  ✗ Orthogonality score mismatch")

                # Check signal data (with tolerance for numerical precision)
                signals_match = True
                for i, (ref_sig, loaded_sig) in enumerate(
                    zip(reference_signals, loaded_signal_set.signals)
                ):
                    if not np.allclose(ref_sig, loaded_sig, rtol=1e-10):
                        signals_match = False
                        break

                if signals_match:
                    print("  ✓ All signal data matches")
                else:
                    print("  ✗ Signal data mismatch detected")

            except Exception as e:
                print(f"  ✗ Verification failed for {format_name}: {e}")


def main():
    """Main demonstration function."""
    print("OFDM Signal Export and Visualization Demo")
    print("=" * 60)

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        print(f"Working directory: {output_dir}")

        try:
            # Create sample signals
            signal_set = create_sample_signals()

            # Demonstrate export functionality
            exported_files = demonstrate_signal_export(signal_set, output_dir)

            # Demonstrate phase configuration management
            phase_config_path = demonstrate_phase_configuration_management(signal_set, output_dir)

            # Demonstrate signal loading
            demonstrate_signal_loading(exported_files, output_dir)

            # Demonstrate visualization
            viz_dir, report_dir = demonstrate_visualization(signal_set, output_dir)

            # Demonstrate data integrity verification
            demonstrate_data_integrity_verification(signal_set, exported_files)

            print("\n" + "=" * 60)
            print("DEMONSTRATION SUMMARY")
            print("=" * 60)
            print("✓ Signal export in multiple formats")
            print("✓ Phase configuration save/load")
            print("✓ Signal loading and verification")
            if viz_dir:
                print("✓ Signal visualization and reporting")
            else:
                print("- Signal visualization (matplotlib not available)")
            print("✓ Data integrity verification")

            print(f"\nAll demonstration files created in: {output_dir}")
            print("\nTo preserve the output files, copy them before the program exits.")

            # Optionally keep files for inspection
            input("\nPress Enter to continue and clean up temporary files...")

        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
