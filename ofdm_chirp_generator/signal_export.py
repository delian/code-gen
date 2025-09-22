"""
Signal export and visualization utilities for OFDM chirp signals.

This module provides functionality to export generated signals in various formats,
visualize signal characteristics, and save/load orthogonal phase configurations.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .config_manager import get_config
from .models import ChirpConfig, OFDMConfig, SignalSet


class SignalExporter:
    """Handles exporting OFDM signals to various file formats."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the signal exporter.

        Args:
            output_dir: Directory for exported files. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_signal_set(
        self,
        signal_set: SignalSet,
        filename: str,
        format: str = "numpy",
        include_metadata: bool = True,
    ) -> Path:
        """Export a complete signal set to file.

        Args:
            signal_set: SignalSet to export
            filename: Base filename (without extension)
            format: Export format ("numpy", "csv", "json", "pickle")
            include_metadata: Whether to include metadata in export

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
            IOError: If file cannot be written
        """
        supported_formats = ["numpy", "csv", "json", "pickle"]
        if format not in supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {supported_formats}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename}_{timestamp}"

        if format == "numpy":
            return self._export_numpy(signal_set, base_filename, include_metadata)
        elif format == "csv":
            return self._export_csv(signal_set, base_filename, include_metadata)
        elif format == "json":
            return self._export_json(signal_set, base_filename, include_metadata)
        elif format == "pickle":
            return self._export_pickle(signal_set, base_filename)

    def _export_numpy(self, signal_set: SignalSet, filename: str, include_metadata: bool) -> Path:
        """Export signal set as NumPy arrays."""
        filepath = self.output_dir / f"{filename}.npz"

        export_data = {
            "signals": np.array(signal_set.signals),
            "phases": signal_set.phases,
            "orthogonality_score": signal_set.orthogonality_score,
        }

        if include_metadata:
            export_data.update(
                {
                    "generation_timestamp": signal_set.generation_timestamp.isoformat(),
                    "num_subcarriers": signal_set.config.num_subcarriers,
                    "subcarrier_spacing": signal_set.config.subcarrier_spacing,
                    "center_frequency": signal_set.config.center_frequency,
                    "sampling_rate": signal_set.config.sampling_rate,
                    "signal_duration": signal_set.config.signal_duration,
                    "metadata": json.dumps(signal_set.metadata),
                }
            )

        np.savez_compressed(filepath, **export_data)
        return filepath

    def _export_csv(self, signal_set: SignalSet, filename: str, include_metadata: bool) -> Path:
        """Export signal set as CSV files."""
        # Create directory for CSV files
        csv_dir = self.output_dir / f"{filename}_csv"
        csv_dir.mkdir(exist_ok=True)

        # Export signals
        for i, signal in enumerate(signal_set.signals):
            signal_file = csv_dir / f"signal_{i:03d}.csv"
            np.savetxt(
                signal_file,
                signal,
                delimiter=",",
                header=f"Signal {i} - Real,Imaginary" if np.iscomplexobj(signal) else f"Signal {i}",
            )

        # Export phases
        phases_file = csv_dir / "phases.csv"
        np.savetxt(
            phases_file,
            signal_set.phases,
            delimiter=",",
            header="Phase matrix (signals x subcarriers)",
        )

        # Export metadata if requested
        if include_metadata:
            metadata_file = csv_dir / "metadata.json"
            metadata = {
                "orthogonality_score": float(signal_set.orthogonality_score),
                "generation_timestamp": signal_set.generation_timestamp.isoformat(),
                "config": {
                    "num_subcarriers": signal_set.config.num_subcarriers,
                    "subcarrier_spacing": signal_set.config.subcarrier_spacing,
                    "center_frequency": signal_set.config.center_frequency,
                    "sampling_rate": signal_set.config.sampling_rate,
                    "signal_duration": signal_set.config.signal_duration,
                },
                "metadata": signal_set.metadata,
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        return csv_dir

    def _export_json(self, signal_set: SignalSet, filename: str, include_metadata: bool) -> Path:
        """Export signal set as JSON (metadata only, signals referenced)."""
        filepath = self.output_dir / f"{filename}.json"

        # Save signals as separate numpy files and reference them
        signals_dir = self.output_dir / f"{filename}_signals"
        signals_dir.mkdir(exist_ok=True)

        signal_files = []
        for i, signal in enumerate(signal_set.signals):
            signal_file = signals_dir / f"signal_{i:03d}.npy"
            np.save(signal_file, signal)
            signal_files.append(str(signal_file.relative_to(self.output_dir)))

        export_data = {
            "signal_files": signal_files,
            "phases": signal_set.phases.tolist(),
            "orthogonality_score": float(signal_set.orthogonality_score),
            "generation_timestamp": signal_set.generation_timestamp.isoformat(),
        }

        if include_metadata:
            export_data.update(
                {
                    "config": {
                        "num_subcarriers": signal_set.config.num_subcarriers,
                        "subcarrier_spacing": signal_set.config.subcarrier_spacing,
                        "bandwidth_per_subcarrier": signal_set.config.bandwidth_per_subcarrier,
                        "center_frequency": signal_set.config.center_frequency,
                        "sampling_rate": signal_set.config.sampling_rate,
                        "signal_duration": signal_set.config.signal_duration,
                    },
                    "metadata": signal_set.metadata,
                }
            )

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        return filepath

    def _export_pickle(self, signal_set: SignalSet, filename: str) -> Path:
        """Export signal set as pickle file."""
        filepath = self.output_dir / f"{filename}.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(signal_set, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath


class PhaseConfigurationManager:
    """Manages saving and loading of orthogonal phase configurations."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize the phase configuration manager.

        Args:
            config_dir: Directory for phase configuration files. If None, uses ./phase_configs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("phase_configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_phase_configuration(
        self,
        phases: np.ndarray,
        config_name: str,
        ofdm_config: OFDMConfig,
        orthogonality_score: float,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save an orthogonal phase configuration.

        Args:
            phases: Phase matrix [num_signals x num_subcarriers]
            config_name: Name for the configuration
            ofdm_config: OFDM configuration used
            orthogonality_score: Achieved orthogonality score
            metadata: Additional metadata

        Returns:
            Path to saved configuration file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config_name}_{timestamp}.json"
        filepath = self.config_dir / filename

        config_data = {
            "config_name": config_name,
            "phases": phases.tolist(),
            "orthogonality_score": float(orthogonality_score),
            "generation_timestamp": datetime.now().isoformat(),
            "ofdm_config": {
                "num_subcarriers": ofdm_config.num_subcarriers,
                "subcarrier_spacing": ofdm_config.subcarrier_spacing,
                "bandwidth_per_subcarrier": ofdm_config.bandwidth_per_subcarrier,
                "center_frequency": ofdm_config.center_frequency,
                "sampling_rate": ofdm_config.sampling_rate,
                "signal_duration": ofdm_config.signal_duration,
            },
            "metadata": metadata or {},
        }

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)

        return filepath

    def load_phase_configuration(self, config_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Load a phase configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (phase_matrix, config_metadata)

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            phases = np.array(config_data["phases"])
            return phases, config_data

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")

    def list_configurations(self) -> List[Dict]:
        """List all available phase configurations.

        Returns:
            List of configuration summaries
        """
        configs = []
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)

                summary = {
                    "filename": config_file.name,
                    "config_name": config_data.get("config_name", "Unknown"),
                    "num_signals": len(config_data["phases"]),
                    "num_subcarriers": (
                        len(config_data["phases"][0]) if config_data["phases"] else 0
                    ),
                    "orthogonality_score": config_data.get("orthogonality_score", 0.0),
                    "generation_timestamp": config_data.get("generation_timestamp", "Unknown"),
                }
                configs.append(summary)

            except (json.JSONDecodeError, KeyError, IndexError):
                # Skip invalid configuration files
                continue

        return sorted(configs, key=lambda x: x["generation_timestamp"], reverse=True)


class SignalVisualizer:
    """Provides visualization tools for OFDM signal analysis."""

    def __init__(self):
        """Initialize the signal visualizer."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Matplotlib is required for visualization. Install with: uv add matplotlib"
            )

    def plot_signal_time_domain(
        self,
        signals: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: float,
        title: str = "OFDM Signals - Time Domain",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot signals in time domain.

        Args:
            signals: Signal array(s) to plot
            sampling_rate: Sampling rate in Hz
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            signals = [signals]
        elif isinstance(signals, np.ndarray) and signals.ndim == 2:
            signals = list(signals)

        fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2 * len(signals)))
        if len(signals) == 1:
            axes = [axes]

        for i, signal in enumerate(signals):
            time_axis = np.arange(len(signal)) / sampling_rate

            if np.iscomplexobj(signal):
                axes[i].plot(time_axis, np.real(signal), label=f"Signal {i} - Real", alpha=0.7)
                axes[i].plot(time_axis, np.imag(signal), label=f"Signal {i} - Imaginary", alpha=0.7)
            else:
                axes[i].plot(time_axis, signal, label=f"Signal {i}", alpha=0.7)

            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude")
            axes[i].set_title(f"Signal {i}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_signal_frequency_domain(
        self,
        signals: Union[np.ndarray, List[np.ndarray]],
        sampling_rate: float,
        title: str = "OFDM Signals - Frequency Domain",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot signals in frequency domain.

        Args:
            signals: Signal array(s) to plot
            sampling_rate: Sampling rate in Hz
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            signals = [signals]
        elif isinstance(signals, np.ndarray) and signals.ndim == 2:
            signals = list(signals)

        fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2 * len(signals)))
        if len(signals) == 1:
            axes = [axes]

        for i, signal in enumerate(signals):
            # Compute FFT
            fft_signal = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)

            # Plot magnitude spectrum (positive frequencies only)
            positive_freqs = freqs[: len(freqs) // 2]
            magnitude = np.abs(fft_signal[: len(fft_signal) // 2])

            axes[i].plot(positive_freqs, 20 * np.log10(magnitude + 1e-12), label=f"Signal {i}")
            axes[i].set_xlabel("Frequency (Hz)")
            axes[i].set_ylabel("Magnitude (dB)")
            axes[i].set_title(f"Signal {i} - Frequency Spectrum")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_phase_matrix(
        self,
        phase_matrix: np.ndarray,
        title: str = "Orthogonal Phase Configuration",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot phase matrix as a heatmap.

        Args:
            phase_matrix: Phase matrix [num_signals x num_subcarriers]
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(phase_matrix, cmap="hsv", aspect="auto", vmin=0, vmax=2 * np.pi)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Phase (radians)")

        # Set labels and ticks
        ax.set_xlabel("Subcarrier Index")
        ax.set_ylabel("Signal Index")
        ax.set_title(title)

        # Add grid
        ax.set_xticks(np.arange(phase_matrix.shape[1]))
        ax.set_yticks(np.arange(phase_matrix.shape[0]))
        ax.grid(True, alpha=0.3)

        # Add phase values as text
        for i in range(phase_matrix.shape[0]):
            for j in range(phase_matrix.shape[1]):
                text = ax.text(
                    j,
                    i,
                    f"{phase_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_orthogonality_matrix(
        self,
        signals: List[np.ndarray],
        title: str = "Signal Orthogonality Matrix",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot cross-correlation matrix between signals.

        Args:
            signals: List of signal arrays
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure object
        """
        num_signals = len(signals)
        correlation_matrix = np.zeros((num_signals, num_signals))

        # Compute cross-correlations
        for i in range(num_signals):
            for j in range(num_signals):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Normalized cross-correlation
                    corr = np.corrcoef(np.real(signals[i]), np.real(signals[j]))[0, 1]
                    correlation_matrix[i, j] = abs(corr)

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(correlation_matrix, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|Correlation Coefficient|")

        # Set labels and ticks
        ax.set_xlabel("Signal Index")
        ax.set_ylabel("Signal Index")
        ax.set_title(title)

        # Add correlation values as text
        for i in range(num_signals):
            for j in range(num_signals):
                text = ax.text(
                    j,
                    i,
                    f"{correlation_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if correlation_matrix[i, j] > 0.5 else "black",
                )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_subcarrier_analysis(
        self,
        signal: np.ndarray,
        ofdm_config: OFDMConfig,
        title: str = "Subcarrier Analysis",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot detailed subcarrier analysis.

        Args:
            signal: OFDM signal to analyze
            ofdm_config: OFDM configuration parameters
            title: Plot title
            save_path: Optional path to save the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Time domain signal
        time_axis = np.arange(len(signal)) / ofdm_config.sampling_rate
        if np.iscomplexobj(signal):
            ax1.plot(time_axis, np.real(signal), label="Real", alpha=0.7)
            ax1.plot(time_axis, np.imag(signal), label="Imaginary", alpha=0.7)
        else:
            ax1.plot(time_axis, signal, alpha=0.7)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Time Domain Signal")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Frequency spectrum
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1 / ofdm_config.sampling_rate)
        positive_freqs = freqs[: len(freqs) // 2]
        magnitude = np.abs(fft_signal[: len(fft_signal) // 2])

        ax2.plot(positive_freqs, 20 * np.log10(magnitude + 1e-12))
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_title("Frequency Spectrum")
        ax2.grid(True, alpha=0.3)

        # Mark subcarrier frequencies
        center_freq = ofdm_config.center_frequency
        spacing = ofdm_config.subcarrier_spacing
        num_subcarriers = ofdm_config.num_subcarriers

        subcarrier_freqs = []
        for i in range(num_subcarriers):
            freq = center_freq + (i - num_subcarriers // 2) * spacing
            if freq > 0 and freq < ofdm_config.sampling_rate / 2:
                subcarrier_freqs.append(freq)
                ax2.axvline(freq, color="red", alpha=0.5, linestyle="--")

        # Spectrogram
        from matplotlib import mlab

        Pxx, freqs_spec, bins, im = ax3.specgram(
            np.real(signal) if np.iscomplexobj(signal) else signal,
            NFFT=256,
            Fs=ofdm_config.sampling_rate,
            noverlap=128,
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Frequency (Hz)")
        ax3.set_title("Spectrogram")

        # Phase plot (if complex signal)
        if np.iscomplexobj(signal):
            phase = np.angle(signal)
            ax4.plot(time_axis, phase)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Phase (radians)")
            ax4.set_title("Instantaneous Phase")
            ax4.grid(True, alpha=0.3)
        else:
            # Show autocorrelation for real signals
            autocorr = np.correlate(signal, signal, mode="full")
            autocorr = autocorr[autocorr.size // 2 :]
            autocorr = autocorr / autocorr[0]  # Normalize

            lags = np.arange(len(autocorr)) / ofdm_config.sampling_rate
            ax4.plot(lags[: len(lags) // 4], autocorr[: len(autocorr) // 4])  # Show first quarter
            ax4.set_xlabel("Lag (s)")
            ax4.set_ylabel("Normalized Autocorrelation")
            ax4.set_title("Autocorrelation")
            ax4.grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_signal_report(
        self, signal_set: SignalSet, output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Create a comprehensive signal analysis report.

        Args:
            signal_set: SignalSet to analyze
            output_dir: Directory for report files

        Returns:
            Path to report directory
        """
        if output_dir is None:
            output_dir = Path(f"signal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all plots
        plots = {}

        # Time domain plot
        fig_time = self.plot_signal_time_domain(
            signal_set.signals,
            signal_set.config.sampling_rate,
            save_path=output_dir / "time_domain.png",
        )
        plots["time_domain"] = fig_time
        plt.close(fig_time)

        # Frequency domain plot
        fig_freq = self.plot_signal_frequency_domain(
            signal_set.signals,
            signal_set.config.sampling_rate,
            save_path=output_dir / "frequency_domain.png",
        )
        plots["frequency_domain"] = fig_freq
        plt.close(fig_freq)

        # Phase matrix plot
        fig_phase = self.plot_phase_matrix(
            signal_set.phases, save_path=output_dir / "phase_matrix.png"
        )
        plots["phase_matrix"] = fig_phase
        plt.close(fig_phase)

        # Orthogonality matrix plot
        fig_ortho = self.plot_orthogonality_matrix(
            signal_set.signals, save_path=output_dir / "orthogonality_matrix.png"
        )
        plots["orthogonality_matrix"] = fig_ortho
        plt.close(fig_ortho)

        # Individual subcarrier analysis for first signal
        if signal_set.signals:
            fig_subcarrier = self.plot_subcarrier_analysis(
                signal_set.signals[0],
                signal_set.config,
                title="Subcarrier Analysis - Signal 0",
                save_path=output_dir / "subcarrier_analysis.png",
            )
            plots["subcarrier_analysis"] = fig_subcarrier
            plt.close(fig_subcarrier)

        # Generate summary report
        self._generate_summary_report(signal_set, output_dir)

        return output_dir

    def _generate_summary_report(self, signal_set: SignalSet, output_dir: Path):
        """Generate a text summary report."""
        report_path = output_dir / "summary_report.txt"

        with open(report_path, "w") as f:
            f.write("OFDM Signal Set Analysis Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Generation Timestamp: {signal_set.generation_timestamp}\n")
            f.write(f"Number of Signals: {signal_set.num_signals}\n")
            f.write(f"Signal Length: {signal_set.signal_length} samples\n")
            f.write(f"Orthogonality Score: {signal_set.orthogonality_score:.6f}\n\n")

            f.write("OFDM Configuration:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  Number of Subcarriers: {signal_set.config.num_subcarriers}\n")
            f.write(f"  Subcarrier Spacing: {signal_set.config.subcarrier_spacing} Hz\n")
            f.write(
                f"  Bandwidth per Subcarrier: {signal_set.config.bandwidth_per_subcarrier} Hz\n"
            )
            f.write(f"  Center Frequency: {signal_set.config.center_frequency} Hz\n")
            f.write(f"  Sampling Rate: {signal_set.config.sampling_rate} Hz\n")
            f.write(f"  Signal Duration: {signal_set.config.signal_duration} s\n\n")

            f.write("Phase Matrix:\n")
            f.write("-" * 15 + "\n")
            for i, phase_row in enumerate(signal_set.phases):
                f.write(f"  Signal {i}: {phase_row}\n")
            f.write("\n")

            if signal_set.metadata:
                f.write("Additional Metadata:\n")
                f.write("-" * 20 + "\n")
                for key, value in signal_set.metadata.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            f.write("Generated Files:\n")
            f.write("-" * 15 + "\n")
            f.write("  - time_domain.png: Time domain signal plots\n")
            f.write("  - frequency_domain.png: Frequency spectrum plots\n")
            f.write("  - phase_matrix.png: Phase configuration heatmap\n")
            f.write("  - orthogonality_matrix.png: Signal correlation matrix\n")
            f.write("  - subcarrier_analysis.png: Detailed subcarrier analysis\n")
            f.write("  - summary_report.txt: This summary report\n")


class SignalLoader:
    """Handles loading exported signal sets from various file formats."""

    def __init__(self):
        """Initialize the signal loader."""
        pass

    def load_signal_set(self, filepath: Union[str, Path]) -> SignalSet:
        """Load a signal set from file.

        Args:
            filepath: Path to signal set file

        Returns:
            Loaded SignalSet object

        Raises:
            ValueError: If file format is not supported or invalid
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Signal file not found: {filepath}")

        if filepath.suffix == ".npz":
            return self._load_numpy(filepath)
        elif filepath.suffix == ".pkl":
            return self._load_pickle(filepath)
        elif filepath.suffix == ".json":
            return self._load_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def _load_numpy(self, filepath: Path) -> SignalSet:
        """Load signal set from NumPy file."""
        data = np.load(filepath)

        # Reconstruct SignalSet
        signals = [data["signals"][i] for i in range(data["signals"].shape[0])]
        phases = data["phases"]
        orthogonality_score = float(data["orthogonality_score"])

        # Reconstruct timestamp
        if "generation_timestamp" in data:
            timestamp = datetime.fromisoformat(str(data["generation_timestamp"]))
        else:
            timestamp = datetime.now()

        # Reconstruct OFDM config
        ofdm_config = OFDMConfig(
            num_subcarriers=int(data["num_subcarriers"]),
            subcarrier_spacing=float(data["subcarrier_spacing"]),
            bandwidth_per_subcarrier=float(data.get("bandwidth_per_subcarrier", 800.0)),
            center_frequency=float(data["center_frequency"]),
            sampling_rate=float(data["sampling_rate"]),
            signal_duration=float(data["signal_duration"]),
        )

        # Reconstruct metadata
        metadata = {}
        if "metadata" in data:
            try:
                metadata = json.loads(str(data["metadata"]))
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        return SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=orthogonality_score,
            generation_timestamp=timestamp,
            config=ofdm_config,
            metadata=metadata,
        )

    def _load_pickle(self, filepath: Path) -> SignalSet:
        """Load signal set from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _load_json(self, filepath: Path) -> SignalSet:
        """Load signal set from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Load individual signal files
        signals = []
        base_dir = filepath.parent
        for signal_file in data["signal_files"]:
            signal_path = base_dir / signal_file
            signal = np.load(signal_path)
            signals.append(signal)

        phases = np.array(data["phases"])
        orthogonality_score = float(data["orthogonality_score"])
        timestamp = datetime.fromisoformat(data["generation_timestamp"])

        # Reconstruct OFDM config
        config_data = data["config"]
        ofdm_config = OFDMConfig(
            num_subcarriers=config_data["num_subcarriers"],
            subcarrier_spacing=config_data["subcarrier_spacing"],
            bandwidth_per_subcarrier=config_data["bandwidth_per_subcarrier"],
            center_frequency=config_data["center_frequency"],
            sampling_rate=config_data["sampling_rate"],
            signal_duration=config_data["signal_duration"],
        )

        metadata = data.get("metadata", {})

        return SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=orthogonality_score,
            generation_timestamp=timestamp,
            config=ofdm_config,
            metadata=metadata,
        )
