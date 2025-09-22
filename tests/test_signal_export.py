"""
Tests for signal export and visualization utilities.

This module tests the functionality of signal export, phase configuration management,
visualization tools, and signal loading capabilities.
"""

import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.models import OFDMConfig, SignalSet
from ofdm_chirp_generator.signal_export import (
    PhaseConfigurationManager,
    SignalExporter,
    SignalLoader,
    SignalVisualizer,
)


class TestSignalExporter:
    """Test cases for SignalExporter class."""

    @pytest.fixture
    def sample_signal_set(self):
        """Create a sample signal set for testing."""
        config = OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,
        )

        # Create sample signals
        num_samples = int(config.sampling_rate * config.signal_duration)
        signals = [
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples),
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples),
        ]

        phases = np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            ]
        )

        return SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=0.95,
            generation_timestamp=datetime.now(),
            config=config,
            metadata={"test_param": "test_value"},
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_init(self, temp_dir):
        """Test SignalExporter initialization."""
        exporter = SignalExporter(temp_dir)
        assert exporter.output_dir == temp_dir
        assert temp_dir.exists()

        # Test default directory
        exporter_default = SignalExporter()
        assert exporter_default.output_dir == Path.cwd()

    def test_export_numpy_format(self, sample_signal_set, temp_dir):
        """Test exporting signal set in NumPy format."""
        exporter = SignalExporter(temp_dir)

        filepath = exporter.export_signal_set(sample_signal_set, "test_signals", format="numpy")

        assert filepath.exists()
        assert filepath.suffix == ".npz"

        # Verify exported data
        data = np.load(filepath)
        assert "signals" in data
        assert "phases" in data
        assert "orthogonality_score" in data
        assert "generation_timestamp" in data

        np.testing.assert_array_equal(data["signals"], np.array(sample_signal_set.signals))
        np.testing.assert_array_equal(data["phases"], sample_signal_set.phases)
        assert float(data["orthogonality_score"]) == sample_signal_set.orthogonality_score

    def test_export_csv_format(self, sample_signal_set, temp_dir):
        """Test exporting signal set in CSV format."""
        exporter = SignalExporter(temp_dir)

        csv_dir = exporter.export_signal_set(sample_signal_set, "test_signals", format="csv")

        assert csv_dir.is_dir()

        # Check individual signal files
        signal_files = list(csv_dir.glob("signal_*.csv"))
        assert len(signal_files) == len(sample_signal_set.signals)

        # Check phases file
        phases_file = csv_dir / "phases.csv"
        assert phases_file.exists()

        # Check metadata file
        metadata_file = csv_dir / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            assert metadata["orthogonality_score"] == sample_signal_set.orthogonality_score

    def test_export_json_format(self, sample_signal_set, temp_dir):
        """Test exporting signal set in JSON format."""
        exporter = SignalExporter(temp_dir)

        filepath = exporter.export_signal_set(sample_signal_set, "test_signals", format="json")

        assert filepath.exists()
        assert filepath.suffix == ".json"

        with open(filepath, "r") as f:
            data = json.load(f)

        assert "signal_files" in data
        assert "phases" in data
        assert "orthogonality_score" in data
        assert len(data["signal_files"]) == len(sample_signal_set.signals)

        # Check that signal files exist
        for signal_file in data["signal_files"]:
            signal_path = temp_dir / signal_file
            assert signal_path.exists()

    def test_export_pickle_format(self, sample_signal_set, temp_dir):
        """Test exporting signal set in pickle format."""
        exporter = SignalExporter(temp_dir)

        filepath = exporter.export_signal_set(sample_signal_set, "test_signals", format="pickle")

        assert filepath.exists()
        assert filepath.suffix == ".pkl"

        # Verify exported data
        with open(filepath, "rb") as f:
            loaded_signal_set = pickle.load(f)

        assert isinstance(loaded_signal_set, SignalSet)
        assert len(loaded_signal_set.signals) == len(sample_signal_set.signals)
        assert loaded_signal_set.orthogonality_score == sample_signal_set.orthogonality_score

    def test_export_unsupported_format(self, sample_signal_set, temp_dir):
        """Test error handling for unsupported export format."""
        exporter = SignalExporter(temp_dir)

        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_signal_set(sample_signal_set, "test_signals", format="unsupported")

    def test_export_without_metadata(self, sample_signal_set, temp_dir):
        """Test exporting without metadata."""
        exporter = SignalExporter(temp_dir)

        filepath = exporter.export_signal_set(
            sample_signal_set, "test_signals", format="numpy", include_metadata=False
        )

        data = np.load(filepath)
        assert "signals" in data
        assert "phases" in data
        assert "orthogonality_score" in data
        # Metadata fields should not be present
        assert "generation_timestamp" not in data
        assert "num_subcarriers" not in data


class TestPhaseConfigurationManager:
    """Test cases for PhaseConfigurationManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration data."""
        return OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,
        )

    def test_init(self, temp_dir):
        """Test PhaseConfigurationManager initialization."""
        manager = PhaseConfigurationManager(temp_dir)
        assert manager.config_dir == temp_dir
        assert temp_dir.exists()

        # Test default directory
        manager_default = PhaseConfigurationManager()
        assert manager_default.config_dir == Path("phase_configs")

    def test_save_phase_configuration(self, sample_config, temp_dir):
        """Test saving phase configuration."""
        manager = PhaseConfigurationManager(temp_dir)

        phases = np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            ]
        )

        filepath = manager.save_phase_configuration(
            phases=phases,
            config_name="test_config",
            ofdm_config=sample_config,
            orthogonality_score=0.95,
            metadata={"test": "value"},
        )

        assert filepath.exists()
        assert filepath.suffix == ".json"

        # Verify saved data
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["config_name"] == "test_config"
        assert data["orthogonality_score"] == 0.95
        np.testing.assert_array_equal(np.array(data["phases"]), phases)
        assert data["ofdm_config"]["num_subcarriers"] == sample_config.num_subcarriers
        assert data["metadata"]["test"] == "value"

    def test_load_phase_configuration(self, sample_config, temp_dir):
        """Test loading phase configuration."""
        manager = PhaseConfigurationManager(temp_dir)

        phases = np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            ]
        )

        # Save configuration first
        filepath = manager.save_phase_configuration(
            phases=phases,
            config_name="test_config",
            ofdm_config=sample_config,
            orthogonality_score=0.95,
        )

        # Load configuration
        loaded_phases, config_data = manager.load_phase_configuration(filepath)

        np.testing.assert_array_equal(loaded_phases, phases)
        assert config_data["config_name"] == "test_config"
        assert config_data["orthogonality_score"] == 0.95

    def test_load_nonexistent_configuration(self, temp_dir):
        """Test error handling for nonexistent configuration file."""
        manager = PhaseConfigurationManager(temp_dir)

        with pytest.raises(FileNotFoundError):
            manager.load_phase_configuration("nonexistent.json")

    def test_load_invalid_configuration(self, temp_dir):
        """Test error handling for invalid configuration file."""
        manager = PhaseConfigurationManager(temp_dir)

        # Create invalid JSON file
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(ValueError, match="Invalid configuration file format"):
            manager.load_phase_configuration(invalid_file)

    def test_list_configurations(self, sample_config, temp_dir):
        """Test listing available configurations."""
        manager = PhaseConfigurationManager(temp_dir)

        # Initially empty
        configs = manager.list_configurations()
        assert len(configs) == 0

        # Save some configurations
        phases1 = np.array([[0.0, np.pi / 2], [np.pi / 4, 3 * np.pi / 4]])
        phases2 = np.array([[0.0, np.pi], [np.pi / 2, 3 * np.pi / 2]])

        manager.save_phase_configuration(phases1, "config1", sample_config, 0.95)
        manager.save_phase_configuration(phases2, "config2", sample_config, 0.90)

        # List configurations
        configs = manager.list_configurations()
        assert len(configs) == 2

        config_names = [c["config_name"] for c in configs]
        assert "config1" in config_names
        assert "config2" in config_names

        # Check configuration details
        for config in configs:
            assert "num_signals" in config
            assert "num_subcarriers" in config
            assert "orthogonality_score" in config
            assert config["num_signals"] == 2
            assert config["num_subcarriers"] == 2


class TestSignalVisualizer:
    """Test cases for SignalVisualizer class."""

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for testing."""
        t = np.linspace(0, 1, 1000)
        signal1 = np.sin(2 * np.pi * 10 * t) + 1j * np.cos(2 * np.pi * 10 * t)
        signal2 = np.sin(2 * np.pi * 20 * t) + 1j * np.cos(2 * np.pi * 20 * t)
        return [signal1, signal2]

    @pytest.fixture
    def sample_signal_set(self, sample_signals):
        """Create a sample signal set for testing."""
        config = OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,  # Increased to satisfy Nyquist limit
            signal_duration=0.02,  # Adjusted duration
        )

        phases = np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            ]
        )

        return SignalSet(
            signals=sample_signals,
            phases=phases,
            orthogonality_score=0.95,
            generation_timestamp=datetime.now(),
            config=config,
        )

    @patch("ofdm_chirp_generator.signal_export.MATPLOTLIB_AVAILABLE", True)
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_init_with_matplotlib(self, mock_tight_layout, mock_subplots):
        """Test SignalVisualizer initialization with matplotlib available."""
        visualizer = SignalVisualizer()
        assert visualizer is not None

    @patch("ofdm_chirp_generator.signal_export.MATPLOTLIB_AVAILABLE", False)
    def test_init_without_matplotlib(self):
        """Test SignalVisualizer initialization without matplotlib."""
        with pytest.raises(ImportError, match="Matplotlib is required"):
            SignalVisualizer()

    @patch("ofdm_chirp_generator.signal_export.MATPLOTLIB_AVAILABLE", True)
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_signal_time_domain(self, mock_tight_layout, mock_subplots, sample_signals):
        """Test time domain plotting."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        visualizer = SignalVisualizer()
        fig = visualizer.plot_signal_time_domain(sample_signals, 1000.0)

        assert fig == mock_fig
        mock_subplots.assert_called_once()

        # Verify that plot methods were called on axes
        for ax in mock_axes:
            ax.plot.assert_called()
            ax.set_xlabel.assert_called()
            ax.set_ylabel.assert_called()
            ax.set_title.assert_called()

    @patch("ofdm_chirp_generator.signal_export.MATPLOTLIB_AVAILABLE", True)
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_phase_matrix(self, mock_tight_layout, mock_colorbar, mock_subplots):
        """Test phase matrix plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_colorbar.return_value = Mock()

        visualizer = SignalVisualizer()
        phases = np.array([[0.0, np.pi / 2], [np.pi, 3 * np.pi / 2]])

        fig = visualizer.plot_phase_matrix(phases)

        assert fig == mock_fig
        mock_ax.imshow.assert_called_once()
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()
        mock_ax.set_title.assert_called()

    @patch("ofdm_chirp_generator.signal_export.MATPLOTLIB_AVAILABLE", True)
    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.pyplot.colorbar")
    @patch("matplotlib.pyplot.tight_layout")
    def test_plot_orthogonality_matrix(
        self, mock_tight_layout, mock_colorbar, mock_subplots, sample_signals
    ):
        """Test orthogonality matrix plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_colorbar.return_value = Mock()

        visualizer = SignalVisualizer()
        fig = visualizer.plot_orthogonality_matrix(sample_signals)

        assert fig == mock_fig
        mock_ax.imshow.assert_called_once()
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()
        mock_ax.set_title.assert_called()

    @patch("ofdm_chirp_generator.signal_export.MATPLOTLIB_AVAILABLE", True)
    @patch("ofdm_chirp_generator.signal_export.SignalVisualizer.plot_signal_time_domain")
    @patch("ofdm_chirp_generator.signal_export.SignalVisualizer.plot_signal_frequency_domain")
    @patch("ofdm_chirp_generator.signal_export.SignalVisualizer.plot_phase_matrix")
    @patch("ofdm_chirp_generator.signal_export.SignalVisualizer.plot_orthogonality_matrix")
    @patch("ofdm_chirp_generator.signal_export.SignalVisualizer.plot_subcarrier_analysis")
    @patch("matplotlib.pyplot.close")
    def test_create_signal_report(
        self,
        mock_close,
        mock_subcarrier,
        mock_ortho,
        mock_phase,
        mock_freq,
        mock_time,
        sample_signal_set,
    ):
        """Test comprehensive signal report creation."""
        # Mock all plot methods to return Mock figures
        mock_fig = Mock()
        mock_time.return_value = mock_fig
        mock_freq.return_value = mock_fig
        mock_phase.return_value = mock_fig
        mock_ortho.return_value = mock_fig
        mock_subcarrier.return_value = mock_fig

        visualizer = SignalVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = visualizer.create_signal_report(sample_signal_set, temp_dir)

            assert report_dir.exists()
            assert report_dir.is_dir()

            # Check that summary report was created
            summary_file = report_dir / "summary_report.txt"
            assert summary_file.exists()

            # Verify summary content
            with open(summary_file, "r") as f:
                content = f.read()
                assert "OFDM Signal Set Analysis Report" in content
                assert f"Number of Signals: {sample_signal_set.num_signals}" in content
                assert f"Orthogonality Score: {sample_signal_set.orthogonality_score}" in content


class TestSignalLoader:
    """Test cases for SignalLoader class."""

    @pytest.fixture
    def sample_signal_set(self):
        """Create a sample signal set for testing."""
        config = OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,
        )

        num_samples = int(config.sampling_rate * config.signal_duration)
        signals = [np.random.randn(num_samples), np.random.randn(num_samples)]

        phases = np.array(
            [
                [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            ]
        )

        return SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=0.95,
            generation_timestamp=datetime.now(),
            config=config,
            metadata={"test": "value"},
        )

    def test_init(self):
        """Test SignalLoader initialization."""
        loader = SignalLoader()
        assert loader is not None

    def test_load_numpy_format(self, sample_signal_set):
        """Test loading signal set from NumPy format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export signal set
            exporter = SignalExporter(temp_path)
            filepath = exporter.export_signal_set(sample_signal_set, "test", format="numpy")

            # Load signal set
            loader = SignalLoader()
            loaded_signal_set = loader.load_signal_set(filepath)

            assert isinstance(loaded_signal_set, SignalSet)
            assert len(loaded_signal_set.signals) == len(sample_signal_set.signals)
            assert loaded_signal_set.orthogonality_score == sample_signal_set.orthogonality_score
            np.testing.assert_array_equal(loaded_signal_set.phases, sample_signal_set.phases)

    def test_load_pickle_format(self, sample_signal_set):
        """Test loading signal set from pickle format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export signal set
            exporter = SignalExporter(temp_path)
            filepath = exporter.export_signal_set(sample_signal_set, "test", format="pickle")

            # Load signal set
            loader = SignalLoader()
            loaded_signal_set = loader.load_signal_set(filepath)

            assert isinstance(loaded_signal_set, SignalSet)
            assert len(loaded_signal_set.signals) == len(sample_signal_set.signals)
            assert loaded_signal_set.orthogonality_score == sample_signal_set.orthogonality_score
            np.testing.assert_array_equal(loaded_signal_set.phases, sample_signal_set.phases)

    def test_load_json_format(self, sample_signal_set):
        """Test loading signal set from JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export signal set
            exporter = SignalExporter(temp_path)
            filepath = exporter.export_signal_set(sample_signal_set, "test", format="json")

            # Load signal set
            loader = SignalLoader()
            loaded_signal_set = loader.load_signal_set(filepath)

            assert isinstance(loaded_signal_set, SignalSet)
            assert len(loaded_signal_set.signals) == len(sample_signal_set.signals)
            assert loaded_signal_set.orthogonality_score == sample_signal_set.orthogonality_score
            np.testing.assert_array_equal(loaded_signal_set.phases, sample_signal_set.phases)

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        loader = SignalLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_signal_set("nonexistent.npz")

    def test_load_unsupported_format(self):
        """Test error handling for unsupported file format."""
        loader = SignalLoader()

        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load_signal_set(temp_file.name)


class TestIntegration:
    """Integration tests for export and visualization functionality."""

    def test_export_load_roundtrip(self):
        """Test complete export-load roundtrip for all formats."""
        # Create sample signal set
        config = OFDMConfig(
            num_subcarriers=2,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=10000.0,
            signal_duration=0.001,
        )

        signals = [np.random.randn(10), np.random.randn(10)]
        phases = np.array([[0.0, np.pi], [np.pi / 2, 3 * np.pi / 2]])

        original_signal_set = SignalSet(
            signals=signals,
            phases=phases,
            orthogonality_score=0.95,
            generation_timestamp=datetime.now(),
            config=config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exporter = SignalExporter(temp_path)
            loader = SignalLoader()

            # Test each format
            for format_name in ["numpy", "pickle", "json"]:
                filepath = exporter.export_signal_set(
                    original_signal_set, f"test_{format_name}", format=format_name
                )

                loaded_signal_set = loader.load_signal_set(filepath)

                # Verify loaded data matches original
                assert len(loaded_signal_set.signals) == len(original_signal_set.signals)
                assert (
                    loaded_signal_set.orthogonality_score == original_signal_set.orthogonality_score
                )
                np.testing.assert_array_equal(loaded_signal_set.phases, original_signal_set.phases)

                # Verify signals (with some tolerance for floating point precision)
                for orig_sig, loaded_sig in zip(
                    original_signal_set.signals, loaded_signal_set.signals
                ):
                    np.testing.assert_array_almost_equal(orig_sig, loaded_sig, decimal=10)

    def test_phase_configuration_roundtrip(self):
        """Test phase configuration save-load roundtrip."""
        config = OFDMConfig(
            num_subcarriers=3,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.001,
        )

        original_phases = np.array(
            [[0.0, np.pi / 3, 2 * np.pi / 3], [np.pi / 6, np.pi / 2, 5 * np.pi / 6]]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = PhaseConfigurationManager(temp_dir)

            # Save configuration
            filepath = manager.save_phase_configuration(
                phases=original_phases,
                config_name="test_roundtrip",
                ofdm_config=config,
                orthogonality_score=0.92,
                metadata={"test": "roundtrip"},
            )

            # Load configuration
            loaded_phases, config_data = manager.load_phase_configuration(filepath)

            # Verify loaded data
            np.testing.assert_array_equal(loaded_phases, original_phases)
            assert config_data["config_name"] == "test_roundtrip"
            assert config_data["orthogonality_score"] == 0.92
            assert config_data["metadata"]["test"] == "roundtrip"
