"""
Input validation for OFDM chirp generator configuration parameters.

This module provides comprehensive validation for all configuration classes
to ensure parameter compatibility and system constraints.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models import ChirpConfig, OFDMConfig, SignalSet


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""

    pass


class ConfigValidator:
    """Validator class for all configuration parameters."""

    # Configuration constraints
    MIN_SUBCARRIERS = 1
    MAX_SUBCARRIERS = 1024
    MIN_SAMPLING_RATE = 1000.0  # Hz
    MAX_SAMPLING_RATE = 1e9  # Hz
    MIN_SIGNAL_DURATION = 1e-6  # seconds
    MAX_SIGNAL_DURATION = 3600.0  # seconds
    MIN_FREQUENCY = 0.0  # Hz
    MAX_FREQUENCY = 1e12  # Hz
    MIN_AMPLITUDE = 1e-10
    MAX_AMPLITUDE = 1e10
    MIN_CHIRP_LENGTH = 10
    MAX_CHIRP_LENGTH = 1000000

    @classmethod
    def validate_ofdm_config(cls, config: "OFDMConfig") -> None:
        """Validate OFDM configuration parameters.

        Args:
            config: OFDMConfig instance to validate

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate num_subcarriers
        if not isinstance(config.num_subcarriers, int):
            raise ValidationError("num_subcarriers must be an integer")
        if config.num_subcarriers < cls.MIN_SUBCARRIERS:
            raise ValidationError(f"num_subcarriers must be >= {cls.MIN_SUBCARRIERS}")
        if config.num_subcarriers > cls.MAX_SUBCARRIERS:
            raise ValidationError(f"num_subcarriers must be <= {cls.MAX_SUBCARRIERS}")

        # Validate subcarrier_spacing
        if not isinstance(config.subcarrier_spacing, (int, float)):
            raise ValidationError("subcarrier_spacing must be a number")
        if config.subcarrier_spacing <= 0:
            raise ValidationError("subcarrier_spacing must be positive")

        # Validate bandwidth_per_subcarrier
        if not isinstance(config.bandwidth_per_subcarrier, (int, float)):
            raise ValidationError("bandwidth_per_subcarrier must be a number")
        if config.bandwidth_per_subcarrier <= 0:
            raise ValidationError("bandwidth_per_subcarrier must be positive")

        # Validate center_frequency
        if not isinstance(config.center_frequency, (int, float)):
            raise ValidationError("center_frequency must be a number")
        if config.center_frequency < cls.MIN_FREQUENCY:
            raise ValidationError(f"center_frequency must be >= {cls.MIN_FREQUENCY}")
        if config.center_frequency > cls.MAX_FREQUENCY:
            raise ValidationError(f"center_frequency must be <= {cls.MAX_FREQUENCY}")

        # Validate sampling_rate
        if not isinstance(config.sampling_rate, (int, float)):
            raise ValidationError("sampling_rate must be a number")
        if config.sampling_rate < cls.MIN_SAMPLING_RATE:
            raise ValidationError(f"sampling_rate must be >= {cls.MIN_SAMPLING_RATE}")
        if config.sampling_rate > cls.MAX_SAMPLING_RATE:
            raise ValidationError(f"sampling_rate must be <= {cls.MAX_SAMPLING_RATE}")

        # Validate signal_duration
        if not isinstance(config.signal_duration, (int, float)):
            raise ValidationError("signal_duration must be a number")
        if config.signal_duration < cls.MIN_SIGNAL_DURATION:
            raise ValidationError(f"signal_duration must be >= {cls.MIN_SIGNAL_DURATION}")
        if config.signal_duration > cls.MAX_SIGNAL_DURATION:
            raise ValidationError(f"signal_duration must be <= {cls.MAX_SIGNAL_DURATION}")

        # Cross-parameter validation
        total_bandwidth = config.num_subcarriers * config.subcarrier_spacing
        if total_bandwidth > config.sampling_rate / 2:
            raise ValidationError(
                f"Total signal bandwidth ({total_bandwidth:.0f} Hz) exceeds "
                f"Nyquist limit ({config.sampling_rate/2:.0f} Hz)"
            )

        if config.bandwidth_per_subcarrier > config.subcarrier_spacing:
            raise ValidationError(
                "bandwidth_per_subcarrier cannot exceed subcarrier_spacing "
                "(would cause subcarrier overlap)"
            )

    @classmethod
    def validate_chirp_config(cls, config: "ChirpConfig") -> None:
        """Validate chirp configuration parameters.

        Args:
            config: ChirpConfig instance to validate

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate chirp_length
        if not isinstance(config.chirp_length, int):
            raise ValidationError("chirp_length must be an integer")
        if config.chirp_length < cls.MIN_CHIRP_LENGTH:
            raise ValidationError(f"chirp_length must be >= {cls.MIN_CHIRP_LENGTH}")
        if config.chirp_length > cls.MAX_CHIRP_LENGTH:
            raise ValidationError(f"chirp_length must be <= {cls.MAX_CHIRP_LENGTH}")

        # Validate phase_matrix
        if not isinstance(config.phase_matrix, np.ndarray):
            raise ValidationError("phase_matrix must be a numpy array")
        if config.phase_matrix.ndim != 2:
            raise ValidationError("phase_matrix must be 2-dimensional")
        if config.phase_matrix.size == 0:
            raise ValidationError("phase_matrix cannot be empty")

        # Validate phase values are in valid range [0, 2π]
        if np.any(config.phase_matrix < 0) or np.any(config.phase_matrix > 2 * np.pi):
            raise ValidationError("All phase values must be in range [0, 2π]")

        # Validate amplitude
        if not isinstance(config.amplitude, (int, float)):
            raise ValidationError("amplitude must be a number")
        if config.amplitude < cls.MIN_AMPLITUDE:
            raise ValidationError(f"amplitude must be >= {cls.MIN_AMPLITUDE}")
        if config.amplitude > cls.MAX_AMPLITUDE:
            raise ValidationError(f"amplitude must be <= {cls.MAX_AMPLITUDE}")

    @classmethod
    def validate_signal_set(cls, signal_set: "SignalSet") -> None:
        """Validate signal set parameters.

        Args:
            signal_set: SignalSet instance to validate

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate signals list
        if not isinstance(signal_set.signals, list):
            raise ValidationError("signals must be a list")
        if len(signal_set.signals) == 0:
            raise ValidationError("signals list cannot be empty")

        # Validate all signals are numpy arrays with same shape
        first_shape = None
        for i, signal in enumerate(signal_set.signals):
            if not isinstance(signal, np.ndarray):
                raise ValidationError(f"Signal {i} must be a numpy array")
            if signal.ndim != 1:
                raise ValidationError(f"Signal {i} must be 1-dimensional")
            if first_shape is None:
                first_shape = signal.shape
            elif signal.shape != first_shape:
                raise ValidationError(
                    f"All signals must have the same shape, "
                    f"signal {i} has shape {signal.shape}, "
                    f"expected {first_shape}"
                )

        # Validate phases array
        if not isinstance(signal_set.phases, np.ndarray):
            raise ValidationError("phases must be a numpy array")
        if signal_set.phases.ndim != 2:
            raise ValidationError("phases must be 2-dimensional")
        if signal_set.phases.shape[0] != len(signal_set.signals):
            raise ValidationError(
                f"phases array first dimension ({signal_set.phases.shape[0]}) "
                f"must match number of signals ({len(signal_set.signals)})"
            )

        # Validate orthogonality_score
        if not isinstance(signal_set.orthogonality_score, (int, float)):
            raise ValidationError("orthogonality_score must be a number")
        if not (0.0 <= signal_set.orthogonality_score <= 1.0):
            raise ValidationError("orthogonality_score must be in range [0, 1]")

        # Validate generation_timestamp
        from datetime import datetime

        if not isinstance(signal_set.generation_timestamp, datetime):
            raise ValidationError("generation_timestamp must be a datetime object")

        # Validate config
        if not hasattr(signal_set.config, "num_subcarriers"):
            raise ValidationError("config must be an OFDMConfig instance")

        # Validate metadata
        if not isinstance(signal_set.metadata, dict):
            raise ValidationError("metadata must be a dictionary")

    @classmethod
    def validate_parameter_compatibility(
        cls, ofdm_config: "OFDMConfig", chirp_config: "ChirpConfig"
    ) -> None:
        """Validate compatibility between OFDM and chirp configurations.

        Args:
            ofdm_config: OFDM configuration
            chirp_config: Chirp configuration

        Raises:
            ValidationError: If configurations are incompatible
        """
        # Check if phase matrix dimensions match OFDM configuration
        if chirp_config.phase_matrix.shape[1] != ofdm_config.num_subcarriers:
            raise ValidationError(
                f"Phase matrix has {chirp_config.phase_matrix.shape[1]} columns, "
                f"but OFDM config specifies {ofdm_config.num_subcarriers} subcarriers"
            )

        # Check if chirp length is reasonable for signal duration
        expected_samples = int(ofdm_config.sampling_rate * ofdm_config.signal_duration)
        if chirp_config.chirp_length > expected_samples:
            raise ValidationError(
                f"Chirp length ({chirp_config.chirp_length}) exceeds "
                f"total signal samples ({expected_samples})"
            )
