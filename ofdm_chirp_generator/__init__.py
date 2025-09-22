"""
OFDM Chirp Signal Generator Package

A GPU-accelerated OFDM signal generator that creates signals with chirp-modulated
subcarriers for orthogonal signal transmission and separation.
"""

from .chirp_modulator import ChirpModulator
from .config_manager import ConfigurationError, ConfigurationManager, get_config
from .correlation_analyzer import CorrelationAnalyzer
from .gpu_backend import GPUBackend, MemoryManager
from .main import (
    OFDMChirpGenerator,
    create_generator,
    quick_generate_orthogonal_signals,
    quick_test_separation,
)
from .models import ChirpConfig, OFDMConfig, SignalSet
from .ofdm_generator import OFDMGenerator
from .orthogonal_signal_generator import (
    OrthogonalSetConfig,
    OrthogonalSignalGenerator,
    PhaseMatrixManager,
)
from .orthogonality_tester import OrthogonalityTester
from .phase_optimizer import OptimizationConfig, OptimizationResult, PhaseOptimizer
from .signal_export import PhaseConfigurationManager, SignalExporter, SignalLoader, SignalVisualizer
from .signal_separator import SeparationQualityMetrics, SignalSeparator
from .subcarrier_manager import SubcarrierManager
from .validation import ConfigValidator

__version__ = "0.1.0"
__all__ = [
    # Core data models
    "OFDMConfig",
    "ChirpConfig",
    "SignalSet",
    # Validation and configuration
    "ConfigValidator",
    "ConfigurationManager",
    "ConfigurationError",
    "get_config",
    # GPU backend
    "GPUBackend",
    "MemoryManager",
    # Signal generation components
    "ChirpModulator",
    "SubcarrierManager",
    "OFDMGenerator",
    # Orthogonality and optimization
    "CorrelationAnalyzer",
    "OrthogonalityTester",
    "PhaseOptimizer",
    "OptimizationResult",
    "OptimizationConfig",
    # Orthogonal signal generation
    "OrthogonalSignalGenerator",
    "PhaseMatrixManager",
    "OrthogonalSetConfig",
    # Signal separation
    "SignalSeparator",
    "SeparationQualityMetrics",
    # Export and visualization
    "SignalExporter",
    "PhaseConfigurationManager",
    "SignalVisualizer",
    "SignalLoader",
    # Main interface (primary API)
    "OFDMChirpGenerator",
    "create_generator",
    "quick_generate_orthogonal_signals",
    "quick_test_separation",
]
