"""
Tests for phase optimization algorithms.

This module tests the PhaseOptimizer class and its optimization strategies
for finding orthogonal phase configurations.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ofdm_chirp_generator.gpu_backend import GPUBackend
from ofdm_chirp_generator.models import OFDMConfig
from ofdm_chirp_generator.phase_optimizer import (
    OptimizationConfig,
    OptimizationResult,
    PhaseOptimizer,
)


class TestPhaseOptimizer:
    """Test cases for PhaseOptimizer class."""

    @pytest.fixture
    def ofdm_config(self):
        """Create test OFDM configuration."""
        return OFDMConfig(
            num_subcarriers=4,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=50000.0,
            signal_duration=0.002,  # Increased to meet minimum chirp length requirements
        )

    @pytest.fixture
    def gpu_backend(self):
        """Create test GPU backend."""
        return GPUBackend()

    @pytest.fixture
    def phase_optimizer(self, ofdm_config, gpu_backend):
        """Create test PhaseOptimizer instance."""
        return PhaseOptimizer(ofdm_config, gpu_backend)

    def test_initialization(self, ofdm_config, gpu_backend):
        """Test PhaseOptimizer initialization."""
        optimizer = PhaseOptimizer(ofdm_config, gpu_backend)

        assert optimizer.ofdm_config == ofdm_config
        assert optimizer.gpu_backend == gpu_backend
        assert optimizer.ofdm_generator is not None
        assert optimizer.orthogonality_tester is not None
        assert optimizer.optimization_history == []
        assert optimizer.best_result is None

    def test_initialization_without_backend(self, ofdm_config):
        """Test PhaseOptimizer initialization without explicit backend."""
        optimizer = PhaseOptimizer(ofdm_config)

        assert optimizer.gpu_backend is not None
        assert isinstance(optimizer.gpu_backend, GPUBackend)

    def test_evaluate_phase_configuration(self, phase_optimizer):
        """Test phase configuration evaluation."""
        # Create test phase matrix
        phase_matrix = np.random.uniform(0, 2 * np.pi, size=(2, 4))

        # Evaluate configuration
        score = phase_optimizer._evaluate_phase_configuration(phase_matrix)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_evaluate_phase_configuration_error_handling(self, phase_optimizer):
        """Test error handling in phase configuration evaluation."""
        # Create invalid phase matrix (wrong shape)
        invalid_phase_matrix = np.random.uniform(0, 2 * np.pi, size=(2, 10))

        # Should handle error gracefully and return 0.0
        score = phase_optimizer._evaluate_phase_configuration(invalid_phase_matrix)
        assert score == 0.0

    def test_initialize_population(self, phase_optimizer):
        """Test population initialization for genetic algorithm."""
        num_signals = 3
        population_size = 10

        population = phase_optimizer._initialize_population(num_signals, population_size)

        assert len(population) == population_size
        for individual in population:
            assert individual.shape == (num_signals, 4)  # 4 subcarriers
            assert np.all(individual >= 0)
            assert np.all(individual < 2 * np.pi)

    def test_tournament_selection(self, phase_optimizer):
        """Test tournament selection for genetic algorithm."""
        # Create test population
        population = [np.random.uniform(0, 2 * np.pi, size=(2, 4)) for _ in range(5)]
        fitness_scores = np.array([0.1, 0.8, 0.3, 0.9, 0.2])

        # Run tournament selection multiple times
        selected_count = {}
        for _ in range(100):
            selected = phase_optimizer._tournament_selection(population, fitness_scores)
            # Find which individual was selected
            for i, individual in enumerate(population):
                if np.array_equal(selected, individual):
                    selected_count[i] = selected_count.get(i, 0) + 1
                    break

        # Higher fitness individuals should be selected more often
        # Individual 3 (fitness 0.9) should be selected most often
        assert selected_count.get(3, 0) > selected_count.get(0, 0)

    def test_crossover(self, phase_optimizer):
        """Test crossover operation for genetic algorithm."""
        parent1 = np.ones((2, 4)) * 1.0
        parent2 = np.ones((2, 4)) * 2.0

        child = phase_optimizer._crossover(parent1, parent2)

        assert child.shape == parent1.shape
        # Child should contain values from both parents
        unique_values = np.unique(child)
        assert len(unique_values) <= 2  # Should only contain parent values

    def test_mutate(self, phase_optimizer):
        """Test mutation operation for genetic algorithm."""
        original = np.ones((2, 4)) * np.pi

        mutated = phase_optimizer._mutate(original, mutation_strength=0.1)

        assert mutated.shape == original.shape
        assert np.all(mutated >= 0)
        assert np.all(mutated < 2 * np.pi)
        # Should be different from original (with high probability)
        assert not np.allclose(mutated, original, atol=1e-6)

    def test_compute_numerical_gradient(self, phase_optimizer):
        """Test numerical gradient computation."""
        # Create test phase matrix
        phase_matrix = np.random.uniform(0, 2 * np.pi, size=(2, 4))

        gradient = phase_optimizer._compute_numerical_gradient(phase_matrix, epsilon=1e-3)

        assert gradient.shape == phase_matrix.shape
        assert np.all(np.isfinite(gradient))

    def test_brute_force_optimization_small(self, phase_optimizer):
        """Test brute-force optimization with small problem."""
        config = OptimizationConfig(
            max_iterations=100, phase_resolution=8, orthogonality_target=0.8
        )

        result = phase_optimizer._brute_force_optimization(2, config)

        assert isinstance(result, OptimizationResult)
        assert result.optimal_phases.shape == (2, 4)
        assert 0.0 <= result.orthogonality_score <= 1.0
        assert len(result.convergence_history) > 0
        assert result.iterations > 0
        assert result.method_used == "brute_force"

    def test_genetic_optimization(self, phase_optimizer):
        """Test genetic algorithm optimization."""
        config = OptimizationConfig(
            max_iterations=20,
            population_size=10,
            orthogonality_target=0.8,
            mutation_rate=0.2,
            crossover_rate=0.8,
        )

        result = phase_optimizer._genetic_optimization(2, config)

        assert isinstance(result, OptimizationResult)
        assert result.optimal_phases.shape == (2, 4)
        assert 0.0 <= result.orthogonality_score <= 1.0
        assert len(result.convergence_history) > 0
        assert result.iterations > 0
        assert result.method_used == "genetic"
        assert "population_size" in result.metadata

    def test_gradient_optimization(self, phase_optimizer):
        """Test gradient-based optimization."""
        config = OptimizationConfig(
            max_iterations=20, orthogonality_target=0.8, convergence_threshold=1e-4
        )

        result = phase_optimizer._gradient_optimization(2, config)

        assert isinstance(result, OptimizationResult)
        assert result.optimal_phases.shape == (2, 4)
        assert 0.0 <= result.orthogonality_score <= 1.0
        assert len(result.convergence_history) > 0
        assert result.iterations > 0
        assert result.method_used == "gradient"

    def test_hybrid_optimization(self, phase_optimizer):
        """Test hybrid optimization strategy."""
        config = OptimizationConfig(max_iterations=30, population_size=5, orthogonality_target=0.8)

        result = phase_optimizer._hybrid_optimization(2, config)

        assert isinstance(result, OptimizationResult)
        assert result.optimal_phases.shape == (2, 4)
        assert 0.0 <= result.orthogonality_score <= 1.0
        assert len(result.convergence_history) > 0
        assert result.iterations > 0
        assert result.method_used == "hybrid"
        assert "genetic_iterations" in result.metadata

    def test_find_orthogonal_phases(self, phase_optimizer):
        """Test main phase optimization interface."""
        config = OptimizationConfig(max_iterations=20, orthogonality_target=0.7)

        result = phase_optimizer.find_orthogonal_phases(2, config, method="genetic")

        assert isinstance(result, OptimizationResult)
        assert result.optimal_phases.shape == (2, 4)
        assert result.optimization_time > 0
        assert result.method_used == "genetic"

        # Should be stored as best result
        assert phase_optimizer.best_result == result
        assert len(phase_optimizer.optimization_history) == 1

    def test_find_orthogonal_phases_invalid_method(self, phase_optimizer):
        """Test error handling for invalid optimization method."""
        with pytest.raises(ValueError, match="Unknown optimization method"):
            phase_optimizer.find_orthogonal_phases(2, method="invalid_method")

    def test_test_orthogonality(self, phase_optimizer):
        """Test orthogonality testing interface."""
        phase_matrix = np.random.uniform(0, 2 * np.pi, size=(2, 4))

        result = phase_optimizer.test_orthogonality(phase_matrix)

        assert isinstance(result, dict)
        assert "overall_orthogonality_score" in result
        assert "correlation_matrix" in result
        assert "is_set_orthogonal" in result

    def test_optimize_phase_set(self, phase_optimizer):
        """Test optimization of existing phase set."""
        initial_phases = np.random.uniform(0, 2 * np.pi, size=(2, 4))
        config = OptimizationConfig(max_iterations=10)

        result = phase_optimizer.optimize_phase_set(initial_phases, config, method="gradient")

        assert isinstance(result, OptimizationResult)
        assert result.optimal_phases.shape == initial_phases.shape
        assert "initial_score" in result.metadata

    def test_get_separation_quality(self, phase_optimizer):
        """Test separation quality measurement."""
        # Generate test signals
        phase_matrix = np.random.uniform(0, 2 * np.pi, size=(2, 4))
        signals = []
        for i in range(2):
            signal = phase_optimizer.ofdm_generator.generate_single_signal(phase_matrix[i, :])
            signals.append(signal)

        quality = phase_optimizer.get_separation_quality(signals)

        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0

    def test_generate_orthogonal_signal_set(self, phase_optimizer):
        """Test complete orthogonal signal set generation."""
        config = OptimizationConfig(max_iterations=10)

        signal_set = phase_optimizer.generate_orthogonal_signal_set(2, config, method="genetic")

        assert signal_set.num_signals == 2
        assert signal_set.phases.shape == (2, 4)
        assert "optimization_method" in signal_set.metadata
        assert "optimization_iterations" in signal_set.metadata
        assert "converged" in signal_set.metadata

    def test_analyze_convergence(self, phase_optimizer):
        """Test convergence analysis."""
        # Create test convergence history
        convergence_history = [0.1, 0.3, 0.5, 0.7, 0.75, 0.76, 0.76, 0.76]

        analysis = phase_optimizer.analyze_convergence(convergence_history)

        assert "initial_value" in analysis
        assert "final_value" in analysis
        assert "total_improvement" in analysis
        assert "convergence_point" in analysis
        assert "convergence_rate" in analysis
        assert "converged" in analysis
        assert analysis["initial_value"] == 0.1
        assert analysis["final_value"] == 0.76

    def test_analyze_convergence_insufficient_data(self, phase_optimizer):
        """Test convergence analysis with insufficient data."""
        convergence_history = [0.5]

        analysis = phase_optimizer.analyze_convergence(convergence_history)

        assert "error" in analysis

    def test_get_optimization_report(self, phase_optimizer):
        """Test optimization report generation."""
        # Create test result
        result = OptimizationResult(
            optimal_phases=np.random.uniform(0, 2 * np.pi, size=(2, 4)),
            orthogonality_score=0.85,
            convergence_history=[0.1, 0.5, 0.8, 0.85],
            optimization_time=1.5,
            iterations=4,
            converged=True,
            method_used="test_method",
            metadata={"test_param": "test_value"},
        )

        report = phase_optimizer.get_optimization_report(result)

        assert isinstance(report, str)
        assert "PHASE OPTIMIZATION REPORT" in report
        assert "test_method" in report
        assert "0.85" in report  # orthogonality score
        assert "YES" in report  # converged

    def test_save_and_load_optimization_result(self, phase_optimizer, tmp_path):
        """Test saving and loading optimization results."""
        # Create test result
        result = OptimizationResult(
            optimal_phases=np.random.uniform(0, 2 * np.pi, size=(2, 4)),
            orthogonality_score=0.75,
            convergence_history=[0.1, 0.5, 0.75],
            optimization_time=1.0,
            iterations=3,
            converged=False,
            method_used="test",
            metadata={},
        )

        # Save result
        filepath = tmp_path / "test_result.pkl"
        phase_optimizer.save_optimization_result(result, str(filepath))

        assert filepath.exists()

        # Load result
        loaded_result = phase_optimizer.load_optimization_result(str(filepath))

        assert np.array_equal(loaded_result.optimal_phases, result.optimal_phases)
        assert loaded_result.orthogonality_score == result.orthogonality_score
        assert loaded_result.method_used == result.method_used

    def test_get_best_result(self, phase_optimizer):
        """Test getting best optimization result."""
        # Initially no best result
        assert phase_optimizer.get_best_result() is None

        # Run optimization
        config = OptimizationConfig(max_iterations=5)
        result = phase_optimizer.find_orthogonal_phases(2, config)

        # Should now have best result
        best = phase_optimizer.get_best_result()
        assert best == result

    def test_get_optimization_history(self, phase_optimizer):
        """Test getting optimization history."""
        # Initially empty history
        assert phase_optimizer.get_optimization_history() == []

        # Run multiple optimizations
        config = OptimizationConfig(max_iterations=5)
        result1 = phase_optimizer.find_orthogonal_phases(2, config)
        result2 = phase_optimizer.find_orthogonal_phases(2, config)

        history = phase_optimizer.get_optimization_history()
        assert len(history) == 2
        # Check that results are in history by comparing unique attributes
        assert any(
            r.optimization_time == result1.optimization_time and r.iterations == result1.iterations
            for r in history
        )
        assert any(
            r.optimization_time == result2.optimization_time and r.iterations == result2.iterations
            for r in history
        )

    def test_clear_history(self, phase_optimizer):
        """Test clearing optimization history."""
        # Run optimization to create history
        config = OptimizationConfig(max_iterations=5)
        phase_optimizer.find_orthogonal_phases(2, config)

        assert len(phase_optimizer.optimization_history) > 0
        assert phase_optimizer.best_result is not None

        # Clear history
        phase_optimizer.clear_history()

        assert phase_optimizer.optimization_history == []
        assert phase_optimizer.best_result is None

    def test_context_manager(self, ofdm_config):
        """Test PhaseOptimizer as context manager."""
        with PhaseOptimizer(ofdm_config) as optimizer:
            assert optimizer is not None
            # Should work normally within context
            config = OptimizationConfig(max_iterations=5)
            result = optimizer.find_orthogonal_phases(2, config)
            assert isinstance(result, OptimizationResult)

        # Resources should be cleaned up after context exit
        # (We can't easily test this without mocking, but the context manager should work)

    def test_repr(self, phase_optimizer):
        """Test string representation."""
        repr_str = repr(phase_optimizer)

        assert "PhaseOptimizer" in repr_str
        assert "subcarriers=4" in repr_str
        assert "backend=" in repr_str
        assert "history_length=0" in repr_str


class TestOptimizationConfig:
    """Test cases for OptimizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.max_iterations == 1000
        assert config.convergence_threshold == 1e-6
        assert config.orthogonality_target == 0.95
        assert config.population_size == 50
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.8
        assert config.early_stopping_patience == 50
        assert config.phase_resolution == 32

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            max_iterations=500, orthogonality_target=0.8, population_size=20
        )

        assert config.max_iterations == 500
        assert config.orthogonality_target == 0.8
        assert config.population_size == 20
        # Other values should remain default
        assert config.convergence_threshold == 1e-6


class TestOptimizationResult:
    """Test cases for OptimizationResult dataclass."""

    def test_creation(self):
        """Test OptimizationResult creation."""
        phases = np.random.uniform(0, 2 * np.pi, size=(2, 4))
        history = [0.1, 0.5, 0.8]
        metadata = {"method": "test"}

        result = OptimizationResult(
            optimal_phases=phases,
            orthogonality_score=0.8,
            convergence_history=history,
            optimization_time=1.5,
            iterations=3,
            converged=True,
            method_used="test_method",
            metadata=metadata,
        )

        assert np.array_equal(result.optimal_phases, phases)
        assert result.orthogonality_score == 0.8
        assert result.convergence_history == history
        assert result.optimization_time == 1.5
        assert result.iterations == 3
        assert result.converged is True
        assert result.method_used == "test_method"
        assert result.metadata == metadata


@pytest.mark.integration
class TestPhaseOptimizerIntegration:
    """Integration tests for PhaseOptimizer with real optimization scenarios."""

    @pytest.fixture
    def large_ofdm_config(self):
        """Create larger OFDM configuration for integration tests."""
        return OFDMConfig(
            num_subcarriers=8,
            subcarrier_spacing=1000.0,
            bandwidth_per_subcarrier=800.0,
            center_frequency=10000.0,
            sampling_rate=100000.0,
            signal_duration=0.003,  # Increased to meet minimum chirp length requirements
        )

    def test_full_optimization_workflow(self, large_ofdm_config):
        """Test complete optimization workflow."""
        optimizer = PhaseOptimizer(large_ofdm_config)

        # Test different optimization methods
        methods = ["brute_force", "genetic", "gradient", "hybrid"]
        config = OptimizationConfig(max_iterations=20, orthogonality_target=0.6)

        results = {}
        for method in methods:
            try:
                result = optimizer.find_orthogonal_phases(2, config, method=method)
                results[method] = result
                assert result.orthogonality_score >= 0.0
            except Exception as e:
                pytest.fail(f"Method {method} failed: {e}")

        # All methods should produce results
        assert len(results) == len(methods)

        # Generate report for best result
        best_result = optimizer.get_best_result()
        report = optimizer.get_optimization_report(best_result)
        assert len(report) > 0

    def test_convergence_behavior(self, large_ofdm_config):
        """Test optimization convergence behavior."""
        optimizer = PhaseOptimizer(large_ofdm_config)

        config = OptimizationConfig(
            max_iterations=50, orthogonality_target=0.7, early_stopping_patience=10
        )

        result = optimizer.find_orthogonal_phases(3, config, method="genetic")

        # Check convergence properties
        history = result.convergence_history
        assert len(history) > 0

        # Score should generally improve or stay stable
        # (allowing for some noise in optimization)
        final_score = history[-1]
        initial_score = history[0]

        # Final score should be at least as good as initial (with some tolerance)
        assert final_score >= initial_score - 0.1

        # Analyze convergence
        convergence_analysis = optimizer.analyze_convergence(history)
        assert convergence_analysis["total_improvement"] >= -0.1  # Allow small degradation

    def test_scalability(self, large_ofdm_config):
        """Test optimization scalability with different problem sizes."""
        optimizer = PhaseOptimizer(large_ofdm_config)

        config = OptimizationConfig(max_iterations=10, orthogonality_target=0.5)

        # Test different numbers of signals
        signal_counts = [2, 3, 4]
        times = []

        for num_signals in signal_counts:
            start_time = time.time()
            result = optimizer.find_orthogonal_phases(num_signals, config, method="genetic")
            end_time = time.time()

            times.append(end_time - start_time)

            # Should produce valid results
            assert result.optimal_phases.shape == (num_signals, 8)  # 8 subcarriers
            assert result.orthogonality_score >= 0.0

        # Time should increase with problem size (generally)
        # But we won't enforce strict ordering due to randomness
        assert all(t > 0 for t in times)
