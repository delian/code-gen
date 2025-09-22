"""
Phase optimization algorithms for OFDM signal orthogonality.

This module implements systematic phase combination search algorithms to find
optimal phase configurations that maximize orthogonality between OFDM signals.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .config_manager import ConfigurationError, get_config
from .error_handling import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    OrthogonalityError,
    create_error_context,
    with_error_handling,
)
from .gpu_backend import GPUBackend
from .models import OFDMConfig, SignalSet
from .ofdm_generator import OFDMGenerator
from .orthogonality_tester import OrthogonalityTester

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Container for phase optimization results."""

    optimal_phases: np.ndarray
    orthogonality_score: float
    convergence_history: List[float]
    optimization_time: float
    iterations: int
    converged: bool
    method_used: str
    metadata: Dict[str, any]


@dataclass
class OptimizationConfig:
    """Configuration for phase optimization algorithms."""

    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    orthogonality_target: float = 0.95
    population_size: int = 50  # For genetic algorithm
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    early_stopping_patience: int = 50
    phase_resolution: int = 32  # Number of discrete phase values for brute force


class PhaseOptimizer:
    """Systematic phase combination search for orthogonal OFDM signals.

    This class implements multiple optimization strategies to find phase
    configurations that maximize orthogonality between OFDM signals.

    Requirements addressed:
    - 5.1: Systematic phase combination search
    - 5.3: Store phase configuration when valid orthogonal set found
    - 5.4: Report best available approximation if no orthogonal solution exists
    """

    def __init__(
        self,
        ofdm_config: Optional[OFDMConfig] = None,
        gpu_backend: Optional[GPUBackend] = None,
        config_file: Optional[str] = None,
    ):
        """Initialize phase optimizer.

        Args:
            ofdm_config: OFDM configuration parameters (loads from config if None)
            gpu_backend: GPU backend for acceleration (creates new if None)
            config_file: Path to configuration file (uses default if None)
        """
        self._error_handler = ErrorHandler()

        # Load configuration if not provided
        if ofdm_config is None:
            try:
                config_manager = get_config(config_file)
                self.ofdm_config = config_manager.create_ofdm_config_object()
                logger.info("Loaded OFDM configuration from config file")
            except (ConfigurationError, Exception) as e:
                context = create_error_context(
                    "config_load", "PhaseOptimizer", config_file=config_file
                )
                self._error_handler.handle_error(e, context)
                logger.warning(f"Could not load configuration: {e}. Using provided ofdm_config.")
                if ofdm_config is None:
                    raise ValueError(
                        "Either ofdm_config must be provided or configuration file must be available"
                    )
        else:
            self.ofdm_config = ofdm_config

        self.gpu_backend = gpu_backend or GPUBackend()

        # Initialize components
        try:
            self.ofdm_generator = OFDMGenerator(self.ofdm_config, self.gpu_backend)
            self.orthogonality_tester = OrthogonalityTester(self.gpu_backend)
        except Exception as e:
            context = create_error_context("component_initialization", "PhaseOptimizer")
            self._error_handler.handle_error(e, context)
            raise

        # Optimization state
        self.optimization_history = []
        self.best_result = None

        logger.info(
            f"PhaseOptimizer initialized: {self.ofdm_config.num_subcarriers} subcarriers, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}"
        )

    def find_orthogonal_phases(
        self,
        num_signals: int,
        optimization_config: Optional[OptimizationConfig] = None,
        method: str = "hybrid",
        config_file: Optional[str] = None,
    ) -> OptimizationResult:
        """Find optimal phase combinations for orthogonal signals.

        Requirements:
        - 5.1: Test different phase combinations systematically
        - 5.3: Store phase configuration when valid orthogonal set found
        - 5.4: Report best available approximation if no solution exists

        Args:
            num_signals: Number of orthogonal signals to generate
            optimization_config: Optimization parameters (loads from config if None)
            method: Optimization method ('brute_force', 'genetic', 'gradient', 'hybrid')
            config_file: Path to configuration file (uses default if None)

        Returns:
            OptimizationResult with best phase configuration found
        """
        if optimization_config is None:
            try:
                config_manager = get_config(config_file)
                optimization_config = config_manager.create_optimization_config_object()
                logger.info("Loaded optimization configuration from config file")
            except (ConfigurationError, Exception) as e:
                logger.warning(f"Could not load optimization configuration: {e}. Using defaults.")
                optimization_config = OptimizationConfig()

        start_time = time.time()

        logger.info(f"Starting phase optimization: {num_signals} signals, method={method}")

        # Select optimization method
        if method == "brute_force":
            result = self._brute_force_optimization(num_signals, optimization_config)
        elif method == "genetic":
            result = self._genetic_optimization(num_signals, optimization_config)
        elif method == "gradient":
            result = self._gradient_optimization(num_signals, optimization_config)
        elif method == "hybrid":
            result = self._hybrid_optimization(num_signals, optimization_config)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Update timing
        result.optimization_time = time.time() - start_time
        result.method_used = method

        # Store best result
        if (
            self.best_result is None
            or result.orthogonality_score > self.best_result.orthogonality_score
        ):
            self.best_result = result

        # Add to history
        self.optimization_history.append(result)

        logger.info(
            f"Phase optimization completed: score={result.orthogonality_score:.6f}, "
            f"converged={result.converged}, time={result.optimization_time:.2f}s"
        )

        return result

    def _brute_force_optimization(
        self, num_signals: int, config: OptimizationConfig
    ) -> OptimizationResult:
        """Brute-force search over discrete phase space.

        Args:
            num_signals: Number of signals to optimize
            config: Optimization configuration

        Returns:
            OptimizationResult with best configuration found
        """
        logger.debug(
            f"Starting brute-force optimization with {config.phase_resolution} phase levels"
        )

        # Generate discrete phase values
        phase_values = np.linspace(0, 2 * np.pi, config.phase_resolution, endpoint=False)

        best_phases = None
        best_score = -1.0
        convergence_history = []
        iterations = 0

        # For small problems, try all combinations
        if num_signals <= 3 and self.ofdm_config.num_subcarriers <= 8:
            # Full combinatorial search
            total_combinations = config.phase_resolution ** (
                num_signals * self.ofdm_config.num_subcarriers
            )
            logger.debug(f"Full brute-force: {total_combinations} combinations")

            # Generate all phase combinations (limited to prevent memory issues)
            if total_combinations <= 1e6:
                best_phases, best_score, convergence_history, iterations = self._exhaustive_search(
                    num_signals, phase_values, config
                )
            else:
                # Fall back to random sampling
                best_phases, best_score, convergence_history, iterations = self._random_sampling(
                    num_signals, phase_values, config
                )
        else:
            # Random sampling for larger problems
            best_phases, best_score, convergence_history, iterations = self._random_sampling(
                num_signals, phase_values, config
            )

        # Check convergence
        converged = best_score >= config.orthogonality_target

        return OptimizationResult(
            optimal_phases=best_phases,
            orthogonality_score=best_score,
            convergence_history=convergence_history,
            optimization_time=0.0,  # Will be set by caller
            iterations=iterations,
            converged=converged,
            method_used="brute_force",
            metadata={
                "phase_resolution": config.phase_resolution,
                "search_type": "exhaustive" if total_combinations <= 1e6 else "random_sampling",
            },
        )

    def _exhaustive_search(
        self, num_signals: int, phase_values: np.ndarray, config: OptimizationConfig
    ) -> Tuple[np.ndarray, float, List[float], int]:
        """Exhaustive search over all phase combinations."""
        best_phases = None
        best_score = -1.0
        convergence_history = []
        iterations = 0

        # Generate all combinations using itertools-like approach
        from itertools import product

        phase_indices = list(range(len(phase_values)))
        total_phases = num_signals * self.ofdm_config.num_subcarriers

        for combination in product(phase_indices, repeat=total_phases):
            if iterations >= config.max_iterations:
                break

            # Convert indices to phase matrix
            phase_matrix = np.array(combination).reshape(
                num_signals, self.ofdm_config.num_subcarriers
            )
            phase_matrix = phase_values[phase_matrix]

            # Evaluate orthogonality
            score = self._evaluate_phase_configuration(phase_matrix)
            convergence_history.append(score)

            if score > best_score:
                best_score = score
                best_phases = phase_matrix.copy()

            iterations += 1

            # Early stopping if target reached
            if score >= config.orthogonality_target:
                logger.debug(f"Target orthogonality reached at iteration {iterations}")
                break

        return best_phases, best_score, convergence_history, iterations

    def _random_sampling(
        self, num_signals: int, phase_values: np.ndarray, config: OptimizationConfig
    ) -> Tuple[np.ndarray, float, List[float], int]:
        """Random sampling of phase combinations."""
        best_phases = None
        best_score = -1.0
        convergence_history = []
        no_improvement_count = 0

        for iteration in range(config.max_iterations):
            # Generate random phase matrix
            phase_indices = np.random.randint(
                0, len(phase_values), size=(num_signals, self.ofdm_config.num_subcarriers)
            )
            phase_matrix = phase_values[phase_indices]

            # Evaluate orthogonality
            score = self._evaluate_phase_configuration(phase_matrix)
            convergence_history.append(score)

            if score > best_score:
                best_score = score
                best_phases = phase_matrix.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if score >= config.orthogonality_target:
                logger.debug(f"Target orthogonality reached at iteration {iteration}")
                break

            if no_improvement_count >= config.early_stopping_patience:
                logger.debug(f"Early stopping at iteration {iteration}")
                break

        return best_phases, best_score, convergence_history, iteration + 1

    def _genetic_optimization(
        self, num_signals: int, config: OptimizationConfig
    ) -> OptimizationResult:
        """Genetic algorithm for phase optimization.

        Args:
            num_signals: Number of signals to optimize
            config: Optimization configuration

        Returns:
            OptimizationResult with best configuration found
        """
        logger.debug(f"Starting genetic optimization with population size {config.population_size}")

        # Initialize population
        population = self._initialize_population(num_signals, config.population_size)
        fitness_scores = np.array(
            [self._evaluate_phase_configuration(individual) for individual in population]
        )

        best_score = np.max(fitness_scores)
        best_individual = population[np.argmax(fitness_scores)].copy()
        convergence_history = [best_score]
        no_improvement_count = 0

        for generation in range(config.max_iterations):
            # Selection, crossover, and mutation
            new_population = []

            for _ in range(config.population_size):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.random() < config.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if np.random.random() < config.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            # Evaluate new population
            population = new_population
            fitness_scores = np.array(
                [self._evaluate_phase_configuration(individual) for individual in population]
            )

            # Update best solution
            current_best_score = np.max(fitness_scores)
            if current_best_score > best_score:
                best_score = current_best_score
                best_individual = population[np.argmax(fitness_scores)].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            convergence_history.append(best_score)

            # Early stopping
            if best_score >= config.orthogonality_target:
                logger.debug(f"Target orthogonality reached at generation {generation}")
                break

            if no_improvement_count >= config.early_stopping_patience:
                logger.debug(f"Early stopping at generation {generation}")
                break

        converged = best_score >= config.orthogonality_target

        return OptimizationResult(
            optimal_phases=best_individual,
            orthogonality_score=best_score,
            convergence_history=convergence_history,
            optimization_time=0.0,  # Will be set by caller
            iterations=generation + 1,
            converged=converged,
            method_used="genetic",
            metadata={
                "population_size": config.population_size,
                "mutation_rate": config.mutation_rate,
                "crossover_rate": config.crossover_rate,
            },
        )

    def _gradient_optimization(
        self, num_signals: int, config: OptimizationConfig
    ) -> OptimizationResult:
        """Gradient-based optimization for phase optimization.

        Args:
            num_signals: Number of signals to optimize
            config: Optimization configuration

        Returns:
            OptimizationResult with best configuration found
        """
        logger.debug("Starting gradient-based optimization")

        # Initialize with random phases
        current_phases = np.random.uniform(
            0, 2 * np.pi, size=(num_signals, self.ofdm_config.num_subcarriers)
        )

        learning_rate = 0.1
        best_phases = current_phases.copy()
        best_score = self._evaluate_phase_configuration(current_phases)
        convergence_history = [best_score]
        no_improvement_count = 0

        for iteration in range(config.max_iterations):
            # Compute numerical gradient
            gradient = self._compute_numerical_gradient(current_phases)

            # Update phases
            current_phases += learning_rate * gradient

            # Keep phases in [0, 2π] range
            current_phases = current_phases % (2 * np.pi)

            # Evaluate new configuration
            current_score = self._evaluate_phase_configuration(current_phases)
            convergence_history.append(current_score)

            # Update best solution
            if current_score > best_score:
                best_score = current_score
                best_phases = current_phases.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                # Reduce learning rate if no improvement
                learning_rate *= 0.95

            # Early stopping
            if best_score >= config.orthogonality_target:
                logger.debug(f"Target orthogonality reached at iteration {iteration}")
                break

            if no_improvement_count >= config.early_stopping_patience:
                logger.debug(f"Early stopping at iteration {iteration}")
                break

        converged = best_score >= config.orthogonality_target

        return OptimizationResult(
            optimal_phases=best_phases,
            orthogonality_score=best_score,
            convergence_history=convergence_history,
            optimization_time=0.0,  # Will be set by caller
            iterations=iteration + 1,
            converged=converged,
            method_used="gradient",
            metadata={"final_learning_rate": learning_rate, "gradient_method": "numerical"},
        )

    def _hybrid_optimization(
        self, num_signals: int, config: OptimizationConfig
    ) -> OptimizationResult:
        """Hybrid optimization combining multiple methods.

        Args:
            num_signals: Number of signals to optimize
            config: Optimization configuration

        Returns:
            OptimizationResult with best configuration found
        """
        logger.debug("Starting hybrid optimization")

        # Phase 1: Genetic algorithm for global search
        genetic_config = OptimizationConfig(
            max_iterations=config.max_iterations // 3,
            population_size=config.population_size,
            orthogonality_target=config.orthogonality_target * 0.8,  # Lower target for first phase
        )
        genetic_result = self._genetic_optimization(num_signals, genetic_config)

        # Phase 2: Gradient refinement
        gradient_config = OptimizationConfig(
            max_iterations=config.max_iterations // 3,
            orthogonality_target=config.orthogonality_target,
        )

        # Initialize gradient search with genetic result
        current_phases = genetic_result.optimal_phases.copy()
        learning_rate = 0.05
        best_phases = current_phases.copy()
        best_score = genetic_result.orthogonality_score
        convergence_history = genetic_result.convergence_history.copy()

        for iteration in range(gradient_config.max_iterations):
            # Compute numerical gradient
            gradient = self._compute_numerical_gradient(current_phases)

            # Update phases
            current_phases += learning_rate * gradient
            current_phases = current_phases % (2 * np.pi)

            # Evaluate new configuration
            current_score = self._evaluate_phase_configuration(current_phases)
            convergence_history.append(current_score)

            if current_score > best_score:
                best_score = current_score
                best_phases = current_phases.copy()
            else:
                learning_rate *= 0.9

            if best_score >= config.orthogonality_target:
                break

        # Phase 3: Local random search for fine-tuning
        local_search_iterations = config.max_iterations - len(convergence_history)
        for iteration in range(local_search_iterations):
            # Small random perturbation
            perturbed_phases = best_phases + np.random.normal(0, 0.1, best_phases.shape)
            perturbed_phases = perturbed_phases % (2 * np.pi)

            score = self._evaluate_phase_configuration(perturbed_phases)
            convergence_history.append(score)

            if score > best_score:
                best_score = score
                best_phases = perturbed_phases.copy()

            if best_score >= config.orthogonality_target:
                break

        converged = best_score >= config.orthogonality_target
        total_iterations = len(convergence_history)

        return OptimizationResult(
            optimal_phases=best_phases,
            orthogonality_score=best_score,
            convergence_history=convergence_history,
            optimization_time=0.0,  # Will be set by caller
            iterations=total_iterations,
            converged=converged,
            method_used="hybrid",
            metadata={
                "genetic_iterations": len(genetic_result.convergence_history),
                "gradient_iterations": gradient_config.max_iterations,
                "local_search_iterations": local_search_iterations,
            },
        )

    def _evaluate_phase_configuration(self, phase_matrix: np.ndarray) -> float:
        """Evaluate orthogonality score for a phase configuration.

        Args:
            phase_matrix: Phase configuration [num_signals x num_subcarriers]

        Returns:
            Orthogonality score (0-1, higher is better)
        """
        try:
            # Validate input
            if phase_matrix.ndim != 2:
                raise ValueError("Phase matrix must be 2D")

            if phase_matrix.shape[1] != self.ofdm_config.num_subcarriers:
                raise ValueError(
                    f"Phase matrix must have {self.ofdm_config.num_subcarriers} columns"
                )

            # Check for NaN or Inf values
            if np.any(np.isnan(phase_matrix)) or np.any(np.isinf(phase_matrix)):
                raise ValueError("Phase matrix contains NaN or Inf values")

            # Generate signals for this phase configuration
            signals = []
            for i in range(phase_matrix.shape[0]):
                signal = self.ofdm_generator.generate_single_signal(phase_matrix[i, :])
                signals.append(signal)

            # Test orthogonality
            orthogonality_result = self.orthogonality_tester.test_signal_set_orthogonality(signals)

            score = orthogonality_result["overall_orthogonality_score"]

            # Validate score
            if np.isnan(score) or np.isinf(score):
                raise ValueError("Orthogonality evaluation produced invalid score")

            return max(0.0, min(1.0, score))  # Clamp to valid range

        except Exception as e:
            context = create_error_context(
                "phase_evaluation", "PhaseOptimizer", phase_matrix_shape=phase_matrix.shape
            )
            self._error_handler.handle_error(e, context)
            logger.warning(f"Error evaluating phase configuration: {e}")
            return 0.0

    def _initialize_population(self, num_signals: int, population_size: int) -> List[np.ndarray]:
        """Initialize population for genetic algorithm.

        Args:
            num_signals: Number of signals
            population_size: Size of population

        Returns:
            List of phase matrices
        """
        population = []
        for _ in range(population_size):
            phases = np.random.uniform(
                0, 2 * np.pi, size=(num_signals, self.ofdm_config.num_subcarriers)
            )
            population.append(phases)
        return population

    def _tournament_selection(
        self, population: List[np.ndarray], fitness_scores: np.ndarray, tournament_size: int = 3
    ) -> np.ndarray:
        """Tournament selection for genetic algorithm.

        Args:
            population: Current population
            fitness_scores: Fitness scores for population
            tournament_size: Size of tournament

        Returns:
            Selected individual
        """
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness_scores[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover operation for genetic algorithm.

        Args:
            parent1: First parent phase matrix
            parent2: Second parent phase matrix

        Returns:
            Child phase matrix
        """
        # Uniform crossover
        mask = np.random.random(parent1.shape) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, individual: np.ndarray, mutation_strength: float = 0.5) -> np.ndarray:
        """Mutation operation for genetic algorithm.

        Args:
            individual: Phase matrix to mutate
            mutation_strength: Strength of mutation

        Returns:
            Mutated phase matrix
        """
        # Add Gaussian noise to phases
        noise = np.random.normal(0, mutation_strength, individual.shape)
        mutated = individual + noise
        # Keep phases in [0, 2π] range
        mutated = mutated % (2 * np.pi)
        return mutated

    def _compute_numerical_gradient(
        self, phase_matrix: np.ndarray, epsilon: float = 1e-4
    ) -> np.ndarray:
        """Compute numerical gradient of orthogonality score.

        Args:
            phase_matrix: Current phase configuration
            epsilon: Step size for numerical differentiation

        Returns:
            Gradient matrix
        """
        gradient = np.zeros_like(phase_matrix)
        base_score = self._evaluate_phase_configuration(phase_matrix)

        for i in range(phase_matrix.shape[0]):
            for j in range(phase_matrix.shape[1]):
                # Forward difference
                perturbed_phases = phase_matrix.copy()
                perturbed_phases[i, j] += epsilon
                forward_score = self._evaluate_phase_configuration(perturbed_phases)

                # Compute gradient
                gradient[i, j] = (forward_score - base_score) / epsilon

        return gradient

    def test_orthogonality(self, phase_matrix: np.ndarray) -> Dict[str, any]:
        """Test orthogonality of a phase configuration.

        Requirements:
        - 5.2: Compute cross-correlation between signal pairs

        Args:
            phase_matrix: Phase configuration to test

        Returns:
            Comprehensive orthogonality analysis
        """
        # Generate signals
        signals = []
        for i in range(phase_matrix.shape[0]):
            signal = self.ofdm_generator.generate_single_signal(phase_matrix[i, :])
            signals.append(signal)

        # Test orthogonality
        return self.orthogonality_tester.test_signal_set_orthogonality(signals)

    def optimize_phase_set(
        self,
        initial_phases: np.ndarray,
        optimization_config: Optional[OptimizationConfig] = None,
        method: str = "gradient",
    ) -> OptimizationResult:
        """Optimize an existing phase set.

        Args:
            initial_phases: Starting phase configuration
            optimization_config: Optimization parameters
            method: Optimization method to use

        Returns:
            OptimizationResult with improved configuration
        """
        if optimization_config is None:
            optimization_config = OptimizationConfig()

        num_signals = initial_phases.shape[0]

        if method == "gradient":
            # Use gradient optimization starting from initial phases
            current_phases = initial_phases.copy()
            learning_rate = 0.1
            best_phases = current_phases.copy()
            best_score = self._evaluate_phase_configuration(current_phases)
            convergence_history = [best_score]

            start_time = time.time()

            for iteration in range(optimization_config.max_iterations):
                # Compute gradient
                gradient = self._compute_numerical_gradient(current_phases)

                # Update phases
                current_phases += learning_rate * gradient
                current_phases = current_phases % (2 * np.pi)

                # Evaluate
                current_score = self._evaluate_phase_configuration(current_phases)
                convergence_history.append(current_score)

                if current_score > best_score:
                    best_score = current_score
                    best_phases = current_phases.copy()
                else:
                    learning_rate *= 0.95

                # Check convergence
                if len(convergence_history) > 1:
                    improvement = convergence_history[-1] - convergence_history[-2]
                    if abs(improvement) < optimization_config.convergence_threshold:
                        break

                if best_score >= optimization_config.orthogonality_target:
                    break

            converged = best_score >= optimization_config.orthogonality_target

            return OptimizationResult(
                optimal_phases=best_phases,
                orthogonality_score=best_score,
                convergence_history=convergence_history,
                optimization_time=time.time() - start_time,
                iterations=iteration + 1,
                converged=converged,
                method_used=f"gradient_refinement",
                metadata={"initial_score": convergence_history[0]},
            )
        else:
            # Use general optimization method
            return self.find_orthogonal_phases(num_signals, optimization_config, method)

    def get_separation_quality(self, signals: List[Union[np.ndarray, "cp.ndarray"]]) -> float:
        """Get separation quality metric for a set of signals.

        Requirements:
        - 5.4: Optimization quality metrics

        Args:
            signals: List of signal arrays

        Returns:
            Separation quality score (0-1, higher is better)
        """
        orthogonality_result = self.orthogonality_tester.test_signal_set_orthogonality(signals)
        return orthogonality_result["overall_orthogonality_score"]

    def generate_orthogonal_signal_set(
        self,
        num_signals: int,
        optimization_config: Optional[OptimizationConfig] = None,
        method: str = "hybrid",
    ) -> SignalSet:
        """Generate a complete orthogonal signal set.

        Requirements:
        - 5.1: Systematic phase combination search
        - 5.3: Store phase configuration when valid orthogonal set found

        Args:
            num_signals: Number of orthogonal signals to generate
            optimization_config: Optimization parameters
            method: Optimization method to use

        Returns:
            SignalSet with orthogonal signals
        """
        # Find optimal phases
        optimization_result = self.find_orthogonal_phases(num_signals, optimization_config, method)

        # Create signal set
        signal_set = self.ofdm_generator.create_signal_set(
            optimization_result.optimal_phases, optimization_result.orthogonality_score
        )

        # Add optimization metadata
        signal_set.metadata.update(
            {
                "optimization_method": method,
                "optimization_iterations": optimization_result.iterations,
                "optimization_time": optimization_result.optimization_time,
                "converged": optimization_result.converged,
                "convergence_history": optimization_result.convergence_history,
            }
        )

        return signal_set

    def analyze_convergence(self, convergence_history: List[float]) -> Dict[str, any]:
        """Analyze convergence characteristics of optimization.

        Args:
            convergence_history: History of objective function values

        Returns:
            Convergence analysis results
        """
        if len(convergence_history) < 2:
            return {"error": "Insufficient data for convergence analysis"}

        history = np.array(convergence_history)

        # Calculate convergence metrics
        final_value = history[-1]
        initial_value = history[0]
        improvement = final_value - initial_value

        # Find convergence point (where improvement becomes small)
        convergence_point = len(history)
        threshold = 1e-6

        for i in range(1, len(history)):
            if i >= 10:  # Look for sustained convergence
                recent_improvement = np.max(history[i - 10 : i]) - np.min(history[i - 10 : i])
                if recent_improvement < threshold:
                    convergence_point = i
                    break

        # Calculate convergence rate
        if convergence_point > 1:
            convergence_rate = improvement / convergence_point
        else:
            convergence_rate = 0.0

        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_improvement": improvement,
            "convergence_point": convergence_point,
            "convergence_rate": convergence_rate,
            "converged": convergence_point < len(history),
            "optimization_efficiency": improvement / len(history) if len(history) > 0 else 0.0,
        }

    def get_optimization_report(self, result: OptimizationResult) -> str:
        """Generate comprehensive optimization report.

        Args:
            result: Optimization result to analyze

        Returns:
            Formatted report string
        """
        convergence_analysis = self.analyze_convergence(result.convergence_history)

        report = []
        report.append("=" * 60)
        report.append("PHASE OPTIMIZATION REPORT")
        report.append("=" * 60)
        report.append(f"Method: {result.method_used}")
        report.append(f"Optimization time: {result.optimization_time:.2f} seconds")
        report.append(f"Iterations: {result.iterations}")
        report.append(f"Converged: {'YES' if result.converged else 'NO'}")
        report.append("")

        report.append("RESULTS:")
        report.append(f"  Final orthogonality score: {result.orthogonality_score:.6f}")
        report.append(f"  Phase matrix shape: {result.optimal_phases.shape}")
        report.append(f"  Number of signals: {result.optimal_phases.shape[0]}")
        report.append(f"  Subcarriers per signal: {result.optimal_phases.shape[1]}")
        report.append("")

        report.append("CONVERGENCE ANALYSIS:")
        report.append(f"  Initial score: {convergence_analysis['initial_value']:.6f}")
        report.append(f"  Final score: {convergence_analysis['final_value']:.6f}")
        report.append(f"  Total improvement: {convergence_analysis['total_improvement']:.6f}")
        report.append(f"  Convergence point: iteration {convergence_analysis['convergence_point']}")
        report.append(f"  Convergence rate: {convergence_analysis['convergence_rate']:.8f}")
        report.append(
            f"  Optimization efficiency: {convergence_analysis['optimization_efficiency']:.8f}"
        )
        report.append("")

        if result.metadata:
            report.append("ALGORITHM PARAMETERS:")
            for key, value in result.metadata.items():
                report.append(f"  {key}: {value}")
            report.append("")

        report.append("BACKEND INFORMATION:")
        device_info = self.gpu_backend.device_info
        report.append(f"  Compute backend: {device_info['backend']}")
        if device_info["backend"] == "GPU":
            report.append(f"  Device: {device_info['device_name']}")

        report.append("=" * 60)

        return "\n".join(report)

    def save_optimization_result(self, result: OptimizationResult, filepath: str) -> None:
        """Save optimization result to file.

        Args:
            result: Optimization result to save
            filepath: Path to save file
        """
        import pickle

        save_data = {"result": result, "ofdm_config": self.ofdm_config, "timestamp": time.time()}

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Optimization result saved to {filepath}")

    def load_optimization_result(self, filepath: str) -> OptimizationResult:
        """Load optimization result from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded optimization result
        """
        import pickle

        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        logger.info(f"Optimization result loaded from {filepath}")
        return save_data["result"]

    def get_best_result(self) -> Optional[OptimizationResult]:
        """Get the best optimization result found so far.

        Returns:
            Best optimization result or None if no optimization has been run
        """
        return self.best_result

    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of all optimization runs.

        Returns:
            List of optimization results
        """
        return self.optimization_history.copy()

    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        self.best_result = None
        logger.debug("Optimization history cleared")

    def cleanup_resources(self) -> None:
        """Clean up GPU resources and memory."""
        self.ofdm_generator.cleanup_resources()
        self.gpu_backend.cleanup_memory()
        logger.debug("PhaseOptimizer resources cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_resources()

    def __repr__(self) -> str:
        """String representation of PhaseOptimizer."""
        return (
            f"PhaseOptimizer(subcarriers={self.ofdm_config.num_subcarriers}, "
            f"backend={'GPU' if self.gpu_backend.is_gpu_available else 'CPU'}, "
            f"history_length={len(self.optimization_history)})"
        )
