#!/usr/bin/env python3
"""
Phase Optimization Demo

This script demonstrates the phase optimization algorithms for finding
orthogonal OFDM signal configurations using various optimization strategies.
"""

import time

import numpy as np

from ofdm_chirp_generator import OFDMConfig, OptimizationConfig, PhaseOptimizer

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("OFDM CHIRP GENERATOR - PHASE OPTIMIZATION DEMO")
    print("=" * 60)

    # Create OFDM configuration
    ofdm_config = OFDMConfig(
        num_subcarriers=6,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002,
    )

    print(f"OFDM Configuration:")
    print(f"  Subcarriers: {ofdm_config.num_subcarriers}")
    print(f"  Subcarrier spacing: {ofdm_config.subcarrier_spacing} Hz")
    print(f"  Bandwidth per subcarrier: {ofdm_config.bandwidth_per_subcarrier} Hz")
    print(f"  Center frequency: {ofdm_config.center_frequency} Hz")
    print(f"  Sampling rate: {ofdm_config.sampling_rate} Hz")
    print(f"  Signal duration: {ofdm_config.signal_duration} s")
    print()

    # Initialize phase optimizer
    print("Initializing Phase Optimizer...")
    optimizer = PhaseOptimizer(ofdm_config)
    print(f"Backend: {'GPU' if optimizer.gpu_backend.is_gpu_available else 'CPU'}")
    print()

    # Demonstration parameters
    num_signals = 3
    print(f"Optimizing for {num_signals} orthogonal signals")
    print()

    # Test different optimization methods
    methods = ["brute_force", "genetic", "gradient", "hybrid"]
    results = {}

    for method in methods:
        print(f"Testing {method.upper()} optimization...")

        # Configure optimization parameters based on method
        if method == "brute_force":
            config = OptimizationConfig(
                max_iterations=200,
                phase_resolution=16,
                orthogonality_target=0.8,
                early_stopping_patience=20,
            )
        elif method == "genetic":
            config = OptimizationConfig(
                max_iterations=50,
                population_size=20,
                mutation_rate=0.15,
                crossover_rate=0.8,
                orthogonality_target=0.8,
                early_stopping_patience=15,
            )
        elif method == "gradient":
            config = OptimizationConfig(
                max_iterations=100,
                convergence_threshold=1e-5,
                orthogonality_target=0.8,
                early_stopping_patience=25,
            )
        else:  # hybrid
            config = OptimizationConfig(
                max_iterations=75,
                population_size=15,
                orthogonality_target=0.8,
                early_stopping_patience=20,
            )

        # Run optimization
        start_time = time.time()
        try:
            result = optimizer.find_orthogonal_phases(num_signals, config, method=method)
            results[method] = result

            print(f"  ✓ Completed in {result.optimization_time:.2f}s")
            print(f"  ✓ Orthogonality score: {result.orthogonality_score:.6f}")
            print(f"  ✓ Iterations: {result.iterations}")
            print(f"  ✓ Converged: {'Yes' if result.converged else 'No'}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[method] = None

        print()

    # Compare results
    print("OPTIMIZATION RESULTS COMPARISON:")
    print("-" * 60)
    print(f"{'Method':<12} {'Score':<10} {'Time (s)':<10} {'Iterations':<12} {'Converged'}")
    print("-" * 60)

    for method, result in results.items():
        if result is not None:
            print(
                f"{method:<12} {result.orthogonality_score:<10.6f} "
                f"{result.optimization_time:<10.2f} {result.iterations:<12} "
                f"{'Yes' if result.converged else 'No'}"
            )
        else:
            print(f"{method:<12} {'Failed':<10} {'-':<10} {'-':<12} {'-'}")

    print()

    # Get best result
    best_result = optimizer.get_best_result()
    if best_result is not None:
        print("BEST OPTIMIZATION RESULT:")
        print("-" * 40)
        print(f"Method: {best_result.method_used}")
        print(f"Orthogonality Score: {best_result.orthogonality_score:.6f}")
        print(f"Optimization Time: {best_result.optimization_time:.2f}s")
        print(f"Converged: {'Yes' if best_result.converged else 'No'}")
        print()

        # Test the optimized phases
        print("TESTING OPTIMIZED PHASE CONFIGURATION:")
        print("-" * 40)

        orthogonality_analysis = optimizer.test_orthogonality(best_result.optimal_phases)

        print(f"Number of signals: {orthogonality_analysis['num_signals']}")
        print(f"Maximum cross-correlation: {orthogonality_analysis['max_cross_correlation']:.6f}")
        print(f"Mean cross-correlation: {orthogonality_analysis['mean_cross_correlation']:.6f}")
        print(
            f"Orthogonal pairs: {orthogonality_analysis['orthogonal_pairs']}/{orthogonality_analysis['total_pairs']}"
        )
        print(f"Orthogonality ratio: {orthogonality_analysis['orthogonality_ratio']:.2%}")
        print(
            f"Set is orthogonal: {'Yes' if orthogonality_analysis['is_set_orthogonal'] else 'No'}"
        )
        print()

        # Generate complete signal set
        print("GENERATING ORTHOGONAL SIGNAL SET:")
        print("-" * 40)

        signal_set = optimizer.generate_orthogonal_signal_set(
            num_signals,
            OptimizationConfig(max_iterations=10),  # Quick optimization for demo
            method="gradient",
        )

        print(f"Generated {signal_set.num_signals} orthogonal signals")
        print(f"Signal length: {signal_set.signal_length} samples")
        print(f"Orthogonality score: {signal_set.orthogonality_score:.6f}")
        print(f"Generation timestamp: {signal_set.generation_timestamp}")
        print()

        # Analyze convergence
        print("CONVERGENCE ANALYSIS:")
        print("-" * 40)

        convergence_analysis = optimizer.analyze_convergence(best_result.convergence_history)

        print(f"Initial score: {convergence_analysis['initial_value']:.6f}")
        print(f"Final score: {convergence_analysis['final_value']:.6f}")
        print(f"Total improvement: {convergence_analysis['total_improvement']:.6f}")
        print(f"Convergence point: iteration {convergence_analysis['convergence_point']}")
        print(f"Convergence rate: {convergence_analysis['convergence_rate']:.8f}")
        print(f"Optimization efficiency: {convergence_analysis['optimization_efficiency']:.8f}")
        print()

        # Generate comprehensive report
        print("DETAILED OPTIMIZATION REPORT:")
        print("-" * 40)
        report = optimizer.get_optimization_report(best_result)
        print(report)

        # Demonstrate phase refinement
        print("\nDEMONSTRATING PHASE REFINEMENT:")
        print("-" * 40)

        # Start with random phases
        initial_phases = np.random.uniform(
            0, 2 * np.pi, size=(num_signals, ofdm_config.num_subcarriers)
        )
        initial_score = optimizer._evaluate_phase_configuration(initial_phases)

        print(f"Initial random phases score: {initial_score:.6f}")

        # Refine using gradient optimization
        refinement_config = OptimizationConfig(max_iterations=30, orthogonality_target=0.8)
        refined_result = optimizer.optimize_phase_set(
            initial_phases, refinement_config, method="gradient"
        )

        print(f"Refined phases score: {refined_result.orthogonality_score:.6f}")
        print(f"Improvement: {refined_result.orthogonality_score - initial_score:.6f}")
        print(f"Refinement iterations: {refined_result.iterations}")
        print()

        # Plot convergence histories if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            try:
                plot_convergence_comparison(results)
                print("Convergence plots saved as 'phase_optimization_convergence.png'")
            except Exception as e:
                print(f"Plotting failed: {e}")
        else:
            print("Matplotlib not available - skipping convergence plots")

    else:
        print("No successful optimization results to analyze.")

    # Cleanup
    optimizer.cleanup_resources()

    print("\nPhase optimization demonstration completed!")


def plot_convergence_comparison(results):
    """Plot convergence comparison for different optimization methods."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.figure(figsize=(12, 8))

    # Plot convergence histories
    plt.subplot(2, 2, 1)
    for method, result in results.items():
        if result is not None and len(result.convergence_history) > 0:
            plt.plot(result.convergence_history, label=method, marker="o", markersize=3)

    plt.xlabel("Iteration")
    plt.ylabel("Orthogonality Score")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot final scores
    plt.subplot(2, 2, 2)
    methods = []
    scores = []
    times = []

    for method, result in results.items():
        if result is not None:
            methods.append(method)
            scores.append(result.orthogonality_score)
            times.append(result.optimization_time)

    if methods:
        bars = plt.bar(methods, scores, alpha=0.7)
        plt.ylabel("Final Orthogonality Score")
        plt.title("Final Optimization Scores")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

    # Plot optimization times
    plt.subplot(2, 2, 3)
    if methods:
        bars = plt.bar(methods, times, alpha=0.7, color="orange")
        plt.ylabel("Optimization Time (s)")
        plt.title("Optimization Times")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{time_val:.2f}s",
                ha="center",
                va="bottom",
            )

    # Plot efficiency (score per second)
    plt.subplot(2, 2, 4)
    if methods:
        efficiency = [s / t if t > 0 else 0 for s, t in zip(scores, times)]
        bars = plt.bar(methods, efficiency, alpha=0.7, color="green")
        plt.ylabel("Efficiency (Score/Second)")
        plt.title("Optimization Efficiency")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, eff in zip(bars, efficiency):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{eff:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig("phase_optimization_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()


def demonstrate_advanced_features():
    """Demonstrate advanced phase optimization features."""
    print("\nADVANCED FEATURES DEMONSTRATION:")
    print("=" * 50)

    # Create configuration for advanced demo
    ofdm_config = OFDMConfig(
        num_subcarriers=4,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=40000.0,
        signal_duration=0.002,  # Increased to meet minimum chirp length requirements
    )

    optimizer = PhaseOptimizer(ofdm_config)

    # Demonstrate custom optimization configuration
    print("1. Custom Optimization Configuration:")
    custom_config = OptimizationConfig(
        max_iterations=30,
        convergence_threshold=1e-4,
        orthogonality_target=0.85,
        population_size=15,
        mutation_rate=0.2,
        crossover_rate=0.7,
        early_stopping_patience=10,
        phase_resolution=24,
    )

    result = optimizer.find_orthogonal_phases(2, custom_config, method="hybrid")
    print(f"   Custom config result: {result.orthogonality_score:.6f}")
    print()

    # Demonstrate optimization history tracking
    print("2. Optimization History Tracking:")

    # Run multiple optimizations
    for i in range(3):
        config = OptimizationConfig(max_iterations=10)
        optimizer.find_orthogonal_phases(2, config, method="genetic")

    history = optimizer.get_optimization_history()
    print(f"   Total optimization runs: {len(history)}")

    best = optimizer.get_best_result()
    print(f"   Best score achieved: {best.orthogonality_score:.6f}")
    print(f"   Best method: {best.method_used}")
    print()

    # Demonstrate separation quality measurement
    print("3. Separation Quality Measurement:")

    # Generate signals with optimized phases
    phase_matrix = best.optimal_phases
    signals = []
    for i in range(phase_matrix.shape[0]):
        signal = optimizer.ofdm_generator.generate_single_signal(phase_matrix[i, :])
        signals.append(signal)

    separation_quality = optimizer.get_separation_quality(signals)
    print(f"   Separation quality: {separation_quality:.6f}")
    print()

    # Cleanup
    optimizer.cleanup_resources()


if __name__ == "__main__":
    main()
    demonstrate_advanced_features()
