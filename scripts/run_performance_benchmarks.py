#!/usr/bin/env python3
"""
Performance Benchmark Runner

This script runs comprehensive performance benchmarks for the OFDM Chirp Generator,
including GPU vs CPU comparisons, scalability tests, and regression detection.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_performance_benchmarks import (
    PerformanceBenchmarkSuite, BenchmarkResult, ScalabilityTestConfig
)


def run_pytest_benchmarks(output_file: str, test_filter: str = None) -> bool:
    """Run pytest performance benchmarks.
    
    Args:
        output_file: File to save results
        test_filter: Optional test filter pattern
        
    Returns:
        True if benchmarks completed successfully
    """
    cmd = [
        "uv", "run", "pytest", 
        "tests/test_performance_benchmarks.py",
        "-v", "-m", "performance",
        "--tb=short"
    ]
    
    if test_filter:
        cmd.extend(["-k", test_filter])
    
    print(f"Running performance benchmarks...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return False


def run_custom_benchmarks(output_file: str) -> PerformanceBenchmarkSuite:
    """Run custom performance benchmarks.
    
    Args:
        output_file: File to save results
        
    Returns:
        Benchmark suite with results
    """
    from ofdm_chirp_generator.models import OFDMConfig
    from ofdm_chirp_generator.gpu_backend import GPUBackend
    from ofdm_chirp_generator.ofdm_generator import OFDMGenerator
    from ofdm_chirp_generator.orthogonal_signal_generator import OrthogonalSignalGenerator
    from ofdm_chirp_generator.performance_optimizer import PerformanceOptimizer
    import numpy as np
    
    suite = PerformanceBenchmarkSuite(output_file)
    
    print("Running custom performance benchmarks...")
    
    # Standard configuration
    config = OFDMConfig(
        num_subcarriers=8,
        subcarrier_spacing=1000.0,
        bandwidth_per_subcarrier=800.0,
        center_frequency=10000.0,
        sampling_rate=50000.0,
        signal_duration=0.002
    )
    
    # Test 1: Basic signal generation comparison
    print("1. Basic signal generation (GPU vs CPU)")
    for backend_name in ['CPU', 'GPU']:
        force_cpu = (backend_name == 'CPU')
        gpu_backend = GPUBackend(force_cpu=force_cpu)
        ofdm_generator = OFDMGenerator(config, gpu_backend)
        
        try:
            phase_array = np.random.uniform(0, 2*np.pi, config.num_subcarriers)
            
            start_time = time.time()
            signal = ofdm_generator.generate_single_signal(phase_array)
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                test_name="basic_signal_generation",
                backend=backend_name,
                parameters={
                    "num_subcarriers": config.num_subcarriers,
                    "signal_length": len(signal)
                },
                execution_time=execution_time,
                throughput=1.0 / execution_time if execution_time > 0 else 0,
                memory_usage_mb=None,
                success=True
            )
            
            print(f"  {backend_name}: {execution_time:.4f}s")
            
        except Exception as e:
            result = BenchmarkResult(
                test_name="basic_signal_generation",
                backend=backend_name,
                parameters={},
                execution_time=0,
                throughput=None,
                memory_usage_mb=None,
                success=False,
                error_message=str(e)
            )
            print(f"  {backend_name}: FAILED - {e}")
        
        finally:
            ofdm_generator.cleanup_resources()
        
        suite.add_result(result)
    
    # Test 2: Batch processing performance
    print("2. Batch processing performance")
    batch_sizes = [1, 5, 10, 20]
    
    for batch_size in batch_sizes:
        for backend_name in ['CPU', 'GPU']:
            force_cpu = (backend_name == 'CPU')
            gpu_backend = GPUBackend(force_cpu=force_cpu)
            optimizer = PerformanceOptimizer(gpu_backend)
            ofdm_generator = OFDMGenerator(config, gpu_backend)
            
            try:
                def generate_signal(phase_array):
                    return ofdm_generator.generate_single_signal(phase_array)
                
                phase_arrays = [
                    np.random.uniform(0, 2*np.pi, config.num_subcarriers)
                    for _ in range(batch_size)
                ]
                
                start_time = time.time()
                results = optimizer.optimize_signal_generation(
                    generate_signal, phase_arrays, f"batch_{batch_size}"
                )
                execution_time = time.time() - start_time
                
                throughput = batch_size / execution_time if execution_time > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"batch_processing_{batch_size}",
                    backend=backend_name,
                    parameters={
                        "batch_size": batch_size,
                        "num_subcarriers": config.num_subcarriers
                    },
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage_mb=None,
                    success=True
                )
                
                print(f"  Batch {batch_size} ({backend_name}): {execution_time:.4f}s, {throughput:.1f} signals/s")
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"batch_processing_{batch_size}",
                    backend=backend_name,
                    parameters={"batch_size": batch_size},
                    execution_time=0,
                    throughput=None,
                    memory_usage_mb=None,
                    success=False,
                    error_message=str(e)
                )
                print(f"  Batch {batch_size} ({backend_name}): FAILED - {e}")
            
            finally:
                optimizer.cleanup_resources()
                ofdm_generator.cleanup_resources()
            
            suite.add_result(result)
    
    # Test 3: Orthogonal signal set generation
    print("3. Orthogonal signal set generation")
    set_sizes = [2, 4, 8]
    
    for set_size in set_sizes:
        for backend_name in ['CPU', 'GPU']:
            force_cpu = (backend_name == 'CPU')
            gpu_backend = GPUBackend(force_cpu=force_cpu)
            orthogonal_generator = OrthogonalSignalGenerator(config, gpu_backend)
            
            try:
                start_time = time.time()
                signal_set = orthogonal_generator.generate_orthogonal_signal_set(set_size)
                execution_time = time.time() - start_time
                
                throughput = set_size / execution_time if execution_time > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"orthogonal_set_{set_size}",
                    backend=backend_name,
                    parameters={
                        "set_size": set_size,
                        "orthogonality_score": signal_set.orthogonality_score
                    },
                    execution_time=execution_time,
                    throughput=throughput,
                    memory_usage_mb=None,
                    success=True
                )
                
                print(f"  Set {set_size} ({backend_name}): {execution_time:.4f}s, score: {signal_set.orthogonality_score:.4f}")
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"orthogonal_set_{set_size}",
                    backend=backend_name,
                    parameters={"set_size": set_size},
                    execution_time=0,
                    throughput=None,
                    memory_usage_mb=None,
                    success=False,
                    error_message=str(e)
                )
                print(f"  Set {set_size} ({backend_name}): FAILED - {e}")
            
            finally:
                orthogonal_generator.cleanup_resources()
            
            suite.add_result(result)
    
    # Test 4: Scalability with subcarrier count
    print("4. Scalability with subcarrier count")
    subcarrier_counts = [4, 8, 16, 32]
    
    for num_subcarriers in subcarrier_counts:
        test_config = OFDMConfig(
            num_subcarriers=num_subcarriers,
            subcarrier_spacing=config.subcarrier_spacing,
            bandwidth_per_subcarrier=config.bandwidth_per_subcarrier,
            center_frequency=config.center_frequency,
            sampling_rate=config.sampling_rate,
            signal_duration=0.001  # Shorter for scalability tests
        )
        
        # Test with CPU for consistent results
        gpu_backend = GPUBackend(force_cpu=True)
        ofdm_generator = OFDMGenerator(test_config, gpu_backend)
        
        try:
            phase_array = np.random.uniform(0, 2*np.pi, num_subcarriers)
            
            start_time = time.time()
            signal = ofdm_generator.generate_single_signal(phase_array)
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                test_name=f"scalability_subcarriers_{num_subcarriers}",
                backend="CPU",
                parameters={
                    "num_subcarriers": num_subcarriers,
                    "signal_length": len(signal),
                    "complexity_factor": num_subcarriers / 8.0  # Relative to base case
                },
                execution_time=execution_time,
                throughput=1.0 / execution_time if execution_time > 0 else 0,
                memory_usage_mb=None,
                success=True
            )
            
            print(f"  {num_subcarriers} subcarriers: {execution_time:.4f}s")
            
        except Exception as e:
            result = BenchmarkResult(
                test_name=f"scalability_subcarriers_{num_subcarriers}",
                backend="CPU",
                parameters={"num_subcarriers": num_subcarriers},
                execution_time=0,
                throughput=None,
                memory_usage_mb=None,
                success=False,
                error_message=str(e)
            )
            print(f"  {num_subcarriers} subcarriers: FAILED - {e}")
        
        finally:
            ofdm_generator.cleanup_resources()
        
        suite.add_result(result)
    
    return suite


def generate_performance_report(suite: PerformanceBenchmarkSuite, output_dir: str):
    """Generate comprehensive performance report.
    
    Args:
        suite: Benchmark suite with results
        output_dir: Directory to save report files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary
    summary = suite.get_performance_summary()
    
    # Create detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "detailed_results": [
            {
                "test_name": result.test_name,
                "backend": result.backend,
                "parameters": result.parameters,
                "execution_time": result.execution_time,
                "throughput": result.throughput,
                "success": result.success,
                "error_message": result.error_message
            }
            for result in suite.results
        ]
    }
    
    # Save detailed report
    report_file = os.path.join(output_dir, "performance_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    md_file = os.path.join(output_dir, "performance_report.md")
    with open(md_file, 'w') as f:
        f.write("# OFDM Chirp Generator Performance Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total tests: {summary.get('total_tests', 0)}\n")
        f.write(f"- Successful tests: {summary.get('successful_tests', 0)}\n")
        f.write(f"- Failed tests: {summary.get('failed_tests', 0)}\n")
        f.write(f"- Success rate: {summary.get('successful_tests', 0) / max(summary.get('total_tests', 1), 1) * 100:.1f}%\n\n")
        
        if "gpu_vs_cpu" in summary:
            gpu_vs_cpu = summary["gpu_vs_cpu"]
            f.write("## GPU vs CPU Performance\n\n")
            f.write(f"- GPU speedup: {gpu_vs_cpu['speedup']:.2f}x\n")
            f.write(f"- GPU faster: {'Yes' if gpu_vs_cpu['gpu_faster'] else 'No'}\n\n")
        
        if "gpu_performance" in summary:
            gpu_perf = summary["gpu_performance"]
            f.write("## GPU Performance\n\n")
            f.write(f"- Average execution time: {gpu_perf['average_time']:.4f}s\n")
            f.write(f"- Median execution time: {gpu_perf['median_time']:.4f}s\n")
            f.write(f"- Min execution time: {gpu_perf['min_time']:.4f}s\n")
            f.write(f"- Max execution time: {gpu_perf['max_time']:.4f}s\n")
            if 'average_throughput' in gpu_perf:
                f.write(f"- Average throughput: {gpu_perf['average_throughput']:.1f} ops/s\n")
            f.write("\n")
        
        if "cpu_performance" in summary:
            cpu_perf = summary["cpu_performance"]
            f.write("## CPU Performance\n\n")
            f.write(f"- Average execution time: {cpu_perf['average_time']:.4f}s\n")
            f.write(f"- Median execution time: {cpu_perf['median_time']:.4f}s\n")
            f.write(f"- Min execution time: {cpu_perf['min_time']:.4f}s\n")
            f.write(f"- Max execution time: {cpu_perf['max_time']:.4f}s\n")
            if 'average_throughput' in cpu_perf:
                f.write(f"- Average throughput: {cpu_perf['average_throughput']:.1f} ops/s\n")
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| Test Name | Backend | Execution Time (s) | Throughput | Success |\n")
        f.write("|-----------|---------|-------------------|------------|----------|\n")
        
        for result in suite.results:
            throughput_str = f"{result.throughput:.1f}" if result.throughput else "N/A"
            success_str = "✓" if result.success else "✗"
            f.write(f"| {result.test_name} | {result.backend} | {result.execution_time:.4f} | {throughput_str} | {success_str} |\n")
    
    print(f"Performance report saved to: {output_dir}")
    print(f"  - JSON report: {report_file}")
    print(f"  - Markdown report: {md_file}")


def check_regression(suite: PerformanceBenchmarkSuite, tolerance: float = 0.2):
    """Check for performance regression and print results.
    
    Args:
        suite: Benchmark suite with results
        tolerance: Acceptable performance degradation threshold
    """
    regression_analysis = suite.check_regression(tolerance)
    
    print("\n" + "="*60)
    print("PERFORMANCE REGRESSION ANALYSIS")
    print("="*60)
    
    if "message" in regression_analysis:
        print(regression_analysis["message"])
        return
    
    print(f"Tolerance: {regression_analysis['tolerance_percent']:.1f}%")
    print(f"Regressions detected: {regression_analysis['regressions_detected']}")
    print(f"Improvements detected: {regression_analysis['improvements_detected']}")
    
    if regression_analysis["regressions"]:
        print("\nREGRESSIONS:")
        for reg in regression_analysis["regressions"]:
            print(f"  - {reg['test_name']} ({reg['backend']}): {reg['degradation_percent']:.1f}% slower")
            print(f"    Baseline: {reg['baseline_time']:.4f}s → Current: {reg['current_time']:.4f}s")
    
    if regression_analysis["improvements"]:
        print("\nIMPROVEMENTS:")
        for imp in regression_analysis["improvements"]:
            print(f"  + {imp['test_name']} ({imp['backend']}): {imp['improvement_percent']:.1f}% faster")
            print(f"    Baseline: {imp['baseline_time']:.4f}s → Current: {imp['current_time']:.4f}s")
    
    if not regression_analysis["regressions"] and not regression_analysis["improvements"]:
        print("No significant performance changes detected.")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run OFDM Chirp Generator performance benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results", 
                       help="Directory to save benchmark results")
    parser.add_argument("--test-filter", help="Filter tests by pattern")
    parser.add_argument("--custom-only", action="store_true", 
                       help="Run only custom benchmarks (skip pytest)")
    parser.add_argument("--pytest-only", action="store_true", 
                       help="Run only pytest benchmarks (skip custom)")
    parser.add_argument("--save-baseline", action="store_true", 
                       help="Save results as baseline for regression testing")
    parser.add_argument("--check-regression", action="store_true", 
                       help="Check for performance regression against baseline")
    parser.add_argument("--tolerance", type=float, default=0.2, 
                       help="Regression tolerance (default: 0.2 = 20%%)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Results file
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    
    print("OFDM Chirp Generator Performance Benchmarks")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Results file: {results_file}")
    
    # Initialize benchmark suite
    suite = PerformanceBenchmarkSuite(results_file)
    
    # Run pytest benchmarks
    if not args.custom_only:
        print("\nRunning pytest performance benchmarks...")
        pytest_success = run_pytest_benchmarks(results_file, args.test_filter)
        if not pytest_success:
            print("Warning: Some pytest benchmarks failed")
    
    # Run custom benchmarks
    if not args.pytest_only:
        print("\nRunning custom performance benchmarks...")
        suite = run_custom_benchmarks(results_file)
    
    # Save results
    suite.save_results()
    
    # Generate report
    generate_performance_report(suite, args.output_dir)
    
    # Save as baseline if requested
    if args.save_baseline:
        suite.save_as_baseline()
        print(f"\nResults saved as baseline for future regression testing")
    
    # Check for regression if requested
    if args.check_regression:
        check_regression(suite, args.tolerance)
    
    # Print final summary
    summary = suite.get_performance_summary()
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total tests: {summary.get('total_tests', 0)}")
    print(f"Successful: {summary.get('successful_tests', 0)}")
    print(f"Failed: {summary.get('failed_tests', 0)}")
    
    if summary.get('total_tests', 0) > 0:
        success_rate = summary.get('successful_tests', 0) / summary['total_tests'] * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if "gpu_vs_cpu" in summary:
        gpu_vs_cpu = summary["gpu_vs_cpu"]
        print(f"GPU speedup: {gpu_vs_cpu['speedup']:.2f}x")
    
    print(f"\nDetailed results available in: {args.output_dir}")


if __name__ == "__main__":
    main()