#!/usr/bin/env python3
"""
MPI Benchmark Runner for HW3
Runs strong and weak scaling experiments and generates CSV files
"""

import subprocess
import sys
import os
import csv
import re
import tempfile
from pathlib import Path

class MPIBenchmarkRunner:
    def __init__(self, executable='./hw3'):
        self.executable = executable
        if not Path(executable).exists():
            print(f"Error: {executable} not found. Run 'make' first.")
            sys.exit(1)
    
    def run_single_test(self, n_size, n_procs):
        """Run a single MPI test with given matrix size and process count"""
        # Write matrix size to input.txt
        with open('input.txt', 'w') as f:
            f.write(f"{n_size}\n")
        
        try:
            # Run MPI program
            result = subprocess.run(
                ['mpirun', '-np', str(n_procs), self.executable],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # Python 3.6 compatible (same as text=True)
                timeout=300  # 5 minute timeout
            )
            
            # Parse output
            output = result.stdout
            time_us = None
            gflops = None
            
            # Look for: Time = XXX us
            time_match = re.search(r'Time = ([\d.]+) us', output)
            if time_match:
                time_us = float(time_match.group(1))
            
            # Look for: Performance = XXX Gflop/s
            perf_match = re.search(r'Performance = ([\d.]+) Gflop/s', output)
            if perf_match:
                gflops = float(perf_match.group(1))
            
            if time_us is None or gflops is None:
                print(f"Warning: Could not parse output for N={n_size}, procs={n_procs}")
                print(f"Output: {output}")
                return None
            
            return {
                'size': n_size,
                'procs': n_procs,
                'time_us': time_us,
                'gflops': gflops
            }
        
        except subprocess.TimeoutExpired:
            print(f"Timeout for N={n_size}, procs={n_procs}")
            return None
        except Exception as e:
            print(f"Error running N={n_size}, procs={n_procs}: {e}")
            return None
    
    def run_strong_scaling(self, matrix_sizes=[1000, 2000, 4000, 8000], 
                          process_counts=[1, 2, 4, 8]):
        """Run strong scaling experiments (fixed size, varying processes)"""
        print("=== Running Strong Scaling Experiments ===")
        print(f"Matrix sizes: {matrix_sizes}")
        print(f"Process counts: {process_counts}")
        results = []
        
        for size in matrix_sizes:
            baseline_time = None
            baseline_gflops = None
            
            for procs in process_counts:
                print(f"Testing N={size}, procs={procs}...", end=' ', flush=True)
                result = self.run_single_test(size, procs)
                
                if result is None:
                    print("FAILED")
                    continue
                
                # Calculate speedup and efficiency
                if procs == 1:
                    baseline_time = result['time_us']
                    baseline_gflops = result['gflops']
                    speedup = 1.0
                    efficiency = 100.0
                else:
                    if baseline_time:
                        speedup = baseline_time / result['time_us']
                        efficiency = (speedup / procs) * 100.0
                    else:
                        speedup = 0.0
                        efficiency = 0.0
                
                result['speedup'] = speedup
                result['efficiency'] = efficiency
                results.append(result)
                
                print(f"Time={result['time_us']:.1f}us, Perf={result['gflops']:.2f} Gflop/s, "
                      f"Speedup={speedup:.2f}x, Eff={efficiency:.1f}%")
        
        return results
    
    def run_weak_scaling(self, base_sizes=[1000], process_counts=[1, 2, 4, 8]):
        """Run weak scaling experiments (work per process constant)"""
        print("\n=== Running Weak Scaling Experiments ===")
        print(f"Base sizes: {base_sizes}")
        print(f"Process counts: {process_counts}")
        results = []
        
        for base_n in base_sizes:
            work_per_proc = base_n * base_n  # Work = N^2
            baseline_gflops = None
            
            for procs in process_counts:
                # Scale problem size to maintain constant work per process
                # For weak scaling: total_work = procs * work_per_proc
                # So total_N^2 = procs * base_N^2
                # Therefore: total_N = base_N * sqrt(procs)
                import math
                total_n = int(base_n * math.sqrt(procs))
                
                print(f"Testing base_N={base_n}, procs={procs}, total_N={total_n}...", 
                      end=' ', flush=True)
                result = self.run_single_test(total_n, procs)
                
                if result is None:
                    print("FAILED")
                    continue
                
                # Calculate efficiency (should be constant for ideal weak scaling)
                if procs == 1:
                    baseline_gflops = result['gflops']
                    efficiency = 100.0
                else:
                    if baseline_gflops:
                        efficiency = (result['gflops'] / baseline_gflops) * 100.0
                    else:
                        efficiency = 0.0
                
                result['work_per_proc'] = work_per_proc
                result['efficiency'] = efficiency
                results.append(result)
                
                print(f"Time={result['time_us']:.1f}us, Perf={result['gflops']:.2f} Gflop/s, "
                      f"Eff={efficiency:.1f}%")
        
        return results
    
    def save_to_csv(self, results, filename):
        """Save results to CSV file atomically to prevent corruption"""
        if not results:
            print(f"Warning: No results to save to {filename}")
            return
        
        # Get all keys from first result
        fieldnames = list(results[0].keys())
        
        # Use atomic write to prevent corruption
        temp_fd, temp_path = tempfile.mkstemp(dir='.', suffix='.csv')
        try:
            with os.fdopen(temp_fd, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            # Atomic rename - if we crash before this, no data is lost
            os.rename(temp_path, filename)
            print(f"\nSaved {len(results)} results to {filename}")
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

def main():
    print("MPI Benchmark Runner for HW3")
    print("=" * 50)
    
    # Check if hw3 executable exists
    if not Path('./hw3').exists():
        print("\nError: ./hw3 not found!")
        print("Please run 'make' first to build the executable.")
        sys.exit(1)
    
    # Check if mpirun is available
    try:
        subprocess.run(['mpirun', '--version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nError: mpirun not found!")
        print("Please ensure MPI is installed and in your PATH.")
        print("Try: module load mpi/openmpi-x86_64")
        sys.exit(1)
    
    runner = MPIBenchmarkRunner()
    
    # Run strong scaling experiments
    print("\nStarting strong scaling tests...")
    print("This will test multiple matrix sizes with varying process counts.")
    
    # Determine max processes based on available resources
    # Default: test up to 8 processes (suitable for single node)
    # For multi-node: can go higher (16, 32, etc.)
    import os
    max_procs = int(os.environ.get('HW3_MAX_PROCS', '8'))
    process_list = [1, 2, 4]
    if max_procs >= 8:
        process_list.append(8)
    if max_procs >= 16:
        process_list.extend([16])
    if max_procs >= 32:
        process_list.extend([32])
    
    strong_results = runner.run_strong_scaling(
        matrix_sizes=[1000, 2000, 4000, 8000],
        process_counts=process_list
    )
    runner.save_to_csv(strong_results, 'strong_scaling_results.csv')
    
    # Run weak scaling experiments
    print("\nStarting weak scaling tests...")
    print("This maintains constant work per process.")
    weak_results = runner.run_weak_scaling(
        base_sizes=[1000],
        process_counts=process_list
    )
    runner.save_to_csv(weak_results, 'weak_scaling_results.csv')
    
    print("\n" + "=" * 50)
    print("Benchmark complete!")
    print("\nGenerated files:")
    print("  - strong_scaling_results.csv")
    print("  - weak_scaling_results.csv")
    print("\nNext steps:")
    print("  1. Run 'python3 report.py' to generate hw3.pdf")

if __name__ == '__main__':
    main()
