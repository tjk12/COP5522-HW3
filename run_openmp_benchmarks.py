#!/usr/bin/env python3
"""
OpenMP Benchmark Runner for comparison with MPI
"""

import subprocess
import sys
import os
import csv
import re
from pathlib import Path

class OpenMPBenchmarkRunner:
    def __init__(self, executable='./hw3_openmp'):
        self.executable = executable
        if not Path(executable).exists():
            print(f"Error: {executable} not found. Run 'make openmp' first.")
            sys.exit(1)
    
    def run_single_test(self, n_size, n_threads):
        """Run a single OpenMP test with given matrix size and thread count"""
        # Write matrix size to input.txt
        with open('input.txt', 'w') as f:
            f.write(f"{n_size}\n")
        
        try:
            # Set OMP_NUM_THREADS environment
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(n_threads)
            
            # Run OpenMP program
            result = subprocess.run(
                [self.executable, str(n_threads)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # Python 3.6 compatible (same as text=True)
                timeout=300,
                env=env
            )
            
            # Parse output
            output = result.stdout
            time_us = None
            gflops = None
            
            time_match = re.search(r'Time = ([\d.]+) us', output)
            if time_match:
                time_us = float(time_match.group(1))
            
            perf_match = re.search(r'Performance = ([\d.]+) Gflop/s', output)
            if perf_match:
                gflops = float(perf_match.group(1))
            
            if time_us is None or gflops is None:
                print(f"Warning: Could not parse output for N={n_size}, threads={n_threads}")
                return None
            
            return {
                'size': n_size,
                'threads': n_threads,
                'time_us': time_us,
                'gflops': gflops
            }
        
        except subprocess.TimeoutExpired:
            print(f"Timeout for N={n_size}, threads={n_threads}")
            return None
        except Exception as e:
            print(f"Error running N={n_size}, threads={n_threads}: {e}")
            return None
    
    def run_openmp_scaling(self, matrix_sizes=[1000, 2000, 4000, 8000], 
                           thread_counts=[1, 2, 4, 8]):
        """Run OpenMP scaling experiments"""
        print("=== Running OpenMP Scaling Experiments ===")
        print(f"Matrix sizes: {matrix_sizes}")
        print(f"Thread counts: {thread_counts}")
        results = []
        
        for size in matrix_sizes:
            baseline_time = None
            
            for threads in thread_counts:
                print(f"Testing N={size}, threads={threads}...", end=' ', flush=True)
                result = self.run_single_test(size, threads)
                
                if result is None:
                    print("FAILED")
                    continue
                
                # Calculate speedup and efficiency
                if threads == 1:
                    baseline_time = result['time_us']
                    speedup = 1.0
                    efficiency = 100.0
                else:
                    if baseline_time:
                        speedup = baseline_time / result['time_us']
                        efficiency = (speedup / threads) * 100.0
                    else:
                        speedup = 0.0
                        efficiency = 0.0
                
                result['speedup'] = speedup
                result['efficiency'] = efficiency
                results.append(result)
                
                print(f"Time={result['time_us']:.1f}us, Perf={result['gflops']:.2f} Gflop/s, "
                      f"Speedup={speedup:.2f}x, Eff={efficiency:.1f}%")
        
        return results
    
    def save_to_csv(self, results, filename):
        """Save results to CSV file"""
        if not results:
            print(f"Warning: No results to save to {filename}")
            return
        
        fieldnames = list(results[0].keys())
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nSaved {len(results)} results to {filename}")

def main():
    print("OpenMP Benchmark Runner")
    print("=" * 50)
    
    # Check if hw3_openmp executable exists
    if not Path('./hw3_openmp').exists():
        print("\nError: ./hw3_openmp not found!")
        print("Please run 'make openmp' first to build the executable.")
        sys.exit(1)
    
    runner = OpenMPBenchmarkRunner()
    
    # Determine thread counts to test
    import os
    max_threads = int(os.environ.get('HW3_MAX_THREADS', '8'))
    thread_list = [1, 2, 4]
    if max_threads >= 8:
        thread_list.append(8)
    if max_threads >= 16:
        thread_list.extend([16])
    
    print(f"\nTesting with thread counts: {thread_list}")
    
    # Run OpenMP experiments
    openmp_results = runner.run_openmp_scaling(
        matrix_sizes=[1000, 2000, 4000, 8000],
        thread_counts=thread_list
    )
    runner.save_to_csv(openmp_results, 'openmp_results.csv')
    
    print("\n" + "=" * 50)
    print("OpenMP benchmark complete!")
    print("\nGenerated files:")
    print("  - openmp_results.csv")
    print("\nNext: Run 'python3 run_benchmarks.py' for MPI results")
    print("Then: Run 'python3 report.py' to generate hw3.pdf with comparison")

if __name__ == '__main__':
    main()
