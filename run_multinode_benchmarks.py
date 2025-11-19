#!/usr/bin/env python3
"""
Multi-Node MPI Benchmark Runner for HW3
Runs tests with 16, 24, 32, 40 processes across multiple nodes
"""

import subprocess
import sys
import csv
import re
import os
import tempfile
from pathlib import Path

class MultiNodeBenchmarkRunner:
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
                universal_newlines=True,
                timeout=600  # 10 minute timeout for large tests
            )
            
            output = result.stdout
            time_us = None
            gflops = None
            
            # Parse output
            time_match = re.search(r'Time = ([\d.]+) us', output)
            if time_match:
                time_us = float(time_match.group(1))
            
            perf_match = re.search(r'Performance = ([\d.]+) Gflop/s', output)
            if perf_match:
                gflops = float(perf_match.group(1))
            
            if time_us is None or gflops is None:
                print(f"Warning: Could not parse output for N={n_size}, procs={n_procs}")
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
    
    def run_multinode_strong_scaling(self):
        """Run strong scaling with 16, 24, 32 processes (2-4 node multi-node)"""
        print("=== Running Multi-Node Strong Scaling ===")
        print("Matrix sizes: [4000, 8000, 16000]")
        print("Process counts: [16, 24, 32]")
        
        matrix_sizes = [4000, 8000, 16000]
        process_counts = [16, 24, 32]
        
        # Load existing single-node data
        existing_data = []
        try:
            with open('strong_scaling_results.csv', 'r') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
        except FileNotFoundError:
            pass
        
        results = existing_data.copy()
        
        for size in matrix_sizes:
            baseline_time = None
            
            # Find baseline from single-process run
            for row in existing_data:
                if int(row['size']) == size and int(row['procs']) == 1:
                    baseline_time = float(row['time_us'])
                    break
            
            for procs in process_counts:
                print(f"Testing N={size}, procs={procs}...", end=' ', flush=True)
                result = self.run_single_test(size, procs)
                
                if result is None:
                    print("FAILED")
                    continue
                
                # Calculate speedup and efficiency
                if baseline_time:
                    speedup = baseline_time / result['time_us']
                    efficiency = (speedup / procs) * 100.0
                else:
                    speedup = 1.0
                    efficiency = 100.0
                
                result['speedup'] = speedup
                result['efficiency'] = efficiency
                
                print(f"Time={result['time_us']:.2f}us, Perf={result['gflops']:.2f} Gflop/s, Speedup={speedup:.2f}x, Eff={efficiency:.1f}%")
                results.append(result)
        
        # Write combined results atomically to prevent corruption
        temp_fd, temp_path = tempfile.mkstemp(dir='.', suffix='.csv')
        try:
            with os.fdopen(temp_fd, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['size', 'procs', 'time_us', 'gflops', 'speedup', 'efficiency'])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        'size': r['size'],
                        'procs': r['procs'],
                        'time_us': f"{r['time_us']:.1f}" if isinstance(r['time_us'], float) else r['time_us'],
                        'gflops': f"{r['gflops']:.5f}" if isinstance(r['gflops'], float) else r['gflops'],
                        'speedup': f"{r['speedup']:.1f}" if isinstance(r['speedup'], float) else r['speedup'],
                        'efficiency': f"{r['efficiency']:.1f}" if isinstance(r['efficiency'], float) else r['efficiency']
                    })
            # Atomic rename - if we crash before this, old file is preserved
            os.rename(temp_path, 'strong_scaling_results.csv')
            print(f"\nUpdated strong_scaling_results.csv with {len(results)} total entries")
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def run_multinode_weak_scaling(self):
        """Run weak scaling with 16, 24, 32 processes (2-4 node multi-node)"""
        print("\n=== Running Multi-Node Weak Scaling ===")
        print("Base work per process: ~1M elements")
        print("Process counts: [16, 24, 32]")
        
        # Load existing single-node data
        existing_data = []
        try:
            with open('weak_scaling_results.csv', 'r') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
        except FileNotFoundError:
            pass
        
        results = existing_data.copy()
        
        # Base: 1M elements per process → N = sqrt(1M * procs)
        base_work = 1000000
        baseline_perf = None
        
        # Find baseline
        for row in existing_data:
            if int(row['procs']) == 1:
                baseline_perf = float(row['gflops'])
                break
        
        process_counts = [16, 24, 32]
        for procs in process_counts:
            total_elements = base_work * procs
            n_size = int(total_elements ** 0.5)
            
            print(f"Testing procs={procs}, N={n_size}...", end=' ', flush=True)
            result = self.run_single_test(n_size, procs)
            
            if result is None:
                print("FAILED")
                continue
            
            result['work_per_proc'] = base_work
            
            if baseline_perf:
                efficiency = (result['gflops'] / baseline_perf) * 100.0
            else:
                efficiency = 100.0
            
            result['efficiency'] = efficiency
            
            print(f"Time={result['time_us']:.2f}us, Perf={result['gflops']:.2f} Gflop/s, Eff={efficiency:.1f}%")
            results.append(result)
        
        # Write combined results atomically to prevent corruption
        temp_fd, temp_path = tempfile.mkstemp(dir='.', suffix='.csv')
        try:
            with os.fdopen(temp_fd, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['size', 'procs', 'time_us', 'gflops', 'work_per_proc', 'efficiency'])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        'size': r['size'],
                        'procs': r['procs'],
                        'time_us': f"{r['time_us']:.1f}" if isinstance(r['time_us'], float) else r['time_us'],
                        'gflops': f"{r['gflops']:.5f}" if isinstance(r['gflops'], float) else r['gflops'],
                        'work_per_proc': r.get('work_per_proc', r['size'] // r['procs']),
                        'efficiency': f"{r['efficiency']:.1f}" if isinstance(r['efficiency'], float) else r['efficiency']
                    })
            # Atomic rename - if we crash before this, old file is preserved
            os.rename(temp_path, 'weak_scaling_results.csv')
            print(f"\nUpdated weak_scaling_results.csv with {len(results)} total entries")
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e

def main():
    print("Multi-Node MPI Benchmark Runner")
    print("=" * 50)
    print("\nThis script extends existing single-node results with multi-node data.")
    print("Requires: MPI job with multiple nodes allocated\n")
    
    runner = MultiNodeBenchmarkRunner()
    
    # Run multi-node strong scaling
    runner.run_multinode_strong_scaling()
    
    # Run multi-node weak scaling
    runner.run_multinode_weak_scaling()
    
    print("\n" + "=" * 50)
    print("Multi-node benchmarks complete!")
    print("\nNext: Run 'python3 report.py' to regenerate hw3.pdf with multi-node data")

if __name__ == '__main__':
    main()
